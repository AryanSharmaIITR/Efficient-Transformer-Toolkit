"""Quantize a trained transformer checkpoint for efficient inference.

Loads a checkpoint produced by ``scripts/run_train.py`` (or any
``load_checkpoint``-compatible ``.pt`` file), applies one of the
post-training quantization strategies in ``src/inference/quantization.py``,
runs a quick sanity forward pass, and saves the quantized model.

Usage
-----
```
# Dynamic int8 quantization -- no calibration data needed, good default.
python -m scripts.run_quantize \\
    --checkpoint outputs/checkpoints/final.pt \\
    --method dynamic \\
    --output outputs/quantized/model_int8.pt

# Dynamic 8-bit quantization on GPU via bitsandbytes (falls back to CPU
# int8 automatically if bitsandbytes isn't installed).
python -m scripts.run_quantize \\
    --checkpoint outputs/checkpoints/final.pt \\
    --method dynamic --device cuda \\
    --output outputs/quantized/model_int8_cuda.pt

# Static int8 quantization with calibration data (best accuracy/size).
python -m scripts.run_quantize \\
    --checkpoint outputs/checkpoints/final.pt \\
    --method static \\
    --calibration_data datasets/wikitext-2/wiki.valid.tokens \\
    --output outputs/quantized/model_static.pt

# FP16 conversion for GPU (Tensor Core) inference.
python -m scripts.run_quantize \\
    --checkpoint outputs/checkpoints/final.pt \\
    --method fp16 --device cuda \\
    --output outputs/quantized/model_fp16.pt
```

The quantized model can then be loaded back and used with
``src/inference/engine.py``'s ``InferenceEngine`` -- see ``--output`` below
and the README's "Quantization" section.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.data.collator import DataCollator
from src.data.dataset import TextDataset
from src.data.tokenizer import Tokenizer
from src.inference.quantization import convert_to_fp16, quantize_dynamic, quantize_static
from src.models.transformer import Transformer, TransformerConfig
from src.training.checkpoint import load_checkpoint
from src.utils.logger import setup_logger
from src.utils.seeding import set_seed

logger: logging.Logger | None = None


# ======================================================================
# CLI
# ======================================================================

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Quantize a trained transformer checkpoint.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Checkpoint / model -----------------------------------------------
    p.add_argument("--checkpoint", type=str, required=True, help="Path to a .pt checkpoint (from run_train.py).")
    p.add_argument("--config_json", type=str, default=None, help="Optional config.json (same dir as model save).")
    p.add_argument("--model_type", choices=["encoder_only", "decoder_only", "encoder_decoder"], default="decoder_only")
    p.add_argument("--attn_type", type=str, default="flashv2")
    p.add_argument("--pos_encoding", type=str, default="rope", choices=["rope", "sinusoidal", "learned"])
    p.add_argument("--n_heads", type=int, default=12)
    p.add_argument("--n_layers", type=int, default=6)
    p.add_argument("--d_model", type=int, default=768)
    p.add_argument("--vocab_size", type=int, default=50257)

    # Quantization method ------------------------------------------------
    p.add_argument("--method", type=str, required=True, choices=["dynamic", "static", "fp16"], help="Quantization strategy to apply.")
    p.add_argument("--dtype", type=str, default="qint8", choices=["qint8", "float16"], help="Target dtype for --method dynamic.")
    p.add_argument("--exclude", type=str, nargs="+", default=None, help="Module name prefixes to skip (e.g. lm_head).")

    # Static-quantization calibration -----------------------------------
    p.add_argument("--calibration_data", type=str, default=None, help="Plain-text file used to build calibration batches for --method static.")
    p.add_argument("--tokenizer_name", type=str, default="gpt2")
    p.add_argument("--num_calib_batches", type=int, default=32)
    p.add_argument("--calib_batch_size", type=int, default=8)
    p.add_argument("--calib_seq_len", type=int, default=512)

    # Output ---------------------------------------------------------
    p.add_argument("--output", type=str, default=None, help="Destination .pt path (default: <checkpoint_dir>/quantized_<method>.pt).")
    p.add_argument("--device", type=str, default="cpu", help="Device to load/quantize on ('cuda' enables bitsandbytes int8 for --method dynamic, or is required for --method fp16).")
    p.add_argument("--seed", type=int, default=42)

    return p


def _load_model_config(args: argparse.Namespace) -> TransformerConfig:
    """Load TransformerConfig from a config.json if available, else from CLI."""
    config_path = args.config_json
    if config_path is None and Path(args.checkpoint).parent.joinpath("config.json").exists():
        config_path = str(Path(args.checkpoint).parent / "config.json")

    if config_path and Path(config_path).exists():
        with open(config_path, "r", encoding="utf-8") as fh:
            cfg_dict = json.load(fh)
        # Transformer.save_pretrained() stores two bookkeeping keys
        # (_use_encoder/_use_decoder) alongside the real TransformerConfig
        # fields -- passing them through would crash TransformerConfig(**cfg_dict).
        cfg_dict.pop("_use_encoder", None)
        cfg_dict.pop("_use_decoder", None)
        logger.info("Loaded model config from %s", config_path)
        return TransformerConfig(**cfg_dict)

    return TransformerConfig(
        vocab_size=args.vocab_size,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        attn_type=args.attn_type,
        pos_encoding=args.pos_encoding,
    )


def _build_model(args: argparse.Namespace, config: TransformerConfig) -> Transformer:
    use_enc = args.model_type in ("encoder_only", "encoder_decoder")
    use_dec = args.model_type in ("decoder_only", "encoder_decoder")
    return Transformer(config, use_encoder=use_enc, use_decoder=use_dec)


def _resolve_device(method: str, requested: str) -> str:
    """Pick the actual device to quantize/run on for *method*.

    Static quantization (PyTorch's fbgemm backend) and plain dynamic int8
    quantization only run on CPU; the bitsandbytes dynamic-quantization path
    is the only one that actually uses CUDA.
    """
    if method == "static" and requested != "cpu":
        logger.warning("Static quantization only supports CPU; ignoring --device=%s.", requested)
        return "cpu"
    if method == "dynamic" and requested == "cuda":
        if importlib.util.find_spec("bitsandbytes") is None:
            logger.warning("--device=cuda for dynamic quantization requires bitsandbytes (not installed); falling back to CPU int8.")
            return "cpu"
    return requested


def _build_calibration_data(args: argparse.Namespace, tokenizer: Tokenizer) -> list[torch.Tensor]:
    """Tokenize --calibration_data into input_ids batches for quantize_static()."""
    dataset = TextDataset(tokenizer, args.calibration_data, max_seq_len=args.calib_seq_len)
    collator = DataCollator(tokenizer)
    dl = DataLoader(dataset, batch_size=args.calib_batch_size, shuffle=False, collate_fn=collator)

    batches: list[torch.Tensor] = []
    for i, batch in enumerate(dl):
        if i >= args.num_calib_batches:
            break
        batches.append(batch["input_ids"])
    return batches


def _sanity_forward(model: torch.nn.Module, config: TransformerConfig, device: str) -> None:
    """Run a tiny forward pass to catch shape/dtype breakage early."""
    model.eval()
    input_ids = torch.randint(0, config.vocab_size, (1, 16), device=device)
    with torch.no_grad():
        out = model(input_ids=input_ids, return_encoder_output=False)
    logits = out["logits"] if isinstance(out, dict) else out
    logger.info("Sanity check OK: output shape %s", tuple(logits.shape))


# ======================================================================
# Main
# ======================================================================

def main() -> None:
    global logger

    args = _build_parser().parse_args()

    output_path = Path(args.output) if args.output else Path(args.checkpoint).parent / f"quantized_{args.method}.pt"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    logger = setup_logger("quantize", log_file=str(output_path.parent / "quantize.log"), level="INFO")
    # src/inference/quantization.py logs (before/after param count & memory)
    # through its own module-level logger, which has no handler of its own;
    # propagate it up to a root console handler so those logs are visible.
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(name)s | %(levelname)s | %(message)s")

    set_seed(args.seed)

    device = _resolve_device(args.method, args.device)
    if args.method == "fp16" and device != "cuda":
        logger.warning("--method fp16 is intended for CUDA Tensor Cores; running on %s will still work but won't be faster.", device)

    # ------------------------------------------------------------------
    # Load model + checkpoint
    # ------------------------------------------------------------------
    config = _load_model_config(args)
    model = _build_model(args, config)

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        logger.error("Checkpoint not found: %s", ckpt_path)
        sys.exit(1)

    meta = load_checkpoint(ckpt_path, model, map_location="cpu")
    model = model.to(device)
    model.eval()
    logger.info("Checkpoint loaded (step=%s, epoch=%s) onto %s", meta.get("global_step"), meta.get("epoch"), device)

    # ------------------------------------------------------------------
    # Quantize
    # ------------------------------------------------------------------
    if args.method == "dynamic":
        dtype = torch.qint8 if args.dtype == "qint8" else torch.float16
        quantized = quantize_dynamic(model, dtype=dtype, exclude=args.exclude)

    elif args.method == "static":
        if not args.calibration_data:
            logger.warning("No --calibration_data provided; quantizing with weight-derived ranges only (lower accuracy).")
            calib_batches = None
        else:
            tokenizer = Tokenizer(args.tokenizer_name)
            calib_batches = _build_calibration_data(args, tokenizer)
            logger.info("Built %d calibration batches from %s", len(calib_batches), args.calibration_data)
        quantized = quantize_static(model, calibration_data=calib_batches, num_calib_batches=args.num_calib_batches)

    else:  # fp16
        quantized = convert_to_fp16(model)

    # ------------------------------------------------------------------
    # Sanity check + save
    # ------------------------------------------------------------------
    result_device = next(quantized.parameters()).device
    _sanity_forward(quantized, config, str(result_device))

    torch.save(
        {
            "model": quantized,
            "config": config.__dict__,
            "model_type": args.model_type,
            "quantization_method": args.method,
            "source_checkpoint": str(ckpt_path),
        },
        output_path,
    )
    logger.info("Quantized model saved to %s", output_path)


if __name__ == "__main__":
    main()
