"""Evaluate a trained transformer checkpoint.

Usage
-----
```
python -m scripts.run_eval --checkpoint outputs/checkpoints/best.pt --data_path data/val.txt
python -m scripts.run_eval --checkpoint outputs/checkpoints/final.pt --compute_bleu
```
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import sys
import time
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.data.collator import DataCollator
from src.data.dataset import TextDataset
from src.data.tokenizer import Tokenizer
from src.models.transformer import Transformer, TransformerConfig
from src.training.checkpoint import load_checkpoint
from src.utils.logger import setup_logger

logger: logging.Logger | None = None


# ======================================================================
# CLI
# ======================================================================

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Evaluate a trained transformer model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--checkpoint", type=str, required=True, help="Path to .pt checkpoint.")
    p.add_argument("--config_json", type=str, default=None, help="Optional config.json (same dir as model save).")
    p.add_argument("--data_path", type=str, default=None, help="Plain-text evaluation file.")
    p.add_argument("--tokenizer_name", type=str, default="gpt2")
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--max_seq_len", type=int, default=512)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--model_type", choices=["encoder_only", "decoder_only", "encoder_decoder"], default="decoder_only")
    p.add_argument("--compute_bleu", action="store_true", default=False, help="Compute BLEU score (encoder-decoder).")
    p.add_argument("--num_beams", type=int, default=1, help="Beam search width (1 = greedy).")
    p.add_argument("--max_gen_tokens", type=int, default=128, help="Max tokens to generate for BLEU.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output_file", type=str, default=None, help="Path to write JSON results.")
    p.add_argument("--attn_type", type=str, default="flashv2")
    p.add_argument("--n_heads", type=int, default=12)
    p.add_argument("--n_layers", type=int, default=6)
    p.add_argument("--d_model", type=int, default=768)
    p.add_argument("--vocab_size", type=int, default=50257)
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
        # fields -- passing them through would crash TransformerConfig(**cfg_dict)
        # with an unexpected-keyword-argument TypeError.
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
    )


def _build_model(args: argparse.Namespace, config: TransformerConfig) -> Transformer:
    use_enc = args.model_type in ("encoder_only", "encoder_decoder")
    use_dec = args.model_type in ("decoder_only", "encoder_decoder")
    model = Transformer(config, use_encoder=use_enc, use_decoder=use_dec)
    return model


@torch.no_grad()
def _evaluate_causal(
    model: Transformer,
    dataloader: DataLoader,
    device: torch.device,
) -> dict[str, float]:
    """Compute loss and perplexity for causal / masked LM."""
    model.eval()
    criterion = torch.nn.CrossEntropyLoss(ignore_index=-100)
    total_loss = 0.0
    total_tokens = 0
    correct = 0
    total_pred = 0

    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)
        attention_mask = batch.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)

        logits = model(input_ids=input_ids, attention_mask=attention_mask, return_encoder_output=False)
        if isinstance(logits, dict):
            logits = logits["logits"]

        # DataCollator._shift_labels_for_causal() already right-shifts
        # `labels` (labels[t] is the prediction target for logits[t]);
        # slicing [:-1]/[1:] here would shift an already-shifted tensor a
        # second time, silently comparing logits[t] against the token at
        # t+2 instead of t+1. Compare the full tensors directly, matching
        # Trainer._compute_loss's convention for the same collator output.
        flat_logits = logits.reshape(-1, logits.size(-1))
        flat_labels = labels.reshape(-1)
        loss = criterion(flat_logits, flat_labels)
        n_tokens = (flat_labels != -100).sum().item()
        total_loss += loss.item() * n_tokens
        total_tokens += max(n_tokens, 1)

        # Token accuracy (ignoring -100)
        preds = flat_logits.argmax(dim=-1)
        mask = flat_labels != -100
        correct += (preds[mask] == flat_labels[mask]).sum().item()
        total_pred += mask.sum().item()

    avg_loss = total_loss / max(total_tokens, 1)
    ppl = math.exp(min(avg_loss, 20))
    accuracy = correct / max(total_pred, 1)

    return {
        "loss": round(avg_loss, 4),
        "perplexity": round(ppl, 4),
        "token_accuracy": round(accuracy, 4),
        "num_tokens": total_tokens,
    }


def _compute_bleu(
    model: Transformer,
    tokenizer: Tokenizer,
    dataset: TextDataset,
    device: torch.device,
    max_gen_tokens: int = 128,
    num_beams: int = 1,
) -> dict[str, float]:
    """Compute corpus-level BLEU for encoder-decoder models."""
    try:
        from nltk.translate.bleu_score import SmoothingFunction, corpus_bleu
    except ImportError:
        logger.error("nltk is required for BLEU.  pip install nltk")
        return {"bleu": 0.0}

    references: list[list[list[str]]] = []
    hypotheses: list[list[str]] = []

    model.eval()
    for i in range(min(len(dataset), 200)):
        item = dataset[i]
        src_ids = torch.tensor([item["input_ids"]], dtype=torch.long, device=device)
        tgt_ids = item.get("labels", item.get("decoder_input_ids", []))
        if not tgt_ids:
            continue

        generated = model.generate(
            src_ids,
            max_new_tokens=max_gen_tokens,
            do_sample=False,
        )
        gen_text = tokenizer.decode(generated[0].tolist(), skip_special_tokens=True)
        ref_text = tokenizer.decode(tgt_ids, skip_special_tokens=True)

        hypotheses.append(gen_text.split())
        references.append([ref_text.split()])

    smooth = SmoothingFunction().method1
    bleu1 = corpus_bleu(references, hypotheses, weights=(1.0, 0, 0, 0), smoothing_function=smooth)
    bleu4 = corpus_bleu(references, hypotheses, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smooth)

    return {
        "bleu-1": round(bleu1, 4),
        "bleu-4": round(bleu4, 4),
        "num_samples": len(references),
    }


# ======================================================================
# Main
# ======================================================================

def main() -> None:
    global logger

    args = _build_parser().parse_args()

    log_file = str(Path(args.checkpoint).parent / "eval.log")
    logger = setup_logger("eval", log_file=log_file, level="INFO")

    # Seed
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if args.device == "cuda" and torch.cuda.is_available() else args.device)
    logger.info("Device: %s  Checkpoint: %s", device, args.checkpoint)

    # ------------------------------------------------------------------
    # Load model
    # ------------------------------------------------------------------
    config = _load_model_config(args)
    model = _build_model(args, config)

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        logger.error("Checkpoint not found: %s", ckpt_path)
        sys.exit(1)

    meta = load_checkpoint(ckpt_path, model, map_location=device)
    model = model.to(device)
    model.eval()
    logger.info("Checkpoint loaded (step=%s, epoch=%s)", meta.get("global_step"), meta.get("epoch"))

    # ------------------------------------------------------------------
    # Tokenizer / Data
    # ------------------------------------------------------------------
    tokenizer: Tokenizer | None = None
    try:
        tokenizer = Tokenizer(args.tokenizer_name)
        logger.info("Tokenizer: %s  vocab=%d", args.tokenizer_name, tokenizer.vocab_size)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Tokenizer unavailable: %s", exc)

    results: dict[str, Any] = {"checkpoint": str(ckpt_path), "device": str(device)}

    if args.data_path is not None and tokenizer is not None:
        dataset = TextDataset(tokenizer, args.data_path, max_seq_len=args.max_seq_len)
        collator = DataCollator(tokenizer)
        dl = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collator)

        logger.info("Evaluating on %d samples …", len(dataset))
        t0 = time.perf_counter()
        eval_metrics = _evaluate_causal(model, dl, device)
        elapsed = time.perf_counter() - t0

        results.update(eval_metrics)
        results["eval_time_sec"] = round(elapsed, 2)
        logger.info("Loss=%.4f  PPL=%.4f  Acc=%.4f  time=%.1fs", eval_metrics["loss"], eval_metrics["perplexity"], eval_metrics["token_accuracy"], elapsed)

        # BLEU (encoder-decoder only)
        if args.compute_bleu and args.model_type == "encoder_decoder" and tokenizer is not None:
            logger.info("Computing BLEU …")
            bleu_metrics = _compute_bleu(model, tokenizer, dataset, device, args.max_gen_tokens, args.num_beams)
            results.update(bleu_metrics)
            logger.info("BLEU-1=%.4f  BLEU-4=%.4f", bleu_metrics["bleu-1"], bleu_metrics["bleu-4"])
    else:
        logger.info("No data_path provided; skipping loss/perplexity evaluation.")

    # ------------------------------------------------------------------
    # Save results
    # ------------------------------------------------------------------
    output_path = args.output_file or str(Path(args.checkpoint).parent / "eval_results.json")
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2, default=str)
    logger.info("Results written to %s", output_path)

    # Pretty-print to stdout
    print("\n========== Evaluation Results ==========")
    print(json.dumps(results, indent=2, default=str))
    print("=" * 40)


if __name__ == "__main__":
    main()
