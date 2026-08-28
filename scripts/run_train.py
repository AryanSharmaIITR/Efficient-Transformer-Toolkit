"""Train a transformer model.

Usage
-----
```
python -m scripts.run_train --config experiments/exp_001_baseline/config.yaml
python -m scripts.run_train --model_type decoder_only --attn_type alibi --data_path data/corpus.txt
```
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, random_split

# ---------------------------------------------------------------------------
# Ensure the project root is on sys.path so that ``src`` is importable.
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.data.collator import DataCollator
from src.data.dataset import TextDataset
from src.data.tokenizer import Tokenizer
from src.models.transformer import Transformer, TransformerConfig
from src.training.trainer import EarlyStopping, Trainer, TrainerConfig
from src.utils.logger import setup_logger
from src.utils.seeding import set_seed

logger: logging.Logger | None = None


# ======================================================================
# Helpers
# ======================================================================

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Train an Efficient Transformer model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Config file takes precedence; individual flags override.
    p.add_argument("--config", type=str, default=None, help="Path to YAML config file.")

    # Model ---------------------------------------------------------
    m = p.add_argument_group("model")
    m.add_argument("--model_type", choices=["encoder_only", "decoder_only", "encoder_decoder"], default="decoder_only")
    m.add_argument("--vocab_size", type=int, default=50257)
    m.add_argument("--d_model", type=int, default=768)
    m.add_argument("--n_heads", type=int, default=12)
    m.add_argument("--n_layers", type=int, default=6)
    m.add_argument("--d_ff", type=int, default=None)
    m.add_argument("--max_seq_len", type=int, default=2048)
    m.add_argument("--attn_type", type=str, default="flashv2", choices=["flashv1", "flashv2", "alibi", "gqa", "mqa"])
    m.add_argument("--pos_encoding", type=str, default="rope", choices=["rope", "sinusoidal", "learned"])
    m.add_argument("--dropout", type=float, default=0.1)
    m.add_argument("--causal", action="store_true", default=True)
    m.add_argument("--no_causal", dest="causal", action="store_false")
    m.add_argument("--use_flash", action="store_true", default=True)
    m.add_argument("--no_flash", dest="use_flash", action="store_false")
    m.add_argument("--n_kv_heads", type=int, default=None, help="Number of KV heads for GQA/MQA. Defaults to n_heads.")
    m.add_argument("--use_bias", action="store_true", default=False)
    m.add_argument("--no_bias", dest="use_bias", action="store_false")
    m.add_argument("--tie_embeddings", action="store_true", default=True)
    m.add_argument("--no_tie_embeddings", dest="tie_embeddings", action="store_false")
    m.add_argument("--ffn_activation", type=str, default="swiglu")
    m.add_argument("--layer_norm_type", type=str, default="pre", choices=["pre", "post"])

    # Data ---------------------------------------------------------
    d = p.add_argument_group("data")
    d.add_argument("--data_path", type=str, default=None, help="Path to a plain-text file for training.")
    d.add_argument("--tokenizer_name", type=str, default="gpt2", help="HuggingFace tokenizer name or path.")
    d.add_argument("--max_seq_len_data", type=int, default=None, help="Max tokens per sample (defaults to model max_seq_len).")
    d.add_argument("--stride", type=int, default=0, help="Sliding-window stride for long documents.")
    d.add_argument("--val_split", type=float, default=0.05, help="Fraction of data used for validation.")
    d.add_argument("--batch_size", type=int, default=8)
    d.add_argument("--num_workers", type=int, default=0)

    # Optimizer -----------------------------------------------------
    o = p.add_argument_group("optimizer")
    o.add_argument("--lr", type=float, default=3e-4)
    o.add_argument("--weight_decay", type=float, default=0.01)
    o.add_argument("--warmup_ratio", type=float, default=0.03)
    o.add_argument("--max_grad_norm", type=float, default=1.0)

    # Trainer -------------------------------------------------------
    t = p.add_argument_group("trainer")
    t.add_argument("--num_epochs", type=int, default=10)
    t.add_argument("--gradient_accumulation_steps", type=int, default=1)
    t.add_argument("--use_amp", action="store_true", default=True)
    t.add_argument("--no_amp", dest="use_amp", action="store_false")
    t.add_argument("--device", type=str, default="cuda")
    t.add_argument("--compile_model", action="store_true", default=False)
    t.add_argument("--seed", type=int, default=42)
    t.add_argument("--output_dir", type=str, default="outputs/checkpoints")
    t.add_argument("--save_every", type=int, default=500)
    t.add_argument("--log_every", type=int, default=10)
    t.add_argument("--eval_every", type=int, default=None)
    t.add_argument("--early_stopping_patience", type=int, default=None)

    # Logging backends ----------------------------------------------
    l = p.add_argument_group("logging")
    l.add_argument("--tensorboard", action="store_true", default=False, help="Enable TensorBoard logging.")
    l.add_argument("--wandb", action="store_true", default=False, help="Enable Weights & Biases logging.")
    l.add_argument("--wandb_project", type=str, default="efficient-transformer-toolkit")
    l.add_argument("--wandb_run_name", type=str, default=None)

    return p


def _load_yaml_config(path: str) -> dict[str, Any]:
    """Load a YAML config and flatten into a flat dict of CLI-style keys."""
    try:
        import yaml
    except ImportError:
        logger.error("PyYAML is required when --config is used.  pip install pyyaml")
        sys.exit(1)

    with open(path, "r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    if not isinstance(data, dict):
        logger.error("Config file must be a YAML mapping, got %s", type(data).__name__)
        sys.exit(1)
    return data


def _flatten_yaml_config(yaml_data: dict[str, Any]) -> dict[str, Any]:
    """Flatten nested config sections (model/training/data/logging/...) into
    a flat dict of CLI-style keys, and resolve a few keys that don't map
    1:1 onto CLI flag names.
    """
    # Only sections this script consumes; "benchmark" / "inference" reuse
    # key names (e.g. batch_size, output_dir, device) for unrelated scripts
    # and must not be flattened in here.
    _RELEVANT_SECTIONS = {"model", "training", "data", "logging"}

    # Within "data", these keys mean something different from (and would
    # collide with) a same-named key elsewhere -- e.g. this repo's own
    # config files aren't consistent: default_config.yaml uses
    # "max_seq_length" for the data section, training_config.yaml uses
    # "max_seq_len" -- and a bare merge would let it silently clobber
    # model.max_seq_len (a real, load-bearing bug: it corrupted the
    # model's positional-encoding range to the dataset truncation length).
    _DATA_KEY_RENAMES = {"max_seq_len": "max_seq_len_data", "max_seq_length": "max_seq_len_data"}

    flat: dict[str, Any] = {}
    for section, value in yaml_data.items():
        if not isinstance(value, dict):
            flat[section] = value
        elif section == "data":
            for k, v in value.items():
                flat[_DATA_KEY_RENAMES.get(k, k)] = v
        elif section in _RELEVANT_SECTIONS:
            flat.update(value)

    data_section = yaml_data.get("data", {})
    if isinstance(data_section, dict):
        dataset_path = data_section.get("dataset_path")
        train_file = data_section.get("train_file")
        if dataset_path and train_file:
            flat["data_path"] = str(Path(dataset_path) / train_file)

    return flat


def _merge_args(parser: argparse.ArgumentParser, args: argparse.Namespace, yaml_data: dict[str, Any]) -> argparse.Namespace:
    """Override CLI defaults with YAML values; CLI explicit flags still win."""
    # NOT `{a.dest: a.default for a in parser._actions}`: several flags share
    # a dest via a --flag/--no_flag pair (e.g. --use_bias/--no_bias both
    # write to "use_bias"), and a raw dict comprehension over _actions keeps
    # whichever action was *added last*, which for every one of these pairs
    # is the --no_x action -- store_false with no explicit default=, so
    # argparse implicitly defaults it to True. That silently flipped
    # use_bias's real default (False) to True for any --config run that
    # doesn't explicitly set use_bias (in the YAML or on the CLI), training
    # extra bias parameters nobody asked for. parser.parse_args([]) asks
    # argparse itself to resolve these dest collisions the way it does
    # during a real parse, which is correct by construction.
    defaults = vars(parser.parse_args([]))

    # Start from CLI defaults, then overlay YAML, then re-apply explicit CLI.
    merged = {**defaults}

    for key, val in _flatten_yaml_config(yaml_data).items():
        if key in merged:
            merged[key] = val

    # Re-apply anything the user explicitly passed on the command line.
    explicit = set(sys.argv[1:])
    for action in parser._actions:
        for opt in action.option_strings:
            if opt in explicit:
                merged[action.dest] = getattr(args, action.dest)
                break

    return argparse.Namespace(**merged)


def _resolve_tokenizer(args: argparse.Namespace, model_config: TransformerConfig) -> Tokenizer | None:
    """Optionally create a Tokenizer from HuggingFace."""
    try:
        tok = Tokenizer(args.tokenizer_name)
        model_config.vocab_size = tok.vocab_size
        logger.info("Tokenizer loaded: %s  vocab_size=%d", args.tokenizer_name, tok.vocab_size)
        return tok
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not load tokenizer %r: %s. Using vocab_size=%d from config.", args.tokenizer_name, exc, model_config.vocab_size)
        return None


def _build_dataloaders(
    args: argparse.Namespace,
    tokenizer: Tokenizer | None,
    model_type: str,
) -> tuple[DataLoader, DataLoader | None]:
    """Build train and optional validation DataLoaders."""
    max_seq = args.max_seq_len_data or args.max_seq_len

    if tokenizer is not None:
        dataset = TextDataset(
            tokenizer,
            args.data_path,
            max_seq_len=max_seq,
            stride=args.stride,
        )
    else:
        raise RuntimeError("--data_path is required when no tokenizer is available.")

    # Train / val split.
    val_size = max(1, int(len(dataset) * args.val_split)) if args.val_split > 0 and len(dataset) > 1 else 0
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size]) if val_size else (dataset, None)

    collator = DataCollator(tokenizer) if tokenizer else None

    train_dl = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collator,
        pin_memory=(args.device == "cuda"),
        drop_last=True,
    )

    val_dl: DataLoader | None = None
    if val_ds is not None:
        val_dl = DataLoader(
            val_ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            collate_fn=collator,
            pin_memory=(args.device == "cuda"),
        )

    logger.info("Data: train=%d  val=%d  batch_size=%d", train_size, val_size, args.batch_size)
    return train_dl, val_dl


def _setup_wandb(args: argparse.Namespace) -> Any:
    """Initialise Weights & Biases if requested."""
    try:
        import wandb

        wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name or f"run-{int(time.time())}",
            config=vars(args),
        )
        return wandb
    except ImportError:
        logger.error("wandb is not installed.  pip install wandb")
        sys.exit(1)


def _setup_tensorboard(log_dir: str) -> Any:
    """Initialise TensorBoard SummaryWriter if possible."""
    try:
        from torch.utils.tensorboard import SummaryWriter

        writer = SummaryWriter(log_dir=log_dir)
        logger.info("TensorBoard logging to %s", log_dir)
        return writer
    except ImportError:
        logger.warning("tensorboard is not installed.  pip install tensorboard")
        return None


def _tb_wandb_callback(wandb_mod: Any, tb_writer: Any):
    """Return a trainer callback that logs metrics to TB / W&B."""

    def _cb(trainer: Trainer, step: int, metrics: dict[str, Any]) -> None:
        if tb_writer is not None:
            for k, v in metrics.items():
                if isinstance(v, (int, float)):
                    tb_writer.add_scalar(f"train/{k}", v, step)
        if wandb_mod is not None:
            wandb_mod.log(metrics, step=step)

    return _cb


# ======================================================================
# Main
# ======================================================================

def main() -> None:
    global logger

    parser = _build_parser()
    args = parser.parse_args()

    # Merge YAML config if provided.
    if args.config is not None:
        yaml_data = _load_yaml_config(args.config)
        args = _merge_args(parser, args, yaml_data)

    # Setup logging / seed.
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    log_file = str(Path(args.output_dir) / "train.log")
    logger = setup_logger("train", log_file=log_file, level="INFO")
    set_seed(args.seed)

    if args.device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA requested but unavailable; falling back to CPU.")
        args.device = "cpu"
    device = torch.device(args.device)
    logger.info("Device: %s  Seed: %d", device, args.seed)

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    model_config = TransformerConfig(
        vocab_size=args.vocab_size,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        d_ff=args.d_ff,
        max_seq_len=args.max_seq_len,
        attn_type=args.attn_type,
        pos_encoding=args.pos_encoding,
        dropout=args.dropout,
        causal=args.causal,
        use_flash=args.use_flash,
        n_kv_heads=args.n_kv_heads,
        use_bias=args.use_bias,
        tie_embeddings=args.tie_embeddings,
        ffn_activation=args.ffn_activation,
        layer_norm_type=args.layer_norm_type,
    )

    # ------------------------------------------------------------------
    # Tokenizer -- must be resolved BEFORE the model is built. It may
    # correct model_config.vocab_size to match the real tokenizer (e.g.
    # experiment configs use a placeholder vocab_size expecting this), and
    # the model's nn.Embedding table is sized once at construction time --
    # mutating vocab_size afterward doesn't resize an already-built
    # embedding table, so a mismatched config vocab_size used to build a
    # too-small embedding table and only fail once real tokens exceeded it
    # (a CUDA device-side assert mid-training, not at model-creation time).
    # ------------------------------------------------------------------
    tokenizer = _resolve_tokenizer(args, model_config)

    use_enc = args.model_type in ("encoder_only", "encoder_decoder")
    use_dec = args.model_type in ("decoder_only", "encoder_decoder")

    model = Transformer(model_config, use_encoder=use_enc, use_decoder=use_dec)
    n_params = sum(p.numel() for p in model.parameters())
    logger.info("Model created: %s  params=%s", args.model_type, f"{n_params:,}")

    train_dl, val_dl = _build_dataloaders(args, tokenizer, args.model_type)

    # ------------------------------------------------------------------
    # Logging backends
    # ------------------------------------------------------------------
    wandb_mod = None
    tb_writer = None

    if args.wandb:
        wandb_mod = _setup_wandb(args)
    if args.tensorboard:
        tb_dir = str(Path(args.output_dir) / "tensorboard")
        tb_writer = _setup_tensorboard(tb_dir)

    # ------------------------------------------------------------------
    # Trainer
    # ------------------------------------------------------------------
    trainer_config = TrainerConfig(
        lr=args.lr,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        max_grad_norm=args.max_grad_norm,
        num_epochs=args.num_epochs,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        use_amp=args.use_amp,
        output_dir=args.output_dir,
        save_every=args.save_every,
        log_every=args.log_every,
        eval_every=args.eval_every,
        early_stopping_patience=args.early_stopping_patience,
        device=str(device),
        compile_model=args.compile_model,
        model_type=args.model_type,
    )

    callbacks = []
    if wandb_mod is not None or tb_writer is not None:
        callbacks.append(_tb_wandb_callback(wandb_mod, tb_writer))

    trainer = Trainer(
        model=model,
        config=trainer_config,
        train_dataloader=train_dl,
        val_dataloader=val_dl,
        callbacks=callbacks if callbacks else None,
    )

    # ------------------------------------------------------------------
    # Train
    # ------------------------------------------------------------------
    try:
        summary = trainer.train()
    except KeyboardInterrupt:
        logger.warning("Training interrupted by user.")
        summary = {"interrupted": True}
    except EarlyStopping as exc:
        # train() raises this from inside the epoch loop, before it reaches
        # its own "final checkpoint" save + summary construction -- do both
        # here instead of letting it crash the script.
        logger.warning("Early stopping: %s", exc)
        final_path = trainer.ckpt_manager.save_final(
            trainer.model, trainer.optimizer, trainer.scheduler,
            trainer.global_step, trainer.current_epoch,
            extra=trainer._resume_state(),
        )
        summary = {
            "early_stopped": True,
            "train_history": trainer.train_history,
            "val_history": trainer.val_history,
            "best_metric": trainer.ckpt_manager.best_value,
            "total_steps": trainer.global_step,
            "final_checkpoint": str(final_path),
        }

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------
    if tb_writer is not None:
        tb_writer.close()
    if wandb_mod is not None:
        wandb_mod.finish()

    logger.info("Done.  Summary: %s", json.dumps({k: v for k, v in summary.items() if k != "train_history"}, default=str, indent=2))

    summary_path = Path(args.output_dir) / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, default=str, indent=2)
    logger.info("Run summary saved to %s", summary_path)


if __name__ == "__main__":
    main()
