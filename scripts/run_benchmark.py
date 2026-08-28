"""Benchmark transformer models across attention types and sequence lengths.

Produces CSV and JSON reports plus matplotlib comparison plots in
``outputs/reports/``.

Usage
-----
```
python -m scripts.run_benchmark                          # defaults
python -m scripts.run_benchmark --attention_types alibi gqa mqa
python -m scripts.run_benchmark --seq_lengths 128 256 512 1024 2048
```
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from pathlib import Path

import torch

# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.benchmark.comparison import (
    BenchmarkConfig,
    compare_models,
)
from src.models.transformer import Transformer, TransformerConfig
from src.utils.logger import setup_logger
from src.utils.seeding import set_seed

logger: logging.Logger | None = None


# ======================================================================
# CLI
# ======================================================================

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Benchmark Efficient Transformer attention variants.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Config file (configs/benchmark_config.yaml format). Individual flags
    # explicitly passed on the command line still override it.
    p.add_argument("--config", type=str, default=None, help="Path to a benchmark YAML config file.")

    # Model
    p.add_argument("--vocab_size", type=int, default=50257)
    p.add_argument("--d_model", type=int, default=768)
    p.add_argument("--n_heads", type=int, default=12)
    p.add_argument("--n_layers", type=int, default=6)
    p.add_argument("--n_kv_heads", type=int, default=None,
                   help="KV heads for GQA (default: n_heads // 4, min 1). "
                        "MQA always uses 1 regardless of this flag.")

    # Attention types to compare
    p.add_argument("--attention_types", nargs="+", default=["flashv2", "alibi", "gqa", "mqa"],
                   help="Attention types to benchmark.")

    # Sequence lengths sweep
    p.add_argument("--seq_lengths", nargs="+", type=int, default=[128, 512, 1024, 2048],
                   help="Sequence lengths to sweep.")

    # Profiling
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--warmup_runs", type=int, default=3)
    p.add_argument("--num_runs", type=int, default=10)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--seed", type=int, default=42)

    # Output
    p.add_argument("--output_dir", type=str, default="outputs/reports")
    p.add_argument("--plot_dir", type=str, default=None,
                   help="Directory for generated plots (default: same as --output_dir).")
    p.add_argument("--no_plots", action="store_true", default=False)

    return p


def _load_yaml_config(path: str) -> dict:
    """Load a YAML config file (see configs/benchmark_config.yaml).

    Called from ``main()`` before ``setup_logger()`` runs (the config can
    itself set ``output_dir``, which the logger's log file lives under), so
    this uses the stdlib root logger rather than the module-global
    ``logger``, which is still ``None`` at that point.
    """
    fallback_logger = logging.getLogger(__name__)
    try:
        import yaml
    except ImportError:
        fallback_logger.error("PyYAML is required when --config is used.  pip install pyyaml")
        sys.exit(1)

    with open(path, "r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    if not isinstance(data, dict):
        fallback_logger.error("Config file must be a YAML mapping, got %s", type(data).__name__)
        sys.exit(1)
    return data


def _apply_yaml_config(args: argparse.Namespace, yaml_data: dict) -> argparse.Namespace:
    """Overlay a benchmark_config.yaml's ``model`` / ``benchmark`` sections
    onto the parsed CLI args. Flags explicitly passed on the command line
    still win over the YAML value."""
    sections = {
        "model": yaml_data.get("model", {}) or {},
        "benchmark": yaml_data.get("benchmark", {}) or {},
    }
    # dest -> (yaml section, yaml key), since a couple of names diverge
    # between the CLI and configs/benchmark_config.yaml.
    key_map = {
        "d_model": ("model", "d_model"),
        "n_heads": ("model", "n_heads"),
        "n_layers": ("model", "n_layers"),
        "vocab_size": ("model", "vocab_size"),
        "seq_lengths": ("benchmark", "seq_lengths"),
        "attention_types": ("benchmark", "compare_attentions"),
        "batch_size": ("benchmark", "batch_size"),
        "warmup_runs": ("benchmark", "warmup"),
        "num_runs": ("benchmark", "iterations"),
        "device": ("benchmark", "device"),
        "output_dir": ("benchmark", "output_dir"),
        "plot_dir": ("benchmark", "plot_dir"),
    }

    explicit = set(sys.argv[1:])
    for dest, (section, key) in key_map.items():
        if f"--{dest}" in explicit:
            continue
        value = sections[section].get(key)
        if value is not None:
            setattr(args, dest, value)
    return args


# ======================================================================
# Plotting helpers (optional matplotlib)
# ======================================================================

def _make_plots(
    results: list[dict],
    output_dir: Path,
) -> None:
    """Generate comparison bar charts for latency, memory, and throughput."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib is not installed; skipping plots.")
        return

    names = [r["name"] for r in results]
    n = len(names)
    x = range(n)

    # --- Latency -------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 5))
    latencies = [r.get("latency_mean_ms", 0) for r in results]
    lat_errs = [r.get("latency_std_ms", 0) for r in results]
    bars = ax.bar(x, latencies, yerr=lat_errs, capsize=4, color=plt.cm.Set2.colors[:n])
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=30, ha="right")
    ax.set_ylabel("Forward Latency (ms)")
    ax.set_title("Forward Pass Latency")
    ax.bar_label(bars, fmt="%.1f", padding=3)
    fig.tight_layout()
    fig.savefig(output_dir / "latency_comparison.png", dpi=150)
    plt.close(fig)

    # --- Memory --------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 5))
    mem = [r.get("peak_gpu_memory_mb", 0) for r in results]
    bars = ax.bar(x, mem, color=plt.cm.Set3.colors[:n])
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=30, ha="right")
    ax.set_ylabel("Peak GPU Memory (MB)")
    ax.set_title("Peak GPU Memory Usage")
    ax.bar_label(bars, fmt="%.0f", padding=3)
    fig.tight_layout()
    fig.savefig(output_dir / "memory_comparison.png", dpi=150)
    plt.close(fig)

    # --- Throughput ----------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 5))
    tp = [r.get("throughput_mean_tokens_per_sec", 0) for r in results]
    bars = ax.bar(x, tp, color=plt.cm.Pastel1.colors[:n])
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=30, ha="right")
    ax.set_ylabel("Throughput (tokens/s)")
    ax.set_title("Inference Throughput")
    ax.bar_label(bars, fmt="%.0f", padding=3)
    fig.tight_layout()
    fig.savefig(output_dir / "throughput_comparison.png", dpi=150)
    plt.close(fig)

    logger.info("Plots saved to %s", output_dir)


# ======================================================================
# Seq-length sweep
# ======================================================================

def _seq_length_sweep(
    attention_types: list[str],
    seq_lengths: list[int],
    vocab_size: int,
    d_model: int,
    n_heads: int,
    n_layers: int,
    batch_size: int,
    warmup_runs: int,
    num_runs: int,
    device: torch.device,
    n_kv_heads: int | None = None,
) -> list[dict]:
    """Run benchmarks across all (attention_type, seq_length) combos."""
    all_rows: list[dict] = []

    for attn_type in attention_types:
        for seq_len in seq_lengths:
            tag = f"{attn_type}_T{seq_len}"
            logger.info("--- Benchmarking %s ---", tag)

            kv_heads = n_kv_heads
            if attn_type == "gqa" and kv_heads is None:
                # TransformerConfig otherwise defaults n_kv_heads to
                # n_heads even for "gqa", which makes the variant
                # indistinguishable from plain MHA in this sweep's
                # latency/memory numbers. Default to a real 4x compression
                # unless the caller asked for a specific KV-head count.
                kv_heads = max(1, n_heads // 4)

            try:
                config = TransformerConfig(
                    vocab_size=vocab_size,
                    d_model=d_model,
                    n_heads=n_heads,
                    n_layers=n_layers,
                    attn_type=attn_type,
                    n_kv_heads=kv_heads,
                    # "rope" is currently broken as a standalone pos_encoding;
                    # see the note in configs/default_config.yaml.
                    pos_encoding="sinusoidal",
                    causal=True,
                )
                model = Transformer(config, use_encoder=False, use_decoder=True)
            except (ValueError, RuntimeError) as exc:
                logger.warning("Could not create model for %s: %s", tag, exc)
                continue

            bench_cfg = BenchmarkConfig(
                name=tag,
                config=model,
                batch_size=batch_size,
                seq_len=seq_len,
                vocab_size=vocab_size,
                warmup_runs=warmup_runs,
                num_runs=num_runs,
                causal=True,
                device=device,
            )

            try:
                comp = compare_models([bench_cfg])
                if comp.results:
                    result = comp.results[0].to_dict()
                    # Flatten "metrics" / "profile" onto the row so the
                    # summary table, CSV, and plots (which read flat keys
                    # like latency_mean_ms) actually see the numbers.
                    row = {
                        "name": result.get("name"),
                        "attention_type": attn_type,
                        "seq_len": seq_len,
                        **result.get("metrics", {}),
                        **result.get("profile", {}),
                    }
                    all_rows.append(row)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Benchmark failed for %s: %s", tag, exc)

            # Free memory
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    return all_rows


# ======================================================================
# Report writers
# ======================================================================

def _write_csv(rows: list[dict], path: Path) -> None:
    """Write a flat CSV from a list of row dicts."""
    if not rows:
        return
    # Flatten one level
    flat_keys: list[str] = []
    for row in rows:
        for k in row:
            if k not in flat_keys:
                flat_keys.append(k)

    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=flat_keys)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    logger.info("CSV report: %s", path)


def _write_json(rows: list[dict], path: Path) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(rows, fh, indent=2, default=str)
    logger.info("JSON report: %s", path)


# ======================================================================
# Main
# ======================================================================

def main() -> None:
    global logger

    args = _build_parser().parse_args()

    if args.config is not None:
        yaml_data = _load_yaml_config(args.config)
        args = _apply_yaml_config(args, yaml_data)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logger("benchmark", log_file=str(output_dir / "benchmark.log"), level="INFO")
    set_seed(args.seed)

    device = torch.device("cuda" if args.device == "cuda" and torch.cuda.is_available() else args.device)
    logger.info("Device: %s", device)

    # ------------------------------------------------------------------
    # Run the sweep
    # ------------------------------------------------------------------
    rows = _seq_length_sweep(
        attention_types=args.attention_types,
        seq_lengths=args.seq_lengths,
        vocab_size=args.vocab_size,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        batch_size=args.batch_size,
        warmup_runs=args.warmup_runs,
        num_runs=args.num_runs,
        device=device,
        n_kv_heads=args.n_kv_heads,
    )

    if not rows:
        logger.error("No benchmark results collected.")
        sys.exit(1)

    # ------------------------------------------------------------------
    # Console summary
    # ------------------------------------------------------------------
    print("\n========== Benchmark Summary ==========")
    print(f"  {'Name':<30s}  {'Latency (ms)':>14s}  {'Memory (MB)':>14s}  {'Throughput':>14s}")
    print("  " + "-" * 78)
    for row in rows:
        name = row.get("name", "?")
        lat = row.get("latency_mean_ms", 0)
        mem = row.get("peak_gpu_memory_mb", 0)
        tp = row.get("throughput_mean_tokens_per_sec", 0)
        print(f"  {name:<30s}  {lat:>12.2f}ms  {mem:>12.0f}MB  {tp:>10.0f}tok/s")
    print("=" * 48)

    # ------------------------------------------------------------------
    # Write reports
    # ------------------------------------------------------------------
    _write_csv(rows, output_dir / "benchmark_results.csv")
    _write_json(rows, output_dir / "benchmark_results.json")

    if not args.no_plots:
        plot_dir = Path(args.plot_dir) if args.plot_dir else output_dir
        plot_dir.mkdir(parents=True, exist_ok=True)
        _make_plots(rows, plot_dir)

    logger.info("All done.  Reports in %s", output_dir)


if __name__ == "__main__":
    main()
