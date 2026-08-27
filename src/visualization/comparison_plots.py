from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from ..benchmark.comparison import ComparisonResult


def _get_labels_and_field(
    comparison: ComparisonResult,
    field_name: str,
) -> tuple[list[str], list[float]]:
    labels = []
    values = []
    for r in comparison.results:
        labels.append(r.name)
        if r.metrics is not None:
            val = getattr(r.metrics, field_name, None)
            if val is None:
                prop = getattr(r.metrics, "to_dict", dict)().get(field_name, 0)
                values.append(float(prop))
            else:
                values.append(float(val))
        elif r.profile is not None:
            val = getattr(r.profile, field_name, None)
            if val is not None:
                values.append(float(val))
            else:
                values.append(0.0)
        else:
            values.append(0.0)
    return labels, values


def plot_param_comparison(
    comparison: ComparisonResult,
    save_path: str | Path | None = None,
    ax: plt.Axes | None = None,
    figsize: tuple[int, int] = (10, 5),
    log_scale: bool = False,
) -> plt.Figure:
    """Bar chart of total parameter counts across models."""
    labels, params = _get_labels_and_field(comparison, "total_params")
    params_m = [p / 1e6 for p in params]

    create_fig = ax is None
    if create_fig:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    colors = plt.cm.Set2(np.linspace(0, 1, len(labels)))
    bars = ax.bar(labels, params_m, color=colors, edgecolor="gray", linewidth=0.5)

    for bar, val in zip(bars, params_m):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{val:.1f}M",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    if log_scale:
        ax.set_yscale("log")

    ax.set_ylabel("Parameters (Millions)")
    ax.set_title("Parameter Count Comparison")
    ax.tick_params(axis="x", rotation=30)

    if create_fig:
        fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_latency_comparison(
    comparison: ComparisonResult,
    save_path: str | Path | None = None,
    ax: plt.Axes | None = None,
    figsize: tuple[int, int] = (10, 5),
) -> plt.Figure:
    """Bar chart of forward pass latency with error bars (mean +/- std)."""
    labels = [r.name for r in comparison.results]
    means = []
    stds = []
    for r in comparison.results:
        if r.profile is not None:
            means.append(r.profile.latency_mean_ms)
            stds.append(r.profile.latency_std_ms)
        else:
            means.append(0.0)
            stds.append(0.0)

    create_fig = ax is None
    if create_fig:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    colors = plt.cm.Paired(np.linspace(0, 1, len(labels)))
    bars = ax.bar(labels, means, yerr=stds, capsize=5, color=colors,
                  edgecolor="gray", linewidth=0.5, error_kw={"linewidth": 1.2})

    for bar, mean, std in zip(bars, means, stds):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + std,
            f"{mean:.2f}ms",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    ax.set_ylabel("Latency (ms)")
    ax.set_title("Forward Pass Latency")
    ax.tick_params(axis="x", rotation=30)

    if create_fig:
        fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_throughput_comparison(
    comparison: ComparisonResult,
    save_path: str | Path | None = None,
    ax: plt.Axes | None = None,
    figsize: tuple[int, int] = (10, 5),
) -> plt.Figure:
    """Horizontal bar chart of throughput (tokens/sec)."""
    labels = []
    throughputs = []
    for r in comparison.results:
        labels.append(r.name)
        if r.profile is not None:
            throughputs.append(r.profile.throughput_mean_tokens_per_sec)
        else:
            throughputs.append(0.0)

    create_fig = ax is None
    if create_fig:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(labels)))
    y_pos = np.arange(len(labels))
    bars = ax.barh(y_pos, throughputs, color=colors, edgecolor="gray", linewidth=0.5)

    for bar, val in zip(bars, throughputs):
        ax.text(
            bar.get_width(),
            bar.get_y() + bar.get_height() / 2,
            f" {val:,.0f} tok/s",
            ha="left",
            va="center",
            fontsize=9,
        )

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Throughput (tokens/sec)")
    ax.set_title("Throughput Comparison")

    if create_fig:
        fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_memory_comparison(
    comparison: ComparisonResult,
    save_path: str | Path | None = None,
    ax: plt.Axes | None = None,
    figsize: tuple[int, int] = (10, 5),
) -> plt.Figure:
    """Bar chart of peak GPU memory usage."""
    labels = []
    memory = []
    for r in comparison.results:
        labels.append(r.name)
        if r.profile is not None:
            memory.append(r.profile.peak_gpu_memory_mb)
        else:
            memory.append(0.0)

    create_fig = ax is None
    if create_fig:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    colors = plt.cm.Oranges(np.linspace(0.3, 0.8, len(labels)))
    bars = ax.bar(labels, memory, color=colors, edgecolor="gray", linewidth=0.5)

    for bar, val in zip(bars, memory):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{val:.1f} MB",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    ax.set_ylabel("Peak GPU Memory (MB)")
    ax.set_title("GPU Memory Usage")
    ax.tick_params(axis="x", rotation=30)

    if create_fig:
        fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_latency_vs_params(
    comparison: ComparisonResult,
    save_path: str | Path | None = None,
    ax: plt.Axes | None = None,
    figsize: tuple[int, int] = (8, 6),
) -> plt.Figure:
    """Scatter plot: latency vs parameter count (efficiency frontier)."""
    create_fig = ax is None
    if create_fig:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    for i, r in enumerate(comparison.results):
        if r.metrics is None or r.profile is None:
            continue
        params_m = r.metrics.total_params / 1e6
        latency = r.profile.latency_mean_ms
        ax.scatter(params_m, latency, s=120, zorder=5, label=r.name)
        ax.annotate(
            r.name,
            (params_m, latency),
            textcoords="offset points",
            xytext=(8, 4),
            fontsize=8,
        )

    ax.set_xlabel("Parameters (Millions)")
    ax.set_ylabel("Latency (ms)")
    ax.set_title("Latency vs Parameters")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    if create_fig:
        fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_macs_vs_latency(
    comparison: ComparisonResult,
    save_path: str | Path | None = None,
    ax: plt.Axes | None = None,
    figsize: tuple[int, int] = (8, 6),
) -> plt.Figure:
    """Scatter plot: compute (MACs) vs latency."""
    create_fig = ax is None
    if create_fig:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    for r in comparison.results:
        if r.metrics is None or r.profile is None:
            continue
        macs_g = r.metrics.total_macs_g
        latency = r.profile.latency_mean_ms
        ax.scatter(macs_g, latency, s=120, zorder=5, label=r.name)
        ax.annotate(
            r.name,
            (macs_g, latency),
            textcoords="offset points",
            xytext=(8, 4),
            fontsize=8,
        )

    ax.set_xlabel("MACs (Giga)")
    ax.set_ylabel("Latency (ms)")
    ax.set_title("Compute vs Latency")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    if create_fig:
        fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_radar_chart(
    comparison: ComparisonResult,
    metrics: list[str] | None = None,
    save_path: str | Path | None = None,
    figsize: tuple[int, int] = (8, 8),
) -> plt.Figure:
    """Radar/spider chart comparing multiple metrics across models.

    Args:
        comparison: Benchmark comparison results.
        metrics: List of metric names to include. Defaults to
            ["total_params", "macs_g", "latency_ms", "throughput_tok_s", "peak_gpu_mb"].
        save_path: Path to save.
        figsize: Figure size.
    """
    if metrics is None:
        metrics = ["total_params", "macs_g", "latency_ms", "throughput_tok_s", "peak_gpu_mb"]

    summary = comparison.summary_table()
    if not summary:
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, "No data", ha="center", va="center", fontsize=14)
        return fig

    available = [m for m in metrics if m in summary[0]]
    if not available:
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, "No matching metrics", ha="center", va="center", fontsize=14)
        return fig

    n_metrics = len(available)
    angles = np.linspace(0, 2 * np.pi, n_metrics, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=figsize, subplot_kw={"projection": "polar"})
    colors = plt.cm.Set2(np.linspace(0, 1, len(summary)))

    all_values: list[list[float]] = []
    for row in summary:
        vals = [row.get(m, 0) for m in available]
        all_values.append(vals)

    max_vals = [max(v[i] for v in all_values) if max(v[i] for v in all_values) > 0 else 1
                for i in range(n_metrics)]

    for idx, (row, color) in enumerate(zip(summary, colors)):
        raw = [row.get(m, 0) for m in available]
        normalized = [raw[i] / max_vals[i] for i in range(n_metrics)]
        normalized += normalized[:1]
        ax.plot(angles, normalized, "o-", linewidth=1.5, label=row["name"], color=color)
        ax.fill(angles, normalized, alpha=0.1, color=color)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(available, fontsize=8)
    ax.set_title("Model Comparison Radar", fontsize=13, y=1.08)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=8)

    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_all(
    comparison: ComparisonResult,
    save_dir: str | Path,
    prefix: str = "",
) -> list[Path]:
    """Generate all comparison charts and save to a directory.

    Returns list of saved file paths.
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []

    plots = [
        ("param_comparison.png", plot_param_comparison),
        ("latency_comparison.png", plot_latency_comparison),
        ("throughput_comparison.png", plot_throughput_comparison),
        ("memory_comparison.png", plot_memory_comparison),
        ("latency_vs_params.png", plot_latency_vs_params),
        ("macs_vs_latency.png", plot_macs_vs_latency),
        ("radar_chart.png", plot_radar_chart),
    ]

    for filename, plot_fn in plots:
        path = save_dir / f"{prefix}{filename}"
        try:
            fig = plot_fn(comparison, save_path=path)
            plt.close(fig)
            saved.append(path)
        except (ValueError, TypeError, OSError):
            plt.close("all")

    return saved
