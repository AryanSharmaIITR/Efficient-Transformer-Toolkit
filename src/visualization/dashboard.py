from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec

from ..benchmark.comparison import ComparisonResult
from ..benchmark.metric import ModelMetrics
from ..benchmark.profiler import AggregatedProfileResult
from .comparison_plots import (
    plot_latency_comparison,
    plot_memory_comparison,
    plot_param_comparison,
    plot_throughput_comparison,
)


def create_benchmark_dashboard(
    comparison: ComparisonResult,
    save_path: str | Path | None = None,
    figsize: tuple[int, int] = (18, 12),
    title: str = "Benchmark Dashboard",
) -> plt.Figure:
    """Create a 2x3 grid dashboard with key benchmark metrics.

    Panels: Parameters, Latency, Throughput, Memory, Radar, Summary Table.
    """
    fig = plt.figure(figsize=figsize)
    fig.suptitle(title, fontsize=16, fontweight="bold", y=0.98)

    gs = GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.3)

    ax1 = fig.add_subplot(gs[0, 0])
    plot_param_comparison(comparison, ax=ax1)

    ax2 = fig.add_subplot(gs[0, 1])
    plot_latency_comparison(comparison, ax=ax2)

    ax3 = fig.add_subplot(gs[0, 2])
    plot_throughput_comparison(comparison, ax=ax3)

    ax4 = fig.add_subplot(gs[1, 0])
    plot_memory_comparison(comparison, ax=ax4)

    ax5 = fig.add_subplot(gs[1, 1], projection="polar")
    _draw_radar_in_axes(comparison, ax5)

    ax6 = fig.add_subplot(gs[1, 2])
    _draw_summary_table(comparison, ax6)

    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def create_model_profile_dashboard(
    result_name: str,
    metrics: ModelMetrics | None = None,
    profile: AggregatedProfileResult | None = None,
    save_path: str | Path | None = None,
    figsize: tuple[int, int] = (16, 10),
) -> plt.Figure:
    """Single-model deep-dive dashboard.

    Panels: Parameter breakdown (pie), Latency histogram, Memory gauge, Summary text.
    """
    fig = plt.figure(figsize=figsize)
    fig.suptitle(f"Model Profile: {result_name}", fontsize=16, fontweight="bold", y=0.98)

    gs = GridSpec(2, 3, figure=fig, hspace=0.4, wspace=0.35)

    ax1 = fig.add_subplot(gs[0, 0])
    if metrics and metrics.param_breakdown:
        _draw_param_breakdown_pie(metrics, ax1)
    else:
        ax1.text(0.5, 0.5, "No breakdown data", ha="center", va="center", transform=ax1.transAxes)
        ax1.set_title("Parameter Breakdown")

    ax2 = fig.add_subplot(gs[0, 1])
    if profile and profile.runs:
        _draw_latency_histogram(profile, ax2)
    else:
        ax2.text(0.5, 0.5, "No profile data", ha="center", va="center", transform=ax2.transAxes)
        ax2.set_title("Latency Distribution")

    ax3 = fig.add_subplot(gs[0, 2])
    if metrics:
        _draw_metrics_summary_text(metrics, ax3)
    else:
        ax3.text(0.5, 0.5, "No metrics", ha="center", va="center", transform=ax3.transAxes)
    ax3.set_title("Metrics Summary")

    ax4 = fig.add_subplot(gs[1, :])
    if profile and profile.runs:
        _draw_latency_timeline(profile, ax4)
    else:
        ax4.text(0.5, 0.5, "No run data", ha="center", va="center", transform=ax4.transAxes)
    ax4.set_title("Latency Per Run")

    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def create_comparison_dashboard(
    comparison: ComparisonResult,
    save_path: str | Path | None = None,
    figsize: tuple[int, int] = (16, 10),
) -> plt.Figure:
    """Side-by-side comparison dashboard with efficiency metrics."""
    fig = plt.figure(figsize=figsize)
    fig.suptitle("Model Comparison Dashboard", fontsize=16, fontweight="bold", y=0.98)

    gs = GridSpec(2, 2, figure=fig, hspace=0.35, wspace=0.3)

    ax1 = fig.add_subplot(gs[0, 0])
    _draw_efficiency_scatter(comparison, ax1)

    ax2 = fig.add_subplot(gs[0, 1])
    _draw_compute_comparison(comparison, ax2)

    ax3 = fig.add_subplot(gs[1, 0])
    _draw_memory_vs_throughput(comparison, ax3)

    ax4 = fig.add_subplot(gs[1, 1])
    _draw_summary_table(comparison, ax4)

    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def save_dashboard(
    figure: plt.Figure,
    path: str | Path,
    dpi: int = 150,
    format: str | None = None,
) -> Path:
    """Save a matplotlib figure to disk."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=dpi, bbox_inches="tight", format=format)
    return path


def _draw_radar_in_axes(
    comparison: ComparisonResult,
    ax: plt.Axes,
) -> None:
    """Draw a radar chart in polar axes."""
    summary = comparison.summary_table()
    if not summary:
        return

    metrics = ["total_params", "macs_g", "latency_ms", "throughput_tok_s", "peak_gpu_mb"]
    available = [m for m in metrics if m in summary[0]]
    if not available:
        return

    n = len(available)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
    angles += angles[:1]

    colors = plt.cm.Set2(np.linspace(0, 1, len(summary)))
    all_values = [[row.get(m, 0) for m in available] for row in summary]
    max_vals = [max(v[i] for v in all_values) or 1 for i in range(n)]

    for row, color in zip(summary, colors):
        raw = [row.get(m, 0) for m in available]
        norm = [raw[i] / max_vals[i] for i in range(n)]
        norm += norm[:1]
        ax.plot(angles, norm, "o-", linewidth=1.2, label=row["name"], color=color)
        ax.fill(angles, norm, alpha=0.08, color=color)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(available, fontsize=7)
    ax.set_title("Radar", fontsize=11, pad=15)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=7)


def _draw_summary_table(
    comparison: ComparisonResult,
    ax: plt.Axes,
) -> None:
    """Draw a summary text table in axes."""
    ax.axis("off")
    summary = comparison.summary_table()
    if not summary:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", fontsize=12)
        return

    headers = list(summary[0].keys())
    short_headers = {
        "name": "Model",
        "total_params": "Params",
        "param_size_mb": "Size(MB)",
        "macs_g": "MACs(G)",
        "flops_g": "FLOPs(G)",
        "latency_ms": "Lat(ms)",
        "latency_std_ms": "Std(ms)",
        "throughput_tok_s": "tok/s",
        "peak_gpu_mb": "GPU(MB)",
    }
    display_headers = [short_headers.get(h, h) for h in headers]

    cell_text = []
    for row in summary:
        cells = []
        for h in headers:
            val = row.get(h, "")
            if isinstance(val, float):
                cells.append(f"{val:.2f}")
            else:
                cells.append(str(val))
        cell_text.append(cells)

    table = ax.table(
        cellText=cell_text,
        colLabels=display_headers,
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1.0, 1.5)

    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_facecolor("#4472C4")
            cell.set_text_props(color="white", fontweight="bold")
        elif row % 2 == 0:
            cell.set_facecolor("#D9E2F3")

    ax.set_title("Summary", fontsize=11, pad=10)


def _draw_param_breakdown_pie(metrics: ModelMetrics, ax: plt.Axes) -> None:
    """Pie chart of parameter breakdown by module."""
    breakdown = metrics.param_breakdown
    labels = list(breakdown.keys())
    sizes = list(breakdown.values())

    if not labels:
        ax.text(0.5, 0.5, "No breakdown", ha="center", va="center", transform=ax.transAxes)
        ax.set_title("Parameter Breakdown")
        return

    if len(labels) > 8:
        top_n = 7
        sorted_pairs = sorted(zip(sizes, labels), reverse=True)
        top = sorted_pairs[:top_n]
        rest_sum = sum(s for s, _ in sorted_pairs[top_n:])
        labels = [l for _, l in top] + ["other"]
        sizes = [s for s, _ in top] + [rest_sum]

    colors = plt.cm.Set3(np.linspace(0, 1, len(labels)))
    _wedges, _texts, autotexts = ax.pie(
        sizes,
        labels=labels,
        autopct=lambda pct: f"{pct:.1f}%",
        colors=colors,
        pctdistance=0.85,
        textprops={"fontsize": 7},
    )
    for t in autotexts:
        t.set_fontsize(6)
    ax.set_title("Parameter Breakdown", fontsize=11)


def _draw_latency_histogram(profile: AggregatedProfileResult, ax: plt.Axes) -> None:
    """Histogram of individual run latencies."""
    latencies = [r.latency_ms for r in profile.runs]
    if not latencies:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        ax.set_title("Latency Distribution")
        return

    ax.hist(latencies, bins=min(len(latencies), 15), color="#4472C4",
            edgecolor="white", alpha=0.85)
    ax.axvline(profile.latency_mean_ms, color="red", linestyle="--",
               linewidth=1.2, label=f"Mean: {profile.latency_mean_ms:.2f}ms")
    ax.set_xlabel("Latency (ms)")
    ax.set_ylabel("Count")
    ax.set_title("Latency Distribution")
    ax.legend(fontsize=8)


def _draw_metrics_summary_text(metrics: ModelMetrics, ax: plt.Axes) -> None:
    """Text summary of key metrics."""
    ax.axis("off")
    lines = [
        f"Total Params:  {metrics.total_params:,}",
        f"Trainable:     {metrics.trainable_params:,}",
        f"Param Size:    {metrics.param_size_mb:.2f} MB",
        "",
        f"MACs:          {metrics.total_macs_g:.2f} G",
        f"FLOPs:         {metrics.total_flops_g:.2f} G",
        "",
        f"Peak Act Mem:  {metrics.activation_size_mb:.2f} MB",
    ]
    ax.text(
        0.05, 0.95, "\n".join(lines),
        ha="left", va="top",
        fontsize=10,
        fontfamily="monospace",
        transform=ax.transAxes,
        bbox={"boxstyle": "round,pad=0.5", "facecolor": "#E8EEF7", "alpha": 0.8},
    )


def _draw_latency_timeline(profile: AggregatedProfileResult, ax: plt.Axes) -> None:
    """Line plot of latency across runs."""
    latencies = [r.latency_ms for r in profile.runs]
    if not latencies:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        ax.set_title("Latency Per Run")
        return

    runs = list(range(1, len(latencies) + 1))
    ax.plot(runs, latencies, "o-", color="#4472C4", linewidth=1.5, markersize=6)
    ax.axhline(profile.latency_mean_ms, color="red", linestyle="--",
               linewidth=1, label=f"Mean: {profile.latency_mean_ms:.2f}ms")
    ax.fill_between(
        runs,
        [profile.latency_mean_ms - profile.latency_std_ms] * len(runs),
        [profile.latency_mean_ms + profile.latency_std_ms] * len(runs),
        alpha=0.15,
        color="red",
        label=f"+/- Std: {profile.latency_std_ms:.2f}ms",
    )
    ax.set_xlabel("Run")
    ax.set_ylabel("Latency (ms)")
    ax.set_title("Latency Per Run")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)


def _draw_efficiency_scatter(comparison: ComparisonResult, ax: plt.Axes) -> None:
    """Scatter: latency vs params with throughput as bubble size."""
    for r in comparison.results:
        if r.metrics is None or r.profile is None:
            continue
        params = r.metrics.total_params / 1e6
        latency = r.profile.latency_mean_ms
        throughput = max(r.profile.throughput_mean_tokens_per_sec, 1)
        size = min(throughput / 50, 500)
        ax.scatter(params, latency, s=size, alpha=0.7, label=r.name, edgecolors="gray")
        ax.annotate(r.name, (params, latency), textcoords="offset points",
                    xytext=(6, 3), fontsize=7)

    ax.set_xlabel("Parameters (M)")
    ax.set_ylabel("Latency (ms)")
    ax.set_title("Efficiency (bubble size = throughput)")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)


def _draw_compute_comparison(comparison: ComparisonResult, ax: plt.Axes) -> None:
    """Stacked bar: MACs vs FLOPs per model."""
    labels = []
    macs_vals = []
    flops_vals = []
    for r in comparison.results:
        labels.append(r.name)
        if r.metrics is not None:
            macs_vals.append(r.metrics.total_macs_g)
            flops_vals.append(max(r.metrics.total_flops_g - r.metrics.total_macs_g, 0))
        else:
            macs_vals.append(0)
            flops_vals.append(0)

    x = np.arange(len(labels))
    width = 0.6
    ax.bar(x, macs_vals, width, label="MACs (G)", color="#4472C4", alpha=0.85)
    ax.bar(x, flops_vals, width, bottom=macs_vals, label="Extra FLOPs (G)", color="#ED7D31", alpha=0.7)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, fontsize=8)
    ax.set_ylabel("Compute (G)")
    ax.set_title("Compute Breakdown")
    ax.legend(fontsize=8)


def _draw_memory_vs_throughput(comparison: ComparisonResult, ax: plt.Axes) -> None:
    """Scatter: GPU memory vs throughput."""
    for r in comparison.results:
        if r.profile is None:
            continue
        mem = r.profile.peak_gpu_memory_mb
        tput = r.profile.throughput_mean_tokens_per_sec
        ax.scatter(mem, tput, s=120, zorder=5, label=r.name, edgecolors="gray")
        ax.annotate(r.name, (mem, tput), textcoords="offset points",
                    xytext=(6, 3), fontsize=7)

    ax.set_xlabel("Peak GPU Memory (MB)")
    ax.set_ylabel("Throughput (tok/s)")
    ax.set_title("Memory vs Throughput")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)
