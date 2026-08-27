from __future__ import annotations

from pathlib import Path

import numpy as np
import plotly.colors as pcolors
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from ..benchmark.comparison import ComparisonResult


def interactive_param_chart(
    comparison: ComparisonResult,
    log_scale: bool = False,
) -> go.Figure:
    """Interactive bar chart of parameter counts with hover tooltips."""
    labels = []
    params = []
    for r in comparison.results:
        labels.append(r.name)
        if r.metrics is not None:
            params.append(r.metrics.total_params / 1e6)
        else:
            params.append(0)

    fig = go.Figure(go.Bar(
        x=labels,
        y=params,
        text=[f"{p:.1f}M" for p in params],
        textposition="outside",
        hovertemplate="<b>%{x}</b><br>Params: %{y:.1f}M<extra></extra>",
        marker_color=pcolors.qualitative.Set2[:len(labels)] if len(labels) <= 8 else pcolors.qualitative.Plotly[:len(labels)],
    ))

    fig.update_layout(
        title="Parameter Count Comparison",
        xaxis_title="Model",
        yaxis_title="Parameters (Millions)",
        yaxis_type="log" if log_scale else "linear",
        template="plotly_white",
        hovermode="x",
    )
    return fig


def interactive_latency_chart(
    comparison: ComparisonResult,
) -> go.Figure:
    """Interactive latency bar chart with error bars."""
    labels = []
    means = []
    stds = []
    for r in comparison.results:
        labels.append(r.name)
        if r.profile is not None:
            means.append(r.profile.latency_mean_ms)
            stds.append(r.profile.latency_std_ms)
        else:
            means.append(0)
            stds.append(0)

    fig = go.Figure(go.Bar(
        x=labels,
        y=means,
        error_y={"type": "data", "array": stds, "visible": True},
        text=[f"{m:.2f}ms" for m in means],
        textposition="outside",
        hovertemplate="<b>%{x}</b><br>Latency: %{y:.2f} ms<br>Std: %{error_y.array:.2f} ms<extra></extra>",
        marker_color="#636EFA",
    ))

    fig.update_layout(
        title="Forward Pass Latency",
        xaxis_title="Model",
        yaxis_title="Latency (ms)",
        template="plotly_white",
        hovermode="x",
    )
    return fig


def interactive_scatter(
    comparison: ComparisonResult,
    x: str = "total_params",
    y: str = "latency_ms",
    size: str | None = "throughput_tok_s",
) -> go.Figure:
    """Interactive scatter plot with hover showing all metrics.

    Args:
        comparison: Benchmark results.
        x: X-axis metric name (from summary_table keys).
        y: Y-axis metric name.
        size: Optional metric for bubble size.
    """
    summary = comparison.summary_table()
    if not summary:
        fig = go.Figure()
        fig.update_layout(title="No data")
        return fig

    x_vals = [row.get(x, 0) for row in summary]
    y_vals = [row.get(y, 0) for row in summary]
    names = [row["name"] for row in summary]
    sizes_raw = [row.get(size, 100) if size else 100 for row in summary]

    max_s = max(sizes_raw) if sizes_raw else 1
    sizes_norm = [max(s / max_s * 40, 5) for s in sizes_raw]

    hover_texts = []
    for row in summary:
        lines = [f"<b>{row['name']}</b>"]
        for k, v in row.items():
            if k != "name":
                if isinstance(v, float):
                    lines.append(f"{k}: {v:.2f}")
                else:
                    lines.append(f"{k}: {v}")
        hover_texts.append("<br>".join(lines))

    fig = go.Figure(go.Scatter(
        x=x_vals,
        y=y_vals,
        mode="markers+text",
        text=names,
        textposition="top center",
        textfont={"size": 10},
        marker={"size": sizes_norm, "opacity": 0.7, "line": {"width": 1, "color": "gray"}},
        hovertext=hover_texts,
        hoverinfo="text",
    ))

    fig.update_layout(
        title=f"{y} vs {x}",
        xaxis_title=x,
        yaxis_title=y,
        template="plotly_white",
        hovermode="closest",
    )
    return fig


def interactive_radar(
    comparison: ComparisonResult,
    metrics: list[str] | None = None,
) -> go.Figure:
    """Interactive radar chart comparing multiple metrics."""
    if metrics is None:
        metrics = ["total_params", "macs_g", "latency_ms", "throughput_tok_s", "peak_gpu_mb"]

    summary = comparison.summary_table()
    if not summary:
        fig = go.Figure()
        fig.update_layout(title="No data")
        return fig

    available = [m for m in metrics if m in summary[0]]
    if not available:
        fig = go.Figure()
        fig.update_layout(title="No matching metrics")
        return fig

    all_values = [[row.get(m, 0) for m in available] for row in summary]
    max_vals = [max(v[i] for v in all_values) or 1 for i in range(len(available))]

    fig = go.Figure()
    for row in summary:
        raw = [row.get(m, 0) for m in available]
        norm = [raw[i] / max_vals[i] for i in range(len(available))]
        norm.append(norm[0])
        fig.add_trace(go.Scatterpolar(
            r=norm,
            theta=available + [available[0]],
            name=row["name"],
            fill="toself",
            opacity=0.3,
            hovertemplate="<b>%{theta}</b>: %{r:.3f}<extra>" + row["name"] + "</extra>",
        ))

    fig.update_layout(
        polar={"radialaxis": {"visible": True, "range": [0, 1.1]}},
        title="Model Comparison Radar",
        template="plotly_white",
        showlegend=True,
    )
    return fig


def interactive_heatmap(
    attn_weights: np.ndarray,
    tokens: list[str] | None = None,
    title: str = "Attention Weights",
) -> go.Figure:
    """Interactive attention heatmap with zoom and hover.

    Args:
        attn_weights: [seq_len, seq_len] numpy array.
        tokens: Token labels for axes.
        title: Chart title.
    """
    if tokens is None:
        tokens = [str(i) for i in range(attn_weights.shape[0])]

    fig = go.Figure(go.Heatmap(
        z=attn_weights,
        x=tokens,
        y=tokens,
        colorscale="Viridis",
        hovertemplate="Query: %{y}<br>Key: %{x}<br>Weight: %{z:.4f}<extra></extra>",
    ))

    fig.update_layout(
        title=title,
        xaxis_title="Key position",
        yaxis_title="Query position",
        template="plotly_white",
        width=700,
        height=600,
    )
    return fig


def interactive_dashboard(
    comparison: ComparisonResult,
) -> go.Figure:
    """Full interactive dashboard with subplots."""
    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=(
            "Parameter Count",
            "Latency",
            "Throughput",
            "Memory Usage",
        ),
        specs=[[{"type": "bar"}, {"type": "bar"}],
               [{"type": "bar"}, {"type": "bar"}]],
    )

    labels = [r.name for r in comparison.results]

    params = []
    for r in comparison.results:
        if r.metrics is not None:
            params.append(r.metrics.total_params / 1e6)
        else:
            params.append(0)

    fig.add_trace(go.Bar(
        x=labels, y=params,
        text=[f"{p:.1f}M" for p in params],
        textposition="outside",
        name="Parameters",
        hovertemplate="%{x}<br>%{y:.1f}M params<extra></extra>",
    ), row=1, col=1)

    latencies = []
    stds = []
    for r in comparison.results:
        if r.profile is not None:
            latencies.append(r.profile.latency_mean_ms)
            stds.append(r.profile.latency_std_ms)
        else:
            latencies.append(0)
            stds.append(0)

    fig.add_trace(go.Bar(
        x=labels, y=latencies,
        error_y={"type": "data", "array": stds, "visible": True},
        text=[f"{l:.2f}ms" for l in latencies],
        textposition="outside",
        name="Latency",
        hovertemplate="%{x}<br>%{y:.2f} ms<extra></extra>",
    ), row=1, col=2)

    throughputs = []
    for r in comparison.results:
        if r.profile is not None:
            throughputs.append(r.profile.throughput_mean_tokens_per_sec)
        else:
            throughputs.append(0)

    fig.add_trace(go.Bar(
        x=labels, y=throughputs,
        text=[f"{t:,.0f}" for t in throughputs],
        textposition="outside",
        name="Throughput",
        hovertemplate="%{x}<br>%{y:,.0f} tok/s<extra></extra>",
    ), row=2, col=1)

    memories = []
    for r in comparison.results:
        if r.profile is not None:
            memories.append(r.profile.peak_gpu_memory_mb)
        else:
            memories.append(0)

    fig.add_trace(go.Bar(
        x=labels, y=memories,
        text=[f"{m:.1f}MB" for m in memories],
        textposition="outside",
        name="GPU Memory",
        hovertemplate="%{x}<br>%{y:.1f} MB<extra></extra>",
    ), row=2, col=2)

    fig.update_layout(
        title="Benchmark Dashboard",
        template="plotly_white",
        showlegend=False,
        height=700,
    )
    return fig


def save_interactive(
    fig: go.Figure,
    path: str | Path,
    include_plotlyjs: bool = True,
) -> Path:
    """Save a Plotly figure as a standalone HTML file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(path), include_plotlyjs=include_plotlyjs)
    return path


def show(fig: go.Figure) -> None:
    """Display a Plotly figure (opens in browser or notebook)."""
    fig.show()
