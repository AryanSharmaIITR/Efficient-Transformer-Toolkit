from __future__ import annotations

from pathlib import Path

from .comparison import ComparisonResult
from .metric import ModelMetrics
from .profiler import AggregatedProfileResult


def _format_number(n: float, precision: int = 2) -> str:
    """Format large numbers with SI suffixes."""
    if isinstance(n, int):
        if abs(n) >= 1_000_000_000:
            return f"{n / 1_000_000_000:.{precision}f}B"
        if abs(n) >= 1_000_000:
            return f"{n / 1_000_000:.{precision}f}M"
        if abs(n) >= 1_000:
            return f"{n / 1_000:.{precision}f}K"
        return str(n)
    if abs(n) >= 1_000_000_000:
        return f"{n / 1_000_000_000:.{precision}f}G"
    if abs(n) >= 1_000_000:
        return f"{n / 1_000_000:.{precision}f}M"
    if abs(n) >= 1_000:
        return f"{n / 1_000:.{precision}f}K"
    return f"{n:.{precision}f}"


def _format_bytes(n: int) -> str:
    """Format bytes with appropriate unit."""
    if abs(n) >= 1_073_741_824:
        return f"{n / 1_073_741_824:.2f} GiB"
    if abs(n) >= 1_048_576:
        return f"{n / 1_048_576:.2f} MiB"
    if abs(n) >= 1024:
        return f"{n / 1024:.2f} KiB"
    return f"{n} B"


def _pad(s: str, width: int, align: str = "left") -> str:
    """Pad a string to a given width."""
    if align == "right":
        return s.rjust(width)
    if align == "center":
        return s.center(width)
    return s.ljust(width)


def _make_table(
    headers: list[str],
    rows: list[list[str]],
    col_widths: list[int] | None = None,
) -> str:
    """Build an ASCII table string."""
    if not rows:
        return "(no data)"

    num_cols = len(headers)
    if col_widths is None:
        col_widths = []
        for i in range(num_cols):
            max_w = len(headers[i])
            for row in rows:
                if i < len(row):
                    max_w = max(max_w, len(row[i]))
            col_widths.append(max_w + 2)

    lines: list[str] = []

    # Header
    header_line = " | ".join(_pad(h, col_widths[i]) for i, h in enumerate(headers))
    lines.append(header_line)

    # Separator
    sep = "-+-".join("-" * w for w in col_widths)
    lines.append(sep)

    # Rows
    for row in rows:
        cells = []
        for i in range(num_cols):
            cell = row[i] if i < len(row) else ""
            cells.append(_pad(cell, col_widths[i]))
        lines.append(" | ".join(cells))

    return "\n".join(lines)


def print_metrics(metrics: ModelMetrics, indent: str = "") -> str:
    """Format model metrics as a readable string."""
    lines = [
        f"{indent}Parameters:",
        f"{indent}  Total:     {_format_number(metrics.total_params)}",
        f"{indent}  Trainable: {_format_number(metrics.trainable_params)}",
        f"{indent}  Frozen:    {_format_number(metrics.non_trainable_params)}",
        f"{indent}  Size:      {_format_bytes(metrics.param_bytes)}",
        f"{indent}Compute:",
        f"{indent}  MACs:      {_format_number(metrics.total_macs_g)}G",
        f"{indent}  FLOPs:     {_format_number(metrics.total_flops_g)}G",
        f"{indent}Activations:",
        f"{indent}  Peak:      {_format_bytes(metrics.activations_bytes)}",
    ]
    if metrics.param_breakdown:
        lines.append(f"{indent}Breakdown:")
        for name, count in metrics.param_breakdown.items():
            lines.append(f"{indent}  {name}: {_format_number(count)}")
    return "\n".join(lines)


def print_profile(profile: AggregatedProfileResult, indent: str = "") -> str:
    """Format profiling results as a readable string."""
    lines = [
        f"{indent}Latency:",
        f"{indent}  Mean:  {profile.latency_mean_ms:.2f} ms",
        f"{indent}  Std:   {profile.latency_std_ms:.2f} ms",
        f"{indent}  P50:   {profile.latency_p50_ms:.2f} ms",
        f"{indent}  P95:   {profile.latency_p95_ms:.2f} ms",
        f"{indent}  P99:   {profile.latency_p99_ms:.2f} ms",
        f"{indent}Throughput:",
        f"{indent}  Mean:  {_format_number(profile.throughput_mean_tokens_per_sec)} tok/s",
        f"{indent}Memory:",
        f"{indent}  GPU Peak: {_format_bytes(int(profile.peak_gpu_memory_mb * 1048576))}",
        f"{indent}  CPU Peak: {_format_bytes(int(profile.peak_cpu_memory_mb * 1048576))}",
    ]
    return "\n".join(lines)


def print_comparison(comparison: ComparisonResult) -> str:
    """Format a full comparison result as a readable string."""
    lines: list[str] = []
    lines.append("=" * 60)
    lines.append(f"Benchmark Comparison ({len(comparison.results)} models)")
    lines.append(f"Timestamp: {comparison.timestamp}")
    lines.append("=" * 60)

    for result in comparison.results:
        lines.append("")
        lines.append(f"--- {result.name} ---")
        if result.metrics is not None:
            lines.append(print_metrics(result.metrics, indent="  "))
        if result.profile is not None:
            lines.append(print_profile(result.profile, indent="  "))

    # Summary table
    summary = comparison.summary_table()
    if summary:
        lines.append("")
        lines.append("=" * 60)
        lines.append("SUMMARY TABLE")
        lines.append("=" * 60)

        headers = list(summary[0].keys())
        rows: list[list[str]] = []
        for row in summary:
            cells = []
            for h in headers:
                val = row.get(h, "")
                if isinstance(val, (int, float)):
                    cells.append(str(val))
                else:
                    cells.append(str(val))
            rows.append(cells)

        lines.append(_make_table(headers, rows))

    return "\n".join(lines)


def generate_html_report(
    comparison: ComparisonResult,
    title: str = "Efficient Transformer Benchmark Report",
) -> str:
    """Generate an HTML report from comparison results."""
    summary = comparison.summary_table()

    table_rows = ""
    if summary:
        headers = list(summary[0].keys())
        table_rows += "          <tr>\n"
        for h in headers:
            table_rows += f"            <th>{h}</th>\n"
        table_rows += "          </tr>\n"

        for row in summary:
            table_rows += "          <tr>\n"
            for h in headers:
                val = row.get(h, "")
                table_rows += f"            <td>{val}</td>\n"
            table_rows += "          </tr>\n"

    # Detail sections
    details = ""
    for result in comparison.results:
        details += "      <div class=\"model-detail\">\n"
        details += f"        <h3>{result.name}</h3>\n"

        if result.metrics is not None:
            m = result.metrics
            details += "        <div class=\"metric-group\">\n"
            details += "          <h4>Parameters</h4>\n"
            details += f"          <p>Total: {_format_number(m.total_params)} | "
            details += f"Trainable: {_format_number(m.trainable_params)} | "
            details += f"Size: {_format_bytes(m.param_bytes)}</p>\n"
            details += "          <h4>Compute</h4>\n"
            details += f"          <p>MACs: {_format_number(m.total_macs_g)}G | "
            details += f"FLOPs: {_format_number(m.total_flops_g)}G</p>\n"
            details += "          <h4>Activations</h4>\n"
            details += f"          <p>Peak: {_format_bytes(m.activations_bytes)}</p>\n"

            if m.param_breakdown:
                details += "          <h4>Parameter Breakdown</h4>\n"
                details += "          <ul>\n"
                for name, count in m.param_breakdown.items():
                    details += f"            <li>{name}: {_format_number(count)}</li>\n"
                details += "          </ul>\n"
            details += "        </div>\n"

        if result.profile is not None:
            p = result.profile
            details += "        <div class=\"profile-group\">\n"
            details += "          <h4>Profiling</h4>\n"
            details += f"          <p>Latency: {p.latency_mean_ms:.2f} ms "
            details += f"(+/- {p.latency_std_ms:.2f})</p>\n"
            details += f"          <p>Throughput: {_format_number(p.throughput_mean_tokens_per_sec)} tok/s</p>\n"
            details += f"          <p>GPU Peak: {_format_bytes(int(p.peak_gpu_memory_mb * 1048576))}</p>\n"
            details += "        </div>\n"

        details += "      </div>\n"

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>{title}</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; margin: 2rem; background: #fafafa; color: #333; }}
    h1 {{ border-bottom: 2px solid #2563eb; padding-bottom: 0.5rem; }}
    h2 {{ color: #2563eb; }}
    table {{ border-collapse: collapse; width: 100%; margin: 1rem 0; }}
    th, td {{ border: 1px solid #ddd; padding: 0.5rem 1rem; text-align: left; }}
    th {{ background: #2563eb; color: white; }}
    tr:nth-child(even) {{ background: #f0f0f0; }}
    .model-detail {{ background: white; border: 1px solid #ddd; border-radius: 8px; padding: 1.5rem; margin: 1rem 0; }}
    .model-detail h3 {{ margin-top: 0; color: #2563eb; }}
    .metric-group, .profile-group {{ margin: 0.5rem 0; }}
    h4 {{ margin: 0.5rem 0 0.25rem 0; color: #555; }}
    ul {{ margin: 0.25rem 0; padding-left: 1.5rem; }}
    .timestamp {{ color: #888; font-size: 0.9rem; }}
  </style>
</head>
<body>
  <h1>{title}</h1>
  <p class="timestamp">Generated: {comparison.timestamp}</p>
  <p>Models compared: {len(comparison.results)}</p>

  <h2>Summary</h2>
  <table>
    <tbody>
{table_rows}    </tbody>
  </table>

  <h2>Details</h2>
{details}
</body>
</html>"""
    return html


def save_html_report(
    comparison: ComparisonResult,
    path: str | Path,
    title: str = "Efficient Transformer Benchmark Report",
) -> Path:
    """Generate and save an HTML report."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    html = generate_html_report(comparison, title)
    path.write_text(html, encoding="utf-8")
    return path


def save_json_report(comparison: ComparisonResult, path: str | Path) -> Path:
    """Save comparison results as JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    comparison.save(path)
    return path


def load_and_print(path: str | Path) -> str:
    """Load a saved comparison JSON and return formatted string."""
    result = ComparisonResult.load(path)
    return print_comparison(result)


def ranking_table(
    comparison: ComparisonResult,
    sort_by: str = "latency_ms",
    ascending: bool = True,
) -> str:
    """Generate a ranked table sorted by a specific metric.

    sort_by can be: latency_ms, throughput_tok_s, total_params, macs_g, peak_gpu_mb.
    """
    summary = comparison.summary_table()
    if not summary:
        return "(no data)"

    # Determine sort key
    if sort_by in summary[0]:
        key_func = lambda row: row.get(sort_by, float("inf") if ascending else 0)
        summary.sort(key=key_func, reverse=not ascending)

    headers = list(summary[0].keys())
    rows = []
    for i, row in enumerate(summary):
        cells = [str(i + 1)]
        for h in headers:
            val = row.get(h, "")
            cells.append(str(val))
        rows.append(cells)

    headers = ["Rank"] + headers
    return _make_table(headers, rows)


def compact_summary(comparison: ComparisonResult) -> str:
    """Print a one-line-per-model compact summary."""
    summary = comparison.summary_table()
    lines = []
    for row in summary:
        name = row.get("name", "?")
        params = _format_number(row.get("total_params", 0))
        macs = f"{row.get('macs_g', 0)}G"
        lat = f"{row.get('latency_ms', 0)}ms"
        tok_s = f"{row.get('throughput_tok_s', 0)} tok/s"
        gpu = f"{row.get('peak_gpu_mb', 0)}MB"
        lines.append(f"  {name:<25s} | {params:>8s} params | {macs:>8s} MACs | {lat:>10s} | {tok_s:>15s} | {gpu:>10s}")
    return "\n".join(lines)
