from __future__ import annotations

import json
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from torch import nn

from .metric import (
    ModelMetrics,
    compute_model_metrics,
    count_parameters,
    parameter_breakdown,
)
from .profiler import (
    AggregatedProfileResult,
    profile_model_forward,
)


@dataclass
class BenchmarkConfig:
    """Configuration for a single benchmark run."""

    name: str
    model: nn.Module | None = None
    config: Any | None = None
    batch_size: int = 1
    seq_len: int = 512
    vocab_size: int = 50257
    warmup_runs: int = 3
    num_runs: int = 10
    causal: bool = True
    ffn_activation: str = "swiglu"
    n_encoder_layers: int | None = None
    n_decoder_layers: int | None = None
    device: torch.device | None = None


@dataclass
class BenchmarkResult:
    """Complete result for a single model benchmark."""

    name: str
    metrics: ModelMetrics | None = None
    profile: AggregatedProfileResult | None = None
    config_snapshot: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {"name": self.name, "config": self.config_snapshot}
        if self.metrics is not None:
            result["metrics"] = self.metrics.to_dict()
        if self.profile is not None:
            result["profile"] = self.profile.to_dict()
        return result


@dataclass
class ComparisonResult:
    """Results from comparing multiple models."""

    results: list[BenchmarkResult] = field(default_factory=list)
    timestamp: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "num_models": len(self.results),
            "results": [r.to_dict() for r in self.results],
        }

    def summary_table(self) -> list[dict[str, Any]]:
        """Generate a summary table with key metrics for each model."""
        rows = []
        for r in self.results:
            row: dict[str, Any] = {"name": r.name}
            if r.metrics is not None:
                row["total_params"] = r.metrics.total_params
                row["param_size_mb"] = round(r.metrics.param_size_mb, 2)
                row["macs_g"] = round(r.metrics.total_macs_g, 2)
                row["flops_g"] = round(r.metrics.total_flops_g, 2)
            if r.profile is not None:
                row["latency_ms"] = round(r.profile.latency_mean_ms, 2)
                row["latency_std_ms"] = round(r.profile.latency_std_ms, 2)
                row["throughput_tok_s"] = round(r.profile.throughput_mean_tokens_per_sec, 0)
                row["peak_gpu_mb"] = round(r.profile.peak_gpu_memory_mb, 2)
            rows.append(row)
        return rows

    def save(self, path: str | Path) -> None:
        """Save results to a JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: str | Path) -> ComparisonResult:
        """Load results from a JSON file."""
        with open(path) as f:
            data = json.load(f)
        result = cls(timestamp=data.get("timestamp", ""))
        for r_data in data.get("results", []):
            br = BenchmarkResult(name=r_data["name"])
            br.config_snapshot = r_data.get("config", {})

            # to_dict() writes derived/renamed keys (macs_g, param_size_mb,
            # num_runs, ...) alongside the real dataclass fields -- pull out
            # just the real fields so save() -> load() round-trips instead
            # of silently discarding everything but `name`/`config`.
            m = r_data.get("metrics")
            if m is not None:
                br.metrics = ModelMetrics(
                    total_params=m.get("total_params", 0),
                    trainable_params=m.get("trainable_params", 0),
                    non_trainable_params=m.get("non_trainable_params", 0),
                    param_bytes=m.get("param_bytes", 0),
                    macs=m.get("macs", 0),
                    flops=m.get("flops", 0),
                    activations_bytes=m.get("activations_bytes", 0),
                    param_breakdown=m.get("param_breakdown", {}),
                )

            p = r_data.get("profile")
            if p is not None:
                br.profile = AggregatedProfileResult(
                    latency_mean_ms=p.get("latency_mean_ms", 0.0),
                    latency_std_ms=p.get("latency_std_ms", 0.0),
                    latency_p50_ms=p.get("latency_p50_ms", 0.0),
                    latency_p95_ms=p.get("latency_p95_ms", 0.0),
                    latency_p99_ms=p.get("latency_p99_ms", 0.0),
                    throughput_mean_tokens_per_sec=p.get("throughput_mean_tokens_per_sec", 0.0),
                    peak_gpu_memory_mb=p.get("peak_gpu_memory_mb", 0.0),
                    peak_cpu_memory_mb=p.get("peak_cpu_memory_mb", 0.0),
                    metadata=p.get("metadata", {}),
                )

            result.results.append(br)
        return result


def benchmark_model(
    benchmark_config: BenchmarkConfig,
    profile: bool = True,
    metrics: bool = True,
) -> BenchmarkResult:
    """Run a full benchmark on a single model.

    Args:
        benchmark_config: Configuration specifying model and parameters.
        profile: Whether to run speed/memory profiling.
        metrics: Whether to compute parameter/FLOPS metrics.

    Returns:
        BenchmarkResult with all collected data.
    """
    model = benchmark_config.config
    device = benchmark_config.device

    if model is None:
        raise ValueError("benchmark_config must have a 'config' attribute with the model instance")

    # Compute metrics
    model_metrics = None
    if metrics:
        try:
            model_metrics = compute_model_metrics(
                model=model,
                seq_len=benchmark_config.seq_len,
                batch_size=benchmark_config.batch_size,
                causal=benchmark_config.causal,
                ffn_activation=benchmark_config.ffn_activation,
                n_encoder_layers=benchmark_config.n_encoder_layers,
                n_decoder_layers=benchmark_config.n_decoder_layers,
            )
        except (ValueError, AttributeError, RuntimeError):
            # Fallback to basic parameter count
            param_info = count_parameters(model)
            model_metrics = ModelMetrics(
                total_params=param_info["total"],
                trainable_params=param_info["trainable"],
                non_trainable_params=param_info["non_trainable"],
                param_bytes=param_info["bytes"],
                param_breakdown=parameter_breakdown(model),
            )

    # Profile
    profile_result = None
    if profile:
        try:
            profile_result = profile_model_forward(
                model=model,
                batch_size=benchmark_config.batch_size,
                seq_len=benchmark_config.seq_len,
                vocab_size=benchmark_config.vocab_size,
                warmup_runs=benchmark_config.warmup_runs,
                num_runs=benchmark_config.num_runs,
                device=device,
                causal=benchmark_config.causal,
            )
        except (ValueError, RuntimeError, TypeError) as e:
            # Profiling failed (e.g., model has bugs) - record empty
            profile_result = AggregatedProfileResult(
                metadata={"error": str(e)},
            )

    # Config snapshot
    config_obj = getattr(model, "config", None)
    config_snapshot = {}
    if config_obj is not None:
        if hasattr(config_obj, "__dict__"):
            config_snapshot = {
                k: v for k, v in config_obj.__dict__.items()
                if not k.startswith("_")
            }
        elif hasattr(config_obj, "__dataclass_fields__"):
            config_snapshot = {
                k: getattr(config_obj, k)
                for k in config_obj.__dataclass_fields__
            }

    return BenchmarkResult(
        name=benchmark_config.name,
        metrics=model_metrics,
        profile=profile_result,
        config_snapshot=config_snapshot,
    )


def compare_models(
    configs: list[BenchmarkConfig],
    profile: bool = True,
    metrics: bool = True,
) -> ComparisonResult:
    """Benchmark and compare multiple models.

    Args:
        configs: List of BenchmarkConfig, one per model.
        profile: Whether to run profiling.
        metrics: Whether to compute metrics.

    Returns:
        ComparisonResult with all results.
    """
    results: list[BenchmarkResult] = []

    for cfg in configs:
        result = benchmark_model(cfg, profile=profile, metrics=metrics)
        results.append(result)

    return ComparisonResult(
        results=results,
        timestamp=time.strftime("%Y-%m-%dT%H:%M:%S"),
    )


def ablation_study(
    base_config: Any,
    model_factory: Callable[[Any], nn.Module],
    param_name: str,
    param_values: list[Any],
    fixed_params: dict[str, Any] | None = None,
    seq_len: int = 512,
    batch_size: int = 1,
    vocab_size: int = 50257,
    warmup_runs: int = 3,
    num_runs: int = 10,
    device: torch.device | None = None,
) -> ComparisonResult:
    """Run an ablation study by varying one parameter.

    Creates models with different values of a single parameter
    and benchmarks each.

    Args:
        base_config: Base configuration object (e.g., TransformerConfig).
        model_factory: Callable that takes a config and returns an nn.Module.
        param_name: Name of the parameter to vary.
        param_values: List of values to try for the parameter.
        fixed_params: Additional params to set on each config.
        seq_len: Sequence length for profiling.
        batch_size: Batch size for profiling.
        vocab_size: Vocabulary size.
        warmup_runs: Warmup iterations.
        num_runs: Timed iterations.
        device: Target device.

    Returns:
        ComparisonResult with one entry per parameter value.
    """
    benchmark_configs: list[BenchmarkConfig] = []

    for val in param_values:
        # Create config with the varied parameter
        cfg = _set_config_attr(base_config, param_name, val)
        if fixed_params:
            for k, v in fixed_params.items():
                cfg = _set_config_attr(cfg, k, v)

        try:
            model = model_factory(cfg)
        except (ValueError, TypeError, RuntimeError):
            # Skip configs that fail to instantiate
            continue

        name = f"{param_name}={val}"
        benchmark_configs.append(BenchmarkConfig(
            name=name,
            config=model,
            batch_size=batch_size,
            seq_len=seq_len,
            vocab_size=vocab_size,
            warmup_runs=warmup_runs,
            num_runs=num_runs,
            device=device,
        ))

    return compare_models(benchmark_configs)


def attention_type_comparison(
    vocab_size: int = 50257,
    d_model: int = 768,
    n_heads: int = 12,
    n_layers: int = 6,
    seq_len: int = 512,
    batch_size: int = 1,
    attention_types: list[str] | None = None,
    device: torch.device | None = None,
) -> ComparisonResult:
    """Compare different attention mechanisms side-by-side.

    Creates identical models with only the attention type varying.
    """
    try:
        from ..models.transformer import Transformer, TransformerConfig
    except ImportError as exc:
        raise ImportError("Transformer model not available. Ensure src.models is importable.") from exc

    if attention_types is None:
        attention_types = ["flashv1", "flashv2", "alibi", "gqa", "mqa"]

    benchmark_configs: list[BenchmarkConfig] = []

    for attn_type in attention_types:
        try:
            config = TransformerConfig(
                vocab_size=vocab_size,
                d_model=d_model,
                n_heads=n_heads,
                n_layers=n_layers,
                attn_type=attn_type,
                # get_attention() overrides attn_type with
                # RotaryPEMultiHeadAttention when pos_encoding == "rope"
                # (its default) -- pin sinusoidal so this function actually
                # exercises each requested attn_type instead of silently
                # benchmarking the same rotary attention every time.
                pos_encoding="sinusoidal",
                causal=True,
            )
            model = Transformer(config, use_encoder=False, use_decoder=True)

            benchmark_configs.append(BenchmarkConfig(
                name=f"attn={attn_type}",
                config=model,
                batch_size=batch_size,
                seq_len=seq_len,
                vocab_size=vocab_size,
                device=device,
            ))
        except (ValueError, TypeError, RuntimeError):
            continue

    return compare_models(benchmark_configs)


def _set_config_attr(config: Any, attr_name: str, value: Any) -> Any:
    """Set an attribute on a config object (dataclass or otherwise)."""
    if hasattr(config, attr_name) or hasattr(config, "__dataclass_fields__"):
        # For dataclasses, create a new instance with the updated value
        if hasattr(config, "__dataclass_fields__") and hasattr(config, "replace"):
            return config.replace(**{attr_name: value})
        elif hasattr(config, "__dataclass_fields__"):
            # Manual dataclass update
            import dataclasses
            fields = dataclasses.fields(config)
            field_names = [f.name for f in fields]
            if attr_name not in field_names:
                raise ValueError(f"Unknown field: {attr_name}")
            values = {}
            for f in fields:
                if f.name == attr_name:
                    values[f.name] = value
                else:
                    values[f.name] = getattr(config, f.name)
            return type(config)(**values)
        else:
            # Plain object - set attribute directly
            new_config = type(config)(**{
                k: v for k, v in config.__dict__.items() if not k.startswith("_")
            })
            setattr(new_config, attr_name, value)
            return new_config
    else:
        raise ValueError(f"Cannot set unknown attribute '{attr_name}' on config")
