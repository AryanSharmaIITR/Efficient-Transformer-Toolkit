from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from torch import nn


@dataclass
class ModelMetrics:
    """Aggregated metrics for a model."""

    total_params: int = 0
    trainable_params: int = 0
    non_trainable_params: int = 0
    param_bytes: int = 0
    macs: int = 0
    flops: int = 0
    activations_bytes: int = 0
    param_breakdown: dict[str, int] = field(default_factory=dict)

    @property
    def param_size_mb(self) -> float:
        return self.param_bytes / (1024 * 1024)

    @property
    def activation_size_mb(self) -> float:
        return self.activations_bytes / (1024 * 1024)

    @property
    def total_flops_g(self) -> float:
        return self.flops / 1e9

    @property
    def total_macs_g(self) -> float:
        return self.macs / 1e9

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_params": self.total_params,
            "trainable_params": self.trainable_params,
            "non_trainable_params": self.non_trainable_params,
            "param_bytes": self.param_bytes,
            "param_size_mb": self.param_size_mb,
            "macs": self.macs,
            "macs_g": self.total_macs_g,
            "flops": self.flops,
            "flops_g": self.total_flops_g,
            "activations_bytes": self.activations_bytes,
            "activation_size_mb": self.activation_size_mb,
            "param_breakdown": self.param_breakdown,
        }


def count_parameters(model: nn.Module) -> dict[str, int]:
    """Count total, trainable, and non-trainable parameters.

    Returns:
        Dictionary with keys: total, trainable, non_trainable, bytes.
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    non_trainable = total - trainable
    param_bytes = sum(
        p.numel() * p.element_size() for p in model.parameters()
    )
    return {
        "total": total,
        "trainable": trainable,
        "non_trainable": non_trainable,
        "bytes": param_bytes,
    }


def parameter_breakdown(model: nn.Module) -> dict[str, int]:
    """Break down parameter counts by top-level named module.

    Returns a dict mapping module name -> parameter count for each
    direct child of the model that has parameters.
    """
    breakdown: dict[str, int] = {}
    for name, module in model.named_children():
        count = sum(p.numel() for p in module.parameters())
        if count > 0:
            breakdown[name] = count
    # Add non-module parameters (e.g. top-level buffers used as params)
    top_level_count = sum(
        p.numel() for p in model.parameters(recurse=False)
    )
    if top_level_count > 0:
        breakdown["__self__"] = top_level_count
    return breakdown


def estimate_linear_macs(in_features: int, out_features: int, bias: bool = False) -> int:
    """MACs for a single nn.Linear layer (multiply-accumulate = 1 MAC)."""
    macs = in_features * out_features
    if bias:
        macs += out_features
    return macs


def estimate_attention_macs(
    seq_len: int,
    d_model: int,
    n_heads: int,
    has_mask: bool = True,
    causal: bool = False,
) -> int:
    """Estimate MACs for standard scaled dot-product attention.

    Includes Q/K/V projections, attention scores, and output projection.
    Causal masking halves the average attention MACs approximately.
    """
    head_dim = d_model // n_heads
    # Q, K, V projections: 3 * seq_len * d_model * d_model
    qkv_proj = 3 * seq_len * d_model * d_model
    # Attention scores: Q @ K^T -> [B, n_heads, T, T] each element is head_dim MACs
    attn_scores = n_heads * seq_len * seq_len * head_dim
    if causal:
        attn_scores = attn_scores // 2
    # Attention @ V: same shape as scores
    attn_weights_v = n_heads * seq_len * seq_len * head_dim
    if causal:
        attn_weights_v = attn_weights_v // 2
    # Output projection: seq_len * d_model * d_model
    out_proj = seq_len * d_model * d_model
    return qkv_proj + attn_scores + attn_weights_v + out_proj


def estimate_ffn_macs(
    seq_len: int,
    d_model: int,
    d_ff: int,
    activation: str = "swiglu",
) -> int:
    """Estimate MACs for a feed-forward network block.

    For standard FFN: Linear1 (d_model -> d_ff) + activation + Linear2 (d_ff -> d_model).
    For SwiGLU: W1 (d_model -> d_ff) + W2 (d_model -> d_ff) + SiLU + elementwise mul + W3 (d_ff -> d_model).
    """
    linear1 = seq_len * d_model * d_ff
    linear2 = seq_len * d_ff * d_model
    if activation == "swiglu":
        # W1 + W2 + W3
        w1 = seq_len * d_model * d_ff
        w2 = seq_len * d_model * d_ff
        w3 = seq_len * d_ff * d_model
        return w1 + w2 + w3
    return linear1 + linear2


def estimate_layer_norm_macs(seq_len: int, d_model: int) -> int:
    """Approximate MACs for LayerNorm (mean, variance, normalize, affine)."""
    return 5 * seq_len * d_model


def estimate_transformer_macs(
    seq_len: int,
    d_model: int,
    n_heads: int,
    d_ff: int,
    n_layers: int,
    causal: bool = True,
    ffn_activation: str = "swiglu",
    n_encoder_layers: int | None = None,
    n_decoder_layers: int | None = None,
) -> dict[str, int]:
    """Estimate total MACs for a transformer model.

    Returns dict with per-component and total MACs.
    """
    per_layer = 0

    # Attention
    attn_macs = estimate_attention_macs(seq_len, d_model, n_heads, causal=causal)
    per_layer += attn_macs

    # FFN
    ffn_macs = estimate_ffn_macs(seq_len, d_model, d_ff, ffn_activation)
    per_layer += ffn_macs

    # Layer norms (2 for encoder, 3 for decoder with cross-attention)
    ln_macs = 2 * estimate_layer_norm_macs(seq_len, d_model)
    per_layer += ln_macs

    total_encoder = 0
    total_decoder = 0

    if n_encoder_layers is not None and n_decoder_layers is not None:
        # Encoder-decoder
        total_encoder = n_encoder_layers * per_layer
        # Decoder layers have extra cross-attention + extra layernorm
        cross_attn_macs = estimate_attention_macs(seq_len, d_model, n_heads, causal=False)
        decoder_ln_extra = estimate_layer_norm_macs(seq_len, d_model)
        per_decoder_layer = per_layer + cross_attn_macs + decoder_ln_extra
        total_decoder = n_decoder_layers * per_decoder_layer
    elif n_encoder_layers is not None:
        total_encoder = n_encoder_layers * per_layer
    elif n_decoder_layers is not None:
        total_decoder = n_decoder_layers * per_layer
    else:
        total_encoder = n_layers * per_layer

    total = total_encoder + total_decoder

    # Embedding MACs (lookup is essentially zero, but projection if tied)
    return {
        "attention_per_layer": attn_macs,
        "ffn_per_layer": ffn_macs,
        "layernorm_per_layer": ln_macs,
        "encoder_total": total_encoder,
        "decoder_total": total_decoder,
        "total": total,
        "total_flops": total * 2,
    }


def estimate_peak_activation_memory(
    batch_size: int,
    seq_len: int,
    d_model: int,
    n_heads: int,
    d_ff: int,
    n_layers: int,
    dtype_bytes: int = 2,
    causal: bool = True,
) -> dict[str, int]:
    """Estimate peak activation memory in bytes.

    Rough estimate of intermediate activations stored during a forward pass.
    """
    per_layer = 0

    # Attention: Q, K, V, scores, output
    head_dim = d_model // n_heads
    qkv = 3 * batch_size * n_heads * seq_len * head_dim * dtype_bytes
    scores = batch_size * n_heads * seq_len * seq_len * dtype_bytes
    attn_out = batch_size * n_heads * seq_len * head_dim * dtype_bytes
    per_layer += qkv + scores + attn_out

    # FFN: two linear intermediates
    ffn_act = batch_size * seq_len * d_ff * dtype_bytes * 2
    per_layer += ffn_act

    # LayerNorm intermediates
    ln_act = batch_size * seq_len * d_model * dtype_bytes * 2
    per_layer += ln_act

    total = n_layers * per_layer

    # Embedding
    embed = batch_size * seq_len * d_model * dtype_bytes

    return {
        "per_layer": per_layer,
        "embeddings": embed,
        "total": total + embed,
    }


def compute_model_metrics(
    model: nn.Module,
    seq_len: int,
    batch_size: int = 1,
    causal: bool = True,
    ffn_activation: str = "swiglu",
    n_encoder_layers: int | None = None,
    n_decoder_layers: int | None = None,
    dtype_bytes: int = 2,
) -> ModelMetrics:
    """Compute comprehensive metrics for a transformer model.

    Combines parameter counting, MACs estimation, and activation memory estimation.
    """
    param_info = count_parameters(model)
    breakdown = parameter_breakdown(model)

    # Try to infer config from model
    config = getattr(model, "config", None)
    d_model = getattr(config, "d_model", None)
    n_heads = getattr(config, "n_heads", None)
    d_ff = getattr(config, "d_ff", None)
    n_layers = getattr(config, "n_layers", None)

    if d_model is None or n_heads is None or d_ff is None or n_layers is None:
        # Fallback: try to extract from model attributes
        if not hasattr(model, "token_embedding"):
            raise ValueError(
                "Cannot infer model dimensions. Pass config or ensure model has a .config attribute."
            )
        d_model = d_model or model.token_embedding.embedding_dim
        n_heads = n_heads or getattr(model.config, "n_heads", 12)
        d_ff = d_ff or getattr(model.config, "d_ff", 4 * d_model)
        n_layers = n_layers or getattr(model.config, "n_layers", 6)

    macs_info = estimate_transformer_macs(
        seq_len=seq_len,
        d_model=d_model,
        n_heads=n_heads,
        d_ff=d_ff,
        n_layers=n_layers,
        causal=causal,
        ffn_activation=ffn_activation,
        n_encoder_layers=n_encoder_layers,
        n_decoder_layers=n_decoder_layers,
    )

    act_info = estimate_peak_activation_memory(
        batch_size=batch_size,
        seq_len=seq_len,
        d_model=d_model,
        n_heads=n_heads,
        d_ff=d_ff,
        n_layers=n_layers,
        dtype_bytes=dtype_bytes,
        causal=causal,
    )

    return ModelMetrics(
        total_params=param_info["total"],
        trainable_params=param_info["trainable"],
        non_trainable_params=param_info["non_trainable"],
        param_bytes=param_info["bytes"],
        macs=macs_info["total"],
        flops=macs_info["total_flops"],
        activations_bytes=act_info["total"],
        param_breakdown=breakdown,
    )
