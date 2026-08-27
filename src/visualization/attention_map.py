from __future__ import annotations

import math
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from torch import nn


class AttentionExtractor:
    """Captures attention weights from transformer attention modules via hooks.

    Registers forward hooks on the attention modules' softmax computation
    to intercept attention weight matrices before dropout.

    Usage:
        extractor = AttentionExtractor(model)
        attn = extractor.extract(input_ids, layer_idx=0)
        plot_attention_heatmap(attn[0], tokens=["hello", "world", ...])
        extractor.remove_hooks()
    """

    def __init__(self, model: nn.Module):
        self.model = model
        self._hooks: list[torch.utils.hooks.RemovableHook] = []
        self._captured: dict[int, torch.Tensor] = {}
        self._register_hooks()

    def _register_hooks(self) -> None:
        encoder = getattr(self.model, "encoder", None)
        decoder = getattr(self.model, "decoder", None)

        layers: list[nn.Module] = []
        if encoder is not None and hasattr(encoder, "layers"):
            layers.extend(encoder.layers)
        if decoder is not None and hasattr(decoder, "layers"):
            layers.extend(decoder.layers)

        for idx, layer in enumerate(layers):
            attn = getattr(layer, "attention", None)
            if attn is None:
                attn = getattr(layer, "self_attn", None)
            if attn is None:
                continue

            def make_hook(layer_idx: int) -> Any:
                def hook_fn(_module: nn.Module, _input: Any, output: Any) -> None:
                    if isinstance(output, tuple) and len(output) > 1:
                        self._captured[layer_idx] = output[1].detach()
                return hook_fn

            hook = attn.register_forward_hook(make_hook(idx))
            self._hooks.append(hook)

    def remove_hooks(self) -> None:
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

    def extract(
        self,
        input_ids: torch.Tensor,
        layer_idx: int | None = None,
        head_idx: int | None = None,
    ) -> dict[int, torch.Tensor]:
        """Run a forward pass and return captured attention weights.

        Args:
            input_ids: Token ids [batch_size, seq_len].
            layer_idx: If given, only return this layer's weights.
            head_idx: If given, only return this head's weights.

        Returns:
            Dict mapping layer_idx -> attention weights [B, n_heads, T, T].

        Raises:
            RuntimeError: None of this repo's attention modules return
                ``(output, attn_weights)`` tuples from ``forward()`` (the
                shape this hook-based extractor expects), so nothing is
                ever captured -- use :func:`compute_attention_manually`
                instead, which recomputes the weights directly rather than
                relying on a hook.
        """
        self._captured.clear()
        self.model.eval()
        with torch.no_grad():
            self.model(input_ids)

        if not self._captured:
            raise RuntimeError(
                "No attention weights were captured. This repo's attention "
                "modules return a plain tensor from forward(), not "
                "(output, attn_weights), so the forward-hook approach used "
                "here can't intercept weights. Use compute_attention_manually() "
                "instead."
            )

        result: dict[int, torch.Tensor] = {}
        for k, v in self._captured.items():
            if layer_idx is not None and k != layer_idx:
                continue
            w = v
            if head_idx is not None and w.dim() >= 2:
                w = w[:, head_idx : head_idx + 1]
            result[k] = w

        return result


def compute_attention_manually(
    model: nn.Module,
    input_ids: torch.Tensor,
    layer_idx: int = 0,
    head_idx: int | None = None,
) -> torch.Tensor:
    """Compute attention weights manually for models where hooks fail.

    Extracts Q, K from the attention layer and computes softmax(Q @ K^T / scale).
    Useful for Flash Attention which doesn't materialize attention weights.

    Returns:
        Attention weights [n_heads, seq_len, seq_len] (first batch element).
    """
    encoder = getattr(model, "encoder", None)
    decoder = getattr(model, "decoder", None)

    layers: list[nn.Module] = []
    if encoder is not None and hasattr(encoder, "layers"):
        layers.extend(encoder.layers)
    if decoder is not None and hasattr(decoder, "layers"):
        layers.extend(decoder.layers)

    if layer_idx >= len(layers):
        raise ValueError(f"layer_idx {layer_idx} out of range (model has {len(layers)} layers)")

    attn = getattr(layers[layer_idx], "attention", None)
    if attn is None:
        attn = getattr(layers[layer_idx], "self_attn", None)
    if attn is None:
        raise ValueError(f"No attention module found in layer {layer_idx}")

    model.eval()
    with torch.no_grad():
        x = model.token_embedding(input_ids)
        x = model.dropout(x)

        if encoder is not None and hasattr(encoder, "pos_encoding") and encoder.pos_encoding is not None:
            x = encoder.pos_encoding(x)

        for i, layer in enumerate(layers):
            if i == layer_idx:
                break
            x = layer(x)

        Wq = getattr(attn, "Wq", getattr(attn, "W_q", None))
        Wk = getattr(attn, "Wk", getattr(attn, "W_k", None))
        if Wq is None or Wk is None:
            raise ValueError("Cannot find Q/K projections in attention module")

        # The real model computes Q/K from the pre-attention LayerNorm's
        # output (norm1(x)) when layer_norm_type == "pre" (this repo's
        # default), not from the raw residual stream -- match that or the
        # "attention weights" here won't reflect what the model actually
        # computed at this layer.
        target_layer = layers[layer_idx]
        x_in = x
        if getattr(target_layer, "layer_norm_type", "pre") == "pre" and hasattr(target_layer, "norm1"):
            x_in = target_layer.norm1(x)

        Q = Wq(x_in)
        K = Wk(x_in)

        head_dim = Q.shape[-1] // getattr(attn, "n_heads", getattr(attn, "num_heads", 1))
        n_heads = getattr(attn, "n_heads", getattr(attn, "num_heads", 1))
        B, T, _ = Q.shape

        Q = Q.view(B, T, n_heads, head_dim).transpose(1, 2)
        K = K.view(B, T, n_heads, head_dim).transpose(1, 2)

        scale = math.sqrt(head_dim)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / scale

        if getattr(attn, "causal", False):
            causal_mask = torch.triu(torch.ones(T, T, device=scores.device), diagonal=1).bool()
            scores.masked_fill_(causal_mask.unsqueeze(0).unsqueeze(0), float("-inf"))

        attn_weights = torch.softmax(scores, dim=-1)

    w = attn_weights[0]
    if head_idx is not None:
        w = w[head_idx]
    return w


def plot_attention_heatmap(
    attn_weights: torch.Tensor,
    tokens: Sequence[str] | None = None,
    title: str = "Attention Weights",
    head_idx: int | None = None,
    ax: plt.Axes | None = None,
    save_path: str | Path | None = None,
    cmap: str = "viridis",
    figsize: tuple[int, int] = (8, 6),
    vmin: float | None = None,
    vmax: float | None = None,
) -> plt.Figure:
    """Plot a single attention head heatmap.

    Args:
        attn_weights: [seq_len, seq_len] or [n_heads, seq_len, seq_len].
        tokens: Optional token labels for axes.
        title: Plot title.
        head_idx: If attn_weights has multiple heads, which one to plot.
        ax: Existing matplotlib axes. If None, creates new figure.
        save_path: Path to save the figure.
        cmap: Colormap name.
        figsize: Figure size if creating new figure.
        vmin: Minimum colormap value.
        vmax: Maximum colormap value.
    """
    if attn_weights.dim() == 3:
        if head_idx is not None:
            attn_weights = attn_weights[head_idx]
        else:
            attn_weights = attn_weights[0]

    data = attn_weights.cpu().numpy()

    create_fig = ax is None
    if create_fig:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    sns.heatmap(
        data,
        ax=ax,
        cmap=cmap,
        xticklabels=tokens if tokens else "auto",
        yticklabels=tokens if tokens else "auto",
        vmin=vmin,
        vmax=vmax,
        square=True,
        linewidths=0.5,
        linecolor="white",
        cbar_kws={"shrink": 0.8},
    )
    ax.set_xlabel("Key position")
    ax.set_ylabel("Query position")
    ax.set_title(title)

    if create_fig:
        fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_attention_grid(
    attn_weights: torch.Tensor,
    tokens: Sequence[str] | None = None,
    title: str = "Attention Heads",
    max_heads: int = 16,
    save_path: str | Path | None = None,
    cmap: str = "viridis",
    figsize_per_subplot: float = 3.0,
) -> plt.Figure:
    """Plot a grid of attention heatmaps, one per head.

    Args:
        attn_weights: [n_heads, seq_len, seq_len] or [seq_len, seq_len].
        tokens: Optional token labels.
        title: Super-title for the figure.
        max_heads: Maximum number of heads to display.
        save_path: Path to save.
        cmap: Colormap.
        figsize_per_subplot: Size of each subplot.
    """
    if attn_weights.dim() == 2:
        attn_weights = attn_weights.unsqueeze(0)

    n_heads = min(attn_weights.shape[0], max_heads)
    ncols = min(n_heads, 4)
    nrows = math.ceil(n_heads / ncols)

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(figsize_per_subplot * ncols, figsize_per_subplot * nrows),
    )
    if nrows * ncols == 1:
        axes = np.array([[axes]])
    axes = axes.flatten()

    for i in range(nrows * ncols):
        ax = axes[i]
        if i < n_heads:
            data = attn_weights[i].cpu().numpy()
            sns.heatmap(
                data,
                ax=ax,
                cmap=cmap,
                xticklabels=False,
                yticklabels=False,
                square=True,
                linewidths=0.3,
                cbar=False,
            )
            ax.set_title(f"Head {i}", fontsize=9)
        else:
            ax.set_visible(False)

    fig.suptitle(title, fontsize=14, y=1.02)
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_layer_comparison(
    attn_weights_by_layer: dict[int, torch.Tensor],
    tokens: Sequence[str] | None = None,
    head_idx: int = 0,
    title: str = "Attention Across Layers",
    save_path: str | Path | None = None,
    cmap: str = "viridis",
    max_layers: int = 12,
) -> plt.Figure:
    """Compare attention patterns across layers for a specific head.

    Args:
        attn_weights_by_layer: Dict mapping layer_idx -> [n_heads, T, T] or [T, T].
        tokens: Token labels.
        head_idx: Which head to compare across layers.
        title: Figure title.
        save_path: Path to save.
        cmap: Colormap.
        max_layers: Maximum layers to display.
    """
    layers = sorted(attn_weights_by_layer.keys())[:max_layers]
    ncols = min(len(layers), 4)
    nrows = math.ceil(len(layers) / ncols)

    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(3.0 * ncols, 3.0 * nrows),
    )
    if nrows * ncols == 1:
        axes = np.array([[axes]])
    axes = np.atleast_2d(axes)
    flat_axes = axes.flatten()

    for i, layer_idx in enumerate(layers):
        ax = flat_axes[i]
        w = attn_weights_by_layer[layer_idx]
        if w.dim() == 3:
            w = w[head_idx] if head_idx < w.shape[0] else w[0]
        data = w.cpu().numpy()
        sns.heatmap(
            data, ax=ax, cmap=cmap,
            xticklabels=tokens if tokens and len(tokens) <= 20 else False,
            yticklabels=tokens if tokens and len(tokens) <= 20 else False,
            square=True, linewidths=0.3, cbar=False,
        )
        ax.set_title(f"Layer {layer_idx}", fontsize=9)

    for i in range(len(layers), len(flat_axes)):
        flat_axes[i].set_visible(False)

    fig.suptitle(title, fontsize=14, y=1.02)
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_head_importance(
    attn_weights: torch.Tensor,
    tokens: Sequence[str] | None = None,
    title: str = "Head Importance (Entropy)",
    ax: plt.Axes | None = None,
    save_path: str | Path | None = None,
    figsize: tuple[int, int] = (10, 4),
) -> plt.Figure:
    """Bar chart of head importance measured by attention entropy.

    Lower entropy = more focused attention = more important head.

    Args:
        attn_weights: [n_heads, seq_len, seq_len].
        tokens: Not used directly; kept for API consistency.
        title: Plot title.
        ax: Existing axes.
        save_path: Path to save.
        figsize: Figure size.
    """
    if attn_weights.dim() == 2:
        attn_weights = attn_weights.unsqueeze(0)

    n_heads = attn_weights.shape[0]
    entropies = []
    for h in range(n_heads):
        w = attn_weights[h].cpu().float()
        w = torch.clamp(w, min=1e-9)
        ent = -(w * torch.log(w)).sum(dim=-1).mean()
        entropies.append(ent.item())

    create_fig = ax is None
    if create_fig:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, n_heads))
    sorted_idx = np.argsort(entropies)
    sorted_ent = [entropies[i] for i in sorted_idx]
    sorted_colors = [colors[i] for i in range(n_heads)]

    ax.barh(range(n_heads), sorted_ent, color=sorted_colors, edgecolor="gray", linewidth=0.5)
    ax.set_yticks(range(n_heads))
    ax.set_yticklabels([f"Head {i}" for i in sorted_idx], fontsize=8)
    ax.set_xlabel("Attention Entropy")
    ax.set_title(title)
    ax.invert_yaxis()

    if create_fig:
        fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig
