from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from torch.optim import AdamW
from torch.optim.optimizer import Optimizer


def get_optimizer(
    model,
    lr: float = 3e-4,
    weight_decay: float = 0.01,
    betas: tuple[float, float] = (0.9, 0.999),
    eps: float = 1e-8,
    no_decay: Iterable[str] | None = None,
) -> Optimizer:
    """Build an AdamW optimizer with sensible weight-decay defaults.

    Parameters whose names match any pattern in *no_decay* (bias,
    LayerNorm, layernorm) are excluded from weight decay, which is the
    standard practice for transformer training.

    Args:
        model: The ``nn.Module`` whose parameters to optimise.
        lr: Peak learning rate.
        weight_decay: L2 weight-decay coefficient.
        betas: Adam momentum coefficients.
        eps: Adam epsilon for numerical stability.
        no_decay: Name substrings that should **not** receive weight
            decay.  Defaults to ``["bias", "LayerNorm", "layernorm"]``.

    Returns:
        An ``AdamW`` optimizer instance.
    """
    if no_decay is None:
        no_decay = ["bias", "LayerNorm", "layernorm", "norm"]

    decay_params: list[dict[str, Any]] = []
    no_decay_params: list[dict[str, Any]] = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if any(nd in name for nd in no_decay):
            no_decay_params.append({"params": [param], "weight_decay": 0.0})
        else:
            decay_params.append({"params": [param], "weight_decay": weight_decay})

    param_groups = [*decay_params, *no_decay_params]

    return AdamW(param_groups, lr=lr, betas=betas, eps=eps)
