"""Reproducibility utilities — seeding for Python, NumPy, and PyTorch."""

from __future__ import annotations

import random

import torch

try:
    import numpy as np

    _NP_AVAILABLE = True
except ImportError:
    _NP_AVAILABLE = False


def set_seed(seed: int) -> None:
    """Set random seeds across Python, NumPy, and PyTorch for reproducibility.

    Also configures cuDNN for deterministic behaviour.  Call this at the
    very beginning of a training or evaluation script.

    Args:
        seed: Integer seed value.
    """
    random.seed(seed)

    if _NP_AVAILABLE:
        np.random.seed(seed)  # type: ignore[attr-defined]

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Deterministic cuDNN (may slow down training slightly).
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
