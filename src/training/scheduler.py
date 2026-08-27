from __future__ import annotations

import math

from torch.optim.lr_scheduler import LambdaLR


def get_cosine_schedule_with_warmup(
    optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    min_lr_ratio: float = 0.0,
    num_cycles: float = 0.5,
) -> LambdaLR:
    """Create a cosine-decay learning-rate schedule with linear warmup.

    During the warmup phase the learning rate increases linearly from 0
    to the base LR of each param-group.  After warmup the LR follows a
    cosine curve from 1.0 down to *min_lr_ratio*.

    Args:
        optimizer: A wrapped ``torch.optim.Optimizer``.
        num_warmup_steps: Number of steps for the linear warmup.
        num_training_steps: Total number of training steps.
        min_lr_ratio: Minimum learning-rate as a fraction of the
            base LR (0.0 = decay to zero).
        num_cycles: Number of cosine cycles.  ``0.5`` decays to
            the minimum once; ``1.0`` does one full cosine cycle.

    Returns:
        A ``torch.optim.lr_scheduler.LambdaLR`` instance.
    """

    def lr_lambda(current_step: int) -> float:
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))

        progress = float(current_step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps)
        )
        cosine_factor = 0.5 * (1.0 + math.cos(math.pi * num_cycles * 2.0 * progress))
        return max(min_lr_ratio, min_lr_ratio + (1.0 - min_lr_ratio) * cosine_factor)

    return LambdaLR(optimizer, lr_lambda)


def get_constant_schedule_with_warmup(
    optimizer,
    num_warmup_steps: int,
) -> LambdaLR:
    """Constant learning rate with linear warmup.

    Args:
        optimizer: A wrapped ``torch.optim.Optimizer``.
        num_warmup_steps: Number of warmup steps.

    Returns:
        A ``LambdaLR`` scheduler.
    """

    def lr_lambda(current_step: int) -> float:
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return 1.0

    return LambdaLR(optimizer, lr_lambda)


def get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
) -> LambdaLR:
    """Linear warmup then linear decay to zero.

    Args:
        optimizer: A wrapped ``torch.optim.Optimizer``.
        num_warmup_steps: Number of warmup steps.
        num_training_steps: Total training steps.

    Returns:
        A ``LambdaLR`` scheduler.
    """

    def lr_lambda(current_step: int) -> float:
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return max(
            0.0,
            float(num_training_steps - current_step)
            / float(max(1, num_training_steps - num_warmup_steps)),
        )

    return LambdaLR(optimizer, lr_lambda)
