from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import torch
from torch import nn

logger = logging.getLogger(__name__)


def save_checkpoint(
    model: nn.Module,
    optimizer,
    scheduler,
    global_step: int,
    epoch: int,
    path: str | Path,
    *,
    extra: dict | None = None,
) -> Path:
    """Save a full training checkpoint to disk.

    The checkpoint contains the model state dict, optimizer state,
    scheduler state, and metadata (step, epoch, timestamp).  It can be
    loaded later with :func:`load_checkpoint`.

    Args:
        model: The model (or DDP wrapper) to save.
        optimizer: The optimizer state dict.
        scheduler: The scheduler state dict (or ``None``).
        global_step: Current global training step.
        epoch: Current epoch number.
        path: Destination file path (should end in ``.pt``).
        extra: Arbitrary extra data to store (e.g. best metric).

    Returns:
        The resolved :class:`Path` that was written.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    state = {
        "model_state_dict": _unwrap_model(model).state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        "global_step": global_step,
        "epoch": epoch,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    if extra:
        state["extra"] = extra

    torch.save(state, path)
    logger.info("Checkpoint saved: %s (step %d, epoch %d)", path, global_step, epoch)
    return path


def load_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer=None,
    scheduler=None,
    *,
    strict: bool = True,
    map_location: str | torch.device | None = None,
) -> dict:
    """Load a training checkpoint, restoring model / optimizer / scheduler state.

    Args:
        path: Path to the ``.pt`` checkpoint file.
        model: Model whose state dict to restore.
        optimizer: If provided, restore its state.
        scheduler: If provided, restore its state.
        strict: Passed to ``load_state_dict``.
        map_location: Device remapping (e.g. ``"cpu"``).

    Returns:
        The metadata dict from the checkpoint (keys: ``global_step``,
        ``epoch``, ``timestamp``, and any ``extra``).
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    checkpoint = torch.load(path, map_location=map_location, weights_only=False)

    _unwrap_model(model).load_state_dict(checkpoint["model_state_dict"], strict=strict)

    if optimizer is not None and checkpoint.get("optimizer_state_dict") is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    if scheduler is not None and checkpoint.get("scheduler_state_dict") is not None:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    logger.info(
        "Checkpoint loaded: %s (step %d, epoch %d)",
        path,
        checkpoint.get("global_step", 0),
        checkpoint.get("epoch", 0),
    )
    return {
        "global_step": checkpoint.get("global_step", 0),
        "epoch": checkpoint.get("epoch", 0),
        "timestamp": checkpoint.get("timestamp", ""),
        "extra": checkpoint.get("extra", {}),
    }


def save_best_model(
    model: nn.Module,
    path: str | Path,
    metric_name: str,
    metric_value: float,
    *,
    higher_is_better: bool = False,
) -> Path:
    """Save the best model checkpoint and a metadata sidecar.

    Writes two files:
    - ``<path>`` – model weights (``state_dict``).
    - ``<path>.meta.json`` – metric info for easy comparison.

    Args:
        model: The model to save.
        path: Destination file path.
        metric_name: Name of the tracked metric (e.g. ``"val_loss"``).
        metric_value: Current value of the metric.
        higher_is_better: ``True`` if a higher metric is better.

    Returns:
        The resolved :class:`Path`.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(_unwrap_model(model).state_dict(), path)

    meta = {
        "metric_name": metric_name,
        "metric_value": metric_value,
        "higher_is_better": higher_is_better,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    meta_path = path.with_suffix(path.suffix + ".meta.json")
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    logger.info("Best model saved: %s (%s=%.4f)", path, metric_name, metric_value)
    return path


class CheckpointManager:
    """Manages checkpoint rotation, best-model tracking, and resume logic.

    Args:
        output_dir: Directory where checkpoints are written.
        save_every: Save a checkpoint every *n* steps (0 to disable).
        keep_last: Number of recent checkpoints to keep.
        metric_name: Metric tracked for best-model selection.
        higher_is_better: Whether a higher metric value is better.
    """

    def __init__(
        self,
        output_dir: str | Path,
        *,
        save_every: int = 500,
        keep_last: int = 3,
        metric_name: str = "val_loss",
        higher_is_better: bool = False,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.save_every = save_every
        self.keep_last = keep_last
        self.metric_name = metric_name
        self.higher_is_better = higher_is_better

        self._best_value: float | None = None
        self._checkpoints: list[Path] = []

    @property
    def best_value(self) -> float | None:
        """Best metric value seen so far (``None`` until first eval)."""
        return self._best_value

    def is_best(self, metric_value: float) -> bool:
        """Return ``True`` if *metric_value* is the new best, and update the tracked best."""
        if self._best_value is None:
            self._best_value = metric_value
            return True
        if self.higher_is_better:
            is_better = metric_value > self._best_value
        else:
            is_better = metric_value < self._best_value
        if is_better:
            self._best_value = metric_value
        return is_better

    def step(
        self,
        model,
        optimizer,
        scheduler,
        global_step: int,
        epoch: int,
        metric_value: float | None = None,
        extra: dict | None = None,
    ) -> tuple[Path | None, bool]:
        """Possibly save a checkpoint; returns ``(path if saved, was_best)``.

        Saves unconditionally every ``save_every`` steps.  If
        *metric_value* is provided and is the new best, saves a
        ``best.pt`` in addition.

        ``was_best`` reflects the single ``is_best()`` call made here --
        callers (e.g. early stopping) should use it instead of calling
        ``is_best()`` again themselves, since ``is_best()`` mutates the
        tracked best value as a side effect and a second call with the
        same metric_value would always compare equal-to-itself and
        report "not better".
        """
        saved_path: Path | None = None
        was_best = False
        ckpt_path = self.output_dir / f"checkpoint-step{global_step}.pt"

        if self.save_every > 0 and global_step % self.save_every == 0:
            save_checkpoint(model, optimizer, scheduler, global_step, epoch, ckpt_path, extra=extra)
            self._track(ckpt_path)
            saved_path = ckpt_path

        if metric_value is not None:
            was_best = self.is_best(metric_value)
            if was_best:
                best_path = self.output_dir / "best.pt"
                # Save the full, resumable checkpoint (model + optimizer +
                # scheduler state) and a metadata sidecar with the metric
                # info -- NOT save_best_model(), which would overwrite
                # best_path with a bare state_dict, clobbering the
                # optimizer/scheduler state just written and leaving a file
                # load_checkpoint() can't parse (no "model_state_dict" key).
                save_checkpoint(model, optimizer, scheduler, global_step, epoch, best_path, extra=extra)
                meta = {
                    "metric_name": self.metric_name,
                    "metric_value": metric_value,
                    "higher_is_better": self.higher_is_better,
                    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
                }
                meta_path = best_path.with_suffix(best_path.suffix + ".meta.json")
                meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
                saved_path = best_path

        return saved_path, was_best

    def save_final(self, model, optimizer, scheduler, global_step: int, epoch: int, extra: dict | None = None) -> Path:
        """Save the final checkpoint at the end of training."""
        path = self.output_dir / "final.pt"
        save_checkpoint(model, optimizer, scheduler, global_step, epoch, path, extra=extra)
        self._track(path)
        return path

    def _track(self, path: Path) -> None:
        self._checkpoints.append(path)
        while len(self._checkpoints) > self.keep_last:
            old = self._checkpoints.pop(0)
            if old.exists() and "best" not in old.name:
                old.unlink()
                meta = old.with_suffix(old.suffix + ".meta.json")
                if meta.exists():
                    meta.unlink()

    def find_latest(self) -> Path | None:
        """Find the most recent checkpoint in the output directory."""
        ckpts = list(self.output_dir.glob("checkpoint-step*.pt"))
        if ckpts:
            # Sort numerically by step, not lexicographically by filename
            # ("step100" < "step50" as strings despite 100 > 50).
            return max(ckpts, key=lambda p: int(p.stem.rsplit("step", 1)[-1]))
        if (self.output_dir / "final.pt").exists():
            return self.output_dir / "final.pt"
        return None


def _unwrap_model(model: nn.Module) -> nn.Module:
    """Strip DDP / DDP wrapper to get the underlying module."""
    if hasattr(model, "module"):
        return model.module
    return model
