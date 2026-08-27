from __future__ import annotations

import logging
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from tqdm.auto import tqdm

from .checkpoint import CheckpointManager, _unwrap_model, load_checkpoint
from .optimizer import get_optimizer
from .scheduler import (
    get_constant_schedule_with_warmup,
    get_cosine_schedule_with_warmup,
    get_linear_schedule_with_warmup,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class TrainerConfig:
    """All hyper-parameters that control the training run.

    Sensible defaults are provided so that a minimal ``TrainerConfig()``
    can be passed straight into the :class:`Trainer`.
    """

    # Optimization
    lr: float = 3e-4
    weight_decay: float = 0.01
    betas: tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-8
    max_grad_norm: float = 1.0

    # Schedule
    schedule: str = "cosine"
    warmup_ratio: float = 0.03
    min_lr_ratio: float = 0.1

    # Training loop
    num_epochs: int = 10
    gradient_accumulation_steps: int = 1
    use_amp: bool = True

    # Checkpointing
    output_dir: str = "outputs/checkpoints"
    save_every: int = 500
    keep_last: int = 3

    # Early stopping
    early_stopping_patience: int | None = None
    early_stopping_metric: str = "val_loss"
    early_stopping_higher_is_better: bool = False

    # Logging
    log_every: int = 10
    eval_every: int | None = None

    # Device
    device: str | torch.device = "cuda"
    compile_model: bool = False

    # DDP
    ddp: bool = False

    # Model type (affects loss computation)
    model_type: str = "decoder_only"  # "encoder_only", "decoder_only", "encoder_decoder"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_batch_keys(model_type: str) -> tuple[list[str], list[str]]:
    """Return (forward_keys, label_keys) for the given model type."""
    if model_type == "encoder_decoder":
        fwd = ["input_ids", "attention_mask", "decoder_input_ids", "decoder_attention_mask"]
        lbl = ["labels"]
    elif model_type == "encoder_only":
        fwd = ["input_ids", "attention_mask"]
        lbl = ["labels"]
    else:
        fwd = ["input_ids", "attention_mask"]
        lbl = ["labels"]
    return fwd, lbl


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class Trainer:
    """General-purpose training loop for encoder, decoder, and encoder-decoder
    transformer models.

    Features:
        - Gradient accumulation
        - Mixed-precision (AMP) via ``torch.cuda.amp``
        - Cosine / linear / constant LR with warmup
        - Checkpoint management (periodic + best-model)
        - Early stopping
        - Optional DDP support
        - tqdm progress bars

    Args:
        model: The ``nn.Module`` to train.
        config: A :class:`TrainerConfig` instance.
        train_dataloader: DataLoader for the training set.
        val_dataloader: Optional DataLoader for validation.
        callbacks: Optional list of callables invoked as
            ``callback(trainer, step, metrics)`` after each log step.
    """

    def __init__(
        self,
        model: nn.Module,
        config: TrainerConfig,
        train_dataloader: DataLoader,
        val_dataloader: DataLoader | None = None,
        callbacks: list | None = None,
    ) -> None:
        self.config = config
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.callbacks = callbacks or []

        # ---- Device / DDP ----
        # TrainerConfig has no local_rank field; fall back to the LOCAL_RANK
        # env var torchrun/torch.distributed.launch set, so DDP actually
        # binds each process to its own GPU instead of every rank hitting GPU 0.
        self.local_rank = int(getattr(config, "local_rank", None) or os.environ.get("LOCAL_RANK", 0))
        self.device = self._resolve_device(config.device)
        self._ddp = config.ddp

        self.model = model.to(self.device)
        if self._ddp and not isinstance(self.model, DDP):
            self.model = DDP(self.model, device_ids=[self.local_rank])

        if config.compile_model and hasattr(torch, "compile"):
            self.model = torch.compile(self.model)

        # ---- Loss ----
        self._criterion = nn.CrossEntropyLoss(ignore_index=-100)

        # ---- Optimizer / Scheduler ----
        raw_model = _unwrap_model(self.model)
        self.optimizer = get_optimizer(
            raw_model,
            lr=config.lr,
            weight_decay=config.weight_decay,
            betas=config.betas,
            eps=config.eps,
        )

        self._train_steps_per_epoch = max(
            1, len(train_dataloader) // config.gradient_accumulation_steps
        )
        total_steps = self._train_steps_per_epoch * config.num_epochs
        warmup_steps = int(total_steps * config.warmup_ratio)

        if config.schedule == "cosine":
            self.scheduler = get_cosine_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=warmup_steps,
                num_training_steps=total_steps,
                min_lr_ratio=config.min_lr_ratio,
            )
        elif config.schedule == "linear":
            self.scheduler = get_linear_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=warmup_steps,
                num_training_steps=total_steps,
            )
        elif config.schedule == "constant":
            self.scheduler = get_constant_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=warmup_steps,
            )
        else:
            raise ValueError(f"Unknown schedule: {config.schedule!r}. Choose from 'cosine', 'linear', 'constant'.")

        # ---- AMP ----
        self._use_amp = config.use_amp and self.device.type == "cuda"
        self._scaler = GradScaler(self.device.type, enabled=self._use_amp)

        # ---- Checkpointing ----
        self.ckpt_manager = CheckpointManager(
            config.output_dir,
            save_every=config.save_every,
            keep_last=config.keep_last,
            metric_name=config.early_stopping_metric,
            higher_is_better=config.early_stopping_higher_is_better,
        )

        # ---- State ----
        self.global_step = 0
        self.current_epoch = 0
        self.train_history: list[dict[str, Any]] = []
        self.val_history: list[dict[str, Any]] = []
        self._early_stop_counter = 0

    # ------------------------------------------------------------------
    # Device helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_device(device: str | torch.device) -> torch.device:
        if isinstance(device, torch.device):
            return device
        if device == "cuda" and not torch.cuda.is_available():
            logger.warning("CUDA unavailable, falling back to CPU.")
            return torch.device("cpu")
        return torch.device(device)

    # ------------------------------------------------------------------
    # Forward / backward
    # ------------------------------------------------------------------

    def _forward_batch(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """Run a forward pass and return the scalar loss.

        Handles encoder-only, decoder-only, and encoder-decoder models
        transparently.
        """
        fwd_keys, _ = _get_batch_keys(self.config.model_type)
        kwargs: dict[str, Any] = {}
        for k in fwd_keys:
            if k in batch:
                kwargs[k] = batch[k].to(self.device)

        outputs = self.model(**kwargs)

        # Model may return a dict (encoder-decoder) or plain tensor
        if isinstance(outputs, dict):
            logits = outputs.get("logits", outputs.get("log_probs", None))
        else:
            logits = outputs

        # Compute loss
        labels = batch.get("labels")
        if labels is not None:
            labels = labels.to(self.device)
            return self._compute_loss(logits, labels)

        # Fallback: return mean of logits as dummy (for models without labels)
        return logits.mean() * 0.0

    def _compute_loss(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Cross-entropy loss with correct reshaping for sequence models."""
        if logits.dim() == 3:
            # [B, T, V] -> [B*T, V] and labels [B, T] -> [B*T]
            shift = logits.size(-1)
            return self._criterion(logits.reshape(-1, shift), labels.reshape(-1))
        return self._criterion(logits, labels)

    # ------------------------------------------------------------------
    # Single training step
    # ------------------------------------------------------------------

    def _training_step(self, batch: dict[str, torch.Tensor]) -> float:
        """Execute one accumulation-averaged gradient step.  Returns the
        (unscaled) loss value."""
        self.model.train()

        with autocast(device_type=self.device.type, enabled=self._use_amp):
            loss = self._forward_batch(batch)
            loss = loss / self.config.gradient_accumulation_steps

        self._scaler.scale(loss).backward()
        return loss.item() * self.config.gradient_accumulation_steps

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _validate(self) -> dict[str, float]:
        """Run one full validation epoch.  Returns ``{"val_loss": ...,
        "val_ppl": ...}``."""
        if self.val_dataloader is None:
            return {}

        self.model.eval()
        total_loss = 0.0
        total_tokens = 0

        desc = "Validating"
        iterator = self.val_dataloader
        if self._is_main_process():
            iterator = tqdm(self.val_dataloader, desc=desc, leave=False)

        for batch in iterator:
            with autocast(device_type=self.device.type, enabled=self._use_amp):
                loss = self._forward_batch(batch)
            labels = batch.get("labels")
            if labels is not None:
                n_tokens = (labels != -100).sum().item()
            else:
                n_tokens = 1
            total_loss += loss.item() * n_tokens
            total_tokens += max(n_tokens, 1)

        avg_loss = total_loss / max(total_tokens, 1)
        ppl = math.exp(min(avg_loss, 20))

        return {"val_loss": avg_loss, "val_ppl": ppl}

    # ------------------------------------------------------------------
    # Main training loop
    # ------------------------------------------------------------------

    def train(self) -> dict[str, Any]:
        """Run the full training loop.

        Returns:
            A summary dict with keys ``train_history``,
            ``val_history``, ``best_metric``, ``total_steps``.
        """
        cfg = self.config
        logger.info("Starting training for %d epochs", cfg.num_epochs)
        logger.info("  Model type       : %s", cfg.model_type)
        logger.info("  Train batches    : %d", len(self.train_dataloader))
        logger.info("  Grad accum steps : %d", cfg.gradient_accumulation_steps)
        logger.info("  Effective steps/ep: %d", self._train_steps_per_epoch)
        logger.info("  AMP              : %s", cfg.use_amp)
        logger.info("  Device           : %s", self.device)

        for epoch in range(cfg.num_epochs):
            self.current_epoch = epoch
            epoch_loss = self._train_epoch(epoch)

            # ---- Validation ----
            val_metrics: dict[str, float] = {}
            do_eval = (
                cfg.eval_every is not None
                and (epoch + 1) % cfg.eval_every == 0
            ) or (epoch == cfg.num_epochs - 1)

            if do_eval and self.val_dataloader is not None:
                val_metrics = self._validate()
                self.val_history.append({"epoch": epoch, **val_metrics})

                metric_val = val_metrics.get(cfg.early_stopping_metric, None)
                if metric_val is not None:
                    _, was_best = self.ckpt_manager.step(
                        self.model, self.optimizer, self.scheduler,
                        self.global_step, epoch, metric_value=metric_val,
                        extra=self._resume_state(),
                    )
                    self._check_early_stopping(was_best)

            logger.info(
                "Epoch %d/%d  train_loss=%.4f  %s",
                epoch + 1,
                cfg.num_epochs,
                epoch_loss,
                "  ".join(f"{k}={v:.4f}" for k, v in val_metrics.items()) if val_metrics else "",
            )

            # Fire callbacks
            for cb in self.callbacks:
                cb(self, self.global_step, {"epoch": epoch, "train_loss": epoch_loss, **val_metrics})

        # ---- Final checkpoint ----
        final_path = self.ckpt_manager.save_final(
            self.model, self.optimizer, self.scheduler,
            self.global_step, self.current_epoch,
            extra=self._resume_state(),
        )

        summary = {
            "train_history": self.train_history,
            "val_history": self.val_history,
            "best_metric": self.ckpt_manager.best_value,
            "total_steps": self.global_step,
            "final_checkpoint": str(final_path),
        }
        logger.info("Training complete. Best %s = %s", cfg.early_stopping_metric, summary["best_metric"])
        return summary

    def _train_epoch(self, epoch: int) -> float:
        """Train one epoch.  Returns the average loss."""
        cfg = self.config
        self.model.train()

        if self._ddp and isinstance(self.train_dataloader.sampler, DistributedSampler):
            self.train_dataloader.sampler.set_epoch(epoch)

        total_loss = 0.0
        num_steps = 0

        iterator = self.train_dataloader
        if self._is_main_process():
            iterator = tqdm(
                self.train_dataloader,
                desc=f"Epoch {epoch + 1}/{cfg.num_epochs}",
                leave=False,
            )

        self.optimizer.zero_grad()
        accumulated = 0
        last_loss_val = 0.0

        def _optimizer_step(loss_val: float) -> None:
            if cfg.max_grad_norm > 0:
                self._scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(self.model.parameters(), cfg.max_grad_norm)

            self._scaler.step(self.optimizer)
            self._scaler.update()
            self.scheduler.step()
            self.optimizer.zero_grad()
            self.global_step += 1

            if self.global_step % cfg.log_every == 0:
                lr = self.scheduler.get_last_lr()[0]
                ppl = math.exp(min(loss_val, 20))
                metrics = {
                    "step": self.global_step,
                    "epoch": epoch,
                    "loss": loss_val,
                    "ppl": ppl,
                    "lr": lr,
                }
                self.train_history.append(metrics)

                if self._is_main_process() and hasattr(iterator, "set_postfix"):
                    iterator.set_postfix(loss=f"{loss_val:.4f}", lr=f"{lr:.2e}")

        for step, batch in enumerate(iterator):
            loss_val = self._training_step(batch)
            total_loss += loss_val
            num_steps += 1
            last_loss_val = loss_val
            accumulated += 1

            if accumulated == cfg.gradient_accumulation_steps:
                _optimizer_step(loss_val)
                accumulated = 0

        if accumulated > 0:
            # Flush the trailing partial accumulation window instead of
            # silently discarding it -- without this, gradients from a
            # dataloader length that isn't a multiple of
            # gradient_accumulation_steps get wiped by the next epoch's
            # leading optimizer.zero_grad() and never influence the weights.
            _optimizer_step(last_loss_val)

        avg_loss = total_loss / max(num_steps, 1)
        return avg_loss

    # ------------------------------------------------------------------
    # Early stopping
    # ------------------------------------------------------------------

    def _check_early_stopping(self, was_best: bool) -> None:
        """Increment or reset the early-stopping counter.

        Takes the ``was_best`` verdict from :meth:`CheckpointManager.step`
        rather than calling ``is_best()`` again here: ``is_best()`` mutates
        the tracked best value as a side effect, so a second call with the
        same metric value would always compare it against itself and
        report "not better", making early stopping fire regardless of
        whether the model was actually improving.
        """
        cfg = self.config
        if cfg.early_stopping_patience is None:
            return

        if was_best:
            self._early_stop_counter = 0
        else:
            self._early_stop_counter += 1
            logger.info(
                "Early stopping patience: %d/%d",
                self._early_stop_counter,
                cfg.early_stopping_patience,
            )
            if self._early_stop_counter >= cfg.early_stopping_patience:
                raise EarlyStopping(
                    f"No improvement in {cfg.early_stopping_metric} for "
                    f"{cfg.early_stopping_patience} evaluations."
                )

    # ------------------------------------------------------------------
    # Resume
    # ------------------------------------------------------------------

    def _resume_state(self) -> dict[str, Any]:
        """State beyond model/optimizer/scheduler needed to resume cleanly:
        best-metric tracking, early-stopping counter, history, and the AMP
        scaler's loss-scale (otherwise a resumed AMP run restarts loss-scale
        discovery from scratch)."""
        return {
            "best_value": self.ckpt_manager.best_value,
            "early_stop_counter": self._early_stop_counter,
            "train_history": self.train_history,
            "val_history": self.val_history,
            "scaler_state_dict": self._scaler.state_dict() if self._use_amp else None,
        }

    def resume_from(self, checkpoint_path: str | Path) -> int:
        """Resume training from a saved checkpoint.

        Args:
            checkpoint_path: Path to a ``.pt`` file saved by
                :func:`checkpoint.save_checkpoint`.

        Returns:
            The global step restored from the checkpoint.
        """
        meta = load_checkpoint(
            checkpoint_path,
            self.model,
            self.optimizer,
            self.scheduler,
            map_location=self.device,
        )
        self.global_step = meta["global_step"]
        self.current_epoch = meta["epoch"]

        extra = meta.get("extra") or {}
        if extra.get("best_value") is not None:
            self.ckpt_manager._best_value = extra["best_value"]
        self._early_stop_counter = extra.get("early_stop_counter", 0)
        self.train_history = extra.get("train_history", [])
        self.val_history = extra.get("val_history", [])
        if extra.get("scaler_state_dict") is not None and self._use_amp:
            self._scaler.load_state_dict(extra["scaler_state_dict"])

        logger.info("Resumed from step %d, epoch %d", self.global_step, self.current_epoch)
        return self.global_step

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def _is_main_process(self) -> bool:
        """Return ``True`` if this is rank 0 (or DDP is off)."""
        if not self._ddp:
            return True
        return self.local_rank == 0


class EarlyStopping(Exception):
    """Raised when early-stopping patience is exhausted."""
