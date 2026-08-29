"""Tests for training step, checkpointing, scheduler, and optimizer."""
from __future__ import annotations

import json
import math
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import torch
from torch.utils.data import DataLoader

from helpers import BATCH, D_MODEL, HEAD_DIM, N_HEADS, SEQ_LEN, VOCAB
from src.models.transformer import Transformer, TransformerConfig
from src.training.checkpoint import (
    CheckpointManager,
    load_checkpoint,
    save_best_model,
    save_checkpoint,
)
from src.training.optimizer import get_optimizer
from src.training.scheduler import (
    get_constant_schedule_with_warmup,
    get_cosine_schedule_with_warmup,
    get_linear_schedule_with_warmup,
)
from src.training.trainer import EarlyStopping, Trainer, TrainerConfig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _small_model(attn_type="alibi", causal=True, model_type="decoder_only"):
    cfg = TransformerConfig(
        vocab_size=VOCAB, d_model=D_MODEL, n_heads=N_HEADS, n_layers=1,
        d_ff=D_MODEL * 4, dropout=0.0, max_seq_len=SEQ_LEN,
        attn_type=attn_type, pos_encoding="sinusoidal", causal=causal,
        use_flash=True, tie_embeddings=True, use_bias=False,
        ffn_activation="swiglu", layer_norm_type="pre",
    )
    use_enc = model_type in ("encoder_only", "encoder_decoder")
    use_dec = model_type in ("decoder_only", "encoder_decoder")
    return Transformer(cfg, use_encoder=use_enc, use_decoder=use_dec)


def _make_dataloader(n_samples=8, seq_len=SEQ_LEN, vocab=VOCAB, batch_size=4):
    input_ids = torch.randint(0, vocab, (n_samples, seq_len))
    attention_mask = torch.ones(n_samples, seq_len, dtype=torch.long)
    labels = input_ids.clone()

    class DictDataset(torch.utils.data.Dataset):
        def __init__(self, input_ids, attention_mask, labels):
            self.input_ids = input_ids
            self.attention_mask = attention_mask
            self.labels = labels
        def __len__(self):
            return len(self.input_ids)
        def __getitem__(self, idx):
            return {
                "input_ids": self.input_ids[idx],
                "attention_mask": self.attention_mask[idx],
                "labels": self.labels[idx],
            }

    ds = DictDataset(input_ids, attention_mask, labels)
    return DataLoader(ds, batch_size=batch_size)


def _dummy_batch(seq_len=SEQ_LEN, vocab=VOCAB, batch_size=2):
    return {
        "input_ids": torch.randint(0, vocab, (batch_size, seq_len)),
        "attention_mask": torch.ones(batch_size, seq_len, dtype=torch.long),
        "labels": torch.randint(0, vocab, (batch_size, seq_len)),
    }


# ---------------------------------------------------------------------------
# Optimizer
# ---------------------------------------------------------------------------

class TestOptimizer:
    def test_creates_optimizer(self):
        model = _small_model()
        opt = get_optimizer(model, lr=1e-3)
        assert len(opt.param_groups) > 0

    def test_weight_decay_separation(self):
        model = _small_model()
        opt = get_optimizer(model, lr=1e-3, weight_decay=0.1)
        decay_found = False
        no_decay_found = False
        for pg in opt.param_groups:
            if pg["weight_decay"] == 0.0:
                no_decay_found = True
            else:
                decay_found = True
        assert decay_found
        assert no_decay_found


# ---------------------------------------------------------------------------
# Schedulers
# ---------------------------------------------------------------------------

class TestSchedulers:
    def test_cosine_warmup(self):
        model = _small_model()
        opt = get_optimizer(model, lr=1e-3)
        sched = get_cosine_schedule_with_warmup(opt, num_warmup_steps=10, num_training_steps=100)
        # During warmup, lr should increase
        lrs = []
        for i in range(20):
            lrs.append(sched.get_last_lr()[0])
            sched.step()
        assert lrs[5] < lrs[10]  # increasing during warmup

    def test_constant_warmup(self):
        model = _small_model()
        opt = get_optimizer(model, lr=1e-3)
        sched = get_constant_schedule_with_warmup(opt, num_warmup_steps=10)
        for _ in range(5):
            sched.step()
        # After warmup, lr should be constant = base_lr
        for _ in range(10):
            sched.step()
        lr = sched.get_last_lr()[0]
        assert abs(lr - 1e-3) < 1e-6

    def test_linear_warmup(self):
        model = _small_model()
        opt = get_optimizer(model, lr=1e-3)
        sched = get_linear_schedule_with_warmup(opt, num_warmup_steps=10, num_training_steps=100)
        lrs = []
        for _ in range(100):
            lrs.append(sched.get_last_lr()[0])
            sched.step()
        # Should decay to near zero
        assert lrs[-1] < 1e-4


# ---------------------------------------------------------------------------
# Checkpoint
# ---------------------------------------------------------------------------

class TestCheckpoint:
    def test_save_and_load(self):
        model = _small_model()
        opt = get_optimizer(model, lr=1e-3)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "ckpt.pt"
            save_checkpoint(model, opt, None, global_step=10, epoch=2, path=path)
            assert path.exists()

            model2 = _small_model()
            opt2 = get_optimizer(model2, lr=1e-3)
            meta = load_checkpoint(path, model2, opt2, map_location="cpu")
            assert meta["global_step"] == 10
            assert meta["epoch"] == 2

            # Weights should match
            for (n1, p1), (n2, p2) in zip(model.named_parameters(), model2.named_parameters()):
                torch.testing.assert_close(p1, p2)

    def test_save_checkpoint_with_extra(self):
        model = _small_model()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "ckpt.pt"
            save_checkpoint(model, None, None, 0, 0, path, extra={"loss": 1.5})
            ckpt = torch.load(path, weights_only=False)
            assert ckpt["extra"]["loss"] == 1.5

    def test_save_best_model(self):
        model = _small_model()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "best.pt"
            save_best_model(model, path, "val_loss", 2.5)
            assert path.exists()
            meta_path = path.with_suffix(".pt.meta.json")
            assert meta_path.exists()
            with open(meta_path) as f:
                meta = json.load(f)
            assert meta["metric_name"] == "val_loss"
            assert meta["metric_value"] == 2.5

    def test_load_nonexistent_raises(self):
        model = _small_model()
        with pytest.raises(FileNotFoundError):
            load_checkpoint("/nonexistent/path.pt", model)


class TestCheckpointManager:
    def test_is_best(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = CheckpointManager(tmpdir, save_every=999, keep_last=3)
            assert mgr.is_best(1.0)
            assert not mgr.is_best(2.0)  # higher is worse
            assert mgr.best_value == 1.0

    def test_is_best_higher_is_better(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = CheckpointManager(tmpdir, save_every=999, higher_is_better=True)
            assert mgr.is_best(1.0)
            assert mgr.is_best(2.0)
            assert not mgr.is_best(1.5)

    def test_save_every(self):
        model = _small_model()
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = CheckpointManager(tmpdir, save_every=5, keep_last=3)
            opt = get_optimizer(model)
            # Step 0 -> save (0 % 5 == 0)
            path = mgr.step(model, opt, None, 0, 0, metric_value=1.0)
            assert path is not None

    def test_keep_last_rotation(self):
        model = _small_model()
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = CheckpointManager(tmpdir, save_every=1, keep_last=2)
            opt = get_optimizer(model)
            for step in range(5):
                mgr.step(model, opt, None, step, 0)
            # Should only have 2 regular checkpoints + best
            regular = [p for p in Path(tmpdir).glob("checkpoint-step*.pt") if "best" not in p.name]
            assert len(regular) <= 2

    def test_save_final(self):
        model = _small_model()
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = CheckpointManager(tmpdir, save_every=999)
            path = mgr.save_final(model, None, None, 100, 5)
            assert path.exists()
            assert path.name == "final.pt"

    def test_find_latest(self):
        model = _small_model()
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = CheckpointManager(tmpdir, save_every=1, keep_last=10)
            opt = get_optimizer(model)
            mgr.step(model, opt, None, 10, 0)
            mgr.step(model, opt, None, 20, 0)
            latest = mgr.find_latest()
            assert latest is not None
            assert "20" in latest.name


# ---------------------------------------------------------------------------
# Trainer – single training step
# ---------------------------------------------------------------------------

class TestTrainer:
    def test_single_training_step(self):
        model = _small_model(attn_type="alibi", causal=True, model_type="decoder_only")
        dl = _make_dataloader(n_samples=4, batch_size=2)
        cfg = TrainerConfig(
            lr=1e-3, num_epochs=1, use_amp=False, device="cpu",
            model_type="decoder_only", log_every=1, save_every=999,
            gradient_accumulation_steps=1,
        )
        trainer = Trainer(model, cfg, dl)
        batch = next(iter(dl))
        loss = trainer._training_step(batch)
        assert isinstance(loss, float)
        assert loss > 0
        assert not math.isnan(loss)

    def test_compute_loss(self):
        model = _small_model(attn_type="alibi", causal=True, model_type="decoder_only")
        dl = _make_dataloader(n_samples=4)
        cfg = TrainerConfig(
            lr=1e-3, num_epochs=1, use_amp=False, device="cpu",
            model_type="decoder_only",
        )
        trainer = Trainer(model, cfg, dl)
        logits = torch.randn(2, SEQ_LEN, VOCAB)
        labels = torch.randint(0, VOCAB, (2, SEQ_LEN))
        loss = trainer._compute_loss(logits, labels)
        assert loss.item() > 0

    def test_gradient_accumulation(self):
        model = _small_model(attn_type="alibi", causal=True, model_type="decoder_only")
        dl = _make_dataloader(n_samples=4, batch_size=2)
        cfg = TrainerConfig(
            lr=1e-3, num_epochs=1, use_amp=False, device="cpu",
            model_type="decoder_only", log_every=1, save_every=999,
            gradient_accumulation_steps=2,
        )
        trainer = Trainer(model, cfg, dl)
        batch = next(iter(dl))
        loss = trainer._training_step(batch)
        # Loss should be divided by grad_accum_steps
        assert loss > 0

    def test_validation(self):
        model = _small_model(attn_type="alibi", causal=True, model_type="decoder_only")
        dl = _make_dataloader(n_samples=4, batch_size=2)
        cfg = TrainerConfig(
            lr=1e-3, num_epochs=1, use_amp=False, device="cpu",
            model_type="decoder_only",
        )
        trainer = Trainer(model, cfg, dl, val_dataloader=dl)
        metrics = trainer._validate()
        assert "val_loss" in metrics
        assert "val_ppl" in metrics
        assert metrics["val_loss"] > 0

    def test_full_train_loop_short(self):
        model = _small_model(attn_type="alibi", causal=True, model_type="decoder_only")
        dl = _make_dataloader(n_samples=4, batch_size=2)
        cfg = TrainerConfig(
            lr=1e-3, num_epochs=2, use_amp=False, device="cpu",
            model_type="decoder_only", log_every=1, save_every=999,
            gradient_accumulation_steps=1,
            eval_every=1,
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg.output_dir = tmpdir
            trainer = Trainer(model, cfg, dl, val_dataloader=dl)
            result = trainer.train()
            assert "train_history" in result
            assert "val_history" in result
            assert result["total_steps"] > 0

    def test_resume_from_checkpoint(self):
        model = _small_model(attn_type="alibi", causal=True, model_type="decoder_only")
        dl = _make_dataloader(n_samples=4, batch_size=2)
        cfg = TrainerConfig(
            lr=1e-3, num_epochs=1, use_amp=False, device="cpu",
            model_type="decoder_only", log_every=1, save_every=999,
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg.output_dir = tmpdir
            trainer = Trainer(model, cfg, dl)
            path = Path(tmpdir) / "ckpt.pt"
            save_checkpoint(model, trainer.optimizer, trainer.scheduler, 42, 1, path)
            step = trainer.resume_from(path)
            assert step == 42

    def test_early_stopping(self):
        model = _small_model(attn_type="alibi", causal=True, model_type="decoder_only")
        dl = _make_dataloader(n_samples=4, batch_size=2)
        cfg = TrainerConfig(
            lr=1e-3, num_epochs=5, use_amp=False, device="cpu",
            model_type="decoder_only", log_every=999, save_every=999,
            eval_every=1, early_stopping_patience=2,
            early_stopping_metric="val_loss",
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg.output_dir = tmpdir
            trainer = Trainer(model, cfg, dl, val_dataloader=dl)
            # Simulate: best already set to 10.0, then two non-improving
            # evals in a row. _check_early_stopping takes the was_best
            # verdict directly (from CheckpointManager.step()) rather than
            # recomputing it via is_best() -- a second is_best() call with
            # the same value would always compare it against itself.
            trainer.ckpt_manager._best_value = 10.0
            trainer._check_early_stopping(was_best=False)
            assert trainer._early_stop_counter == 1
            with pytest.raises(EarlyStopping):
                trainer._check_early_stopping(was_best=False)

    def test_resolve_device_cpu(self):
        dev = Trainer._resolve_device("cpu")
        assert dev == torch.device("cpu")

    def test_resolve_device_object(self):
        dev = Trainer._resolve_device(torch.device("cpu"))
        assert dev == torch.device("cpu")

    def test_callbacks_called(self):
        model = _small_model(attn_type="alibi", causal=True, model_type="decoder_only")
        dl = _make_dataloader(n_samples=4, batch_size=2)
        cfg = TrainerConfig(
            lr=1e-3, num_epochs=1, use_amp=False, device="cpu",
            model_type="decoder_only", log_every=999, save_every=999,
        )
        callback = MagicMock()
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg.output_dir = tmpdir
            trainer = Trainer(model, cfg, dl, callbacks=[callback])
            trainer.train()
            assert callback.call_count > 0


# ---------------------------------------------------------------------------
# TrainerConfig
# ---------------------------------------------------------------------------

class TestTrainerConfig:
    def test_defaults(self):
        cfg = TrainerConfig()
        assert cfg.lr == 3e-4
        assert cfg.num_epochs == 10
        assert cfg.model_type == "decoder_only"
        assert cfg.use_amp is True

    def test_custom(self):
        cfg = TrainerConfig(lr=1e-2, num_epochs=5, model_type="encoder_only")
        assert cfg.lr == 1e-2
        assert cfg.num_epochs == 5
        assert cfg.model_type == "encoder_only"
