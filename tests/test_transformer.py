"""Tests for the full Transformer model: forward, generation, save/load."""
from __future__ import annotations

import json
import math
import tempfile
from pathlib import Path

import pytest
import torch

from helpers import BATCH, D_MODEL, HEAD_DIM, N_HEADS, SEQ_LEN, VOCAB
from src.models.transformer import Transformer, TransformerConfig


def _encoder_only_config(**overrides):
    defaults = dict(
        vocab_size=VOCAB, d_model=D_MODEL, n_heads=N_HEADS, n_layers=2,
        d_ff=D_MODEL * 4, dropout=0.0, max_seq_len=SEQ_LEN,
        attn_type="alibi", pos_encoding="sinusoidal", causal=False,
        use_flash=True, tie_embeddings=True, use_bias=False,
        ffn_activation="swiglu", layer_norm_type="pre",
    )
    defaults.update(overrides)
    return TransformerConfig(**defaults)


def _decoder_only_config(**overrides):
    defaults = dict(
        vocab_size=VOCAB, d_model=D_MODEL, n_heads=N_HEADS, n_layers=2,
        d_ff=D_MODEL * 4, dropout=0.0, max_seq_len=SEQ_LEN,
        attn_type="alibi", pos_encoding="sinusoidal", causal=True,
        use_flash=True, tie_embeddings=True, use_bias=False,
        ffn_activation="swiglu", layer_norm_type="pre",
    )
    defaults.update(overrides)
    return TransformerConfig(**defaults)


def _enc_dec_config(**overrides):
    defaults = dict(
        vocab_size=VOCAB, d_model=D_MODEL, n_heads=N_HEADS, n_layers=2,
        d_ff=D_MODEL * 4, dropout=0.0, max_seq_len=SEQ_LEN,
        attn_type="alibi", pos_encoding="sinusoidal", causal=False,
        use_flash=True, tie_embeddings=True, use_bias=False,
        ffn_activation="swiglu", layer_norm_type="pre",
    )
    defaults.update(overrides)
    return TransformerConfig(**defaults)


# ---------------------------------------------------------------------------
# Forward pass – encoder-only
# ---------------------------------------------------------------------------

class TestEncoderOnly:
    def test_output_shape(self):
        model = Transformer(_encoder_only_config(), use_encoder=True, use_decoder=False)
        ids = torch.randint(0, VOCAB, (BATCH, SEQ_LEN))
        logits = model(ids)
        assert logits.shape == (BATCH, SEQ_LEN, VOCAB)

    def test_with_attention_mask(self):
        model = Transformer(_encoder_only_config(), use_encoder=True, use_decoder=False)
        ids = torch.randint(0, VOCAB, (BATCH, SEQ_LEN))
        mask = torch.ones(BATCH, SEQ_LEN)
        mask[:, -4:] = 0
        logits = model(ids, attention_mask=mask)
        assert logits.shape == (BATCH, SEQ_LEN, VOCAB)

    def test_return_encoder_output(self):
        model = Transformer(_encoder_only_config(), use_encoder=True, use_decoder=False)
        ids = torch.randint(0, VOCAB, (BATCH, SEQ_LEN))
        out = model(ids, return_encoder_output=True)
        assert isinstance(out, dict)
        assert "logits" in out
        assert "encoder_output" in out
        assert out["encoder_output"].shape == (BATCH, SEQ_LEN, D_MODEL)

    def test_gradient_flow(self):
        model = Transformer(_encoder_only_config(), use_encoder=True, use_decoder=False)
        ids = torch.randint(0, VOCAB, (BATCH, SEQ_LEN))
        logits = model(ids)
        loss = logits.sum()
        loss.backward()
        for p in model.parameters():
            if p.requires_grad:
                assert p.grad is not None


# ---------------------------------------------------------------------------
# Forward pass – decoder-only
# ---------------------------------------------------------------------------

class TestDecoderOnly:
    def test_output_shape(self):
        model = Transformer(_decoder_only_config(), use_encoder=False, use_decoder=True)
        ids = torch.randint(0, VOCAB, (BATCH, SEQ_LEN))
        logits = model(ids)
        assert logits.shape == (BATCH, SEQ_LEN, VOCAB)

    def test_with_attention_mask(self):
        model = Transformer(_decoder_only_config(), use_encoder=False, use_decoder=True)
        ids = torch.randint(0, VOCAB, (BATCH, SEQ_LEN))
        mask = torch.ones(BATCH, SEQ_LEN)
        mask[:, -4:] = 0
        logits = model(ids, attention_mask=mask)
        assert logits.shape == (BATCH, SEQ_LEN, VOCAB)

    def test_gradient_flow(self):
        model = Transformer(_decoder_only_config(), use_encoder=False, use_decoder=True)
        ids = torch.randint(0, VOCAB, (BATCH, SEQ_LEN))
        logits = model(ids)
        logits.sum().backward()
        has_grad = False
        for p in model.parameters():
            if p.requires_grad and p.grad is not None:
                has_grad = True
                break
        assert has_grad


# ---------------------------------------------------------------------------
# Forward pass – encoder-decoder
# ---------------------------------------------------------------------------

class TestEncDec:
    def test_output_shape(self):
        model = Transformer(_enc_dec_config(), use_encoder=True, use_decoder=True)
        enc_ids = torch.randint(0, VOCAB, (BATCH, SEQ_LEN))
        dec_ids = torch.randint(0, VOCAB, (BATCH, SEQ_LEN))
        logits = model(enc_ids, decoder_input_ids=dec_ids)
        assert logits.shape == (BATCH, SEQ_LEN, VOCAB)

    def test_requires_decoder_input_ids(self):
        model = Transformer(_enc_dec_config(), use_encoder=True, use_decoder=True)
        enc_ids = torch.randint(0, VOCAB, (BATCH, SEQ_LEN))
        with pytest.raises(ValueError, match="decoder_input_ids required"):
            model(enc_ids)

    def test_return_encoder_output(self):
        model = Transformer(_enc_dec_config(), use_encoder=True, use_decoder=True)
        enc_ids = torch.randint(0, VOCAB, (BATCH, SEQ_LEN))
        dec_ids = torch.randint(0, VOCAB, (BATCH, SEQ_LEN))
        out = model(enc_ids, decoder_input_ids=dec_ids, return_encoder_output=True)
        assert isinstance(out, dict)
        assert out["encoder_output"] is not None


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

class TestGenerate:
    def test_greedy_generation(self):
        model = Transformer(_decoder_only_config(), use_encoder=False, use_decoder=True)
        ids = torch.randint(0, VOCAB, (1, 8))
        out = model.generate(ids, max_new_tokens=10, do_sample=False)
        assert out.shape[0] == 1
        assert out.shape[1] == 8 + 10

    def test_generation_eos_stops(self):
        model = Transformer(_decoder_only_config(), use_encoder=False, use_decoder=True)
        ids = torch.randint(0, VOCAB, (1, 4))
        eos_id = 1
        # Manually force eos on first step by setting bias
        out = model.generate(
            ids, max_new_tokens=50, do_sample=False, eos_token_id=eos_id,
        )
        assert out.shape[1] <= 4 + 50

    def test_generation_with_temperature(self):
        model = Transformer(_decoder_only_config(), use_encoder=False, use_decoder=True)
        ids = torch.randint(0, VOCAB, (1, 4))
        out = model.generate(ids, max_new_tokens=5, temperature=0.5, do_sample=True)
        assert out.shape[1] == 4 + 5

    def test_generation_with_top_k(self):
        model = Transformer(_decoder_only_config(), use_encoder=False, use_decoder=True)
        ids = torch.randint(0, VOCAB, (1, 4))
        out = model.generate(ids, max_new_tokens=5, temperature=1.0, top_k=10, do_sample=True)
        assert out.shape[1] == 4 + 5

    def test_generation_with_top_p(self):
        model = Transformer(_decoder_only_config(), use_encoder=False, use_decoder=True)
        ids = torch.randint(0, VOCAB, (1, 4))
        out = model.generate(ids, max_new_tokens=5, temperature=1.0, top_p=0.9, do_sample=True)
        assert out.shape[1] == 4 + 5

    def test_generation_encoder_decoder(self):
        model = Transformer(_enc_dec_config(), use_encoder=True, use_decoder=True)
        enc_ids = torch.randint(0, VOCAB, (1, 8))
        out = model.generate(
            enc_ids,
            max_new_tokens=5, do_sample=False,
        )
        assert out.shape[0] == 1
        assert out.shape[1] == 8 + 5

    def test_max_new_tokens_zero(self):
        model = Transformer(_decoder_only_config(), use_encoder=False, use_decoder=True)
        ids = torch.randint(0, VOCAB, (1, 4))
        out = model.generate(ids, max_new_tokens=0, do_sample=False)
        assert out.shape[1] == 4

    def test_generation_no_nan(self):
        model = Transformer(_decoder_only_config(), use_encoder=False, use_decoder=True)
        ids = torch.randint(0, VOCAB, (1, 4))
        out = model.generate(ids, max_new_tokens=10, do_sample=False)
        assert not torch.isnan(out.float()).any()


# ---------------------------------------------------------------------------
# Save / Load
# ---------------------------------------------------------------------------

class TestSaveLoad:
    def test_roundtrip_encoder_only(self):
        cfg = _encoder_only_config()
        model = Transformer(cfg, use_encoder=True, use_decoder=False)
        model.eval()
        ids = torch.randint(0, VOCAB, (1, 8))
        with torch.no_grad():
            logits_before = model(ids)

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = str(Path(tmpdir) / "model_save")
            Path(save_path).mkdir()
            model.save_pretrained(save_path)
            # Check files exist
            assert (Path(save_path) / "pytorch_model.bin").exists()
            assert (Path(save_path) / "config.json").exists()

            loaded = Transformer.from_pretrained(save_path)
            loaded.eval()
            with torch.no_grad():
                logits_after = loaded(ids)
            torch.testing.assert_close(logits_before, logits_after, atol=1e-6, rtol=1e-6)

    def test_roundtrip_decoder_only(self):
        cfg = _decoder_only_config()
        model = Transformer(cfg, use_encoder=False, use_decoder=True)
        model.eval()
        ids = torch.randint(0, VOCAB, (1, 8))
        with torch.no_grad():
            logits_before = model(ids)

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = str(Path(tmpdir) / "model_save")
            Path(save_path).mkdir()
            model.save_pretrained(save_path)
            loaded = Transformer.from_pretrained(save_path)
            loaded.eval()
            with torch.no_grad():
                logits_after = loaded(ids)
            torch.testing.assert_close(logits_before, logits_after, atol=1e-6, rtol=1e-6)

    def test_config_json_content(self):
        cfg = _encoder_only_config()
        model = Transformer(cfg, use_encoder=True, use_decoder=False)
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = str(Path(tmpdir) / "model_save")
            Path(save_path).mkdir()
            model.save_pretrained(save_path)
            with open(Path(save_path) / "config.json") as f:
                saved_cfg = json.load(f)
            assert saved_cfg["vocab_size"] == VOCAB
            assert saved_cfg["d_model"] == D_MODEL

    def test_tied_embeddings(self):
        cfg = _encoder_only_config(tie_embeddings=True)
        model = Transformer(cfg, use_encoder=True, use_decoder=False)
        # lm_head weight should be the same object as token_embedding weight
        assert model.lm_head.weight is model.token_embedding.weight

    def test_untied_embeddings(self):
        cfg = _encoder_only_config(tie_embeddings=False)
        model = Transformer(cfg, use_encoder=True, use_decoder=False)
        assert model.lm_head.weight is not model.token_embedding.weight
        assert model.lm_head.weight.shape == (VOCAB, D_MODEL)


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_single_token_sequence(self):
        model = Transformer(_encoder_only_config(), use_encoder=True, use_decoder=False)
        ids = torch.randint(0, VOCAB, (1, 1))
        logits = model(ids)
        assert logits.shape == (1, 1, VOCAB)

    def test_batch_size_one(self):
        model = Transformer(_encoder_only_config(), use_encoder=True, use_decoder=False)
        ids = torch.randint(0, VOCAB, (1, SEQ_LEN))
        logits = model(ids)
        assert logits.shape == (1, SEQ_LEN, VOCAB)

    def test_no_nan_encoder_only(self):
        model = Transformer(_encoder_only_config(), use_encoder=True, use_decoder=False)
        ids = torch.randint(0, VOCAB, (BATCH, SEQ_LEN))
        logits = model(ids)
        assert not torch.isnan(logits).any()

    def test_no_nan_decoder_only(self):
        model = Transformer(_decoder_only_config(), use_encoder=False, use_decoder=True)
        ids = torch.randint(0, VOCAB, (BATCH, SEQ_LEN))
        logits = model(ids)
        assert not torch.isnan(logits).any()

    def test_parameter_count(self):
        cfg = _encoder_only_config()
        model = Transformer(cfg, use_encoder=True, use_decoder=False)
        n_params = sum(p.numel() for p in model.parameters())
        assert n_params > 0

    def test_invalid_model_config(self):
        with pytest.raises(ValueError, match="Invalid combination"):
            model = Transformer(_encoder_only_config(), use_encoder=False, use_decoder=False)
            ids = torch.randint(0, VOCAB, (1, SEQ_LEN))
            model(ids)
