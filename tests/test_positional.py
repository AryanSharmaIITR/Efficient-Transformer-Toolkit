"""Tests for positional encodings: RoPE, ALiBi, sinusoidal, learned."""
from __future__ import annotations

import math

import pytest
import torch

from helpers import D_MODEL, HEAD_DIM, N_HEADS, SEQ_LEN
from src.position.learn import LearnedPositionalEmbedding
from src.position.rope import RotaryPEMultiHeadAttention, RotaryPositionalEmbedding
from src.position.sinusoidal import SinusoidalPositionalEncoding


# ---------------------------------------------------------------------------
# RotaryPositionalEmbedding
# ---------------------------------------------------------------------------

class TestRotaryPE:
    def test_output_shape(self):
        rope = RotaryPositionalEmbedding(d=HEAD_DIM, max_seq_len=SEQ_LEN)
        # RoPE expects [seq_len, batch, heads, head_dim]
        x = torch.randn(SEQ_LEN, 2, N_HEADS, HEAD_DIM)
        out = rope(x)
        assert out.shape == x.shape

    def test_preserves_non_rope_dims(self):
        rope = RotaryPositionalEmbedding(d=8, max_seq_len=64)
        x = torch.randn(16, 2, 4, 32)  # d=8 means only first 8 dims rotated
        out = rope(x)
        # dims beyond d should pass through unchanged
        torch.testing.assert_close(out[:, :, :, 8:], x[:, :, :, 8:])

    def test_rotation_is_orthogonal(self):
        """Rotary embedding should approximately preserve norm."""
        rope = RotaryPositionalEmbedding(d=HEAD_DIM, max_seq_len=SEQ_LEN)
        x = torch.randn(SEQ_LEN, 1, 1, HEAD_DIM)
        out = rope(x)
        # Norms should be approximately equal (rotation preserves norm)
        norms_in = x.norm(dim=-1)
        norms_out = out.norm(dim=-1)
        torch.testing.assert_close(norms_in, norms_out, atol=1e-5, rtol=1e-5)

    def test_cache_building(self):
        rope = RotaryPositionalEmbedding(d=HEAD_DIM, max_seq_len=64)
        x = torch.randn(16, 1, 1, HEAD_DIM)
        rope(x)
        assert rope.cos_cached is not None
        assert rope.sin_cached is not None
        assert rope.cos_cached.shape[0] >= 16

    def test_cache_reuse(self):
        rope = RotaryPositionalEmbedding(d=HEAD_DIM, max_seq_len=64)
        x1 = torch.randn(8, 1, 1, HEAD_DIM)
        rope(x1)
        cos_before = rope.cos_cached.clone()
        x2 = torch.randn(8, 1, 1, HEAD_DIM)
        rope(x2)
        torch.testing.assert_close(rope.cos_cached, cos_before)

    def test_cache_extends(self):
        rope = RotaryPositionalEmbedding(d=HEAD_DIM, max_seq_len=64)
        x1 = torch.randn(8, 1, 1, HEAD_DIM)
        rope(x1)
        assert rope.cos_cached.shape[0] == 8
        x2 = torch.randn(32, 1, 1, HEAD_DIM)
        rope(x2)
        assert rope.cos_cached.shape[0] == 32

    def test_deterministic(self):
        rope = RotaryPositionalEmbedding(d=HEAD_DIM, max_seq_len=SEQ_LEN)
        x = torch.randn(SEQ_LEN, 2, N_HEADS, HEAD_DIM)
        out1 = rope(x)
        out2 = rope(x)
        torch.testing.assert_close(out1, out2)

    def test_gradient_flow(self):
        rope = RotaryPositionalEmbedding(d=HEAD_DIM, max_seq_len=SEQ_LEN)
        x = torch.randn(SEQ_LEN, 1, 1, HEAD_DIM, requires_grad=True)
        out = rope(x)
        out.sum().backward()
        assert x.grad is not None
        assert not torch.all(x.grad == 0)

    def test_no_nan(self):
        rope = RotaryPositionalEmbedding(d=HEAD_DIM, max_seq_len=64)
        x = torch.randn(64, 4, 8, HEAD_DIM) * 100
        out = rope(x)
        assert not torch.isnan(out).any()

    def test_half_precision(self):
        rope = RotaryPositionalEmbedding(d=HEAD_DIM, max_seq_len=SEQ_LEN)
        x = torch.randn(SEQ_LEN, 2, N_HEADS, HEAD_DIM).half()
        out = rope(x)
        assert out.shape == x.shape
        assert not torch.isnan(out).any()


# ---------------------------------------------------------------------------
# RotaryPEMultiHeadAttention
# ---------------------------------------------------------------------------

class TestRotaryMHA:
    def test_output_shape(self):
        attn = RotaryPEMultiHeadAttention(D_MODEL, N_HEADS, rope_percentage=0.5)
        q = torch.randn(SEQ_LEN, 2, N_HEADS, HEAD_DIM)
        k = torch.randn(SEQ_LEN, 2, N_HEADS, HEAD_DIM)
        scores = attn.get_scores(q, k)
        # einsum "ibhd,jbhd->bhij" => [B, H, T_q, T_k], matching every other
        # attention class's score layout so forward() can mask/softmax it
        # directly without an extra permute.
        assert scores.shape == (2, N_HEADS, SEQ_LEN, SEQ_LEN)

    def test_scores_not_nan(self):
        attn = RotaryPEMultiHeadAttention(D_MODEL, N_HEADS)
        q = torch.randn(SEQ_LEN, 2, N_HEADS, HEAD_DIM)
        k = torch.randn(SEQ_LEN, 2, N_HEADS, HEAD_DIM)
        scores = attn.get_scores(q, k)
        assert not torch.isnan(scores).any()


# ---------------------------------------------------------------------------
# SinusoidalPositionalEncoding
# ---------------------------------------------------------------------------

class TestSinusoidal:
    def test_output_shape(self):
        pe = SinusoidalPositionalEncoding(D_MODEL, max_len=100)
        x = torch.randn(2, SEQ_LEN, D_MODEL)
        out = pe(x)
        assert out.shape == x.shape

    def test_additive(self):
        pe = SinusoidalPositionalEncoding(D_MODEL, max_len=100)
        x = torch.randn(2, SEQ_LEN, D_MODEL)
        out = pe(x)
        expected = x + pe.pe[:, :SEQ_LEN]
        torch.testing.assert_close(out, expected)

    def test_buffer_is_buffer(self):
        pe = SinusoidalPositionalEncoding(D_MODEL, max_len=100)
        assert hasattr(pe, "pe")
        # Should be a registered buffer (not a parameter)
        assert "pe" in dict(pe.named_buffers())
        assert "pe" not in dict(pe.named_parameters())

    def test_deterministic(self):
        pe = SinusoidalPositionalEncoding(D_MODEL, max_len=100)
        x = torch.randn(2, SEQ_LEN, D_MODEL)
        out1 = pe(x)
        out2 = pe(x)
        torch.testing.assert_close(out1, out2)

    def test_different_positions_get_different_encodings(self):
        pe = SinusoidalPositionalEncoding(D_MODEL, max_len=100)
        enc0 = pe.pe[:, 0]
        enc1 = pe.pe[:, 1]
        assert not torch.allclose(enc0, enc1)

    def test_no_grad(self):
        pe = SinusoidalPositionalEncoding(D_MODEL, max_len=100)
        assert not pe.pe.requires_grad

    def test_max_len_exceeded(self):
        pe = SinusoidalPositionalEncoding(D_MODEL, max_len=8)
        x = torch.randn(1, 16, D_MODEL)
        out = pe(x)
        # Should still work, just slicing the first 8 positions
        assert out.shape == (1, 16, D_MODEL)


# ---------------------------------------------------------------------------
# LearnedPositionalEmbedding
# ---------------------------------------------------------------------------

class TestLearnedPE:
    def test_output_shape(self):
        pe = LearnedPositionalEmbedding(D_MODEL, max_len=100)
        x = torch.randn(2, SEQ_LEN, D_MODEL)
        out = pe(x)
        assert out.shape == x.shape

    def test_additive(self):
        pe = LearnedPositionalEmbedding(D_MODEL, max_len=100)
        x = torch.randn(2, SEQ_LEN, D_MODEL)
        out = pe(x)
        positions = torch.arange(SEQ_LEN).unsqueeze(0)
        expected = x + pe.pos_embedding(positions)
        torch.testing.assert_close(out, expected)

    def test_gradient_flow(self):
        pe = LearnedPositionalEmbedding(D_MODEL, max_len=100)
        x = torch.randn(2, SEQ_LEN, D_MODEL, requires_grad=True)
        out = pe(x)
        out.sum().backward()
        assert x.grad is not None
        assert not torch.all(x.grad == 0)

    def test_embedding_is_parameter(self):
        pe = LearnedPositionalEmbedding(D_MODEL, max_len=100)
        assert "pos_embedding.weight" in dict(pe.named_parameters())

    def test_init_normal(self):
        pe = LearnedPositionalEmbedding(D_MODEL, max_len=100)
        w = pe.pos_embedding.weight
        assert w.mean().abs() < 0.1
        assert w.std().item() < 0.1  # init std=0.02

    def test_different_positions_get_different_embeddings(self):
        pe = LearnedPositionalEmbedding(D_MODEL, max_len=100)
        e0 = pe.pos_embedding.weight[0]
        e1 = pe.pos_embedding.weight[1]
        assert not torch.allclose(e0, e1)

    def test_deterministic(self):
        pe = LearnedPositionalEmbedding(D_MODEL, max_len=100)
        x = torch.randn(2, SEQ_LEN, D_MODEL)
        out1 = pe(x)
        out2 = pe(x)
        torch.testing.assert_close(out1, out2)
