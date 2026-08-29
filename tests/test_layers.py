"""Tests for feed-forward, SwiGLU, encoder/decoder layers, and factory functions."""
from __future__ import annotations

import pytest
import torch

from helpers import BATCH, D_MODEL, HEAD_DIM, N_HEADS, SEQ_LEN, VOCAB
from src.models.transformer import (
    FeedForward,
    SwiGLU,
    TransformerConfig,
    TransformerDecoderLayer,
    TransformerDecoder,
    TransformerEncoder,
    TransformerEncoderLayer,
    get_attention,
    get_ffn,
    get_positional_encoding,
)


# ---------------------------------------------------------------------------
# FeedForward
# ---------------------------------------------------------------------------

class TestFeedForward:
    @pytest.mark.parametrize("activation", ["gelu", "relu", "silu"])
    def test_output_shape(self, activation):
        ffn = FeedForward(D_MODEL, D_MODEL * 4, activation=activation, dropout=0.0)
        x = torch.randn(BATCH, SEQ_LEN, D_MODEL)
        out = ffn(x)
        assert out.shape == (BATCH, SEQ_LEN, D_MODEL)

    @pytest.mark.parametrize("activation", ["gelu", "relu", "silu"])
    def test_gradient_flow(self, activation):
        ffn = FeedForward(D_MODEL, D_MODEL * 4, activation=activation, dropout=0.0)
        x = torch.randn(BATCH, SEQ_LEN, D_MODEL, requires_grad=True)
        out = ffn(x)
        out.sum().backward()
        assert x.grad is not None
        assert not torch.all(x.grad == 0)

    def test_invalid_activation(self):
        with pytest.raises(ValueError, match="Unknown activation"):
            FeedForward(D_MODEL, D_MODEL * 4, activation="invalid")

    def test_different_d_ff(self):
        ffn = FeedForward(D_MODEL, D_MODEL * 2, activation="gelu", dropout=0.0)
        x = torch.randn(BATCH, SEQ_LEN, D_MODEL)
        out = ffn(x)
        assert out.shape == (BATCH, SEQ_LEN, D_MODEL)


# ---------------------------------------------------------------------------
# SwiGLU
# ---------------------------------------------------------------------------

class TestSwiGLU:
    def test_output_shape(self):
        ffn = SwiGLU(D_MODEL, expansion_factor=4, dropout=0.0)
        x = torch.randn(BATCH, SEQ_LEN, D_MODEL)
        out = ffn(x)
        assert out.shape == (BATCH, SEQ_LEN, D_MODEL)

    def test_gradient_flow(self):
        ffn = SwiGLU(D_MODEL, expansion_factor=4, dropout=0.0)
        x = torch.randn(BATCH, SEQ_LEN, D_MODEL, requires_grad=True)
        out = ffn(x)
        out.sum().backward()
        assert x.grad is not None
        assert not torch.all(x.grad == 0)

    def test_expansion_factor(self):
        ef = 2
        ffn = SwiGLU(D_MODEL, expansion_factor=ef, dropout=0.0)
        hidden = D_MODEL * ef
        assert ffn.W1.weight.shape == (hidden, D_MODEL)
        assert ffn.W2.weight.shape == (hidden, D_MODEL)
        assert ffn.W3.weight.shape == (D_MODEL, hidden)

    def test_no_nan(self):
        ffn = SwiGLU(D_MODEL, expansion_factor=4, dropout=0.0)
        x = torch.randn(BATCH, SEQ_LEN, D_MODEL) * 100
        out = ffn(x)
        assert not torch.isnan(out).any()


# ---------------------------------------------------------------------------
# get_ffn factory
# ---------------------------------------------------------------------------

class TestGetFFN:
    def test_swiglu(self):
        ffn = get_ffn(D_MODEL, D_MODEL * 4, "swiglu", dropout=0.0)
        assert isinstance(ffn, SwiGLU)

    def test_gelu(self):
        ffn = get_ffn(D_MODEL, D_MODEL * 4, "gelu", dropout=0.0)
        assert isinstance(ffn, FeedForward)

    def test_relu(self):
        ffn = get_ffn(D_MODEL, D_MODEL * 4, "relu", dropout=0.0)
        assert isinstance(ffn, FeedForward)

    def test_silu(self):
        ffn = get_ffn(D_MODEL, D_MODEL * 4, "silu", dropout=0.0)
        assert isinstance(ffn, FeedForward)


# ---------------------------------------------------------------------------
# TransformerConfig
# ---------------------------------------------------------------------------

class TestTransformerConfig:
    def test_defaults(self):
        cfg = TransformerConfig()
        assert cfg.d_ff == 4 * cfg.d_model
        assert cfg.ffn_expansion == 4
        assert cfg.n_kv_heads == cfg.n_heads

    def test_mqa_forces_kv_heads_1(self):
        cfg = TransformerConfig(attn_type="mqa", n_heads=8)
        assert cfg.n_kv_heads == 1

    def test_gqa_validates_divisibility(self):
        with pytest.raises(ValueError, match="divisible"):
            TransformerConfig(attn_type="gqa", n_heads=6, n_kv_heads=4)

    def test_explicit_d_ff(self):
        cfg = TransformerConfig(d_model=128, d_ff=256)
        assert cfg.d_ff == 256


# ---------------------------------------------------------------------------
# TransformerEncoderLayer
# ---------------------------------------------------------------------------

class TestEncoderLayer:
    @pytest.mark.parametrize("norm_type", ["pre", "post"])
    def test_output_shape(self, norm_type):
        cfg = TransformerConfig(
            vocab_size=VOCAB, d_model=D_MODEL, n_heads=N_HEADS, n_layers=1,
            dropout=0.0, attn_type="alibi", causal=False,
            layer_norm_type=norm_type, d_ff=D_MODEL * 4,
            pos_encoding=None,
        )
        layer = TransformerEncoderLayer(cfg)
        x = torch.randn(BATCH, SEQ_LEN, D_MODEL)
        out = layer(x)
        assert out.shape == (BATCH, SEQ_LEN, D_MODEL)

    @pytest.mark.parametrize("norm_type", ["pre", "post"])
    def test_gradient_flow(self, norm_type):
        cfg = TransformerConfig(
            vocab_size=VOCAB, d_model=D_MODEL, n_heads=N_HEADS, n_layers=1,
            dropout=0.0, attn_type="alibi", causal=False,
            layer_norm_type=norm_type, d_ff=D_MODEL * 4,
            pos_encoding=None,
        )
        layer = TransformerEncoderLayer(cfg)
        x = torch.randn(BATCH, SEQ_LEN, D_MODEL, requires_grad=True)
        out = layer(x)
        out.sum().backward()
        assert x.grad is not None

    def test_with_mask(self):
        cfg = TransformerConfig(
            vocab_size=VOCAB, d_model=D_MODEL, n_heads=N_HEADS, n_layers=1,
            dropout=0.0, attn_type="alibi", causal=False,
            d_ff=D_MODEL * 4, pos_encoding=None,
        )
        layer = TransformerEncoderLayer(cfg)
        x = torch.randn(BATCH, SEQ_LEN, D_MODEL)
        mask = torch.ones(BATCH, 1, 1, SEQ_LEN)
        out = layer(x, mask=mask)
        assert out.shape == (BATCH, SEQ_LEN, D_MODEL)


# ---------------------------------------------------------------------------
# TransformerDecoderLayer
# ---------------------------------------------------------------------------

class TestDecoderLayer:
    @pytest.mark.parametrize("norm_type", ["pre", "post"])
    def test_output_shape_self_attn_only(self, norm_type):
        cfg = TransformerConfig(
            vocab_size=VOCAB, d_model=D_MODEL, n_heads=N_HEADS, n_layers=1,
            dropout=0.0, attn_type="alibi", causal=True,
            layer_norm_type=norm_type, d_ff=D_MODEL * 4,
            pos_encoding=None,
        )
        layer = TransformerDecoderLayer(cfg)
        x = torch.randn(BATCH, SEQ_LEN, D_MODEL)
        out = layer(x, encoder_output=None)
        assert out.shape == (BATCH, SEQ_LEN, D_MODEL)

    @pytest.mark.parametrize("norm_type", ["pre", "post"])
    def test_with_cross_attention(self, norm_type):
        cfg = TransformerConfig(
            vocab_size=VOCAB, d_model=D_MODEL, n_heads=N_HEADS, n_layers=1,
            dropout=0.0, attn_type="alibi", causal=True,
            layer_norm_type=norm_type, d_ff=D_MODEL * 4,
            pos_encoding=None,
        )
        layer = TransformerDecoderLayer(cfg)
        x = torch.randn(BATCH, SEQ_LEN, D_MODEL)
        enc_out = torch.randn(BATCH, SEQ_LEN, D_MODEL)
        out = layer(x, encoder_output=enc_out)
        assert out.shape == (BATCH, SEQ_LEN, D_MODEL)

    def test_gradient_flow(self):
        cfg = TransformerConfig(
            vocab_size=VOCAB, d_model=D_MODEL, n_heads=N_HEADS, n_layers=1,
            dropout=0.0, attn_type="alibi", causal=True,
            d_ff=D_MODEL * 4, pos_encoding=None,
        )
        layer = TransformerDecoderLayer(cfg)
        x = torch.randn(BATCH, SEQ_LEN, D_MODEL, requires_grad=True)
        out = layer(x)
        out.sum().backward()
        assert x.grad is not None

    def test_with_self_mask(self):
        cfg = TransformerConfig(
            vocab_size=VOCAB, d_model=D_MODEL, n_heads=N_HEADS, n_layers=1,
            dropout=0.0, attn_type="alibi", causal=True,
            d_ff=D_MODEL * 4, pos_encoding=None,
        )
        layer = TransformerDecoderLayer(cfg)
        x = torch.randn(BATCH, SEQ_LEN, D_MODEL)
        mask = torch.ones(BATCH, 1, SEQ_LEN, SEQ_LEN)
        out = layer(x, self_mask=mask)
        assert out.shape == (BATCH, SEQ_LEN, D_MODEL)


# ---------------------------------------------------------------------------
# TransformerEncoder (stack of layers)
# ---------------------------------------------------------------------------

class TestEncoder:
    def test_output_shape(self):
        cfg = TransformerConfig(
            vocab_size=VOCAB, d_model=D_MODEL, n_heads=N_HEADS, n_layers=2,
            dropout=0.0, attn_type="alibi", causal=False,
            d_ff=D_MODEL * 4, pos_encoding="sinusoidal",
        )
        encoder = TransformerEncoder(cfg)
        x = torch.randn(BATCH, SEQ_LEN, D_MODEL)
        out = encoder(x)
        assert out.shape == (BATCH, SEQ_LEN, D_MODEL)

    def test_gradient_flow(self):
        cfg = TransformerConfig(
            vocab_size=VOCAB, d_model=D_MODEL, n_heads=N_HEADS, n_layers=2,
            dropout=0.0, attn_type="alibi", causal=False,
            d_ff=D_MODEL * 4, pos_encoding="sinusoidal",
        )
        encoder = TransformerEncoder(cfg)
        x = torch.randn(BATCH, SEQ_LEN, D_MODEL, requires_grad=True)
        out = encoder(x)
        out.sum().backward()
        assert x.grad is not None

    def test_layer_count(self):
        cfg = TransformerConfig(
            vocab_size=VOCAB, d_model=D_MODEL, n_heads=N_HEADS, n_layers=3,
            dropout=0.0, attn_type="alibi", causal=False,
            d_ff=D_MODEL * 4, pos_encoding=None,
        )
        encoder = TransformerEncoder(cfg)
        assert len(encoder.layers) == 3


# ---------------------------------------------------------------------------
# TransformerDecoder (stack of layers)
# ---------------------------------------------------------------------------

class TestDecoder:
    def test_output_shape(self):
        cfg = TransformerConfig(
            vocab_size=VOCAB, d_model=D_MODEL, n_heads=N_HEADS, n_layers=2,
            dropout=0.0, attn_type="alibi", causal=True,
            d_ff=D_MODEL * 4, pos_encoding="sinusoidal",
        )
        decoder = TransformerDecoder(cfg)
        x = torch.randn(BATCH, SEQ_LEN, D_MODEL)
        out = decoder(x)
        assert out.shape == (BATCH, SEQ_LEN, D_MODEL)

    def test_with_encoder_output(self):
        cfg = TransformerConfig(
            vocab_size=VOCAB, d_model=D_MODEL, n_heads=N_HEADS, n_layers=2,
            dropout=0.0, attn_type="alibi", causal=True,
            d_ff=D_MODEL * 4, pos_encoding="sinusoidal",
        )
        decoder = TransformerDecoder(cfg)
        x = torch.randn(BATCH, SEQ_LEN, D_MODEL)
        enc_out = torch.randn(BATCH, SEQ_LEN, D_MODEL)
        out = decoder(x, encoder_output=enc_out)
        assert out.shape == (BATCH, SEQ_LEN, D_MODEL)


# ---------------------------------------------------------------------------
# get_positional_encoding factory
# ---------------------------------------------------------------------------

class TestGetPosEnc:
    def test_rope(self):
        # RoPE isn't a standalone additive embedding -- it rotates Q/K
        # *inside* attention (see get_attention()), so it has no
        # get_positional_encoding() representation of its own.
        cfg = TransformerConfig(pos_encoding="rope", d_model=D_MODEL, n_heads=N_HEADS)
        enc = get_positional_encoding(cfg)
        assert enc is None

        attn = get_attention(cfg)
        assert isinstance(attn, RotaryPEMultiHeadAttention)

    def test_sinusoidal(self):
        cfg = TransformerConfig(pos_encoding="sinusoidal", d_model=D_MODEL)
        enc = get_positional_encoding(cfg)
        assert isinstance(enc, SinusoidalPositionalEncoding)

    def test_learned(self):
        cfg = TransformerConfig(pos_encoding="learned", d_model=D_MODEL)
        enc = get_positional_encoding(cfg)
        assert isinstance(enc, LearnedPositionalEmbedding)

    def test_none(self):
        cfg = TransformerConfig(pos_encoding=None)
        enc = get_positional_encoding(cfg)
        assert enc is None


# Import needed for get_positional_encoding tests
from src.position.rope import RotaryPEMultiHeadAttention
from src.position.sinusoidal import SinusoidalPositionalEncoding
from src.position.learn import LearnedPositionalEmbedding
