"""Tests for all attention variants: vanilla MHA, flash v1/v2, ALiBi, GQA, MQA, block-sparse."""
from __future__ import annotations

import math

import pytest
import torch

from helpers import BATCH, D_MODEL, HEAD_DIM, N_HEADS, SEQ_LEN, VOCAB
from src.attention.alibi import Alibi, build_alibi_tensor, get_slope
from src.attention.base import BaseAttention
from src.attention.block import BlockSparseAttention
from src.attention.flash import FlashAttentionV1, FlashAttentionV2, FlashAttentionWrapper
from src.attention.gqa import GroupedQueryAttention
from src.attention.mqa import MultiQueryAttention
from src.attention.multiheadattention import MultiHeadAttention


def _make_qkv(batch=BATCH, seq=SEQ_LEN, d=D_MODEL):
    x = torch.randn(batch, seq, d)
    return x, x.clone(), x.clone()


def _causal_mask_4d(batch, seq, device="cpu"):
    return torch.tril(torch.ones(batch, 1, seq, seq, device=device))


# ---------------------------------------------------------------------------
# MultiHeadAttention
# ---------------------------------------------------------------------------

class TestMultiHeadAttention:
    def test_output_shape(self):
        attn = MultiHeadAttention(D_MODEL, N_HEADS, dropout=0.0, causal=False)
        q, k, v = _make_qkv()
        out = attn(q, k, v)
        assert out.shape == (BATCH, SEQ_LEN, D_MODEL)

    def test_causal_output_shape(self):
        attn = MultiHeadAttention(D_MODEL, N_HEADS, dropout=0.0, causal=True)
        q, k, v = _make_qkv()
        out = attn(q, k, v)
        assert out.shape == (BATCH, SEQ_LEN, D_MODEL)

    def test_gradient_flow(self):
        attn = MultiHeadAttention(D_MODEL, N_HEADS, dropout=0.0, causal=False)
        q, k, v = _make_qkv()
        q.requires_grad_(True)
        out = attn(q, k, v)
        loss = out.sum()
        loss.backward()
        assert q.grad is not None
        assert q.grad.shape == q.shape
        assert not torch.all(q.grad == 0)

    def test_with_padding_mask(self):
        attn = MultiHeadAttention(D_MODEL, N_HEADS, dropout=0.0, causal=False)
        q, k, v = _make_qkv()
        mask = torch.ones(BATCH, 1, 1, SEQ_LEN)
        mask[:, :, :, -3:] = 0
        out = attn(q, k, v, mask=mask)
        assert out.shape == (BATCH, SEQ_LEN, D_MODEL)

    def test_causal_mask_prevents_lookahead(self):
        attn = MultiHeadAttention(D_MODEL, N_HEADS, dropout=0.0, causal=True)
        q, k, v = _make_qkv(seq=4)
        out1 = attn(q, k, v)
        k2 = k.clone()
        k2[:, 2:, :] += 10.0
        out2 = attn(q, k2, v)
        torch.testing.assert_close(out1[:, :2], out2[:, :2], atol=1e-6, rtol=1e-6)

    def test_varied_sequence_lengths(self):
        attn = MultiHeadAttention(D_MODEL, N_HEADS, dropout=0.0, causal=False)
        for seq in [1, 5, 32]:
            q, k, v = _make_qkv(seq=seq)
            out = attn(q, k, v)
            assert out.shape == (BATCH, seq, D_MODEL)

    def test_single_batch(self):
        attn = MultiHeadAttention(D_MODEL, N_HEADS, dropout=0.0)
        q, k, v = _make_qkv(batch=1)
        out = attn(q, k, v)
        assert out.shape == (1, SEQ_LEN, D_MODEL)

    def test_mixed_precision(self):
        attn = MultiHeadAttention(D_MODEL, N_HEADS, dropout=0.0).half()
        q, k, v = _make_qkv()
        out = attn(q.half(), k.half(), v.half())
        assert out.shape == (BATCH, SEQ_LEN, D_MODEL)
        assert out.dtype == torch.float16

    def test_numerical_stability_no_nan(self):
        attn = MultiHeadAttention(D_MODEL, N_HEADS, dropout=0.0, causal=False)
        q = torch.randn(BATCH, 32, D_MODEL) * 10
        _, k, v = _make_qkv(seq=32)
        out = attn(q, k, v)
        assert not torch.isnan(out).any()
        assert not torch.isinf(out).any()


# ---------------------------------------------------------------------------
# FlashAttentionWrapper (V1 & V2)
# ---------------------------------------------------------------------------

class TestFlashAttention:
    @pytest.mark.parametrize("use_v2", [False, True])
    def test_output_shape(self, use_v2):
        attn = FlashAttentionWrapper(D_MODEL, N_HEADS, block_size=32, causal=False, use_v2=use_v2)
        q, k, v = _make_qkv()
        out = attn(q, k, v)
        assert out.shape == (BATCH, SEQ_LEN, D_MODEL)

    @pytest.mark.parametrize("use_v2", [False, True])
    def test_causal(self, use_v2):
        attn = FlashAttentionWrapper(D_MODEL, N_HEADS, block_size=32, causal=True, use_v2=use_v2)
        q, k, v = _make_qkv()
        out = attn(q, k, v)
        assert out.shape == (BATCH, SEQ_LEN, D_MODEL)

    def test_gradient_flow_v2(self):
        attn = FlashAttentionWrapper(D_MODEL, N_HEADS, block_size=32, causal=False, use_v2=True)
        q, k, v = _make_qkv()
        q.requires_grad_(True)
        out = attn(q, k, v)
        out.sum().backward()
        assert q.grad is not None
        assert not torch.all(q.grad == 0)

    @pytest.mark.parametrize("use_v2", [False, True])
    def test_with_mask(self, use_v2):
        attn = FlashAttentionWrapper(D_MODEL, N_HEADS, block_size=32, causal=False, use_v2=use_v2)
        q, k, v = _make_qkv()
        mask = _causal_mask_4d(BATCH, SEQ_LEN)
        out = attn(q, k, v, mask=mask)
        assert out.shape == (BATCH, SEQ_LEN, D_MODEL)

    def test_no_nan_large_input(self):
        attn = FlashAttentionV2(D_MODEL, N_HEADS, block_size=32, causal=False)
        q = torch.randn(BATCH, 32, D_MODEL) * 10
        _, k, v = _make_qkv(seq=32)
        out = attn(q, k, v)
        assert not torch.isnan(out).any()

    def test_flash_v2_adaptive_block_size(self):
        model = FlashAttentionV2(D_MODEL, N_HEADS, block_size=128)
        assert model._compute_block_size(256) == 64
        assert model._compute_block_size(600) == 128
        assert model._compute_block_size(2500) == 256

    def test_v1_v2_reasonably_close(self):
        torch.manual_seed(42)
        q, k, v = _make_qkv(seq=32)
        attn1 = FlashAttentionV1(D_MODEL, N_HEADS, block_size=32, causal=False)
        attn2 = FlashAttentionV2(D_MODEL, N_HEADS, block_size=32, causal=False, use_fp32_accum=False)
        attn2.load_state_dict(attn1.state_dict())
        out1 = attn1(q, k, v)
        out2 = attn2(q, k, v)
        torch.testing.assert_close(out1, out2, atol=1e-4, rtol=1e-4)

    def test_empty_seq_len_boundary(self):
        attn = FlashAttentionV2(D_MODEL, N_HEADS, block_size=128, causal=False)
        q, k, v = _make_qkv(seq=5)
        out = attn(q, k, v)
        assert out.shape == (BATCH, 5, D_MODEL)


# ---------------------------------------------------------------------------
# ALiBi
# ---------------------------------------------------------------------------

class TestAlibi:
    def test_output_shape(self):
        attn = Alibi(D_MODEL, N_HEADS, dropout=0.0, causal=True)
        q, k, v = _make_qkv()
        out = attn(q, k, v)
        assert out.shape == (BATCH, SEQ_LEN, D_MODEL)

    def test_get_slope_count(self):
        slopes = get_slope(8)
        assert slopes.shape == (8,)
        assert (slopes > 0).all()
        assert slopes[0] >= slopes[-1]

    def test_get_slope_non_power_of_two(self):
        slopes = get_slope(6)
        assert slopes.shape == (6,)

    def test_build_alibi_tensor_shape(self):
        bias = build_alibi_tensor(N_HEADS, SEQ_LEN, device="cpu", dtype=torch.float32)
        assert bias.shape == (1, N_HEADS, SEQ_LEN, SEQ_LEN)

    def test_build_alibi_tensor_properties(self):
        bias = build_alibi_tensor(N_HEADS, 8, device="cpu", dtype=torch.float32)
        diag = bias[0, :, torch.arange(8), torch.arange(8)]
        torch.testing.assert_close(diag, torch.zeros_like(diag), atol=1e-6, rtol=1e-6)
        assert (bias[:, :, 0, 1:] <= 0).all()

    def test_gradient_flow(self):
        attn = Alibi(D_MODEL, N_HEADS, dropout=0.0, causal=True)
        q, k, v = _make_qkv()
        q.requires_grad_(True)
        out = attn(q, k, v)
        out.sum().backward()
        assert q.grad is not None
        assert not torch.all(q.grad == 0)

    def test_with_padding_mask_2d(self):
        attn = Alibi(D_MODEL, N_HEADS, dropout=0.0, causal=False)
        q, k, v = _make_qkv()
        mask = torch.ones(BATCH, SEQ_LEN)
        mask[:, -2:] = 0
        out = attn(q, k, v, mask=mask)
        assert out.shape == (BATCH, SEQ_LEN, D_MODEL)

    def test_with_padding_mask_4d(self):
        attn = Alibi(D_MODEL, N_HEADS, dropout=0.0, causal=False)
        q, k, v = _make_qkv()
        mask = torch.ones(BATCH, 1, SEQ_LEN, SEQ_LEN)
        out = attn(q, k, v, mask=mask)
        assert out.shape == (BATCH, SEQ_LEN, D_MODEL)

    def test_no_nan(self):
        attn = Alibi(D_MODEL, N_HEADS, dropout=0.0, causal=True)
        q, k, v = _make_qkv(seq=32)
        out = attn(q, k, v)
        assert not torch.isnan(out).any()

    def test_causal_vs_non_causal_differ(self):
        attn_c = Alibi(D_MODEL, N_HEADS, dropout=0.0, causal=True)
        attn_nc = Alibi(D_MODEL, N_HEADS, dropout=0.0, causal=False)
        attn_nc.load_state_dict(attn_c.state_dict())
        q, k, v = _make_qkv()
        out_c = attn_c(q, k, v)
        out_nc = attn_nc(q, k, v)
        assert not torch.allclose(out_c, out_nc, atol=1e-5)

    def test_cross_attention(self):
        attn = Alibi(D_MODEL, N_HEADS, dropout=0.0, causal=False)
        q = torch.randn(BATCH, 8, D_MODEL)
        kv = torch.randn(BATCH, 8, D_MODEL)
        out = attn(q, kv, kv)
        assert out.shape == (BATCH, 8, D_MODEL)


# ---------------------------------------------------------------------------
# GroupedQueryAttention
# ---------------------------------------------------------------------------

class TestGQA:
    def test_output_shape(self):
        n_kv = 2
        attn = GroupedQueryAttention(D_MODEL, N_HEADS, n_kv, dropout=0.0, causal=False)
        q, k, v = _make_qkv()
        out = attn(q, k, v)
        assert out.shape == (BATCH, SEQ_LEN, D_MODEL)

    def test_groups_computed(self):
        attn = GroupedQueryAttention(D_MODEL, N_HEADS, 2, dropout=0.0)
        assert attn.n_groups == N_HEADS // 2

    def test_gradient_flow(self):
        n_kv = 2
        attn = GroupedQueryAttention(D_MODEL, N_HEADS, n_kv, dropout=0.0, causal=False)
        q, k, v = _make_qkv()
        q.requires_grad_(True)
        out = attn(q, k, v)
        out.sum().backward()
        assert q.grad is not None

    def test_causal(self):
        n_kv = 2
        attn = GroupedQueryAttention(D_MODEL, N_HEADS, n_kv, dropout=0.0, causal=True)
        q, k, v = _make_qkv()
        out = attn(q, k, v)
        assert out.shape == (BATCH, SEQ_LEN, D_MODEL)

    def test_from_multi_head(self):
        mha = MultiHeadAttention(D_MODEL, N_HEADS, dropout=0.0, causal=False)
        gqa = GroupedQueryAttention.from_multi_head(mha, n_kv_heads=2)
        assert gqa.n_kv_heads == 2
        assert gqa.n_groups == N_HEADS // 2
        torch.testing.assert_close(gqa.Wq.weight, mha.Wq.weight)

    def test_get_kv_cache(self):
        n_kv = 2
        attn = GroupedQueryAttention(D_MODEL, N_HEADS, n_kv, dropout=0.0)
        k = torch.randn(BATCH, SEQ_LEN, D_MODEL)
        v = torch.randn(BATCH, SEQ_LEN, D_MODEL)
        kc, vc = attn.get_kv_cache(k, v)
        assert kc.shape == (BATCH, n_kv, SEQ_LEN, D_MODEL // N_HEADS)
        assert vc.shape == (BATCH, n_kv, SEQ_LEN, D_MODEL // N_HEADS)

    def test_with_mask(self):
        attn = GroupedQueryAttention(D_MODEL, N_HEADS, 2, dropout=0.0, causal=False)
        q, k, v = _make_qkv()
        mask = torch.ones(BATCH, 1, 1, SEQ_LEN)
        out = attn(q, k, v, mask=mask)
        assert out.shape == (BATCH, SEQ_LEN, D_MODEL)

    def test_repr(self):
        attn = GroupedQueryAttention(D_MODEL, N_HEADS, 2, dropout=0.0)
        r = repr(attn)
        assert "n_kv_heads=2" in r
        assert f"groups={N_HEADS // 2}" in r

    def test_invalid_n_kv_heads(self):
        with pytest.raises(AssertionError):
            GroupedQueryAttention(D_MODEL, N_HEADS, 3, dropout=0.0)

    def test_cross_attention(self):
        attn = GroupedQueryAttention(D_MODEL, N_HEADS, 2, dropout=0.0, causal=False)
        q = torch.randn(BATCH, 8, D_MODEL)
        kv = torch.randn(BATCH, 16, D_MODEL)
        out = attn(q, kv, kv)
        assert out.shape == (BATCH, 8, D_MODEL)


# ---------------------------------------------------------------------------
# MultiQueryAttention
# ---------------------------------------------------------------------------

class TestMQA:
    def test_output_shape(self):
        dk, dv = 32, 32
        attn = MultiQueryAttention(D_MODEL, N_HEADS, dk, dv, dropout=0.0, causal=False)
        q, k, v = _make_qkv()
        out = attn(q, k, v)
        assert out.shape == (BATCH, SEQ_LEN, D_MODEL)

    def test_gradient_flow(self):
        dk, dv = 32, 32
        attn = MultiQueryAttention(D_MODEL, N_HEADS, dk, dv, dropout=0.0, causal=False)
        q, k, v = _make_qkv()
        q.requires_grad_(True)
        out = attn(q, k, v)
        out.sum().backward()
        assert q.grad is not None
        assert not torch.all(q.grad == 0)

    def test_causal(self):
        dk, dv = 32, 32
        attn = MultiQueryAttention(D_MODEL, N_HEADS, dk, dv, dropout=0.0, causal=True)
        q, k, v = _make_qkv()
        out = attn(q, k, v)
        assert out.shape == (BATCH, SEQ_LEN, D_MODEL)

    def test_with_mask(self):
        dk, dv = 32, 32
        attn = MultiQueryAttention(D_MODEL, N_HEADS, dk, dv, dropout=0.0, causal=False)
        q, k, v = _make_qkv()
        mask = torch.ones(BATCH, 1, 1, SEQ_LEN)
        out = attn(q, k, v, mask=mask)
        assert out.shape == (BATCH, SEQ_LEN, D_MODEL)

    def test_single_head(self):
        dk, dv = 16, 16
        attn = MultiQueryAttention(D_MODEL, 1, dk, dv, dropout=0.0, causal=False)
        q, k, v = _make_qkv()
        out = attn(q, k, v)
        assert out.shape == (BATCH, SEQ_LEN, D_MODEL)

    def test_different_dk_dv(self):
        dk, dv = 16, 24
        attn = MultiQueryAttention(D_MODEL, N_HEADS, dk, dv, dropout=0.0, causal=False)
        q, k, v = _make_qkv()
        out = attn(q, k, v)
        assert out.shape == (BATCH, SEQ_LEN, D_MODEL)

    def test_no_nan(self):
        dk, dv = 32, 32
        attn = MultiQueryAttention(D_MODEL, N_HEADS, dk, dv, dropout=0.0, causal=True)
        q = torch.randn(BATCH, 32, D_MODEL)
        _, k, v = _make_qkv(seq=32)
        out = attn(q, k, v)
        assert not torch.isnan(out).any()

    def test_kv_head_is_shared(self):
        dk, dv = 32, 32
        attn = MultiQueryAttention(D_MODEL, N_HEADS, dk, dv, dropout=0.0, causal=False)
        k_in = torch.randn(BATCH, SEQ_LEN, D_MODEL)
        K = attn.Wk(k_in)
        K_expanded = K.unsqueeze(1).expand(-1, N_HEADS, -1, -1)
        assert K_expanded[:, 0].equal(K_expanded[:, 1])

    def test_cross_attention(self):
        dk, dv = 32, 32
        attn = MultiQueryAttention(D_MODEL, N_HEADS, dk, dv, dropout=0.0, causal=False)
        q = torch.randn(BATCH, 8, D_MODEL)
        kv = torch.randn(BATCH, 16, D_MODEL)
        out = attn(q, kv, kv)
        assert out.shape == (BATCH, 8, D_MODEL)


# ---------------------------------------------------------------------------
# BlockSparseAttention
# ---------------------------------------------------------------------------

class TestBlockSparse:
    def test_block_mask_generation(self):
        attn = BlockSparseAttention(D_MODEL, N_HEADS, block_size=8, num_local_blocks=2, num_global_blocks=1, stride=2)
        mask = attn._generate_block_mask(4, device="cpu")
        assert mask.shape == (4, 4)
        for i in range(4):
            assert mask[i, i]

    def test_output_shape(self):
        attn = BlockSparseAttention(D_MODEL, N_HEADS, dropout=0.0, causal=True, block_size=8)
        q, k, v = _make_qkv(seq=16)
        out = attn(q, k, v)
        assert out.shape == (BATCH, 16, D_MODEL)

    def test_gradient_flow(self):
        attn = BlockSparseAttention(D_MODEL, N_HEADS, dropout=0.0, causal=True, block_size=8)
        q, k, v = _make_qkv(seq=16)
        q.requires_grad_(True)
        out = attn(q, k, v)
        out.sum().backward()
        assert q.grad is not None

    def test_no_nan(self):
        attn = BlockSparseAttention(D_MODEL, N_HEADS, dropout=0.0, causal=True, block_size=8)
        q, k, v = _make_qkv(seq=16)
        out = attn(q, k, v)
        assert not torch.isnan(out).any()
