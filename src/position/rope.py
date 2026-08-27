import torch
from torch import nn

from ..attention.multiheadattention import MultiHeadAttention


class RotaryPositionalEmbedding(nn.Module):
    """Rotary position embedding (RoFormer, Su et al., 2021).

    Uses the GPT-NeoX/LLaMA "rotate_half" formulation -- pairing dimension
    i with i + d/2 -- rather than the original paper's interleaved
    (2i, 2i+1) pairing. The two are different (non-interchangeable
    parameterizations of) the same rotation and are equally standard;
    "rotate_half" is what most modern implementations use.

    Expects a *sequence-first* input ``[seq_len, batch, n_heads, dim]``,
    matching :class:`RotaryPEMultiHeadAttention`'s Q/K layout. Only the
    first ``d`` dimensions of the last axis are rotated; the rest pass
    through unchanged (partial-rotary, as in GPT-NeoX/GPT-J).
    """

    def __init__(self, d: int, max_seq_len: int = 2048, base: int = 10_000):
        super().__init__()
        if d % 2 != 0:
            raise ValueError(f"RotaryPositionalEmbedding requires an even rotary dim, got d={d}")
        self.base = base
        self.d = d
        self.max_seq_len = max_seq_len
        # Non-persistent: cheaply recomputed, not real model state. Buffers
        # (rather than plain attributes) so they move with .to(device) and
        # don't silently go stale on a device/dtype change.
        self.register_buffer("cos_cached", None, persistent=False)
        self.register_buffer("sin_cached", None, persistent=False)

    def _angles(self, positions: torch.Tensor) -> torch.Tensor:
        theta = 1.0 / (self.base ** (torch.arange(0, self.d, 2, device=positions.device).float() / self.d))
        idx_theta = torch.einsum("n,d->nd", positions.float(), theta)
        return torch.cat([idx_theta, idx_theta], dim=1)

    def _build_cache(self, seq_len: int, device: torch.device, dtype: torch.dtype) -> None:
        cache = self.cos_cached
        if (
            cache is not None
            and cache.shape[0] >= seq_len
            and cache.device == device
            and cache.dtype == dtype
        ):
            return

        angles = self._angles(torch.arange(seq_len, device=device))
        self.cos_cached = angles.cos()[:, None, None, :].to(dtype)
        self.sin_cached = angles.sin()[:, None, None, :].to(dtype)

    def _neg_half(self, x: torch.Tensor) -> torch.Tensor:
        d_2 = self.d // 2
        return torch.cat([-x[:, :, :, d_2:], x[:, :, :, :d_2]], dim=-1)

    def forward(self, x: torch.Tensor, positions: torch.Tensor | None = None) -> torch.Tensor:
        seq_len = x.shape[0]

        if positions is None:
            self._build_cache(seq_len, x.device, x.dtype)
            cos, sin = self.cos_cached[:seq_len], self.sin_cached[:seq_len]
        else:
            # Explicit absolute positions -- e.g. a KV-cached incremental
            # decoding step where the new token isn't at index 0..seq_len-1
            # -- computed directly rather than assumed from the cache.
            angles = self._angles(positions.to(x.device))
            cos = angles.cos()[:, None, None, :].to(x.dtype)
            sin = angles.sin()[:, None, None, :].to(x.dtype)

        x_rope, x_pass = x[..., :self.d], x[..., self.d:]
        neg_half_x = self._neg_half(x_rope)
        x_rope = (x_rope * cos) + (neg_half_x * sin)

        return torch.cat((x_rope, x_pass), dim=-1)


class RotaryPEMultiHeadAttention(MultiHeadAttention):
    """Multi-head attention with rotary position embeddings applied to
    query/key before the score matmul.

    Unlike the rest of this attention library (batch-first internally,
    ``[batch, heads, seq, head_dim]``), this class works in the
    sequence-first ``[seq, batch, heads, head_dim]`` layout that
    :class:`RotaryPositionalEmbedding` expects, matching the convention of
    the original RoPE reference implementations.
    """

    def __init__(self, d_model: int, n_heads: int, rope_percentage: float = 0.5, dropout_prob: float = 0.0, causal: bool = False, bias: bool = False):
        super().__init__(d_model, n_heads, dropout_prob, causal, bias)

        d_rope = int(self.head_dim * rope_percentage)
        self.query_rotary_pe = RotaryPositionalEmbedding(d_rope)
        self.key_rotary_pe = RotaryPositionalEmbedding(d_rope)

    def _reshape_seq_first(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        x = x.view(batch_size, seq_len, self.n_heads, self.head_dim)
        return x.permute(1, 0, 2, 3)  # (seq, batch, heads, head_dim)

    def get_scores(self, query: torch.Tensor, key: torch.Tensor, positions: torch.Tensor | None = None) -> torch.Tensor:
        """query/key: (seq, batch, heads, head_dim) -> (batch, heads, q_len, k_len)."""
        return torch.einsum(
            "ibhd,jbhd->bhij",
            self.query_rotary_pe(query, positions), self.key_rotary_pe(key, positions),
        )

    def forward(self, query, key, value, mask=None, positions=None):
        batch_size, seq_len, _ = query.shape

        q = self._reshape_seq_first(self.Wq(query))
        k = self._reshape_seq_first(self.Wk(key))
        v = self._reshape_seq_first(self.Wv(value))

        scores = self.get_scores(q, k, positions) / self.scale

        mask = self._normalize_mask(mask)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))
        if self.causal:
            causal_mask = self._get_causal_mask(scores.size(-2), scores.size(-1), scores.device)
            scores = scores.masked_fill(causal_mask, float("-inf"))

        attn_weights = torch.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        v_bhsd = v.permute(1, 2, 0, 3)  # (batch, heads, seq, head_dim)
        output = torch.matmul(attn_weights, v_bhsd)
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)

        return self.Wo(output)
