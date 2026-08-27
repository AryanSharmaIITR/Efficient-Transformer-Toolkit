import math

import torch
import torch.nn.functional as F

from .base import BaseAttention


def get_slope(n_heads: int) -> torch.Tensor:
    """Compute ALiBi head slopes (Press et al., 2021).

    Matches the paper's reference implementation exactly, including the
    interleaved fallback for head counts that aren't a power of 2: the
    extra slopes are every *other* value of the geometric sequence for the
    next power of 2 up, not a fresh sequence starting back at index 1.
    """
    def _slopes_power_of_2(n: int) -> list[float]:
        start = 2.0 ** (-8.0 / n)
        return [start ** i for i in range(1, n + 1)]

    if math.log2(n_heads).is_integer():
        slopes = _slopes_power_of_2(n_heads)
    else:
        closest_pow2 = 2 ** math.floor(math.log2(n_heads))
        slopes = _slopes_power_of_2(closest_pow2)
        extra = get_slope(2 * closest_pow2).tolist()[0::2][: n_heads - closest_pow2]
        slopes = slopes + extra
    return torch.tensor(slopes, dtype=torch.float32)


@torch.no_grad()
def build_alibi_tensor(n_heads: int, seq_len: int, device: torch.device, dtype: torch.dtype):
    slopes = get_slope(n_heads).to(device=device, dtype=dtype)
    pos_i = torch.arange(seq_len, device=device, dtype=dtype).unsqueeze(1)
    pos_j = torch.arange(seq_len, device=device, dtype=dtype).unsqueeze(0)

    distance = torch.abs(pos_i - pos_j)
    bias = -slopes[:, None, None] * distance[None, :, :]
    return bias.unsqueeze(0)


class Alibi(BaseAttention):
    def __init__(self, d_model: int, n_heads: int, dropout=0.1, bias=False, causal=True, max_seq_len=512):
        super().__init__(d_model, n_heads, dropout, causal, bias)
        self.max_seq_len = max_seq_len
        self.register_buffer("alibi_bias", None)

    def forward(self, query, key, value, mask=None, positions=None):
        batch_size, seq_len, _ = query.shape
        kv_seq_len = key.shape[1]

        Q = self.Wq(query)
        K = self.Wk(key)
        V = self.Wv(value)

        Q = self._reshape(Q)
        K = self._reshape(K)
        V = self._reshape(V)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale

        max_len = max(seq_len, kv_seq_len)
        cache = self.alibi_bias
        if (
            cache is None
            or cache.size(-1) < max_len
            or cache.device != query.device
            or cache.dtype != query.dtype
        ):
            self.alibi_bias = build_alibi_tensor(
                self.n_heads, max_len, query.device, query.dtype
            )

        alibi = self.alibi_bias[:, :, :seq_len, :kv_seq_len]
        scores = scores + alibi

        mask = self._normalize_mask(mask)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))

        if self.causal:
            causal_mask = self._get_causal_mask(seq_len, kv_seq_len, scores.device)
            scores = scores.masked_fill(causal_mask, float("-inf"))

        attn_weight = F.softmax(scores, dim=-1)
        attn_weight = self.dropout(attn_weight)

        output = torch.matmul(attn_weight, V)

        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)

        return self.Wo(output)
