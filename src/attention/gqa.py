
import torch
import torch.nn.functional as F
from torch import nn

from .base import BaseAttention


class GroupedQueryAttention(BaseAttention):
    def __init__(self, d_model: int, n_heads: int, n_kv_heads: int, dropout=0.0, bias=False, use_flash=True, causal=False):
        super().__init__(d_model, n_heads, dropout, causal, bias)

        assert n_heads % n_kv_heads == 0, (
            f"n_heads ({n_heads}) must be divisible by n_kv_heads ({n_kv_heads})"
        )
        self.n_kv_heads = n_kv_heads
        self.n_groups = n_heads // n_kv_heads
        self.use_flash = use_flash

        # BaseAttention.__init__ (above) unconditionally creates a full
        # d_model x d_model self.Wk/self.Wv, sized for n_heads KV heads --
        # GQA's whole point is fewer KV heads than query heads, so those
        # are replaced by the smaller W_k/W_v below and would otherwise
        # just sit there as dead, never-forward()'d parameters: still
        # initialized, still optimized, still counted in memory/param-count
        # (verified: a 12-head/4-kv-head GQA model reported *more* total
        # parameters than plain full MHA on the same dims, entirely from
        # this unused pair). Drop them so GQA's real parameter/memory
        # savings actually show up.
        del self.Wk
        del self.Wv

        self.W_k = nn.Linear(d_model, n_kv_heads * self.head_dim, bias=bias)
        self.W_v = nn.Linear(d_model, n_kv_heads * self.head_dim, bias=bias)

    def _reshape_kv(self, x):
        B, T, _ = x.shape
        x = x.view(B, T, self.n_kv_heads, self.head_dim)
        return x.transpose(1, 2)

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, mask: torch.Tensor | None = None, positions: torch.Tensor | None = None) -> torch.Tensor:
        B, T_q, _ = query.shape
        T_kv = key.shape[1]

        q = self.Wq(query)
        q = self._reshape(q)

        k = self.W_k(key)
        v = self.W_v(value)
        k = self._reshape_kv(k)  # (B, n_kv_heads, T_kv, head_dim)
        v = self._reshape_kv(v)

        # Broadcast each KV head across its query-head group instead of
        # materializing repeat_interleave'd copies of K/V: matmul already
        # broadcasts over the singleton "group" dim, so K/V memory stays
        # O(n_kv_heads) rather than O(n_heads), matching GQA's point of
        # sharing K/V across grouped query heads.
        q_grouped = q.view(B, self.n_kv_heads, self.n_groups, T_q, self.head_dim)
        scores = torch.matmul(q_grouped, k.unsqueeze(2).transpose(-2, -1)) / self.scale
        scores = scores.reshape(B, self.n_heads, T_q, T_kv)

        mask = self._normalize_mask(mask)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))
        if self.causal:
            causal_mask = self._get_causal_mask(T_q, T_kv, scores.device)
            scores = scores.masked_fill(causal_mask, float("-inf"))
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        attn_grouped = attn.view(B, self.n_kv_heads, self.n_groups, T_q, T_kv)
        out = torch.matmul(attn_grouped, v.unsqueeze(2))
        out = out.reshape(B, self.n_heads, T_q, self.head_dim)
        out = out.transpose(1, 2).contiguous().view(B, T_q, self.d_model)
        out = self.Wo(out)
        return out

    def get_kv_cache(self, key: torch.Tensor, value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        k = self.W_k(key)
        v = self.W_v(value)
        k = self._reshape_kv(k)
        v = self._reshape_kv(v)
        return k, v

    @classmethod
    def from_multi_head(cls, mha_layer: nn.Module, n_kv_heads: int) -> GroupedQueryAttention:
        d_model = mha_layer.d_model
        n_heads = mha_layer.n_heads
        head_dim = d_model // n_heads

        gqa = cls(d_model, n_heads, n_kv_heads, dropout=0.0)

        gqa.Wq.weight.data.copy_(mha_layer.Wq.weight.data)

        with torch.no_grad():
            k_weight = mha_layer.Wk.weight.view(n_heads, head_dim, d_model)
            v_weight = mha_layer.Wv.weight.view(n_heads, head_dim, d_model)

            group_size = n_heads // n_kv_heads
            k_grouped = k_weight.view(n_kv_heads, group_size, head_dim, d_model).mean(dim=1)
            v_grouped = v_weight.view(n_kv_heads, group_size, head_dim, d_model).mean(dim=1)

            gqa.W_k.weight.data.copy_(k_grouped.view(n_kv_heads * head_dim, d_model))
            gqa.W_v.weight.data.copy_(v_grouped.view(n_kv_heads * head_dim, d_model))

            gqa.Wo.weight.data.copy_(mha_layer.Wo.weight.data)

        return gqa

    def extra_repr(self) -> str:
        return (
            f"d_model={self.d_model}, n_heads={self.n_heads}, "
            f"n_kv_heads={self.n_kv_heads}, groups={self.n_groups}"
        )
