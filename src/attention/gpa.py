import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
from .base import AttentionBase


class GroupedQueryAttention(AttentionBase):
    def __init__(self, d_model: int, n_heads: int, n_kv_heads: int, dropout: float = 0.0, bias: bool = False, use_flash: bool = True, causal: bool = False,):
        super().__init__(d_model, n_heads, dropout, causal, bias)

        assert n_heads % n_kv_heads == 0, (
            f"n_heads ({n_heads}) must be divisible by n_kv_heads ({n_kv_heads})"
        )
        self.n_kv_heads = n_kv_heads
        self.n_groups = n_heads // n_kv_heads
        self.use_flash = use_flash

        # Override K and V projections to use n_kv_heads instead of n_heads
        self.W_k = nn.Linear(d_model, n_kv_heads * self.head_dim, bias=bias)
        self.W_v = nn.Linear(d_model, n_kv_heads * self.head_dim, bias=bias)

        # Optionally try to import FlashAttention
        self.flash_attn = None
        if use_flash:
            try:
                from flash import FlashAttentionV2
                self.flash_attn = FlashAttentionV2(d_model, n_heads, block_size=128, bias=bias, dropout=dropout, casual=causal)
            except ImportError:
                print("Warning: flash_attn not installed. Falling back to manual attention.")
                self.use_flash = False

    def _reshape_kv(self, x):
        B, T, _ = x.shape
        x = x.view(B, T, self.n_kv_heads, self.head_dim)
        return x.transpose(1, 2)  # [B, n_kv_heads, T, D]

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, T_q, _ = query.shape
        T_kv = key.shape[1]

        q = self.Wq(query)
        q = self._reshape(q)  # [B, n_heads, T_q, D]

        k = self.Wk(key)
        v = self.Wv(value)
        k = self._reshape_kv(k)  # [B, n_kv_heads, T_kv, D]
        v = self._reshape_kv(v)  # [B, n_kv_heads, T_kv, D]

        k = k.repeat_interleave(self.n_groups, dim=1)  # [B, n_heads, T_kv, D]
        v = v.repeat_interleave(self.n_groups, dim=1)  # [B, n_heads, T_kv, D]

        if self.use_flash and self.flash_attn is not None:
            q_flash = q.transpose(1, 2)   # [B, T_q, n_heads, D]
            k_flash = k.transpose(1, 2)   # [B, T_kv, n_heads, D]
            v_flash = v.transpose(1, 2)   # [B, T_kv, n_heads, D]
            out = self.flash_attn(q_flash, k_flash, v_flash)
            out = out.transpose(1, 2).contiguous()  # [B, n_heads, T_q, D]
        else:
            # Manual attention (supports mask and dropout)
            # Compute scores: [B, n_heads, T_q, T_kv]
            scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale

            # Apply mask
            if mask is not None:
                if mask.dim() == 3:
                    mask = mask.unsqueeze(1)  # [B, 1, T_q, T_kv]
                scores = scores.masked_fill(mask == 0, float('-inf'))

            if self.causal:
                # Create causal mask (upper triangular)
                causal_mask = torch.triu(
                    torch.ones(T_q, T_kv, device=scores.device),
                    diagonal=T_kv - T_q + 1
                ).bool()
                scores = scores.masked_fill(causal_mask, float('-inf'))

            attn = F.softmax(scores, dim=-1)
            attn = self.dropout(attn)

            out = torch.matmul(attn, v)  # [B, n_heads, T_q, D]

        out = out.transpose(1, 2).contiguous().view(B, T_q, self.d_model)
        out = self.Wo(out)
        return out

    def get_kv_cache(self, key: torch.Tensor, value: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        k = self.Wk(key)
        v = self.Wv(value)
        k = self._reshape_kv(k)
        v = self._reshape_kv(v)
        return k, v

    @classmethod
    def from_multi_head(cls, mha_layer: nn.Module, n_kv_heads: int) -> "GroupedQueryAttention":
        d_model = mha_layer.d_model
        n_heads = mha_layer.n_heads
        head_dim = d_model // n_heads

        # Create new GQA layer
        gqa = cls(d_model, n_heads, n_kv_heads, dropout=0.0)

        # Copy query projection (unchanged)
        gqa.Wq.weight.data.copy_(mha_layer.Wq.weight.data)

        # For K and V, we need to average or slice the original heads.
        # Here we average each group of original heads into one KV head.
        with torch.no_grad():
            # Reshape original K/V weights to [n_heads, head_dim, d_model]
            k_weight = mha_layer.Wk.weight.view(n_heads, head_dim, d_model)
            v_weight = mha_layer.Wv.weight.view(n_heads, head_dim, d_model)

            # Group heads and average
            group_size = n_heads // n_kv_heads
            k_grouped = k_weight.view(n_kv_heads, group_size, head_dim, d_model).mean(dim=1)
            v_grouped = v_weight.view(n_kv_heads, group_size, head_dim, d_model).mean(dim=1)

            # Assign to new projections
            gqa.Wk.weight.data.copy_(k_grouped.view(n_kv_heads * head_dim, d_model))
            gqa.Wv.weight.data.copy_(v_grouped.view(n_kv_heads * head_dim, d_model))

            # Copy output projection
            gqa.Wo.weight.data.copy_(mha_layer.Wo.weight.data)

        return gqa

    def extra_repr(self) -> str:
        return (
            f"d_model={self.d_model}, n_heads={self.n_heads}, "
            f"n_kv_heads={self.n_kv_heads}, groups={self.n_groups}"
        )