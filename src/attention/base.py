import math
from abc import ABC, abstractmethod

import torch
from torch import nn


class BaseAttention(nn.Module, ABC):
    def __init__(self, d_model, n_heads, dropout=0.1, causal=False, bias=False):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.scale = math.sqrt(self.head_dim)
        self.causal = causal

        self.Wq = nn.Linear(d_model, d_model, bias=bias)
        self.Wk = nn.Linear(d_model, d_model, bias=bias)
        self.Wv = nn.Linear(d_model, d_model, bias=bias)
        self.Wo = nn.Linear(d_model, d_model, bias=bias)

        self.dropout = nn.Dropout(dropout)

        # Lazily-grown cache for the self-attention (q_len == k_len) causal
        # mask, so subclasses don't re-allocate + re-triu an identical
        # boolean matrix on every single forward call. Not persistent: it's
        # cheaply derivable, not model state.
        self.register_buffer("_causal_mask_cache", None, persistent=False)

    @abstractmethod
    def forward(self, query, key, value, mask=None, positions=None):
        """``positions`` is accepted uniformly across all attention
        variants so callers (the encoder/decoder layers) can pass it
        unconditionally; only rotary-position-aware attention actually
        uses it, everyone else ignores it."""

    def _reshape(self, x):
        batch_size = x.size(0)
        seq_len = x.size(1)
        x = x.view(batch_size, seq_len, self.n_heads, self.head_dim)
        x = x.transpose(1, 2)
        return x

    @staticmethod
    def _normalize_mask(mask: torch.Tensor | None) -> torch.Tensor | None:
        """Reshape an optional padding mask to a 4-D tensor broadcastable
        against ``[batch, heads, q_len, k_len]`` attention scores.

        Accepts ``[batch, k_len]`` (2-D), ``[batch, q_len, k_len]`` (3-D,
        head-broadcast), or an already 4-D mask (passed through).
        """
        if mask is None:
            return None
        if mask.dim() == 2:
            return mask[:, None, None, :]
        if mask.dim() == 3:
            return mask.unsqueeze(1)
        return mask

    def _get_causal_mask(self, q_len: int, k_len: int, device: torch.device) -> torch.Tensor:
        """Boolean mask, ``True`` where a position must be blocked (i.e. the
        key is in the query's future). Pass directly to ``masked_fill``.

        The common self-attention case (``q_len == k_len``) is cached in a
        buffer that only grows, mirroring how ``Alibi`` caches its bias
        tensor, so repeated forward calls at the same or smaller sequence
        length reuse the same allocation instead of rebuilding it.
        """
        if q_len == k_len:
            cache = self._causal_mask_cache
            if cache is None or cache.size(-1) < q_len or cache.device != device:
                cache = torch.triu(
                    torch.ones(q_len, q_len, device=device, dtype=torch.bool),
                    diagonal=1,
                )
                self._causal_mask_cache = cache
            return cache[:q_len, :q_len]

        # Cross-length case (e.g. incremental decoding against a KV cache):
        # align the query block with the *last* q_len positions of the key
        # sequence, i.e. key position k is masked when k > (k_len - q_len + row).
        return torch.triu(
            torch.ones(q_len, k_len, device=device, dtype=torch.bool),
            diagonal=k_len - q_len + 1,
        )
