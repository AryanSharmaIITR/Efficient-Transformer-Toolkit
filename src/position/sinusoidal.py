import math

import torch
from torch import nn


class SinusoidalPositionalEncoding(nn.Module):
    """Fixed sinusoidal positional encoding (Vaswani et al., 2017)."""

    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        self.d_model = d_model
        self.register_buffer("pe", self._build(max_len), persistent=False)

    @torch.no_grad()
    def _build(self, max_len: int, device=None, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        pe = torch.zeros(max_len, self.d_model, device=device, dtype=dtype)
        position = torch.arange(0, max_len, dtype=dtype, device=device).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, self.d_model, 2, dtype=dtype, device=device) * (-math.log(10000.0) / self.d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)

    def forward(self, x: torch.Tensor, positions: torch.Tensor | None = None) -> torch.Tensor:
        seq_len = x.size(1)
        max_pos = int(positions.max().item()) + 1 if positions is not None else seq_len
        if max_pos > self.pe.size(1):
            # Genuinely extend the table instead of tiling the cached one --
            # the sinusoid is well-defined at any position, but repeating a
            # shorter table would alias position i with position i % max_len.
            self.pe = self._build(max_pos, device=x.device)

        if positions is not None:
            # Explicit absolute positions -- e.g. a single new token during
            # KV-cached incremental decoding, which isn't at index 0..seq_len-1
            # of the full sequence. Without this, every decode step would
            # apply position-0's encoding to whatever token it's given.
            pe = self.pe[:, positions.to(torch.long)].to(dtype=x.dtype)
        else:
            pe = self.pe[:, :seq_len].to(dtype=x.dtype)
        return x + pe
