import math

import torch
from torch import nn
from torch.nn import functional as F

from .base import BaseAttention


class MultiQueryAttention(BaseAttention):
    def __init__(self, d_model: int, n_heads: int, dk: int, dv: int, dropout=0.0, bias=False, causal=False):
        super().__init__(d_model, n_heads, dropout, causal, bias)

        self.dk = dk
        self.dv = dv
        self.scale = math.sqrt(dk)

        self.Wk = nn.Linear(d_model, dk, bias=bias)
        self.Wq = nn.Linear(d_model, dk * n_heads, bias=bias)
        self.Wv = nn.Linear(d_model, dv, bias=bias)
        self.Wo = nn.Linear(dv * n_heads, d_model, bias=bias)

    def forward(self, query, key, value, mask=None, positions=None):
        Q = self.Wq(query)
        K = self.Wk(key)
        V = self.Wv(value)

        Q = Q.view(Q.size(0), Q.size(1), self.n_heads, self.dk).transpose(1, 2)
        K = K.unsqueeze(1)
        V = V.unsqueeze(1)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale

        mask = self._normalize_mask(mask)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))

        if self.causal:
            causal_mask = self._get_causal_mask(scores.size(-2), scores.size(-1), scores.device)
            scores = scores.masked_fill(causal_mask, float("-inf"))

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        output = torch.matmul(attn_weights, V)
        output = output.transpose(1, 2).contiguous().view(output.size(0), output.size(2), self.n_heads * self.dv)
        output = self.Wo(output)

        return output
