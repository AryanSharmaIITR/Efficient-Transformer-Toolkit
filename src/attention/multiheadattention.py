from torch.nn import nn
import torch
from .base import BaseAttention

class MultiHeadAttention(BaseAttention):
    def __init__(self, d_model, n_heads, dropout=0.1, casual=False):
        super().__init__(d_model, n_heads, dropout, casual)

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)

        Q = self.Wq(query)
        K = self.Wk(key)
        V = self.Wv(value)

        Q = self._reshape(Q)
        K = self._reshape(K)
        V = self._reshape(V)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        if self.casual:
            seq_len = scores.size(-1)
            casual_mask = torch.tril(torch.ones(seq_len, seq_len)).to(scores.device)
            scores = scores.masked_fill(casual_mask == 0, float('-inf'))

        attn_weights = torch.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        output = torch.matmul(attn_weights, V)

        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)

        return self.Wo(output)