import math
import torch
from torch import nn
from .multiheadattention import MultiHeadAttention

def get_slope(n_heads):
    """Compute ALiBi slopes for each head."""
    n = 2 ** math.floor(math.log2(n_heads))
    m_0 = 2.0 ** (-8.0 / n)
    m = torch.pow(m_0, torch.arange(1, 1 + n))
    if n < n_heads:
        m_start = 2.0 ** (-4.0 / n)
        m_end = 2.0 ** (-8.0 / n_heads)
        m_rest = torch.pow(m_start, torch.arange(1, 1 + (n_heads - n)))
        m = torch.cat([m, m_rest])
    return m

@torch.no_grad()
def build_alibi_tensor(n_heads: int, seq_len: int, device: torch.device, dtype: torch.dtype):
    slopes = get_slope(n_heads).to(device=device, dtype=dtype)
    pos_i = torch.arange(seq_len, device=device, dtype=dtype).unsqueeze(1)
    pos_j = torch.arange(seq_len, device=device, dtype=dtype).unsqueeze(0)

    distance = torch.abs(pos_i - pos_j)
    bias = -slopes[:, None, None] * distance[None, :, :]
    return bias.unsqueeze(0)

class Alibi(MultiHeadAttention):
    def __init__(self, d_model: int, n_heads: int, dropout=0.1, bias: bool = False, causal: bool = True, max_seq_len: int = 512):
        super().__init__(d_model, n_heads, dropout, causal, bias)
        self.causal = causal
        self.max_seq_len = max_seq_len
        self.register_buffer('alibi_bias', None)  # will be (1, n_heads, L, L)

    def forward(self, query, key, value, mask=None):
        batch_size, seq_len, _ = query.shape

        # Linear projections
        Q = self.W_Q(query)
        K = self.W_K(key)
        V = self.W_V(value)

        # Reshape to (batch, n_heads, seq_len, head_dim)
        Q = self._reshape(Q)
        K = self._reshape(K)
        V = self._reshape(V)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale  # (batch, n_heads, seq_len, seq_len)

        # Build or update the bias cache if sequence length changed
        if self.alibi_bias is None or self.alibi_bias.size(-1) < seq_len:
            self.alibi_bias = build_alibi_tensor(
                self.n_heads, seq_len, query.device, query.dtype
            )

        alibi = self.alibi_bias[:, :, :seq_len, :seq_len]
        scores = scores + alibi

        if mask is not None:
            if mask.dim() == 2:  # (batch, seq_len)
                mask = mask.unsqueeze(1).unsqueeze(2)  # (batch, 1, 1, seq_len)
            # For padding, we want to mask positions where mask == 0
            scores = scores.masked_fill(mask == 0, float('-inf'))

        if self.causal:
            # Create a triangular mask for the current sequence length
            causal_mask = torch.tril(torch.ones(seq_len, seq_len, device=scores.device, dtype=torch.bool))
            # Expand to (1, 1, seq_len, seq_len) for broadcasting
            causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len, seq_len)
            scores = scores.masked_fill(~causal_mask, float('-inf'))

        # Attention weights and dropout
        attn_weight = self.softmax(scores)
        attn_weight = self.dropout(attn_weight)

        # Apply attention to values
        output = torch.matmul(attn_weight, V)  # (batch, n_heads, seq_len, head_dim)

        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)

        return self.Wo(output)