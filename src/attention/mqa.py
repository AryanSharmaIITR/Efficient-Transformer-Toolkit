import torch
from torch import nn
from torch.nn import functional as F

class MultiQueryAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dk: int, dv: int, dropout: float = 0.0, bias: bool = False, causal: bool = False):
        super().__init__()

        self.dk = dk
        self.dv = dv
        self.n_heads = n_heads
        self.causal = causal
        self.scale = torch.sqrt(torch.tensor(dk, dtype=torch.float32))
        self.dropout = nn.Dropout(dropout)

        self.Wk = nn.Linear(d_model, self.dk , bias=bias)
        self.Wq = nn.Linear(d_model, self.dk * self.n_heads, bias=bias)

        self.Wv = nn.Linear(d_model, self.dv , bias=bias)
        self.Wo = nn.Linear(self.dv * self.n_heads, d_model, bias=bias)

    def forward(self, query, key, value, mask) -> torch.Tensor:

        Q = self.Wq(query)  # (B, L, n_heads * dk)
        K = self.Wk(key)    # (B, L, dk)
        V = self.Wv(value)  # (B, L, dv)

        Q =  Q.view(Q.size(0), Q.size(1), self.n_heads, self.dk).transpose(1, 2)  # (B, n_heads, L, dk)
        K =  K.unsqueeze(1)  # (B, 1, L, dk)
        V =  V.unsqueeze(1)  # (B, 1, L, dv)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale  # (B, n_heads, L, L)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        if self.casual:
            causal_mask = torch.tril(torch.ones(scores.size(-2), scores.size(-1), device=scores.device)).unsqueeze(0).unsqueeze(0)
            scores = scores.masked_fill(causal_mask == 0, float('-inf'))

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        output = torch.matmul(attn_weights, V)  # (B, n_heads, L, dv)
        output = output.transpose(1, 2).contiguous().view(output.size(0), output.size(2), self.n_heads * self.dv)  # (B, L, n_heads * dv)
        output = self.Wo(output)  # (B, L, d_model) 

        return output

        


