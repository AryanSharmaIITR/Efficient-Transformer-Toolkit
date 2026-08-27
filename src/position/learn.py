import torch
from torch import nn


class LearnedPositionalEmbedding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        self.pos_embedding = nn.Embedding(max_len, d_model)
        nn.init.normal_(self.pos_embedding.weight, mean=0.0, std=0.02)

    def forward(self, x, positions: torch.Tensor | None = None):
        if positions is None:
            seq_len = x.size(1)
            positions = torch.arange(seq_len, device=x.device).unsqueeze(0)
        else:
            # Explicit absolute positions -- e.g. a single new token during
            # KV-cached incremental decoding, which isn't at index 0 of the
            # full sequence.
            positions = positions.to(device=x.device, dtype=torch.long).unsqueeze(0)
        pos_embeds = self.pos_embedding(positions)
        return x + pos_embeds
