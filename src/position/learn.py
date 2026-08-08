import torch
from torch import nn

class postionEmbedding(nn.modules):
    def __init__(self, d_model:int, max_len: int = 5000):
        super().__init__()
        self.pos_embedding = nn.Embedding(max_len, d_model)
        nn.init.normal_(self.pos_embedding.weight, mean=0.0, std=0.02)

    def forward(self, x):
        seq_len = x.size(1)
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0)  # (1, seq_len)
        pos_embeds = self.pos_embedding(positions)  # (1, seq_len, d_model)
        return x + pos_embeds



