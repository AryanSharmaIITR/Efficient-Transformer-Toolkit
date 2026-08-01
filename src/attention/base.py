import numpy
from torch.nn import nn
from abc import ABC, abstractmethod

class BaseAttention(nn.Module, ABC):
    def __init__(self, d_model, n_heads, dropout=0.1, casual=False):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.scale = numpy.sqrt(self.head_dim)
        self.casual = casual

        self.Wq = nn.Linear(d_model, d_model, bias=False)
        self.Wk = nn.Linear(d_model, d_model, bias=False)
        self.Wv = nn.Linear(d_model, d_model, bias=False)
        self.Wo = nn.Linear(d_model, d_model, bias=False)

        self.dropout = nn.Dropout(dropout)

    @abstractmethod
    def forward(self, query, key, value, mask=None):
        pass    

    def _reshape(self, x):
        batch_size, seq_len = x.size()
        x = x.view(batch_size, seq_len, self.n_heads, self.head_dim)
        x = x.transpose(1, 2)
        return x