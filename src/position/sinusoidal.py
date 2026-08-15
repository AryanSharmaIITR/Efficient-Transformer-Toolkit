import torch
import math

def SinusoidalPositionalEncoding(d_model: int, seq_len: int, batch_first=True):
    pe = torch.zeros(seq_len, d_model)
    position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1) 


    div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)

    if batch_first:
        pe = pe.unsqueeze(0)  # (1, seq_len, d_model)
    else:
        pe = pe.unsqueeze(1)  # (seq_len, 1, d_model)

    return pe
