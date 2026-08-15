import torch
import torch.nn as nn
from .base import BaseAttention
import math

class BlockSparseAttention(BaseAttention):
    def __init__(self, d_model, num_heads, block_size=64, num_local_blocks=2, num_global_blocks=1, stride=2):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.block_size = block_size
        
        self.num_local_blocks = num_local_blocks
        self.num_global_blocks = num_global_blocks
        self.stride = stride

        self.Wq = nn.Linear(self.d_model, self.num_heads*self.d_model, bias=False)
        self.Wk = nn.Linear(self.d_model, self.num_heads*self.d_model, bias=False)
        self.Wv = nn.Linear(self.d_model, self.num_heads*self.d_model, bias=False)
        self.Wo = nn.Linear(self.d_model, self.num_heads*self.d_model, bias=False)

        # Scale Factor
        self.scale = 1.0 / math.sqrt(self.head_dim)

    def _generate_block_mask(self, num_blocks, device):
        mask = torch.zeros((num_blocks, num_blocks), device=device, dtype=torch.bool)
        
        for q_idx in range(num_blocks):
            mask[q_idx, :self.num_global_blocks] = True
            
            start_local = max(0, q_idx - self.num_local_blocks)
            mask[q_idx, start_local:q_idx + 1] = True

            for k_idx in range(0, q_idx, self.stride):
                mask[q_idx, k_idx] = True
                
        return mask

    def forward(self, x):
        b, n, c = x.shape
        block_sz = self.block_size
        
        pad_len = (block_sz - (n % block_sz)) % block_sz
        if pad_len > 0:
            x = torch.nn.functional.pad(x, (0, 0, 0, pad_len))
            n_padded = n + pad_len
        else:
            n_padded = n

        num_blocks = n_padded // block_sz

        q = self.Wq(x).view(b, n_padded, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.Wk(x).view(b, n_padded, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.Wv(x).view(b, n_padded, self.num_heads, self.head_dim).transpose(1, 2)

        q = q.view(b, self.num_heads, num_blocks, block_sz, self.head_dim)
        k = k.view(b, self.num_heads, num_blocks, block_sz, self.head_dim)
        v = v.view(b, self.num_heads, num_blocks, block_sz, self.head_dim)
        
        block_mask = self._generate_block_mask(num_blocks, x.device)
        
        atten_weight = torch.zeros_like(q)

        for q_blk in range(num_blocks):
            valid_k_blks = torch.where(block_mask[q_blk])[0]
            
            if len(valid_k_blks) == 0:
                continue
                
            q_slice = q[:, :, q_blk]        # Shape:[B, Heads, Block_Sz, Head_Dim]
            k_slice = k[:, :, valid_k_blks] # Shape:[B, Heads, Valid_K, Block_Sz, Head_Dim]
            v_slice = v[:, :, valid_k_blks] # Shape:[B, Heads, Valid_K, Block_Sz, Head_Dim]
            
            k_slice = k_slice.permute(0, 1, 2, 4, 3).reshape(b, self.num_heads, -1, self.head_dim)
            v_slice = v_slice.permute(0, 1, 2, 3, 4).reshape(b, self.num_heads, -1, self.head_dim)
            
            scores = torch.matmul(q_slice, k_slice.transpose(-1, -2)) * self.scale
            
            for idx, k_blk in enumerate(valid_k_blks):
                if q_blk == k_blk:
                    causal_mask = torch.triu(torch.full((block_sz, block_sz), float('-inf'), device=x.device), diagonal=1)
                    scores[:, :, :, idx*block_sz : (idx+1)*block_sz] += causal_mask

            attn_probs = torch.softmax(scores, dim=-1)
            
            atten_weight[:, :, q_blk] = attn_probs

        out = atten_weight.view(b, self.num_heads, n_padded, self.head_dim).transpose(1, 2)
        out = out.reshape(b, n_padded, c)

        if pad_len > 0:
            out = out[:, :n, :]
            
        return self.Wo(out)