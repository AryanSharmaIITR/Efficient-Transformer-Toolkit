from typing import Optional
from .base import BaseAttention
import torch
from torch import nn


class FlashAttentionV1(BaseAttention):
    def __init__(self, d_model, num_heads, block_size, bias = False, dropout=0.0, casual=False):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.block_size = block_size
        self.casual = casual
        self.bias = bias


        self.Wq = nn.Linear(d_model, d_model, bias=bias)
        self.Wk = nn.Linear(d_model, d_model, bias=bias)
        self.Wv = nn.Linear(d_model, d_model, bias=bias)
        self.Wo = nn.Linear(d_model, d_model, bias=bias)

        self.scale = self.head_dim ** -0.5
        self.dropout = dropout

    def forward(self, query, key, value, mask= None,):
            batch_size, seq_len, _ = query.shape
            
            q = self._compute_and_reshape(query, self.W_q)
            k = self._compute_and_reshape(key, self.W_k)
            v = self._compute_and_reshape(value, self.W_v)
            
            # Apply Flash Attention
            output = self._flash_attention_forward(q, k, v, mask)
            
            # Reshape back and project
            output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
            output = self.W_o(output)
            
            return output

    def _compute_and_reshape(self, x, linear_layer):
        x = linear_layer(x)
        x = x.view(x.size(0), x.size(1), self.num_heads, self.head_dim)
        x = x.transpose(1, 2)  # (batch_size, num_heads, seq_len, head_dim)
        return x

    def _flash_attention(self, q, k, v, mask=None):
        
        batch_size, _, q_seq_len, _ = q.size()
        kv_seq_len = k.size(2)
        output = torch.zeros_like(q, dtype=torch.float32)
        
        l = torch.zeros((batch_size, self.num_heads, q_seq_len), device=q.device, dtype=torch.float32)        
        m = torch.full((batch_size, self.num_heads, q_seq_len), -float('inf'), device=q.device, dtype=torch.float32)
        
        # Block sizes
        Br = min(self.block_size, q_seq_len)
        Bc = min(self.block_size, kv_seq_len)

        for q_start in range(0, q_seq_len, Br):
            q_end = min(q_start + Br, q_seq_len)
            q_block = q[:, :, q_start:q_end, :]  # [B, H, Br, D]

            l_block = torch.zeros((batch_size, self.num_heads, q_end - q_start), device=q.device, dtype=torch.float32)
            m_block = torch.full((batch_size, self.num_heads, q_end - q_start), -float('inf'), device=q.device, dtype=torch.float32)

            output_block = torch.zeros((batch_size, self.num_heads, q_end - q_start, self.head_dim), device=q.device, dtype=torch.float32)

            for kv_start in range(0, kv_seq_len, Bc):
                kv_end = min(kv_start + Bc, kv_seq_len)
                k_block = k[:, :, kv_start:kv_end, :]  # [B, H, Bc, D]
                v_block = v[:, :, kv_start:kv_end, :]  # [B, H, Bc, D]

                scores = torch.matmul(q_block, k_block.transpose(-2, -1)) * self.scale  # [B, H, Br, Bc]

                scores = self._apply_masks(scores, mask, q_start, q_end, kv_start, kv_end)

                m_new = torch.max(m_block, scores.max(dim=-1).values)

                exp_scores = torch.exp(scores - m_new.unsqueeze(-1))

                l_new = torch.exp(m_block - m_new) * l_block + torch.exp(scores- m_new).sum(dim=-1)

                output_block = self._update_output(output_block, exp_scores, v_block, l_block, l_new, m_block, m_new)

                m_block = m_new
                l_block = l_new

            output[:, :, q_start:q_end, :] = output_block
            l[:, :, q_start:q_end] = l_block
            m[:, :, q_start:q_end] = m_block

        return output

    def _apply_masks(self, scores, mask, q_start, q_end, k_start, k_end):
            if self.causal:
                causal_mask = torch.triu(
                    torch.ones((q_end - q_start, k_end - k_start), device=scores.device),
                    diagonal=k_start - q_start + 1
                ).bool()
                scores = scores.masked_fill(causal_mask, -float('inf'))
            
            # External mask
            if mask is not None:
                mask_block = mask[:, :, q_start:q_end, k_start:k_end]
                scores = scores.masked_fill(mask_block == 0, -float('inf'))
            
            return scores

    def _update_output(self, output_block, exp_scores, v_block, l_block, l_new, m_block, m_new):
            scale = torch.exp(m_block - m_new).unsqueeze(-1)
            
            # Update output with weighted average
            output_block = output_block * (l_block / l_new).unsqueeze(-1)
            output_block = output_block + torch.matmul(exp_scores, v_block) * scale
            
            return output_block



class FlashAttentionV2(FlashAttentionV1):
    def __init__(self, d_model, num_heads, block_size, bias=False, dropout=0.0, casual=False, use_fp32_accum: bool = True,):
        super().__init__(d_model, num_heads, block_size, bias, dropout, casual)
        self.use_fp32_accum = use_fp32_accum

    def _flash_attention(self, q, k, v, mask=None):
        batch_size, n_heads, q_seq_len, head_dim = q.shape
        kv_seq_len = k.shape[2]
        
        dtype = torch.float32 if self.use_fp32_accum else q.dtype
        
        output = torch.zeros_like(q, dtype=dtype)

        l = torch.zeros((batch_size, n_heads, q_seq_len), device=q.device, dtype=dtype)
        m = torch.full((batch_size, n_heads, q_seq_len), -float('inf'), device=q.device, dtype=dtype)
        
        Br = self._compute_block_size(q_seq_len)
        Bc = self._compute_block_size(kv_seq_len)
        
        for q_start in range(0, q_seq_len, Br):
            q_end = min(q_start + Br, q_seq_len)
            q_block = q[:, :, q_start:q_end, :].to(dtype)
            
            l_block = torch.zeros((batch_size, n_heads, q_end - q_start), device=q.device, dtype=dtype)
            m_block = torch.full((batch_size, n_heads, q_end - q_start), -float('inf'), device=q.device, dtype=dtype)
            output_block = torch.zeros_like(q_block, dtype=dtype)
            
            for kv_start in range(0, kv_seq_len, Bc):
                kv_end = min(kv_start + Bc, kv_seq_len)
                k_block = k[:, :, kv_start:kv_end, :].to(dtype)
                v_block = v[:, :, kv_start:kv_end, :].to(dtype)

                scores = torch.matmul(q_block, k_block.transpose(-2, -1)) * self.scale
                
                scores = self._apply_masks(scores, mask, q_start, q_end, kv_start, kv_end)
                
                m_new = torch.maximum(m_block, scores.max(dim=-1).values)
                exp_scores = torch.exp(scores - m_new.unsqueeze(-1))
                l_new = torch.exp(m_block - m_new) * l_block + exp_scores.sum(dim=-1)
                
                output_block = self._update_output(
                    output_block, exp_scores, v_block, l_block, l_new, m_block, m_new
                )
                
                m_block = m_new
                l_block = l_new
            
            # Write back with proper dtype
            output[:, :, q_start:q_end, :] = output_block
            l[:, :, q_start:q_end] = l_block
            m[:, :, q_start:q_end] = m_block
        
        return output.to(q.dtype)
    
    def _compute_block_size(self, seq_len):
        if seq_len < 512:
            return 64
        elif seq_len < 2048:
            return 128
        else:
            return 256

class FlashAttentionWrapper(BaseAttention):
    def __init__(self, d_model, num_heads, block_size, bias=False, dropout=0.0, casual=False, use_v2: bool = True):
        super().__init__()
        if use_v2:
            self.flash_attention = FlashAttentionV2(d_model, num_heads, block_size, bias, dropout, casual, use_fp32_accum=True)
        else:
            self.flash_attention = FlashAttentionV1(d_model, num_heads, block_size, bias, dropout, casual)

    def forward(self, hidden_states, attention_mask=None):
        output = self.flash_attention(hidden_states, hidden_states, hidden_states, mask=attention_mask)
        return output