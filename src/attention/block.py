import torch
import torch.nn.functional as F

from .base import BaseAttention


class BlockSparseAttention(BaseAttention):
    """Block-sparse self-attention combining local, global, and strided
    attention patterns (in the spirit of Sparse Transformer / Longformer /
    BigBird). Each query block attends only to a small set of key blocks
    instead of the full sequence.
    """

    def __init__(self, d_model, num_heads, dropout=0.1, causal=True, bias=False, block_size=64, num_local_blocks=2, num_global_blocks=1, stride=2):
        super().__init__(d_model, num_heads, dropout, causal, bias)
        self.block_size = block_size
        self.num_local_blocks = num_local_blocks
        self.num_global_blocks = num_global_blocks
        self.stride = stride
        self._block_mask_cache: dict[tuple[int, torch.device], torch.Tensor] = {}
        self.register_buffer("_block_causal_bias", None, persistent=False)

    def _generate_block_mask(self, num_blocks, device):
        """Which key blocks each query block is allowed to attend to.
        Cached per (num_blocks, device) since it depends only on those plus
        fixed hyperparameters, not on the actual batch data."""
        cached = self._block_mask_cache.get((num_blocks, device))
        if cached is not None:
            return cached

        mask = torch.zeros((num_blocks, num_blocks), device=device, dtype=torch.bool)

        for q_idx in range(num_blocks):
            # Clamp global blocks to <= q_idx under causal masking, otherwise
            # a global block ahead of the query would leak future tokens.
            n_global = min(self.num_global_blocks, q_idx + 1) if self.causal else self.num_global_blocks
            mask[q_idx, :n_global] = True

            start_local = max(0, q_idx - self.num_local_blocks)
            mask[q_idx, start_local:q_idx + 1] = True

            for k_idx in range(0, q_idx, self.stride):
                mask[q_idx, k_idx] = True

        self._block_mask_cache[(num_blocks, device)] = mask
        return mask

    def _get_block_causal_bias(self, block_sz, device):
        """Additive -inf bias for the diagonal (q_blk == k_blk) block."""
        cache = self._block_causal_bias
        if cache is None or cache.size(-1) != block_sz or cache.device != device:
            cache = torch.triu(
                torch.full((block_sz, block_sz), float("-inf"), device=device),
                diagonal=1,
            )
            self._block_causal_bias = cache
        return cache

    def forward(self, query, key, value, mask=None, positions=None):
        b, n, c = query.shape
        block_sz = self.block_size

        pad_len = (block_sz - (n % block_sz)) % block_sz
        if pad_len > 0:
            query = F.pad(query, (0, 0, 0, pad_len))
            key = F.pad(key, (0, 0, 0, pad_len))
            value = F.pad(value, (0, 0, 0, pad_len))
            n_padded = n + pad_len
        else:
            n_padded = n

        num_blocks = n_padded // block_sz

        q = self.Wq(query).view(b, n_padded, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.Wk(key).view(b, n_padded, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.Wv(value).view(b, n_padded, self.n_heads, self.head_dim).transpose(1, 2)

        q = q.view(b, self.n_heads, num_blocks, block_sz, self.head_dim)
        k = k.view(b, self.n_heads, num_blocks, block_sz, self.head_dim)
        v = v.view(b, self.n_heads, num_blocks, block_sz, self.head_dim)

        padding_mask = self._normalize_mask(mask)
        mask_blocks = None
        if padding_mask is not None:
            if pad_len > 0:
                padding_mask = F.pad(padding_mask, (0, pad_len), value=0)
            mask_blocks = padding_mask.view(padding_mask.size(0), 1, 1, num_blocks, block_sz)

        block_mask = self._generate_block_mask(num_blocks, query.device)
        block_causal_bias = self._get_block_causal_bias(block_sz, query.device) if self.causal else None

        out_blocks = torch.zeros_like(q)

        for q_blk in range(num_blocks):
            valid_k_blks = torch.where(block_mask[q_blk])[0]

            if len(valid_k_blks) == 0:
                continue

            q_slice = q[:, :, q_blk]
            k_slice = k[:, :, valid_k_blks].reshape(b, self.n_heads, -1, self.head_dim)
            v_slice = v[:, :, valid_k_blks].reshape(b, self.n_heads, -1, self.head_dim)

            scores = torch.matmul(q_slice, k_slice.transpose(-1, -2)) / self.scale

            if mask_blocks is not None:
                m_slice = mask_blocks[:, :, :, valid_k_blks].reshape(mask_blocks.size(0), 1, 1, -1)
                scores = scores.masked_fill(m_slice == 0, float("-inf"))

            if self.causal:
                # block_mask guarantees valid_k_blks never exceeds q_blk, so
                # only the diagonal block needs a partial (triangular) mask.
                diag_idx = (valid_k_blks == q_blk).nonzero(as_tuple=True)[0].item()
                col = slice(diag_idx * block_sz, (diag_idx + 1) * block_sz)
                scores[:, :, :, col] = scores[:, :, :, col] + block_causal_bias

            attn_probs = torch.softmax(scores, dim=-1)
            attn_probs = self.dropout(attn_probs)

            out_blocks[:, :, q_blk] = torch.matmul(attn_probs, v_slice)

        out = out_blocks.view(b, self.n_heads, n_padded, self.head_dim).transpose(1, 2)
        out = out.reshape(b, n_padded, c)

        if pad_len > 0:
            out = out[:, :n, :]

        return self.Wo(out)
