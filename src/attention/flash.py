import torch

from .base import BaseAttention


class FlashAttentionV1(BaseAttention):
    def __init__(self, d_model, num_heads, block_size=128, bias=False, dropout=0.0, causal=False):
        super().__init__(d_model, num_heads, dropout, causal, bias)
        self.block_size = block_size

    def forward(self, query, key, value, mask=None, positions=None):
        batch_size, seq_len, _ = query.shape

        q = self._compute_and_reshape(query, self.Wq)
        k = self._compute_and_reshape(key, self.Wk)
        v = self._compute_and_reshape(value, self.Wv)

        output = self._flash_attention_forward(q, k, v, self._normalize_mask(mask))

        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        output = self.Wo(output)

        return output

    def _compute_and_reshape(self, x, linear_layer):
        x = linear_layer(x)
        x = x.view(x.size(0), x.size(1), self.n_heads, self.head_dim)
        x = x.transpose(1, 2)
        return x

    def _flash_attention_forward(self, q, k, v, mask=None):
        batch_size, _, q_seq_len, _ = q.size()
        kv_seq_len = k.size(2)
        output = torch.zeros_like(q, dtype=torch.float32)

        l = torch.zeros((batch_size, self.n_heads, q_seq_len), device=q.device, dtype=torch.float32)
        m = torch.full((batch_size, self.n_heads, q_seq_len), -float("inf"), device=q.device, dtype=torch.float32)

        Br = min(self.block_size, q_seq_len)
        Bc = min(self.block_size, kv_seq_len)

        for q_start in range(0, q_seq_len, Br):
            q_end = min(q_start + Br, q_seq_len)
            q_block = q[:, :, q_start:q_end, :]

            l_block = torch.zeros((batch_size, self.n_heads, q_end - q_start), device=q.device, dtype=torch.float32)
            m_block = torch.full((batch_size, self.n_heads, q_end - q_start), -float("inf"), device=q.device, dtype=torch.float32)
            output_block = torch.zeros((batch_size, self.n_heads, q_end - q_start, self.head_dim), device=q.device, dtype=torch.float32)

            for kv_start in range(0, kv_seq_len, Bc):
                kv_end = min(kv_start + Bc, kv_seq_len)
                k_block = k[:, :, kv_start:kv_end, :]
                v_block = v[:, :, kv_start:kv_end, :]

                scores = torch.matmul(q_block, k_block.transpose(-2, -1)) / self.scale

                scores = self._apply_masks(scores, mask, q_start, q_end, kv_start, kv_end)

                m_new = torch.max(m_block, scores.max(dim=-1).values)
                exp_scores = torch.exp(scores - m_new.unsqueeze(-1))
                l_new = torch.exp(m_block - m_new) * l_block + exp_scores.sum(dim=-1)

                scale = torch.exp(m_block - m_new).unsqueeze(-1)
                # Online-softmax rescale: exp_scores is already exp(S - m_new),
                # so exp_scores @ V needs no further scaling -- only the old,
                # already-normalized output_block needs rescaling back up to
                # an l_block-weighted (un-normalized) sum before adding the
                # new block's contribution, then the whole thing is
                # renormalized by the updated running sum l_new.
                output_block = (output_block * (l_block.unsqueeze(-1) * scale) + torch.matmul(exp_scores, v_block)) / l_new.unsqueeze(-1)

                m_block = m_new
                l_block = l_new

            output[:, :, q_start:q_end, :] = output_block
            l[:, :, q_start:q_end] = l_block
            m[:, :, q_start:q_end] = m_block

        return output

    @staticmethod
    def _slice_broadcastable(t: torch.Tensor, dim: int, start: int, end: int) -> torch.Tensor:
        """Slice ``t`` along ``dim`` by ``[start:end]``, leaving broadcast
        dimensions (size 1) untouched so e.g. a ``[B, 1, 1, K]`` padding
        mask stays broadcastable against per-block query ranges."""
        if t.size(dim) == 1:
            return t
        return t.narrow(dim, start, end - start)

    def _apply_masks(self, scores, mask, q_start, q_end, k_start, k_end):
        if self.causal:
            causal_mask = torch.triu(
                torch.ones((q_end - q_start, k_end - k_start), device=scores.device),
                diagonal=q_start - k_start + 1,
            ).bool()
            scores = scores.masked_fill(causal_mask, -float("inf"))

        if mask is not None:
            mask_block = self._slice_broadcastable(mask, 2, q_start, q_end)
            mask_block = self._slice_broadcastable(mask_block, 3, k_start, k_end)
            scores = scores.masked_fill(mask_block == 0, -float("inf"))

        return scores


class FlashAttentionV2(FlashAttentionV1):
    def __init__(self, d_model, num_heads, block_size=128, bias=False, dropout=0.0, causal=False, use_fp32_accum=True):
        super().__init__(d_model, num_heads, block_size, bias, dropout, causal)
        self.use_fp32_accum = use_fp32_accum

    def _flash_attention_forward(self, q, k, v, mask=None):
        batch_size, n_heads, q_seq_len, _head_dim = q.shape
        kv_seq_len = k.shape[2]

        dtype = torch.float32 if self.use_fp32_accum else q.dtype

        output = torch.zeros_like(q, dtype=dtype)
        l = torch.zeros((batch_size, n_heads, q_seq_len), device=q.device, dtype=dtype)
        m = torch.full((batch_size, n_heads, q_seq_len), -float("inf"), device=q.device, dtype=dtype)

        Br = self._compute_block_size(q_seq_len)
        Bc = self._compute_block_size(kv_seq_len)

        for q_start in range(0, q_seq_len, Br):
            q_end = min(q_start + Br, q_seq_len)
            q_block = q[:, :, q_start:q_end, :].to(dtype)

            l_block = torch.zeros((batch_size, n_heads, q_end - q_start), device=q.device, dtype=dtype)
            m_block = torch.full((batch_size, n_heads, q_end - q_start), -float("inf"), device=q.device, dtype=dtype)
            output_block = torch.zeros_like(q_block, dtype=dtype)

            for kv_start in range(0, kv_seq_len, Bc):
                kv_end = min(kv_start + Bc, kv_seq_len)
                k_block = k[:, :, kv_start:kv_end, :].to(dtype)
                v_block = v[:, :, kv_start:kv_end, :].to(dtype)

                scores = torch.matmul(q_block, k_block.transpose(-2, -1)) / self.scale

                scores = self._apply_masks(scores, mask, q_start, q_end, kv_start, kv_end)

                m_new = torch.maximum(m_block, scores.max(dim=-1).values)
                exp_scores = torch.exp(scores - m_new.unsqueeze(-1))
                l_new = torch.exp(m_block - m_new) * l_block + exp_scores.sum(dim=-1)

                scale = torch.exp(m_block - m_new).unsqueeze(-1)
                # Online-softmax rescale: exp_scores is already exp(S - m_new),
                # so exp_scores @ V needs no further scaling -- only the old,
                # already-normalized output_block needs rescaling back up to
                # an l_block-weighted (un-normalized) sum before adding the
                # new block's contribution, then the whole thing is
                # renormalized by the updated running sum l_new.
                output_block = (output_block * (l_block.unsqueeze(-1) * scale) + torch.matmul(exp_scores, v_block)) / l_new.unsqueeze(-1)

                m_block = m_new
                l_block = l_new

            output[:, :, q_start:q_end, :] = output_block
            l[:, :, q_start:q_end] = l_block
            m[:, :, q_start:q_end] = m_block

        return output.to(q.dtype)

    def _compute_block_size(self, seq_len):
        if seq_len < 512:
            return 64
        if seq_len < 2048:
            return 128
        return 256


class FlashAttentionWrapper(BaseAttention):
    def __init__(self, d_model, num_heads, block_size=128, bias=False, dropout=0.0, causal=False, use_v2=True):
        super().__init__(d_model, num_heads, dropout, causal, bias)
        self.block_size = block_size
        if use_v2:
            self.flash_attention = FlashAttentionV2(d_model, num_heads, block_size, bias, dropout, causal, use_fp32_accum=True)
        else:
            self.flash_attention = FlashAttentionV1(d_model, num_heads, block_size, bias, dropout, causal)

        # BaseAttention.__init__ (above) creates its own Wq/Wk/Wv/Wo, but
        # forward() delegates entirely to self.flash_attention, which has
        # its own separate, independently-initialized copies -- so those
        # base-class ones would sit dead: never used in the forward pass,
        # never receiving a gradient, just extra memory and inflated
        # reported param counts (and anything reading `.Wq` etc off this
        # wrapper, like CachedAttention, would silently grab the wrong,
        # untrained weights). Alias to the copies that are actually used;
        # nn.Module's parameter/state_dict dedup means this doesn't
        # register them twice.
        self.Wq = self.flash_attention.Wq
        self.Wk = self.flash_attention.Wk
        self.Wv = self.flash_attention.Wv
        self.Wo = self.flash_attention.Wo

    def forward(self, query, key, value, mask=None, positions=None):
        return self.flash_attention(query, key, value, mask)
