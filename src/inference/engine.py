from __future__ import annotations

import logging

import torch
import torch.nn.functional as F
from torch import nn

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# KV-Cache
# ---------------------------------------------------------------------------

class KVCache:
    """Stores cached key/value tensors for efficient autoregressive generation.

    During prefill, K/V projections for every token are computed and stored.
    During subsequent decode steps, only the new token's K/V are projected,
    concatenated with the cache, and used for attention — avoiding redundant
    computation over the full sequence.

    Attributes:
        cache: mapping of ``layer_idx -> (key, value)`` tensors, each of shape
            ``[B, n_heads, T_cached, head_dim]``.
    """

    def __init__(self) -> None:
        self.cache: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}

    # ------------------------------------------------------------------
    def get(self, layer_idx: int) -> tuple[torch.Tensor, torch.Tensor] | None:
        """Return cached ``(key, value)`` for *layer_idx*, or ``None``."""
        return self.cache.get(layer_idx)

    def update(
        self,
        layer_idx: int,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Append new K/V to the cache and return the full tensors.

        Args:
            layer_idx: decoder layer index.
            key: new key tensor ``[B, n_heads, T_new, head_dim]``.
            value: new value tensor ``[B, n_heads, T_new, head_dim]``.

        Returns:
            Tuple of the concatenated ``(key, value)`` tensors covering all
            tokens seen so far.
        """
        if layer_idx in self.cache:
            cached_k, cached_v = self.cache[layer_idx]
            key = torch.cat([cached_k, key], dim=2)
            value = torch.cat([cached_v, value], dim=2)
        self.cache[layer_idx] = (key, value)
        return key, value

    def clear(self) -> None:
        """Remove all cached tensors."""
        self.cache.clear()

    def __len__(self) -> int:
        return len(self.cache)

    @property
    def sequence_length(self) -> int:
        """Length of the longest cached sequence (0 if empty)."""
        if not self.cache:
            return 0
        return max(kv[0].size(2) for kv in self.cache.values())


# ---------------------------------------------------------------------------
# CachedAttention wrapper
# ---------------------------------------------------------------------------

class CachedAttention(nn.Module):
    """Drop-in replacement for ``BaseAttention`` subclasses with KV-cache.

    Re-uses the original module's ``Wq``, ``Wk``, ``Wv``, ``Wo`` projection
    weights so that no parameters are duplicated.  The forward pass:

    1. Projects Q, K, V from the input.
    2. Concatenates cached K/V when a ``KVCache`` is provided.
    3. Runs scaled-dot-product attention (with optional causal masking).
    4. Projects the output through ``Wo``.

    Supports plain scaled-dot-product attention with matching Q/K/V head
    counts (``MultiHeadAttention`` / ``FlashAttentionWrapper``) as well as
    rotary-position-embedded attention (``RotaryPEMultiHeadAttention``).
    It has no ALiBi-bias or grouped/shared-KV logic, so
    ``enable_kv_cache()`` refuses to wrap ALiBi/GQA/MQA attention modules
    rather than silently generating wrong output for them.
    """

    def __init__(self, original: nn.Module, layer_idx: int = 0) -> None:
        super().__init__()
        self.Wq = original.Wq
        self.Wk = original.Wk
        self.Wv = original.Wv
        self.Wo = original.Wo
        self.n_heads: int = original.n_heads
        self.head_dim: int = original.head_dim
        self.dropout = original.dropout
        self.scale: float = original.scale
        self.default_causal: bool = getattr(original, "causal", True)

        # Rotary position components (present on RotaryPEMultiHeadAttention).
        # They expect sequence-first tensors and absolute positions; kept so
        # the cached path can rotate the new token's Q/K at its true
        # absolute position (the cached keys were already rotated when they
        # were first stored).
        self.query_rotary_pe = getattr(original, "query_rotary_pe", None)
        self.key_rotary_pe = getattr(original, "key_rotary_pe", None)

        # Which cache to read/write and under what key. `kv_cache` is
        # instance state (not just a forward() kwarg) because
        # TransformerDecoderLayer.forward() calls self.self_attn(...) with
        # a fixed positional signature shared by every attention variant --
        # it has no way to pass extra kv_cache/layer_idx kwargs through.
        # InferenceEngine sets `.kv_cache` on every patched module before a
        # generate() call (see _set_kv_cache), so the cache is actually used
        # instead of forward()'s kv_cache= default silently staying None.
        self._cache_layer_idx = layer_idx
        self.kv_cache: KVCache | None = None

    # ------------------------------------------------------------------
    def _apply_rope(
        self,
        x: torch.Tensor,
        positions: torch.Tensor | None,
        rotary_pe: nn.Module | None,
    ) -> torch.Tensor:
        """Apply rotary position embedding to a ``[B, n_heads, T, head_dim]``
        tensor (batch-first, this wrapper's layout), returning the same
        layout.

        ``RotaryPositionalEmbedding`` works sequence-first
        (``[T, B, n_heads, head_dim]``) and takes absolute positions, so the
        tensor is permuted before and after. With ``rotary_pe is None`` (a
        non-rotary module) this is a no-op.
        """
        if rotary_pe is None:
            return x
        B, n_h, T, hd = x.shape
        x_seq = x.permute(2, 0, 1, 3)  # [T, B, n_heads, head_dim]
        x_seq = rotary_pe(x_seq, positions)
        return x_seq.permute(1, 2, 0, 3)  # [B, n_heads, T, head_dim]

    # ------------------------------------------------------------------
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: torch.Tensor | None = None,
        positions: torch.Tensor | None = None,
        *,
        causal: bool | None = None,
        kv_cache: KVCache | None = None,
        layer_idx: int | None = None,
    ) -> torch.Tensor:
        """Forward pass with optional KV-cache.

        Args:
            query, key, value: raw hidden states ``[B, T, d_model]``.
            mask: unused in the cached path (causal mask is generated
                automatically when ``causal=True``).
            positions: absolute token positions used to apply rotary
                embeddings to Q and new keys when the wrapped attention is
                rotary (``RotaryPEMultiHeadAttention``). ``None`` means
                "assume sequential positions 0..T-1", which is correct for
                a fresh prefill; during incremental decode the engine passes
                the new token's absolute position.
            causal: whether to apply a causal mask. Defaults to the
                wrapped module's own ``causal`` setting.
            kv_cache: shared cache object; overrides ``self.kv_cache`` (the
                cache ``InferenceEngine`` sets automatically) when given
                explicitly. ``None`` here means "use ``self.kv_cache``",
                not "disable caching" -- pass ``kv_cache=KVCache()`` (via
                ``self.kv_cache``) to actually disable it if needed.
            layer_idx: overrides ``self._cache_layer_idx`` when given.

        Returns:
            Output tensor ``[B, T, d_model]``.
        """
        if causal is None:
            causal = self.default_causal
        if kv_cache is None:
            kv_cache = self.kv_cache
        if layer_idx is None:
            layer_idx = self._cache_layer_idx

        Q = self.Wq(query)  # [B, T_q, d_model]

        if kv_cache is not None and not self.training:
            K_new = self.Wk(key)  # [B, T_new, d_model]
            V_new = self.Wv(value)
            # Project then reshape to [B, n_heads, T, head_dim]
            T_new = K_new.size(1)
            K_new = K_new.view(-1, T_new, self.n_heads, self.head_dim).transpose(1, 2)
            V_new = V_new.view(-1, T_new, self.n_heads, self.head_dim).transpose(1, 2)
            # Rotate the new keys at their true absolute positions before
            # caching: keys already in the cache were rotated when first
            # stored, and rotary is a fixed function of absolute position,
            # so appending freshly-rotated keys is exactly correct.
            K_new = self._apply_rope(K_new, positions, self.key_rotary_pe)
            K, V = kv_cache.update(layer_idx, K_new, V_new)
        else:
            K = self.Wk(key)
            V = self.Wv(value)
            # Match the cached branch's reshape -- without this, K/V stay
            # [B, T, d_model] (3D) while Q is reshaped to 4D below, so the
            # SDPA call and T_k = K.size(2) (which would silently read
            # d_model instead of the sequence length) both break.
            T_kv = K.size(1)
            K = K.view(-1, T_kv, self.n_heads, self.head_dim).transpose(1, 2)
            V = V.view(-1, T_kv, self.n_heads, self.head_dim).transpose(1, 2)
            # Also rotate when caching is off so this wrapper never silently
            # drops rotary semantics for a rotary model.
            K = self._apply_rope(K, positions, self.key_rotary_pe)

        B, T_q, _ = Q.shape
        T_k = K.size(2)

        Q = Q.view(B, T_q, self.n_heads, self.head_dim).transpose(1, 2)
        Q = self._apply_rope(Q, positions, self.query_rotary_pe)

        # Use is_causal only when query and key have the same length (full
        # sequence attention).  During incremental decode T_q == 1 while
        # T_k > 1 so we must not set is_causal.
        use_causal = causal and (T_q == T_k)

        drop_p = self.dropout.p if self.training else 0.0
        attn_out = F.scaled_dot_product_attention(
            Q,
            K,
            V,
            is_causal=use_causal,
            dropout_p=drop_p,
        )

        attn_out = attn_out.transpose(1, 2).contiguous().view(B, T_q, -1)
        return self.Wo(attn_out)


# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------

def _patch_attention(module: nn.Module, layer_idx: int) -> list[str]:
    """Replace decoder self-attention modules with ``CachedAttention``.

    Returns a list of attribute paths that were patched so they can be
    restored later.
    """
    patched: list[str] = []

    def _recurse(mod: nn.Module, prefix: str) -> None:
        for name, child in mod.named_children():
            path = f"{prefix}.{name}" if prefix else name
            # Self-attention lives on TransformerDecoderLayer.self_attn
            if name == "self_attn":
                wrapper = CachedAttention(child)
                wrapper._cache_layer_idx = layer_idx  # type: ignore[attr-defined]
                setattr(mod, name, wrapper)
                patched.append(path)
                logger.debug("Patched attention at %s", path)
            else:
                _recurse(child, path)

    _recurse(module, "")
    return patched


def enable_kv_cache(model: nn.Module) -> None:
    """Monkey-patch decoder self-attention layers to support KV-cache.

    After calling this, pass a ``KVCache`` instance via the ``kv_cache``
    keyword argument to the patched ``self_attn`` modules (handled
    automatically by ``InferenceEngine``).

    Only ``TransformerDecoderLayer.self_attn`` modules are patched;
    cross-attention modules are left untouched because their keys/values
    do not change between decode steps.

    Args:
        model: a ``Transformer`` instance (or any ``nn.Module`` whose
            decoder layers expose ``self_attn`` attributes).
    """
    decoder = getattr(model, "decoder", None)
    if decoder is None:
        logger.warning("Model has no decoder; KV-cache has no effect.")
        return

    # CachedAttention replicates plain scaled-dot-product attention
    # (MultiHeadAttention / FlashAttentionWrapper with matching Q/K/V head
    # counts) and rotary-position-embedded attention
    # (RotaryPEMultiHeadAttention). It has no ALiBi-bias or grouped/shared-KV
    # logic, so wrapping those variants wouldn't just be suboptimal -- it
    # would silently produce wrong generations. Fail loudly instead.
    _UNSUPPORTED = {"Alibi", "GroupedQueryAttention", "MultiQueryAttention"}
    for idx, layer in enumerate(decoder.layers):
        cls_name = type(layer.self_attn).__name__
        if cls_name in _UNSUPPORTED:
            raise ValueError(
                f"enable_kv_cache() does not support attn_type/pos_encoding "
                f"producing {cls_name} self-attention (layer {idx}) -- "
                f"CachedAttention doesn't replicate its rotation/bias/grouped-KV "
                f"behavior and would silently generate wrong output. Use a "
                f"plain MultiHeadAttention or FlashAttentionWrapper model "
                f"(attn_type='flashv1'/'flashv2' with pos_encoding != 'rope'), "
                f"or generate without KV-cache."
            )
        # nn.Module.__init__ defaults a freshly-constructed module to
        # training mode. Swapping this in *after* model.eval() was already
        # called would otherwise silently leave it in training mode --
        # active dropout corrupting every cached forward pass -- with no
        # visible error, since the outer model still reports eval() mode.
        original_training = layer.self_attn.training
        wrapper = CachedAttention(layer.self_attn, layer_idx=idx)
        wrapper.train(original_training)
        layer.self_attn = wrapper
        logger.debug("Enabled KV-cache for decoder layer %d", idx)


def _set_kv_cache(model: nn.Module, kv_cache: KVCache | None) -> None:
    """Set (or clear, with ``None``) the active cache on every
    ``CachedAttention`` module in the model's decoder.

    ``TransformerDecoderLayer.forward()`` calls ``self.self_attn(...)``
    with the same fixed positional signature every attention variant
    shares -- it has no ``kv_cache=`` kwarg to pass through. Setting the
    cache as instance state here is what makes the patched modules
    actually use it instead of always taking the no-cache path.
    """
    decoder = getattr(model, "decoder", None)
    if decoder is None:
        return
    for layer in decoder.layers:
        if isinstance(layer.self_attn, CachedAttention):
            layer.self_attn.kv_cache = kv_cache


def disable_kv_cache(model: nn.Module) -> None:
    """Remove KV-cache wrappers and restore original attention modules.

    This is a best-effort operation.  If the original modules were already
    garbage-collected the wrapper weights will still be valid (they share
    the same parameter tensors).
    """
    decoder = getattr(model, "decoder", None)
    if decoder is None:
        return

    for layer in decoder.layers:
        attn = layer.self_attn
        if isinstance(attn, CachedAttention):
            # Reconstruct a plain module sharing the same parameters.
            restored = nn.Module()
            restored.Wq = attn.Wq
            restored.Wk = attn.Wk
            restored.Wv = attn.Wv
            restored.Wo = attn.Wo
            restored.n_heads = attn.n_heads  # type: ignore[attr-defined]
            restored.head_dim = attn.head_dim  # type: ignore[attr-defined]
            restored.dropout = attn.dropout  # type: ignore[attr-defined]
            restored.scale = attn.scale  # type: ignore[attr-defined]
            restored.forward = lambda q, k, v, mask=None, positions=None, *, causal=True, _r=restored: _plain_forward(  # type: ignore[attr-defined]
                _r, q, k, v, causal=causal,
            )
            layer.self_attn = restored


def _plain_forward(mod: nn.Module, x: torch.Tensor, *, causal: bool = True) -> torch.Tensor:
    """Fallback attention forward used after ``disable_kv_cache``."""
    Q = mod.Wq(x)
    K = mod.Wk(x)
    V = mod.Wv(x)

    B, T, _ = Q.shape
    n_heads: int = mod.n_heads  # type: ignore[attr-defined]
    head_dim: int = mod.head_dim  # type: ignore[attr-defined]

    Q = Q.view(B, T, n_heads, head_dim).transpose(1, 2)
    K = K.view(B, T, n_heads, head_dim).transpose(1, 2)
    V = V.view(B, T, n_heads, head_dim).transpose(1, 2)

    drop_p = mod.dropout.p if mod.training else 0.0  # type: ignore[attr-defined]
    attn_out = F.scaled_dot_product_attention(Q, K, V, is_causal=causal, dropout_p=drop_p)
    attn_out = attn_out.transpose(1, 2).contiguous().view(B, T, -1)
    return mod.Wo(attn_out)


# ---------------------------------------------------------------------------
# InferenceEngine
# ---------------------------------------------------------------------------

class InferenceEngine:
    """High-level wrapper for efficient autoregressive text generation.

    Supports encoder-only, decoder-only, and encoder-decoder models.  When
    ``use_kv_cache=True`` and the model has a decoder, the engine patches
    self-attention layers to reuse key/value projections across decode steps.

    Args:
        model: a ``Transformer`` (or compatible ``nn.Module``).
        device: target device (e.g. ``"cuda"``).  If ``None``, uses the
            device of the first model parameter.
        dtype: floating-point dtype for inference (e.g. ``torch.float16``).
            If ``None``, keeps the model's current dtype.
        use_kv_cache: enable KV-cache for decoder-only generation.  Ignored
            when the model has no decoder.
        pad_token_id: default pad token used when batching prompts of
            different lengths.
        eos_token_id: default end-of-sequence token.

    Example::

        engine = InferenceEngine(model, device="cuda", use_kv_cache=True)
        outputs = engine.generate_batch(
            ["Hello, how are you?", "Tell me a joke."],
            max_new_tokens=64,
        )
        for text in outputs:
            print(text)
    """

    def __init__(
        self,
        model: nn.Module,
        device: str | torch.device | None = None,
        dtype: torch.dtype | None = None,
        use_kv_cache: bool = False,
        pad_token_id: int | None = None,
        eos_token_id: int | None = None,
    ) -> None:
        self.model = model
        self.device = device or next(model.parameters()).device
        self.dtype = dtype
        self.pad_token_id = pad_token_id
        self.eos_token_id = eos_token_id

        self._kv_cache_enabled = False
        if use_kv_cache:
            self.enable_kv_cache()

    # ------------------------------------------------------------------
    # KV-cache management
    # ------------------------------------------------------------------

    def enable_kv_cache(self) -> None:
        """Enable KV-cache on the underlying model."""
        if self._kv_cache_enabled:
            return
        enable_kv_cache(self.model)
        self._kv_cache_enabled = True
        logger.info("KV-cache enabled")

    def disable_kv_cache(self) -> None:
        """Disable KV-cache and restore original attention modules."""
        if not self._kv_cache_enabled:
            return
        disable_kv_cache(self.model)
        self._kv_cache_enabled = False
        logger.info("KV-cache disabled")

    # ------------------------------------------------------------------
    # Token helpers
    # ------------------------------------------------------------------

    def _to_device(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor.to(self.device)

    def _top_k_top_p(
        self,
        logits: torch.Tensor,
        temperature: float,
        top_k: int,
        top_p: float,
    ) -> torch.Tensor:
        """Apply temperature, top-k, and top-p filtering then sample."""
        logits = logits / max(temperature, 1e-8)

        if top_k > 0:
            top_k_val = min(top_k, logits.size(-1))
            indices_to_remove = logits < torch.topk(logits, top_k_val)[0][..., -1, None]
            logits[indices_to_remove] = -float("inf")

        if 0.0 < top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            indices_to_remove = sorted_indices_to_remove.scatter(
                1, sorted_indices, sorted_indices_to_remove,
            )
            logits[indices_to_remove] = -float("inf")

        probs = F.softmax(logits, dim=-1)
        return torch.multinomial(probs, num_samples=1)

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 50,
        temperature: float = 1.0,
        top_k: int = 0,
        top_p: float = 0.0,
        do_sample: bool = False,
        eos_token_id: int | None = None,
        pad_token_id: int | None = None,
    ) -> torch.Tensor:
        """Generate tokens for a batch of prompts.

        When KV-cache is enabled and the model is decoder-only, an efficient
        incremental decode loop is used: the prompt is prefilled in one
        forward pass, then each new token is generated from a single-token
        input using the cached key/value state.

        For encoder-decoder or encoder-only models (or when the cache is
        disabled), falls back to the model's native ``generate`` method.

        Args:
            input_ids: ``[B, T]`` prompt token ids.
            max_new_tokens: maximum number of new tokens to produce.
            temperature: sampling temperature (1.0 = greedy).
            top_k: top-k filtering (0 = disabled).
            top_p: nucleus sampling threshold (0.0 = disabled).
            do_sample: if ``True`` sample; otherwise greedy.
            eos_token_id: overrides ``self.eos_token_id``.
            pad_token_id: overrides ``self.pad_token_id``.

        Returns:
            Token ids ``[B, T + generated]``.
        """
        input_ids = self._to_device(input_ids)
        eos = eos_token_id or self.eos_token_id
        pad = pad_token_id or self.pad_token_id

        self.model.eval()

        # Fast path: use model's native generate when cache is off or model
        # is not decoder-only.
        is_decoder_only = getattr(self.model, "use_decoder", False) and not getattr(
            self.model, "use_encoder", False,
        )
        if not self._kv_cache_enabled or not is_decoder_only:
            return self.model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                do_sample=do_sample,
                pad_token_id=pad,
                eos_token_id=eos,
            )

        # --- Cached incremental decode --------------------------------
        generated = input_ids
        kv_cache = KVCache()
        _set_kv_cache(self.model, kv_cache)
        try:
            # Prefill: process the whole prompt (positions default to
            # 0..prompt_len-1, correct for a fresh cache) and use ITS OWN
            # logits for the first next-token prediction. The loop below
            # only ever feeds the single most-recently-generated token, so
            # it never reprocesses -- and never double-appends to the cache
            # -- a token that's already cached.
            logits = self.model(input_ids=generated, return_encoder_output=False)
            if isinstance(logits, dict):
                logits = logits["logits"]

            for _ in range(max_new_tokens):
                next_token_logits = logits[:, -1, :]

                if do_sample:
                    next_token = self._top_k_top_p(next_token_logits, temperature, top_k, top_p)
                else:
                    next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)

                generated = torch.cat([generated, next_token], dim=-1)

                if eos is not None and (next_token == eos).any():
                    break
                if pad is not None and (next_token == pad).any():
                    break

                # `next_token`'s absolute position is its index in `generated`.
                current_pos = torch.tensor([generated.size(1) - 1], device=generated.device)
                logits = self.model(input_ids=next_token, positions=current_pos, return_encoder_output=False)
                if isinstance(logits, dict):
                    logits = logits["logits"]
        finally:
            # Each generate() call gets its own cache; don't leave a stale
            # one attached to the modules for the next unrelated call.
            _set_kv_cache(self.model, None)

        return generated

    # ------------------------------------------------------------------
    # Batch generation
    # ------------------------------------------------------------------

    def generate_batch(
        self,
        prompts: list[str] | list[list[int]],
        tokenizer: object | None = None,
        max_new_tokens: int = 50,
        temperature: float = 1.0,
        top_k: int = 0,
        top_p: float = 0.0,
        do_sample: bool = False,
        batch_size: int | None = None,
        eos_token_id: int | None = None,
        pad_token_id: int | None = None,
        return_sequences: bool = False,
    ) -> list[str] | list[torch.Tensor]:
        """Generate text for a list of prompts, optionally batching them.

        If *tokenizer* is ``None``, *prompts* must already be tokenized
        (i.e. ``list[list[int]]``).

        Args:
            prompts: list of string prompts or pre-tokenized id lists.
            tokenizer: a HuggingFace-compatible tokenizer with
                ``__call__`` / ``encode`` / ``decode`` / ``pad_token_id``.
            max_new_tokens: generation length.
            temperature, top_k, top_p, do_sample: sampling parameters.
            batch_size: prompts per batch (``None`` = all at once).
            eos_token_id, pad_token_id: override engine defaults.
            return_sequences: if ``True``, return raw ``torch.Tensor``
                id tensors instead of decoded strings.

        Returns:
            List of generated strings (or tensors when *return_sequences*).
        """
        # Tokenize if necessary.
        if tokenizer is not None and len(prompts) > 0 and isinstance(prompts[0], str):
            if callable(tokenizer):
                encodings = tokenizer(
                    prompts,  # type: ignore[arg-type]
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                )
                input_ids = self._to_device(encodings["input_ids"])
            elif hasattr(tokenizer, "encode"):
                # Wrapper tokenizers (e.g. Tokenizer) expose ``encode``,
                # ``pad_batch`` instead of ``__call__``.
                ids_list = tokenizer.encode(prompts)  # type: ignore[attr-defined]
                if isinstance(ids_list, torch.Tensor):
                    input_ids = ids_list
                else:
                    padded = tokenizer.pad_batch(ids_list)  # type: ignore[attr-defined]
                    input_ids = self._to_device(padded["input_ids"])
            else:
                raise TypeError(
                    "tokenizer must be callable (HuggingFace) or expose "
                    "encode() / pad_batch() (wrapper).",
                )
        else:
            # Pad pre-tokenized inputs to equal length.
            assert isinstance(prompts[0], list), "prompts must be str or list[int]"
            max_len = max(len(p) for p in prompts)
            pad_id = pad_token_id or self.pad_token_id or 0
            padded = [p + [pad_id] * (max_len - len(p)) for p in prompts]
            input_ids = self._to_device(torch.tensor(padded, dtype=torch.long))

        eos = eos_token_id or self.eos_token_id
        pad = pad_token_id or self.pad_token_id
        if tokenizer is not None and pad is None and hasattr(tokenizer, "pad_token_id"):
            pad = tokenizer.pad_token_id

        bs = batch_size or input_ids.size(0)
        all_output_ids: list[torch.Tensor] = []

        for start in range(0, input_ids.size(0), bs):
            batch = input_ids[start : start + bs]
            out = self.generate(
                batch,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                do_sample=do_sample,
                eos_token_id=eos,
                pad_token_id=pad,
            )
            all_output_ids.extend(out)

        if return_sequences:
            return all_output_ids

        if tokenizer is None:
            raise ValueError(
                "Cannot decode output ids without a tokenizer.  "
                "Pass a tokenizer or set return_sequences=True.",
            )

        texts: list[str] = []
        for ids in all_output_ids:
            # Strip leading prompt tokens and any trailing padding/eos.
            ids_list = ids.tolist()
            if pad is not None:
                ids_list = [t for t in ids_list if t != pad]
            if eos is not None and eos in ids_list:
                ids_list = ids_list[: ids_list.index(eos)]
            texts.append(tokenizer.decode(ids_list, skip_special_tokens=True))  # type: ignore[union-attr]
        return texts
