from __future__ import annotations

import logging

import torch
from torch import nn

logger = logging.getLogger(__name__)

_BNB_AVAILABLE = False
try:
    import bitsandbytes as bnb  # type: ignore[import-untyped]

    _BNB_AVAILABLE = True
except ImportError:
    pass

# Module types that support dynamic quantization.
_QUANTIZABLE = (nn.Linear, nn.LSTM, nn.GRU, nn.RNNCell, nn.LSTMCell, nn.GRUCell)


def _log_size_before_after(model: nn.Module, tag: str) -> None:
    """Log model parameter count and approximate memory footprint."""
    n_params = sum(p.numel() for p in model.parameters())
    n_bytes = sum(p.nelement() * p.element_size() for p in model.parameters())
    logger.info(
        "[%s] params: %s  memory: %.2f MB",
        tag,
        f"{n_params:,}",
        n_bytes / (1024 * 1024),
    )


# ---------------------------------------------------------------------------
# Dynamic quantization
# ---------------------------------------------------------------------------

def quantize_dynamic(
    model: nn.Module,
    dtype: torch.dtype = torch.qint8,
    exclude: list[str] | None = None,
) -> nn.Module:
    """Apply post-training dynamic quantization.

    Dynamic quantization determines the quantization range at runtime for
    activation tensors while keeping weights in low precision.  This is the
    simplest quantization strategy and works without calibration data.

    When ``bitsandbytes`` is installed, its 8-bit matrix multiplication is
    preferred because it integrates more naturally with CUDA acceleration.

    Args:
        model: module to quantize (modified in-place and returned).
        dtype: target quantized dtype (``torch.qint8`` or ``torch.float16``).
            Ignored when ``bitsandbytes`` is used.
        exclude: list of fully-qualified parameter names to skip.  Matching
            is done via ``str.startswith``.

    Returns:
        The quantized model (same object as *model*).

    Raises:
        ValueError: if *dtype* is not supported.
    """
    if dtype not in (torch.qint8, torch.float16):
        raise ValueError(f"Unsupported quantization dtype: {dtype}")

    _log_size_before_after(model, "pre-quantize")

    # Prefer bitsandbytes when available (CUDA only).
    if _BNB_AVAILABLE and next(model.parameters()).is_cuda:
        model = _bnb_quantize(model, exclude=exclude)
        _log_size_before_after(model, "post-bnb-quantize")
        return model

    exclude_names = set(exclude or [])

    def _should_exclude(name: str) -> bool:
        return any(name.startswith(e) for e in exclude_names)

    qconfig = torch.quantization.default_dynamic_qconfig if dtype == torch.qint8 else torch.quantization.float16_dynamic_qconfig
    quantized = torch.quantization.quantize_dynamic(
        model,
        # torch.quantization.quantize_dynamic's qconfig_spec dict is keyed
        # by module *name or type* (per its own docstring) -- it was keyed
        # by module *instance* here, which the underlying propagate_qconfig_/
        # convert machinery never matches against anything, so this quietly
        # quantized zero layers (verified: identical state_dict size before
        # and after). Fully-qualified name strings are what named_modules()
        # already gives us and what quantize_dynamic's per-module lookup
        # actually matches on.
        qconfig_spec={name: qconfig
                       for name, m in model.named_modules()
                       if isinstance(m, _QUANTIZABLE) and not _should_exclude(name)},
        dtype=dtype,
        mapping=None,
    )
    _log_size_before_after(quantized, "post-quantize")
    return quantized


def _set_submodule(root: nn.Module, name: str, new_module: nn.Module) -> None:
    """Replace the submodule at dotted path *name* (from ``named_modules()``)
    with *new_module*, in-place on *root*."""
    parts = name.split(".")
    parent = root
    for p in parts[:-1]:
        parent = getattr(parent, p)
    setattr(parent, parts[-1], new_module)


def _bnb_quantize(model: nn.Module, exclude: list[str] | None = None) -> nn.Module:
    """Replace ``nn.Linear`` layers with ``bnb.nn.Linear8bitLt``."""
    exclude_names = set(exclude or [])

    for name, module in reversed(list(model.named_modules())):
        if any(name.startswith(e) for e in exclude_names):
            continue
        if not isinstance(module, nn.Linear):
            continue
        has_bias = module.bias is not None
        eightbit = bnb.nn.Linear8bitLt(
            module.in_features,
            module.out_features,
            bias=has_bias,
            threshold=6.0,
        )
        # Linear8bitLt's actual int8 quantization is triggered by
        # Int8Params' overridden .cuda()/.to() -- assigning a plain
        # nn.Parameter here skips that entirely, so the layer would
        # silently run in full precision instead of int8.
        eightbit.weight = bnb.nn.Int8Params(
            module.weight.data.clone(), requires_grad=False, has_fp16_weights=False,
        )
        if has_bias:
            eightbit.bias = nn.Parameter(module.bias, requires_grad=False)

        _set_submodule(model, name, eightbit)

    return model


# ---------------------------------------------------------------------------
# Static quantization (simplified)
# ---------------------------------------------------------------------------

def quantize_static(
    model: nn.Module,
    calibration_data: list[torch.Tensor] | None = None,
    dtype: torch.dtype = torch.qint8,
    num_calib_batches: int = 32,
) -> nn.Module:
    """Apply post-training static quantization with optional calibration.

    Static quantization pre-computes quantization ranges using calibration
    data, which yields better accuracy than dynamic quantization at the cost
    of requiring representative input samples.

    Only ``nn.Linear`` submodules are quantized (each individually wrapped
    with :class:`torch.quantization.QuantWrapper`, so its input is quantized
    right before the matmul and dequantized right after). Everything else
    -- embeddings, LayerNorm, residual adds, softmax -- stays ``float32``.
    Eager-mode static quantization requires an explicit float/int8 boundary
    (``QuantStub``/``DeQuantStub``) around whatever *is* quantized; applying
    a blanket qconfig to the whole model without one makes ``convert()``
    swap every Linear/LayerNorm for a quantized module that expects an
    already-quantized input tensor, which nothing upstream (token
    embeddings, residual streams, …) would ever produce. Scoping the
    quantized region to each Linear individually sidesteps that without
    having to instrument the model's forward pass with stubs.

    .. note::

       PyTorch static quantization only works on CPU.  The model is moved
       to CPU for quantization and must be moved back afterward if needed.

    Args:
        model: module to quantize.
        calibration_data: list of input tensors for calibration.  Each
            tensor should match the model's expected input shape.  If
            ``None``, the model is quantized without calibration (ranges
            are estimated from weight distributions).
        dtype: target quantized dtype (``torch.qint8`` only).
        num_calib_batches: maximum number of batches to use from
            *calibration_data*.

    Returns:
        The quantized model (moved back to original device).

    Raises:
        ValueError: if *dtype* is not ``torch.qint8``.
    """
    if dtype != torch.qint8:
        raise ValueError("Static quantization only supports torch.qint8")

    original_device = next(model.parameters()).device
    model.cpu()
    model.eval()

    _log_size_before_after(model, "pre-static-quantize")

    qconfig = torch.quantization.get_default_qconfig("fbgemm")
    for name, module in list(model.named_modules()):
        if isinstance(module, nn.Linear):
            # QuantWrapper reads module.qconfig at construction time to
            # build its internal QuantStub, so this must be set *before*
            # wrapping, not on the wrapper afterward.
            module.qconfig = qconfig
            _set_submodule(model, name, torch.quantization.QuantWrapper(module))

    # Prepare — inserts observers (only inside each QuantWrapper).
    model_prepared = torch.quantization.prepare(model, inplace=False)

    # Calibration pass.
    if calibration_data:
        with torch.no_grad():
            for i, batch in enumerate(calibration_data):
                if i >= num_calib_batches:
                    break
                batch = batch.cpu()
                model_prepared(batch)
        logger.info("Static quantization: used %d calibration batches", min(len(calibration_data), num_calib_batches))
    else:
        logger.info("Static quantization: no calibration data provided; using weight-derived ranges")

    # Convert — fuse layers and quantize.
    model_quantized = torch.quantization.convert(model_prepared, inplace=False)
    _log_size_before_after(model_quantized, "post-static-quantize")

    return model_quantized.to(original_device)


# ---------------------------------------------------------------------------
# FP16 conversion
# ---------------------------------------------------------------------------

def convert_to_fp16(model: nn.Module) -> nn.Module:
    """Convert model parameters to ``float16`` for half-precision inference.

    This is the simplest way to reduce model memory and speed up inference
    on GPUs with Tensor Cores.  No quantization artifacts are introduced.

    Args:
        model: module to convert (modified in-place and returned).

    Returns:
        The model with all parameters in ``torch.float16``.

    Raises:
        RuntimeError: if the model has no parameters.
    """
    params = list(model.parameters())
    if not params:
        raise RuntimeError("Model has no parameters to convert")

    _log_size_before_after(model, "pre-fp16")
    model.half()
    _log_size_before_after(model, "post-fp16")
    logger.info("Converted model to float16")
    return model
