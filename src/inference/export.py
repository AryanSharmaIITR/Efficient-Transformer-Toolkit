from __future__ import annotations

import logging
from pathlib import Path

import torch
from torch import nn

logger = logging.getLogger(__name__)


def _make_example_inputs(
    model: nn.Module,
    vocab_size: int = 50257,
    batch_size: int = 1,
    seq_len: int = 32,
) -> dict[str, torch.Tensor]:
    """Build example inputs for tracing.

    Attempts to infer ``vocab_size`` and ``max_seq_len`` from model
    attributes (``config.vocab_size``, ``config.max_seq_len``) before
    falling back to defaults.

    Returns:
        A dict with ``input_ids`` and ``attention_mask`` shaped
        ``[batch_size, seq_len]``.
    """
    cfg = getattr(model, "config", None)
    if cfg is not None:
        vocab_size = getattr(cfg, "vocab_size", vocab_size)
        seq_len = getattr(cfg, "max_seq_len", seq_len)

    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    attention_mask = torch.ones(batch_size, seq_len, dtype=torch.long)
    return {"input_ids": input_ids, "attention_mask": attention_mask}


# ---------------------------------------------------------------------------
# TorchScript
# ---------------------------------------------------------------------------

def export_torchscript(
    model: nn.Module,
    path: str | Path,
    batch_size: int = 1,
    seq_len: int = 32,
    optimize: bool = True,
) -> Path:
    """Export the model to TorchScript via ``torch.jit.script``.

    ``torch.jit.script`` traces the model's Python source to produce a
    statically-typed IR.  This preserves dynamic control flow (causal
    masks, conditional branches) better than tracing but requires the
    model to use only TorchScript-compatible Python subsets.

    Args:
        model: module to export.  Must be in eval mode.
        path: destination file path (``.pt``).
        batch_size: example batch dimension.
        seq_len: example sequence length dimension.
        optimize: run ``torch.jit.optimize_for_injection`` on the result.

    Returns:
        The ``Path`` to the saved script.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    model.eval()
    model_cpu = model.cpu()

    logger.info("TorchScript: scripting model …")
    scripted = torch.jit.script(model_cpu)

    if optimize:
        scripted = torch.jit.optimize_for_inference(scripted)  # type: ignore[union-attr]

    scripted.save(str(path))
    logger.info("TorchScript model saved to %s", path)
    return path


# ---------------------------------------------------------------------------
# ONNX
# ---------------------------------------------------------------------------

def export_onnx(
    model: nn.Module,
    path: str | Path,
    batch_size: int = 1,
    seq_len: int = 32,
    opset_version: int = 17,
    input_names: list[str] | None = None,
    output_names: list[str] | None = None,
    dynamic_axes: dict[str, dict[int, str]] | None = None,
) -> Path:
    """Export the model to ONNX format via ``torch.onnx.export``.

    ONNX models can be loaded in TensorRT, ONNX Runtime, OpenVINO, and
    many other runtimes for deployment.

    Args:
        model: module to export.  Must be in eval mode.
        path: destination file path (``.onnx``).
        batch_size: example batch dimension.
        seq_len: example sequence length dimension.
        opset_version: ONNX opset version (17 recommended for recent ops).
        input_names: ONNX graph input names (default: ``["input_ids",
            "attention_mask"]``).
        output_names: ONNX graph output names (default: ``["logits"]``).
        dynamic_axes: mapping of ``{name: {dim: label}}`` for variable-
            length dimensions.  Defaults to making ``batch_size`` and
            ``seq_len`` dynamic for ``input_ids``.

    Returns:
        The ``Path`` to the saved ONNX file.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    model.eval()
    model_cpu = model.cpu()

    example_inputs = _make_example_inputs(model, batch_size=batch_size, seq_len=seq_len)

    is_enc_dec = getattr(model, "use_encoder", False) and getattr(
        model, "use_decoder", False,
    )
    if is_enc_dec:
        cfg = getattr(model, "config", None)
        vocab_size = getattr(cfg, "vocab_size", 50257)
        example_inputs["decoder_input_ids"] = torch.randint(0, vocab_size, (batch_size, seq_len))

    input_names = input_names or list(example_inputs.keys())
    output_names = output_names or ["logits"]

    if dynamic_axes is None:
        dynamic_axes = {
            "input_ids": {0: "batch_size", 1: "seq_len"},
            "attention_mask": {0: "batch_size", 1: "seq_len"},
        }
        if is_enc_dec:
            dynamic_axes["decoder_input_ids"] = {0: "batch_size", 1: "decoder_seq_len"}

    logger.info("ONNX: exporting with opset %d …", opset_version)
    torch.onnx.export(
        model_cpu,
        example_inputs,
        str(path),
        opset_version=opset_version,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
    )
    logger.info("ONNX model saved to %s", path)
    return path


# ---------------------------------------------------------------------------
# TorchScript via tracing
# ---------------------------------------------------------------------------

def export_jit(
    model: nn.Module,
    path: str | Path,
    batch_size: int = 1,
    seq_len: int = 32,
    optimize: bool = True,
) -> Path:
    """Export the model to TorchScript via ``torch.jit.trace``.

    Tracing records the tensor operations executed for a specific set of
    example inputs.  This is faster and more compatible than scripting but
    does **not** preserve dynamic control flow (e.g. ``if`` branches that
    depend on tensor values).

    Use :func:`export_torchscript` if your model contains dynamic control
    flow; use this function when you need maximum compatibility with
    deployment runtimes.

    Args:
        model: module to export.  Must be in eval mode.
        path: destination file path (``.pt``).
        batch_size: example batch dimension.
        seq_len: example sequence length dimension.
        optimize: run ``torch.jit.optimize_for_inference`` on the result.

    Returns:
        The ``Path`` to the saved trace.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    model.eval()
    model_cpu = model.cpu()

    example_inputs = _make_example_inputs(model, batch_size=batch_size, seq_len=seq_len)

    is_enc_dec = getattr(model, "use_encoder", False) and getattr(
        model, "use_decoder", False,
    )
    if is_enc_dec:
        cfg = getattr(model, "config", None)
        vocab_size = getattr(cfg, "vocab_size", 50257)
        example_inputs["decoder_input_ids"] = torch.randint(0, vocab_size, (batch_size, seq_len))

    logger.info("JIT trace: tracing with batch_size=%d, seq_len=%d …", batch_size, seq_len)
    # example_inputs is a dict of keyword args (input_ids=..., ...); trace's
    # positional `example_inputs` expects a tensor/tuple of positional args,
    # so passing a dict there lands the whole dict as the first positional
    # argument. Use example_kwarg_inputs instead.
    traced = torch.jit.trace(model_cpu, example_kwarg_inputs=example_inputs, strict=False)

    if optimize:
        traced = torch.jit.optimize_for_inference(traced)  # type: ignore[union-attr]

    traced.save(str(path))
    logger.info("Traced model saved to %s", path)
    return path
