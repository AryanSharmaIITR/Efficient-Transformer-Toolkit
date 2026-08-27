"""Logging configuration for the Efficient Transformer Toolkit."""

from __future__ import annotations

import logging
import sys

_DEFAULT_FORMAT = "%(asctime)s | %(name)s | %(levelname)s | %(message)s"
_DEFAULT_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

_LOGGERS: dict[str, logging.Logger] = {}


def setup_logger(
    name: str = "transformer",
    log_file: str | None = None,
    level: str = "INFO",
    format_str: str = _DEFAULT_FORMAT,
) -> logging.Logger:
    """Create or retrieve a logger with console and optional file handlers.

    Safe to call multiple times with the same *name* — existing handlers are
    not duplicated.

    Args:
        name: Logger name.
        log_file: If provided, a file handler writing to this path is added.
        level: Minimum log level (``"DEBUG"``, ``"INFO"``, ``"WARNING"``,
            ``"ERROR"``, ``"CRITICAL"``).
        format_str: Python logging format string.

    Returns:
        Configured :class:`logging.Logger`.
    """
    if name in _LOGGERS:
        return _LOGGERS[name]

    resolved_level = getattr(logging, level.upper(), logging.INFO)

    logger = logging.getLogger(name)
    logger.setLevel(resolved_level)
    logger.propagate = False

    formatter = logging.Formatter(format_str, datefmt=_DEFAULT_DATE_FORMAT)

    # Console handler --------------------------------------------------
    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(formatter)
    logger.addHandler(console)

    # File handler (optional) ------------------------------------------
    file_handler = None
    if log_file is not None:
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    # Every module in this codebase logs via logging.getLogger(__name__)
    # (e.g. "src.training.trainer", "src.inference.quantization"), which
    # propagates to the root logger by default. Root has no handlers and
    # defaults to WARNING, so without this, Trainer's per-step loss logs,
    # "Training complete", checkpoint-save messages, quantization size
    # reports, etc. all vanish silently even though this "train"/"eval"/...
    # logger itself is fully configured. Mirror this logger's handlers onto
    # root (once per process) so the rest of src/*'s own logging actually
    # surfaces. `propagate = False` above keeps this logger's own records
    # from *also* being handled a second time via root.
    root = logging.getLogger()
    if not getattr(root, "_etk_configured", False):
        root.setLevel(resolved_level)
        root.addHandler(console)
        if file_handler is not None:
            root.addHandler(file_handler)
        root._etk_configured = True

    _LOGGERS[name] = logger
    return logger


def get_logger(name: str = "transformer") -> logging.Logger:
    """Return an existing logger or a basic one if *name* was never set up.

    Args:
        name: Logger name.

    Returns:
        :class:`logging.Logger` instance.
    """
    if name in _LOGGERS:
        return _LOGGERS[name]
    return logging.getLogger(name)
