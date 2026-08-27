"""Utilities for distributed (multi-GPU / multi-node) training."""

from __future__ import annotations

import os
from collections.abc import Generator
from contextlib import contextmanager
from typing import Any

import torch.distributed as dist


def is_distributed() -> bool:
    """Check whether ``torch.distributed`` is initialized.

    Returns:
        ``True`` if the distributed process group has been initialized.
    """
    return dist.is_available() and dist.is_initialized()


def get_world_size() -> int:
    """Return the total number of processes in the current job.

    Returns:
        World size, or ``1`` when running without distributed.
    """
    if is_distributed():
        return dist.get_world_size()
    return 1


def get_rank() -> int:
    """Return the global rank of the current process.

    Returns:
        Rank, or ``0`` when running without distributed.
    """
    if is_distributed():
        return dist.get_rank()
    return 0


def get_local_rank() -> int:
    """Return the local rank of the current process on this node.

    Checks ``LOCAL_RANK`` (used by ``torchrun`` / ``torch.distributed.launch``)
    then ``OMPI_COMM_WORLD_LOCAL_RANK`` (used by OpenMPI).

    Returns:
        Local rank, or ``0`` when the variable is not set.
    """
    rank_str = os.environ.get("LOCAL_RANK") or os.environ.get(
        "OMPI_COMM_WORLD_LOCAL_RANK", ""
    )
    try:
        return int(rank_str)
    except (ValueError, TypeError):
        return 0


def is_main_process() -> bool:
    """Return ``True`` if the current process has global rank 0.

    Returns:
        ``True`` on rank 0 (or when distributed is not in use).
    """
    return get_rank() == 0


def sync_all() -> None:
    """Synchronise all processes with a distributed barrier.

    Does nothing when distributed is not initialized.
    """
    if is_distributed():
        dist.barrier()


@contextmanager
def broadcast_config(config: dict[str, Any]) -> Generator[dict[str, Any]]:
    """Broadcast a configuration dictionary from rank 0 to all ranks.

    This context manager yields the (potentially broadcast) config and
    ensures every rank sees the same values.

    Args:
        config: The configuration dictionary.  Only the value on rank 0 is
            authoritative; other ranks may pass anything (or ``{}``).

    Yields:
        The authoritative configuration dictionary.

    Example::

        cfg = load_config("train.yaml")
        with broadcast_config(cfg) as cfg:
            # cfg is identical on every rank
            train(cfg)
    """
    if is_distributed():
        object_list = [config]
        dist.broadcast_object_list(object_list, src=0)
        config = object_list[0]
    yield config
