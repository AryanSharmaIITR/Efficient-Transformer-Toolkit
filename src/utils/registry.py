"""Generic name-based registry for model components."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, TypeVar

T = TypeVar("T")


class Registry:
    """A name → object registry for attention mechanisms, encodings, etc.

    Objects (classes or functions) are registered by a unique string key and
    retrieved later by the same key.  Duplicate registrations raise an error
    unless ``allow_override`` is set.

    Example::

        ATTENTIONS = Registry("attentions")
        ATTENTIONS.register("flash", FlashAttention)

        @ATTENTIONS.register("alibi")
        class AlibiAttention(nn.Module):
            ...

        cls = ATTENTIONS.get("alibi")
    """

    def __init__(self, namespace: str = "default", *, allow_override: bool = False) -> None:
        """Initialise an empty registry.

        Args:
            namespace: Human-readable name for this registry (used in error
                messages).
            allow_override: If ``True``, re-registering an existing key
                silently overwrites it.
        """
        self._namespace = namespace
        self._allow_override = allow_override
        self._store: dict[str, Any] = {}

    # ------------------------------------------------------------------
    def register(self, name: str, obj: Any) -> None:
        """Register *obj* under *name*.

        Args:
            name: Unique string key.
            obj: Object to store (typically a class or callable).

        Raises:
            KeyError: If *name* is already registered and
                ``allow_override`` is ``False``.
        """
        if not self._allow_override and name in self._store:
            raise KeyError(
                f"'{name}' is already registered in the {self._namespace!r} registry. "
                f"Use allow_override=True to replace it."
            )
        self._store[name] = obj

    # ------------------------------------------------------------------
    def get(self, name: str) -> Any:
        """Retrieve the object registered under *name*.

        Args:
            name: Lookup key.

        Returns:
            The registered object.

        Raises:
            KeyError: If *name* is not found.
        """
        if name not in self._store:
            available = ", ".join(sorted(self._store)) or "(none)"
            raise KeyError(
                f"'{name}' not found in the {self._namespace!r} registry. "
                f"Available: {available}"
            )
        return self._store[name]

    # ------------------------------------------------------------------
    def list_names(self) -> list[str]:
        """Return a sorted list of all registered names.

        Returns:
            Sorted list of string keys.
        """
        return sorted(self._store.keys())

    def __contains__(self, name: str) -> bool:
        return name in self._store

    def __len__(self) -> int:
        return len(self._store)

    def __repr__(self) -> str:
        return f"Registry(namespace={self._namespace!r}, items={self.list_names()})"


# ---------------------------------------------------------------------------
# Module-level decorator factories
# ---------------------------------------------------------------------------

_GLOBAL_REGISTRIES: dict[str, Registry] = {}


def _get_or_create_registry(namespace: str) -> Registry:
    if namespace not in _GLOBAL_REGISTRIES:
        _GLOBAL_REGISTRIES[namespace] = Registry(namespace)
    return _GLOBAL_REGISTRIES[namespace]


def register(
    name: str,
    *,
    namespace: str = "default",
) -> Callable[[T], T]:
    """Decorator that registers a class or function under *name*.

    Args:
        name: Unique key in the registry.
        namespace: Registry namespace (default ``"default"``).

    Returns:
        A decorator that registers the decorated callable and returns it
        unchanged.

    Example::

        @register("flash", namespace="attentions")
        class FlashAttention(nn.Module):
            ...
    """

    def decorator(obj: T) -> T:
        reg = _get_or_create_registry(namespace)
        reg.register(name, obj)
        return obj

    return decorator
