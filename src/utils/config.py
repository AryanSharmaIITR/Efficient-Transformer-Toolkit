"""YAML-based configuration loading and saving utilities."""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError:  # pragma: no cover
    yaml = None  # type: ignore[assignment]


def _ensure_yaml() -> None:
    if yaml is None:
        raise ImportError(
            "PyYAML is required for config utilities.  "
            "Install it with: pip install pyyaml"
        )


def load_config(config_path: str | Path) -> dict[str, Any]:
    """Read a YAML configuration file and return it as a dictionary.

    Args:
        config_path: Path to the YAML file.

    Returns:
        Parsed configuration dictionary.

    Raises:
        FileNotFoundError: If the file does not exist.
        ImportError: If PyYAML is not installed.
        yaml.YAMLError: If the file contains invalid YAML.
    """
    _ensure_yaml()
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path, "r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)

    if data is None:
        return {}
    if not isinstance(data, dict):
        raise TypeError(
            f"Expected a YAML mapping at the top level, got {type(data).__name__}"
        )
    return data


def save_config(config: dict[str, Any], save_path: str | Path) -> None:
    """Write a configuration dictionary to a YAML file.

    Args:
        config: Configuration to serialize.
        save_path: Destination file path.  Parent directories are created
            automatically.

    Raises:
        ImportError: If PyYAML is not installed.
        yaml.YAMLError: If serialization fails.
    """
    _ensure_yaml()
    path = Path(save_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as fh:
        yaml.dump(config, fh, default_flow_style=False, sort_keys=False)


# ---------------------------------------------------------------------------
# Dataclass ↔ dict helpers
# ---------------------------------------------------------------------------


def dataclass_to_dict(obj: Any) -> dict[str, Any]:
    """Convert a dataclass instance to a plain dictionary.

    Args:
        obj: A dataclass instance (recursively converted).

    Returns:
        Nested dictionary.

    Raises:
        TypeError: If *obj* is not a dataclass instance.
    """
    if not dataclasses.is_dataclass(obj):
        raise TypeError(f"Expected a dataclass instance, got {type(obj).__name__}")
    return dataclasses.asdict(obj)


def dict_to_dataclass(data: dict[str, Any], cls: type) -> Any:
    """Populate a dataclass from a dictionary.

    Extra keys in *data* that are not fields of *cls* are silently ignored
    so that forward-compatible configs work without errors.

    Args:
        data: Source dictionary.
        cls: Target dataclass type.

    Returns:
        An instance of *cls*.

    Raises:
        TypeError: If *cls* is not a dataclass type.
    """
    if not dataclasses.is_dataclass(cls):
        raise TypeError(f"Expected a dataclass type, got {cls!r}")
    field_names = {f.name for f in dataclasses.fields(cls)}  # type: ignore[arg-type]
    filtered = {k: v for k, v in data.items() if k in field_names}
    return cls(**filtered)
