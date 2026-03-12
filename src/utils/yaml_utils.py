"""Shared YAML loading/parsing helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def read_yaml_file(path: Path | str, *, default: Any = None) -> Any:
    """Read a YAML file and return parsed data, falling back to default for empty documents."""
    yaml_path = Path(path)
    if not yaml_path.exists():
        raise FileNotFoundError(f"YAML file not found: {yaml_path}")
    with open(yaml_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return default if data is None else data


def parse_yaml_text(text: str, *, default: Any = None) -> Any:
    """Parse YAML content from a string and return default for empty documents."""
    data = yaml.safe_load(text)
    return default if data is None else data
