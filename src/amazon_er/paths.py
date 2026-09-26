"""Project path resolution."""

from pathlib import Path
from typing import Any, Mapping


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def resolve_project_path(config: Mapping[str, Any], key: str) -> Path:
    value = Path(config["paths"][key])
    return value if value.is_absolute() else project_root() / value
