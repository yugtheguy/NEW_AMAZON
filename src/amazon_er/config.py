"""Validated YAML configuration loading with deterministic hashing."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping

import yaml

from amazon_er.infra.hashing import hash_mapping


class ConfigError(ValueError):
    """Configuration is missing or invalid."""


def _merge(base: dict[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    result = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(result.get(key), Mapping):
            result[key] = _merge(dict(result[key]), value)
        else:
            result[key] = deepcopy(value)
    return result


def _read_yaml(path: Path) -> dict[str, Any]:
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except FileNotFoundError as exc:
        raise ConfigError(f"Configuration not found: {path}") from exc
    except yaml.YAMLError as exc:
        raise ConfigError(f"Invalid YAML in {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise ConfigError(f"Top-level configuration must be a mapping: {path}")
    return data


def validate_config(config: Mapping[str, Any]) -> None:
    required = {"project", "paths", "runtime", "sharding", "retrieval", "features", "models", "decision", "audit"}
    missing = sorted(required - config.keys())
    if missing:
        raise ConfigError(f"Missing configuration sections: {missing}")
    warning = float(config["runtime"]["warning_ram_fraction"])
    critical = float(config["runtime"]["critical_ram_fraction"])
    if not 0 < warning < critical < 1:
        raise ConfigError("RAM fractions must satisfy 0 < warning < critical < 1")
    if int(config["runtime"]["seed"]) < 0:
        raise ConfigError("runtime.seed must be non-negative")
    if int(config["sharding"]["default_rows"]) <= 0:
        raise ConfigError("sharding.default_rows must be positive")


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path).resolve()
    data = _read_yaml(config_path)
    extends = data.pop("extends", None)
    if extends:
        parent = Path(extends)
        if not parent.is_absolute():
            parent = config_path.parent / parent
        data = _merge(load_config(parent), data)
    validate_config(data)
    return data


def config_hash(config: Mapping[str, Any]) -> str:
    validate_config(config)
    return hash_mapping(config)
