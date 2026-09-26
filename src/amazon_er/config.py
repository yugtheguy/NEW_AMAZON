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
    required = {"project", "paths", "runtime", "sharding", "retrieval", "features", "models", "decision", "audit", "normalization"}
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
    normalization = config["normalization"]
    if int(normalization["shard_rows"]) <= 0:
        raise ConfigError("normalization.shard_rows must be positive")
    if not 1 <= int(normalization["postal_like_min_digits"]) <= int(normalization["postal_like_max_digits"]):
        raise ConfigError("normalization postal digit lengths are invalid")
    suffixes = normalization["legal_suffixes"]
    if not isinstance(suffixes, list) or not suffixes or len(suffixes) != len(set(suffixes)):
        raise ConfigError("normalization.legal_suffixes must be a non-empty unique list")
    exact_structured = config["retrieval"].get("exact_structured")
    if not isinstance(exact_structured, Mapping):
        raise ConfigError("retrieval.exact_structured must be a mapping")
    for name in (
        "exact_name_max_bucket", "exact_address_max_bucket",
        "structured_max_bucket", "rare_token_max_df",
    ):
        if int(exact_structured.get(name, 0)) <= 0:
            raise ConfigError(f"retrieval.exact_structured.{name} must be positive")
    lexical = config["retrieval"].get("multikey")
    if not isinstance(lexical, Mapping):
        raise ConfigError("retrieval.multikey must be a mapping")
    for name in (
        "target_batch_rows", "min_name_token_length", "min_address_token_length", "max_name_tokens_per_target",
        "max_address_tokens_per_target", "max_numeric_tokens_per_target", "rare_name_token_max_df",
        "rare_address_token_max_df", "rare_name_bucket_cap", "name_number_bucket_cap",
        "number_address_bucket_cap", "name_address_bucket_cap", "address_pair_bucket_cap",
        "translit_name_bucket_cap", "translit_name_number_bucket_cap", "blocker_max_candidates_per_target",
    ):
        if int(lexical.get(name, 0)) <= 0:
            raise ConfigError(f"retrieval.multikey.{name} must be positive")


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
