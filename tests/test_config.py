from copy import deepcopy

import pytest

from amazon_er.config import ConfigError, config_hash, load_config, validate_config


def test_config_loading_and_override():
    config = load_config("configs/smoke.yaml")
    assert config["project"]["profile"] == "smoke"
    assert config["runtime"]["seed"] == 2026
    assert config["paths"]["artifact_root"] == "artifacts/smoke"


def test_config_hash_is_order_independent():
    config = load_config("configs/base.yaml")
    reversed_config = dict(reversed(list(config.items())))
    assert config_hash(config) == config_hash(reversed_config)


def test_config_hash_changes_with_value():
    config = load_config("configs/base.yaml")
    changed = deepcopy(config)
    changed["runtime"]["seed"] += 1
    assert config_hash(config) != config_hash(changed)


def test_invalid_memory_thresholds():
    config = load_config("configs/base.yaml")
    config["runtime"]["warning_ram_fraction"] = 0.9
    with pytest.raises(ConfigError):
        validate_config(config)
