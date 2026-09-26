"""Minimal registry for safe utility stages only."""

from __future__ import annotations

from typing import Any, Mapping

from amazon_er.config import config_hash
from amazon_er.data.audit import run_data_audit
from amazon_er.infra.resources import ResourceMonitor
from amazon_er.infra.seed import set_global_seed
from amazon_er.pipeline.normalization import run_normalization


def run_stage(
    stage: str, config: Mapping[str, Any], *, country: str | None = None,
    source: str | None = None, shard_id: str | None = None,
    data_root: str | None = None, output_dir: str | None = None, split: str | None = None,
) -> dict[str, Any]:
    if stage not in {"healthcheck", "config-check", "data-audit", "normalize"}:
        raise ValueError(f"Stage {stage!r} is not implemented")
    set_global_seed(int(config["runtime"]["seed"]))
    if stage == "data-audit":
        return run_data_audit(config, data_root=data_root, output_dir=output_dir)
    if stage == "normalize":
        return run_normalization(
            config, data_root=data_root, output_dir=output_dir, split=split,
            source=source, country=country, shard_id=shard_id,
        )
    result: dict[str, Any] = {"stage": stage, "status": "ok", "config_hash": config_hash(config)}
    if stage == "healthcheck":
        result["resources"] = ResourceMonitor(stage, country=country, source=source, shard=shard_id).sample().to_json()
    return result
