"""Minimal registry for safe utility stages only."""

from __future__ import annotations

from typing import Any, Mapping

from amazon_er.config import config_hash
from amazon_er.infra.resources import ResourceMonitor
from amazon_er.infra.seed import set_global_seed


def run_stage(stage: str, config: Mapping[str, Any], *, country: str | None = None, source: str | None = None, shard_id: str | None = None) -> dict[str, Any]:
    if stage not in {"healthcheck", "config-check"}:
        raise ValueError(f"Stage {stage!r} is not implemented in Phase 0A")
    set_global_seed(int(config["runtime"]["seed"]))
    result: dict[str, Any] = {"stage": stage, "status": "ok", "config_hash": config_hash(config)}
    if stage == "healthcheck":
        result["resources"] = ResourceMonitor(stage, country=country, source=source, shard=shard_id).sample().to_json()
    return result
