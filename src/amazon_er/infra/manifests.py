"""Shard manifest contract and validation."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ShardManifest:
    stage: str
    stage_version: str
    country: str | None
    source: str | None
    shard_id: str
    input_rows: int
    output_rows: int
    runtime_seconds: float
    peak_ram_bytes: int | None
    peak_vram_bytes: int | None
    config_hash: str
    code_commit: str
    input_fingerprint: str | None
    artifact_path: str
    artifact_size_bytes: int
    artifact_hash: str
    status: str
    created_at: str

    @classmethod
    def complete(cls, **kwargs: Any) -> "ShardManifest":
        kwargs.setdefault("status", "complete")
        kwargs.setdefault("created_at", datetime.now(timezone.utc).isoformat())
        return cls(**kwargs)

    def validate(self) -> list[str]:
        errors: list[str] = []
        for name in ("stage", "stage_version", "shard_id", "config_hash", "code_commit", "artifact_path", "artifact_hash", "created_at"):
            if not getattr(self, name):
                errors.append(f"{name} is required")
        if self.status not in {"pending", "running", "complete", "failed"}:
            errors.append("status is invalid")
        for name in ("input_rows", "output_rows", "artifact_size_bytes"):
            if getattr(self, name) < 0:
                errors.append(f"{name} must be non-negative")
        if self.runtime_seconds < 0:
            errors.append("runtime_seconds must be non-negative")
        return errors

    def write(self, path: str | Path) -> None:
        errors = self.validate()
        if errors:
            raise ValueError("Invalid manifest: " + "; ".join(errors))
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_suffix(target.suffix + ".tmp")
        temporary.write_text(json.dumps(asdict(self), indent=2, sort_keys=True) + "\n", encoding="utf-8")
        temporary.replace(target)


def load_manifest(path: str | Path) -> ShardManifest:
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        manifest = ShardManifest(**payload)
    except (OSError, json.JSONDecodeError, TypeError) as exc:
        raise ValueError(f"Cannot load manifest {path}: {exc}") from exc
    errors = manifest.validate()
    if errors:
        raise ValueError("Invalid manifest: " + "; ".join(errors))
    return manifest
