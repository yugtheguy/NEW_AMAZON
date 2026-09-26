"""Artifact integrity and conservative resume decisions."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from amazon_er.infra.hashing import hash_file
from amazon_er.infra.manifests import load_manifest


@dataclass(frozen=True)
class ValidationResult:
    valid: bool
    reasons: tuple[str, ...]


def validate_artifact(
    manifest_path: str | Path,
    *,
    expected_config_hash: str,
    expected_stage: str | None = None,
    expected_stage_version: str | None = None,
    expected_input_fingerprint: str | None = None,
    schema_validator: Callable[[Path], None] | None = None,
) -> ValidationResult:
    reasons: list[str] = []
    try:
        manifest = load_manifest(manifest_path)
    except ValueError as exc:
        return ValidationResult(False, (str(exc),))
    if manifest.status != "complete":
        reasons.append("manifest status is not complete")
    if manifest.config_hash != expected_config_hash:
        reasons.append("config hash mismatch")
    if expected_stage is not None and manifest.stage != expected_stage:
        reasons.append("stage mismatch")
    if expected_stage_version is not None and manifest.stage_version != expected_stage_version:
        reasons.append("stage version mismatch")
    if expected_input_fingerprint is not None and manifest.input_fingerprint != expected_input_fingerprint:
        reasons.append("input fingerprint mismatch")
    artifact = Path(manifest.artifact_path)
    if not artifact.exists() or not artifact.is_file():
        reasons.append("artifact is missing")
        return ValidationResult(False, tuple(reasons))
    size = artifact.stat().st_size
    if size <= 0:
        reasons.append("artifact is empty")
    if size != manifest.artifact_size_bytes:
        reasons.append("artifact size mismatch")
    if size > 0 and hash_file(artifact) != manifest.artifact_hash:
        reasons.append("artifact hash mismatch")
    if schema_validator is not None and size > 0:
        try:
            schema_validator(artifact)
        except Exception as exc:  # validators expose format-specific failures
            reasons.append(f"artifact schema/open validation failed: {exc}")
    return ValidationResult(not reasons, tuple(reasons))


def can_resume_shard(*args, **kwargs) -> bool:
    return validate_artifact(*args, **kwargs).valid
