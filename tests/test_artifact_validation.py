import json

import pytest

from amazon_er.infra.artifacts import can_resume_shard, validate_artifact
from amazon_er.infra.hashing import hash_file
from amazon_er.infra.manifests import ShardManifest


def make_manifest(tmp_path, *, contents='{"rows": [1]}', config_hash="cfg", artifact_exists=True):
    artifact = tmp_path / "part.json"
    if artifact_exists:
        artifact.write_text(contents, encoding="utf-8")
    size = artifact.stat().st_size if artifact_exists else 10
    digest = hash_file(artifact) if artifact_exists and size else "missing"
    manifest = ShardManifest.complete(
        stage="retrieval", stage_version="v1", country="US", source="S2", shard_id="000",
        input_rows=1, output_rows=1, runtime_seconds=1.0, peak_ram_bytes=100,
        peak_vram_bytes=None, config_hash=config_hash, code_commit="abc123",
        input_fingerprint="input-v1",
        artifact_path=str(artifact), artifact_size_bytes=size, artifact_hash=digest,
    )
    path = tmp_path / "manifest.json"
    manifest.write(path)
    return path, artifact


def open_json(path):
    payload = json.loads(path.read_text(encoding="utf-8"))
    if "rows" not in payload:
        raise ValueError("rows field missing")


def test_valid_artifact_can_resume(tmp_path):
    manifest, _ = make_manifest(tmp_path)
    result = validate_artifact(manifest, expected_config_hash="cfg", expected_stage="retrieval", schema_validator=open_json)
    assert result.valid
    assert can_resume_shard(manifest, expected_config_hash="cfg", schema_validator=open_json)


def test_missing_artifact_cannot_resume(tmp_path):
    manifest, _ = make_manifest(tmp_path, artifact_exists=False)
    result = validate_artifact(manifest, expected_config_hash="cfg")
    assert not result.valid
    assert "artifact is missing" in result.reasons


def test_empty_artifact_cannot_resume(tmp_path):
    manifest, _ = make_manifest(tmp_path, contents="")
    result = validate_artifact(manifest, expected_config_hash="cfg")
    assert not result.valid
    assert "artifact is empty" in result.reasons


def test_corrupted_or_schema_mismatched_artifact_cannot_resume(tmp_path):
    manifest, artifact = make_manifest(tmp_path)
    artifact.write_text("corrupt", encoding="utf-8")
    result = validate_artifact(manifest, expected_config_hash="cfg", schema_validator=open_json)
    assert not result.valid
    assert "artifact size mismatch" in result.reasons or "artifact hash mismatch" in result.reasons
    assert any("schema/open" in reason for reason in result.reasons)


def test_config_hash_mismatch_cannot_resume(tmp_path):
    manifest, _ = make_manifest(tmp_path, config_hash="old")
    result = validate_artifact(manifest, expected_config_hash="new")
    assert not result.valid
    assert "config hash mismatch" in result.reasons


def test_input_fingerprint_mismatch_cannot_resume(tmp_path):
    manifest, _ = make_manifest(tmp_path)
    result = validate_artifact(
        manifest,
        expected_config_hash="cfg",
        expected_input_fingerprint="input-v2",
    )
    assert not result.valid
    assert "input fingerprint mismatch" in result.reasons
