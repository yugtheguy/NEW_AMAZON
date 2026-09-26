from amazon_er.infra.manifests import ShardManifest


def valid_manifest(**updates):
    values = dict(
        stage="unit", stage_version="v1", country="US", source="S2", shard_id="0",
        input_rows=2, output_rows=1, runtime_seconds=0.1, peak_ram_bytes=100,
        peak_vram_bytes=None, config_hash="abc", code_commit="deadbeef",
        input_fingerprint="input-v1",
        artifact_path="artifact.json", artifact_size_bytes=2, artifact_hash="hash",
        status="complete", created_at="2026-01-01T00:00:00+00:00",
    )
    values.update(updates)
    return ShardManifest(**values)


def test_valid_manifest():
    assert valid_manifest().validate() == []


def test_manifest_rejects_negative_counts_and_bad_status():
    errors = valid_manifest(output_rows=-1, status="done").validate()
    assert "output_rows must be non-negative" in errors
    assert "status is invalid" in errors
