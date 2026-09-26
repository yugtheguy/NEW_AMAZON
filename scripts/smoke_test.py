"""Seconds-scale infrastructure smoke test; no competition data required."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from amazon_er.candidates.schema import validate_candidates
from amazon_er.config import config_hash, load_config
from amazon_er.infra.artifacts import validate_artifact
from amazon_er.infra.hashing import hash_file
from amazon_er.infra.manifests import ShardManifest
from amazon_er.infra.resources import ResourceMonitor
from amazon_er.metrics.fbeta import macro_entity_fbeta


def main() -> int:
    config = load_config("configs/smoke.yaml")
    digest = config_hash(config)
    candidates = validate_candidates([{
        "target_entity_id": "t1", "candidate_s1_entity_id": "s1",
        "target_source": "S2", "country": "US", "dense_present": False,
        "dense_score": None, "dense_rank": None,
    }])
    assert candidates.valid
    assert macro_entity_fbeta({"s1": {"t1"}, "s2": set()}, {"s1": {"t1"}}, ["s1", "s2"]) == 1.0
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        artifact = root / "artifact.json"
        artifact.write_text(json.dumps({"ok": True}), encoding="utf-8")
        manifest = ShardManifest.complete(
            stage="smoke", stage_version="v1", country=None, source=None, shard_id="0",
            input_rows=1, output_rows=1, runtime_seconds=0.01, peak_ram_bytes=None,
            peak_vram_bytes=None, config_hash=digest, code_commit="UNCOMMITTED",
            input_fingerprint="synthetic-v1",
            artifact_path=str(artifact), artifact_size_bytes=artifact.stat().st_size,
            artifact_hash=hash_file(artifact),
        )
        manifest_path = root / "manifest.json"
        manifest.write(manifest_path)
        assert validate_artifact(manifest_path, expected_config_hash=digest).valid
    sample = ResourceMonitor("smoke").sample(processed_rows=1, total_rows=1)
    assert sample.rss_bytes > 0
    print(json.dumps({"status": "ok", "config_hash": digest, "gpu_available": sample.gpu_available}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
