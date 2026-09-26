"""Lightweight deterministic fingerprints for very large immutable inputs."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


FINGERPRINT_ALGORITHM = "sha256(size + header + first/middle/last sampled blocks); v1"


def sampled_file_fingerprint(path: str | Path, sample_bytes: int = 1024 * 1024) -> dict[str, object]:
    source = Path(path)
    if sample_bytes <= 0:
        raise ValueError("sample_bytes must be positive")
    size = source.stat().st_size
    offsets = sorted({0, max(0, size // 2 - sample_bytes // 2), max(0, size - sample_bytes)})
    digest = hashlib.sha256()
    digest.update(str(size).encode("ascii"))
    with source.open("rb") as stream:
        header = stream.readline()
        digest.update(len(header).to_bytes(8, "big"))
        digest.update(header)
        sampled: list[dict[str, int]] = []
        for offset in offsets:
            stream.seek(offset)
            block = stream.read(sample_bytes)
            digest.update(offset.to_bytes(8, "big"))
            digest.update(len(block).to_bytes(8, "big"))
            digest.update(block)
            sampled.append({"offset": offset, "bytes": len(block)})
    return {
        "input_fingerprint": digest.hexdigest(),
        "algorithm": FINGERPRINT_ALGORITHM,
        "file_size_bytes": size,
        "sample_bytes_requested": sample_bytes,
        "samples": sampled,
    }


def combined_fingerprint(fingerprints: dict[str, dict[str, object]]) -> str:
    payload = {name: item["input_fingerprint"] for name, item in sorted(fingerprints.items())}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
