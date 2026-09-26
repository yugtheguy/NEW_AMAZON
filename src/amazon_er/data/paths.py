"""Canonical discovery of the seven competition inputs."""

from __future__ import annotations

import os
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, Mapping

from amazon_er.paths import project_root


EXPECTED_FILES = {
    "train_source1": "train_source1.tsv",
    "train_source2": "train_source2.tsv",
    "train_source3": "train_source3.tsv",
    "train_ground_truth": "train_ground_truth.tsv",
    "test_source1": "test_source1.tsv",
    "test_source2": "test_source2.tsv",
    "test_source3": "test_source3.tsv",
}


class DatasetDiscoveryError(FileNotFoundError):
    """The complete dataset could not be resolved without ambiguity."""


@dataclass(frozen=True)
class DatasetPaths:
    root: Path
    train_source1: Path
    train_source2: Path
    train_source3: Path
    train_ground_truth: Path
    test_source1: Path
    test_source2: Path
    test_source3: Path

    def sources(self) -> dict[str, Path]:
        return {field.name: getattr(self, field.name) for field in fields(self) if field.name != "root"}


def _resolve_inside(root: Path) -> DatasetPaths:
    if not root.exists() or not root.is_dir():
        raise DatasetDiscoveryError(f"Dataset root does not exist or is not a directory: {root}")
    resolved: dict[str, Path] = {}
    problems: list[str] = []
    for logical_name, filename in EXPECTED_FILES.items():
        matches = sorted(path.resolve() for path in root.rglob(filename) if path.is_file())
        if not matches:
            problems.append(f"missing {filename}")
        elif len(matches) > 1:
            problems.append(f"ambiguous {filename}: {len(matches)} copies under {root}")
        else:
            resolved[logical_name] = matches[0]
    if problems:
        expected = ", ".join(EXPECTED_FILES.values())
        raise DatasetDiscoveryError(f"Cannot resolve dataset under {root}: {'; '.join(problems)}. Expected: {expected}")
    return DatasetPaths(root=root.resolve(), **resolved)


def resolve_dataset_paths(config: Mapping[str, Any], data_root: str | Path | None = None) -> DatasetPaths:
    """Resolve CLI, environment, configured, then repository-local roots in that order."""
    if data_root is not None:
        return _resolve_inside(Path(data_root).expanduser().resolve())
    environment = os.environ.get("AMAZON_DATA_ROOT")
    if environment:
        return _resolve_inside(Path(environment).expanduser().resolve())
    configured = Path(config["paths"]["data_root"])
    if not configured.is_absolute():
        configured = project_root() / configured
    candidates = [configured.resolve(), (project_root() / "data" / "raw").resolve()]
    errors: list[str] = []
    seen: set[Path] = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        try:
            return _resolve_inside(candidate)
        except DatasetDiscoveryError as exc:
            errors.append(str(exc))
    raise DatasetDiscoveryError("Dataset discovery failed. " + " | ".join(errors))
