"""Streaming, country-partitioned normalization with verified shard resume."""

from __future__ import annotations

import json
import os
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping
from urllib.parse import quote

from amazon_er.config import config_hash
from amazon_er.data.fingerprint import combined_fingerprint, sampled_file_fingerprint
from amazon_er.data.normalize import NORMALIZED_COLUMNS, normalize_record
from amazon_er.data.paths import DatasetPaths, resolve_dataset_paths
from amazon_er.infra.artifacts import validate_artifact
from amazon_er.infra.hashing import hash_file, hash_mapping
from amazon_er.infra.manifests import ShardManifest, load_manifest
from amazon_er.infra.resources import ResourceMonitor
from amazon_er.paths import project_root, resolve_project_path


SOURCE_MAP = {
    ("train", "source1"): "train_source1",
    ("train", "source2"): "train_source2",
    ("train", "source3"): "train_source3",
    ("test", "source1"): "test_source1",
    ("test", "source2"): "test_source2",
    ("test", "source3"): "test_source3",
}


class NormalizationError(RuntimeError):
    """Normalization or its validation failed."""


def _imports() -> tuple[Any, Any, Any]:
    try:
        import pyarrow as pa
        import pyarrow.csv as pacsv
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError("Normalization requires the data extra: pip install -e '.[data]'") from exc
    return pa, pacsv, pq


def normalized_schema() -> Any:
    pa, _, _ = _imports()
    string_fields = [name for name in NORMALIZED_COLUMNS if name not in {"name_missing", "address_missing"}]
    return pa.schema([
        *[pa.field(name, pa.string()) for name in string_fields],
        pa.field("name_missing", pa.bool_()),
        pa.field("address_missing", pa.bool_()),
    ])


def _code_commit() -> str:
    root = project_root()
    try:
        result = subprocess.run(
            ["git", "-c", f"safe.directory={root}", "rev-parse", "HEAD"],
            cwd=root, capture_output=True, text=True, timeout=5, check=True,
        )
        return result.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return os.environ.get("CODE_COMMIT", "UNAVAILABLE")


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def _append_jsonl(path: Path, payload: str, lock: threading.Lock) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with lock, path.open("a", encoding="utf-8") as stream:
        stream.write(payload + "\n")


def _artifact_paths(root: Path, split: str, source: str, country: str, shard_id: int) -> tuple[Path, Path]:
    directory = root / split / source / f"country={quote(country, safe='')}"
    artifact = directory / f"part-{shard_id:05d}.parquet"
    return artifact, artifact.with_suffix(".manifest.json")


def validate_normalized_parquet(path: Path, *, expected_rows: int, expected_country: str) -> None:
    pa, _, pq = _imports()
    parquet = pq.ParquetFile(path)
    if parquet.metadata.num_rows != expected_rows:
        raise ValueError(f"row count mismatch: {parquet.metadata.num_rows} != {expected_rows}")
    names = tuple(parquet.schema_arrow.names)
    if names != NORMALIZED_COLUMNS:
        raise ValueError(f"normalized schema mismatch: {names}")
    for batch in parquet.iter_batches(
        batch_size=65536,
        columns=["entity_id", "country", "name_raw", "name_nfkc", "address_raw", "address_nfkc"],
    ):
        entity_ids = batch.column(0)
        countries = batch.column(1)
        if entity_ids.null_count or countries.null_count:
            raise ValueError("entity_id and country must be non-null")
        if any(value != expected_country for value in countries.to_pylist()):
            raise ValueError("artifact contains the wrong country partition")
        name_raw, name_nfkc = batch.column(2).to_pylist(), batch.column(3).to_pylist()
        address_raw, address_nfkc = batch.column(4).to_pylist(), batch.column(5).to_pylist()
        for raw, view in zip(name_raw, name_nfkc):
            if raw is None and view.casefold() in {"nan", "none", "null"}:
                raise ValueError("missing name became a literal missing-value token")
        for raw, view in zip(address_raw, address_nfkc):
            if raw is None and view.casefold() in {"nan", "none", "null"}:
                raise ValueError("missing address became a literal missing-value token")


@dataclass
class _Writer:
    artifact: Path
    manifest_path: Path
    temporary: Path
    writer: Any
    split: str
    source: str
    country: str
    shard_id: int
    started: float
    rows: int = 0
    peak_ram_bytes: int = 0


def _open_writer(root: Path, split: str, source: str, country: str, shard_id: int, config: Mapping[str, Any]) -> _Writer:
    _, _, pq = _imports()
    artifact, manifest_path = _artifact_paths(root, split, source, country, shard_id)
    artifact.parent.mkdir(parents=True, exist_ok=True)
    temporary = artifact.with_suffix(".parquet.tmp")
    writer = pq.ParquetWriter(
        temporary, normalized_schema(), compression=config["normalization"]["parquet_compression"],
        compression_level=int(config["normalization"]["parquet_compression_level"]),
        use_dictionary=["country"], write_statistics=True,
    )
    return _Writer(artifact, manifest_path, temporary, writer, split, source, country, shard_id, time.monotonic())


def _finish_writer(
    state: _Writer, *, config: Mapping[str, Any], dependency_hash: str, code_commit: str,
) -> None:
    state.writer.close()
    state.temporary.replace(state.artifact)
    validate_normalized_parquet(state.artifact, expected_rows=state.rows, expected_country=state.country)
    manifest = ShardManifest.complete(
        stage="normalization", stage_version=str(config["normalization"]["stage_version"]),
        split=state.split, country=state.country, source=state.source, shard_id=str(state.shard_id),
        input_rows=state.rows, output_rows=state.rows,
        runtime_seconds=time.monotonic() - state.started,
        peak_ram_bytes=state.peak_ram_bytes, peak_vram_bytes=None,
        config_hash=config_hash(config), code_commit=code_commit, input_fingerprint=dependency_hash,
        artifact_path=str(state.artifact.resolve()), artifact_size_bytes=state.artifact.stat().st_size,
        artifact_hash=hash_file(state.artifact),
    )
    manifest.write(state.manifest_path)


def _valid_existing_shard(
    manifest_path: Path, *, config: Mapping[str, Any], dependency_hash: str,
    split: str, source: str, country: str, shard_id: int,
) -> bool:
    if not manifest_path.exists():
        return False
    try:
        manifest = load_manifest(manifest_path)
    except ValueError:
        return False
    if (manifest.split, manifest.source, manifest.country, manifest.shard_id) != (
        split, source, country, str(shard_id),
    ):
        return False
    validator = lambda path: validate_normalized_parquet(
        path, expected_rows=manifest.input_rows, expected_country=country,
    )
    return validate_artifact(
        manifest_path, expected_config_hash=config_hash(config), expected_stage="normalization",
        expected_stage_version=str(config["normalization"]["stage_version"]),
        expected_input_fingerprint=dependency_hash, schema_validator=validator,
    ).valid


def _fingerprints(paths: DatasetPaths, config: Mapping[str, Any]) -> tuple[dict[str, dict[str, object]], str]:
    sample_bytes = int(config["audit"]["fingerprint_sample_bytes"])
    values = {name: sampled_file_fingerprint(path, sample_bytes) for name, path in paths.sources().items()}
    combined = combined_fingerprint(values)
    expected = config["normalization"].get("expected_dataset_fingerprint")
    if expected and combined != expected:
        raise NormalizationError(f"dataset fingerprint mismatch: {combined} != {expected}")
    return values, combined


def _implementation_hash() -> str:
    return hash_mapping({
        "normalization_views": hash_file(project_root() / "src" / "amazon_er" / "data" / "normalize.py"),
        "normalization_pipeline": hash_file(Path(__file__)),
    })


def _selected_inputs(split: str | None, source: str | None) -> list[tuple[str, str, str]]:
    selected: list[tuple[str, str, str]] = []
    for (candidate_split, candidate_source), logical_name in SOURCE_MAP.items():
        if split is not None and split != candidate_split:
            continue
        if source is not None and source != candidate_source:
            continue
        selected.append((candidate_split, candidate_source, logical_name))
    if not selected:
        raise NormalizationError("No source files match the requested split/source filters")
    return selected


def _normalize_source(
    *, input_path: Path, logical_name: str, split: str, source: str,
    output_root: Path, config: Mapping[str, Any], dependency_hash: str,
    country_filter: str | None, shard_filter: int | None,
    monitor: ResourceMonitor, code_commit: str,
) -> dict[str, Any]:
    pa, pacsv, _ = _imports()
    behavior = config["normalization"]
    reader = pacsv.open_csv(
        input_path,
        read_options=pacsv.ReadOptions(block_size=int(behavior["input_block_size_bytes"]), use_threads=False),
        parse_options=pacsv.ParseOptions(delimiter="\t", quote_char=False, newlines_in_values=False),
        convert_options=pacsv.ConvertOptions(
            column_types={name: pa.string() for name in ("entity_id", "business_name", "business_address", "country")},
            strings_can_be_null=False, null_values=[],
        ),
    )
    schema = normalized_schema()
    shard_rows = int(behavior["shard_rows"])
    positions: dict[str, int] = {}
    input_counts: dict[str, int] = {}
    writers: dict[tuple[str, int], _Writer] = {}
    valid_cache: dict[tuple[str, int], bool] = {}
    written, skipped, processed = 0, 0, 0
    peak_rss = 0
    try:
        for batch in reader:
            values = {
                name: batch.column(batch.schema.get_field_index(name)).to_pylist()
                for name in ("entity_id", "business_name", "business_address", "country")
            }
            grouped: dict[tuple[str, int], list[tuple[Any, Any, Any, Any]]] = {}
            for row in zip(values["entity_id"], values["business_name"], values["business_address"], values["country"]):
                entity_id, business_name, business_address, country = row
                if country is None or not country:
                    raise NormalizationError(f"{logical_name} contains null/empty country")
                offset = positions.get(country, 0)
                current_shard = offset // shard_rows
                positions[country] = offset + 1
                input_counts[country] = input_counts.get(country, 0) + 1
                if country_filter is not None and country != country_filter:
                    continue
                if shard_filter is not None and current_shard != shard_filter:
                    continue
                grouped.setdefault((country, current_shard), []).append(row)
            for (country, current_shard), rows in grouped.items():
                key = (country, current_shard)
                artifact, manifest_path = _artifact_paths(output_root, split, source, country, current_shard)
                if key not in valid_cache:
                    valid_cache[key] = _valid_existing_shard(
                        manifest_path, config=config, dependency_hash=dependency_hash,
                        split=split, source=source, country=country, shard_id=current_shard,
                    )
                    if valid_cache[key]:
                        skipped += 1
                if valid_cache[key]:
                    continue
                state = writers.get(key)
                if state is None:
                    state = _open_writer(output_root, split, source, country, current_shard, config)
                    writers[key] = state
                columns: dict[str, list[Any]] = {name: [] for name in NORMALIZED_COLUMNS}
                for entity_id, business_name, business_address, row_country in rows:
                    normalized = normalize_record(
                        entity_id=entity_id, business_name=business_name,
                        business_address=business_address, country=row_country,
                        legal_suffixes=behavior["legal_suffixes"],
                        postal_like_min_digits=int(behavior["postal_like_min_digits"]),
                        postal_like_max_digits=int(behavior["postal_like_max_digits"]),
                    )
                    for name in NORMALIZED_COLUMNS:
                        columns[name].append(normalized[name])
                table = pa.Table.from_pydict(columns, schema=schema)
                state.writer.write_table(table, row_group_size=min(65536, len(rows)))
                state.rows += len(rows)
                processed += len(rows)
                sample = monitor.sample(processed_rows=processed)
                peak_rss = max(peak_rss, sample.rss_bytes)
                state.peak_ram_bytes = max(state.peak_ram_bytes, sample.rss_bytes)
                if state.rows == shard_rows:
                    _finish_writer(state, config=config, dependency_hash=dependency_hash, code_commit=code_commit)
                    del writers[key]
                    written += 1
                elif state.rows > shard_rows:
                    raise NormalizationError("internal shard row overflow")
                if sample.ram_fraction >= float(config["runtime"]["critical_ram_fraction"]):
                    raise MemoryError("RAM reached the configured critical fraction after a safe batch boundary")
        for key, state in list(writers.items()):
            _finish_writer(state, config=config, dependency_hash=dependency_hash, code_commit=code_commit)
            del writers[key]
            written += 1
    finally:
        for state in writers.values():
            try:
                state.writer.close()
            except Exception:
                pass
    return {
        "split": split, "source": source, "input_counts": input_counts,
        "rows_processed": processed, "shards_written": written,
        "shards_skipped": skipped, "peak_rss_bytes": peak_rss,
    }


def _manifest_inventory(
    root: Path, config: Mapping[str, Any], dependency_hashes: Mapping[tuple[str, str], str],
) -> tuple[list[ShardManifest], int]:
    manifests: list[ShardManifest] = []
    total_size = 0
    for manifest_path in sorted(root.rglob("part-*.manifest.json")):
        manifest = load_manifest(manifest_path)
        dependency_hash = dependency_hashes.get((str(manifest.split), str(manifest.source)))
        if dependency_hash is None:
            raise NormalizationError(f"unexpected split/source manifest: {manifest_path}")
        validator = lambda path, item=manifest: validate_normalized_parquet(
            path, expected_rows=item.input_rows, expected_country=str(item.country),
        )
        result = validate_artifact(
            manifest_path, expected_config_hash=config_hash(config), expected_stage="normalization",
            expected_stage_version=str(config["normalization"]["stage_version"]),
            expected_input_fingerprint=dependency_hash, schema_validator=validator,
        )
        if not result.valid:
            raise NormalizationError(f"invalid normalized shard {manifest_path}: {result.reasons}")
        manifests.append(manifest)
        total_size += manifest.artifact_size_bytes
    if not manifests:
        raise NormalizationError("no normalized shard manifests were found")
    return manifests, total_size


def _deterministic_samples(artifact_paths: list[Path], limit_per_category: int = 4) -> list[dict[str, Any]]:
    _, _, pq = _imports()
    categories = (
        "ascii", "non_ascii", "france_accented", "devanagari", "other_indic",
        "punctuation", "missing_address", "numeric_address",
    )
    samples: dict[str, list[dict[str, Any]]] = {category: [] for category in categories}
    columns = [
        "entity_id", "country", "name_raw", "name_casefold", "name_accent_fold",
        "name_compact", "name_core", "name_transliterated", "address_raw", "address_missing",
        "numeric_tokens",
    ]
    for path in sorted(artifact_paths):
        for batch in pq.ParquetFile(path).iter_batches(batch_size=2048, columns=columns):
            for row in batch.to_pylist():
                raw = row["name_raw"] or ""
                selected: list[str] = []
                if raw.isascii():
                    selected.append("ascii")
                else:
                    selected.append("non_ascii")
                if row["country"] == "France" and any("\u00c0" <= char <= "\u024f" for char in raw):
                    selected.append("france_accented")
                if any("\u0900" <= char <= "\u097f" for char in raw):
                    selected.append("devanagari")
                elif any("\u0980" <= char <= "\u0dff" for char in raw):
                    selected.append("other_indic")
                if any(not char.isalnum() and not char.isspace() for char in raw):
                    selected.append("punctuation")
                if row["address_missing"]:
                    selected.append("missing_address")
                if row["numeric_tokens"].count("|") >= 2:
                    selected.append("numeric_address")
                for category in selected:
                    if len(samples[category]) < limit_per_category:
                        samples[category].append({"category": category, **row})
            if all(len(items) >= limit_per_category for items in samples.values()):
                break
        if all(len(items) >= limit_per_category for items in samples.values()):
            break
    return [item for category in categories for item in samples[category]]


def _write_summary(
    root: Path, report_root: Path, config: Mapping[str, Any],
    dependency_hashes: Mapping[tuple[str, str], str], runtime_seconds: float, peak_rss_bytes: int,
) -> dict[str, Any]:
    try:
        import duckdb
    except ImportError as exc:
        raise RuntimeError("Normalization summary requires DuckDB from the data extra") from exc
    manifests, total_size = _manifest_inventory(root, config, dependency_hashes)
    counts: dict[str, dict[str, dict[str, int]]] = {}
    for item in manifests:
        counts.setdefault(str(item.split), {}).setdefault(str(item.source), {})
        target = counts[str(item.split)][str(item.source)]
        target[str(item.country)] = target.get(str(item.country), 0) + item.output_rows
    expected = config["normalization"].get("expected_country_rows")
    if expected is not None and counts != expected:
        raise NormalizationError(f"normalized country counts do not match Phase 0B: {counts}")
    connection = duckdb.connect(":memory:")
    connection.execute("SET memory_limit='1GB'")
    connection.execute("SET threads=2")
    connection.execute("SET preserve_insertion_order=false")
    stats: dict[str, Any] = {}
    rows_table: list[dict[str, Any]] = []
    try:
        for split, sources in sorted(counts.items()):
            for source, countries in sorted(sources.items()):
                pattern = str((root / split / source / "country=*" / "part-*.parquet").resolve()).replace("\\", "/").replace("'", "''")
                row = connection.execute(f"""
                    SELECT count(*) AS rows, count(DISTINCT entity_id) AS unique_ids,
                           sum(CAST(name_missing AS BIGINT)) AS name_missing,
                           sum(CAST(address_missing AS BIGINT)) AS address_missing,
                           count(*) FILTER (WHERE name_accent_fold <> name_casefold) AS accent_changed,
                           count(*) FILTER (WHERE name_transliterated <> name_tokens) AS transliteration_changed,
                           count(*) FILTER (WHERE name_core <> name_tokens) AS core_changed,
                           avg(length(name_tokens)) AS avg_name_length, max(length(name_tokens)) AS max_name_length,
                           avg(length(address_normalized)) AS avg_address_length,
                           max(length(address_normalized)) AS max_address_length
                    FROM read_parquet('{pattern}', hive_partitioning=false)
                """).fetchone()
                names = [item[0] for item in connection.description]
                values = dict(zip(names, row))
                if int(values["rows"]) != int(values["unique_ids"]):
                    raise NormalizationError(f"entity IDs are not unique in {split}/{source}")
                stats[f"{split}/{source}"] = values
                for country, output_rows in sorted(countries.items()):
                    rows_table.append({
                        "split": split, "source": source, "country": country,
                        "input_rows": output_rows, "output_rows": output_rows,
                    })
    finally:
        connection.close()
    artifacts = [Path(item.artifact_path) for item in manifests]
    samples = _deterministic_samples(artifacts)
    summary = {
        "normalization_version": config["normalization"]["contract_version"],
        "rows": rows_table, "statistics": stats,
        "total_artifact_size_bytes": total_size,
        "runtime_seconds": runtime_seconds, "peak_rss_bytes": peak_rss_bytes,
        "cardinality_verified": True, "country_counts_verified": expected is None or counts == expected,
    }
    _atomic_json(report_root / "summary.json", summary)
    _atomic_json(report_root / "samples.json", samples)
    lines = ["# Normalization Summary", "", "| split | source | country | input rows | output rows |", "|---|---|---|---:|---:|"]
    lines.extend(
        f"| {row['split']} | {row['source']} | {row['country']} | {row['input_rows']} | {row['output_rows']} |"
        for row in rows_table
    )
    lines.extend(["", f"Total artifact size: {total_size:,} bytes", f"Peak RSS: {peak_rss_bytes:,} bytes", ""])
    (report_root / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    sample_lines = ["# Deterministic Normalization Samples", ""]
    for item in samples:
        sample_lines.extend([
            f"## {item['category']} — {item['entity_id']}", "",
            f"- Raw: `{item['name_raw']}`", f"- Casefold: `{item['name_casefold']}`",
            f"- Accent fold: `{item['name_accent_fold']}`", f"- Compact: `{item['name_compact']}`",
            f"- Core: `{item['name_core']}`", f"- Transliterated: `{item['name_transliterated']}`", "",
        ])
    (report_root / "samples.md").write_text("\n".join(sample_lines), encoding="utf-8")
    return summary


def run_normalization(
    config: Mapping[str, Any], *, data_root: str | Path | None = None,
    output_dir: str | Path | None = None, split: str | None = None,
    source: str | None = None, country: str | None = None, shard_id: str | None = None,
) -> dict[str, Any]:
    if split not in {None, "train", "test"}:
        raise NormalizationError("split must be train or test")
    if source not in {None, "source1", "source2", "source3"}:
        raise NormalizationError("source must be source1, source2, or source3")
    try:
        shard_filter = None if shard_id is None else int(shard_id)
    except ValueError as exc:
        raise NormalizationError("shard-id must be a non-negative integer") from exc
    if shard_filter is not None and shard_filter < 0:
        raise NormalizationError("shard-id must be a non-negative integer")
    paths = resolve_dataset_paths(config, data_root)
    fingerprints, dataset_fingerprint = _fingerprints(paths, config)
    implementation_hash = _implementation_hash()
    dependency_hashes = {
        (item_split, item_source): hash_mapping({
            "dataset_fingerprint": dataset_fingerprint,
            "source_fingerprint": fingerprints[logical_name]["input_fingerprint"],
            "normalization": config["normalization"],
            "implementation_hash": implementation_hash,
        })
        for (item_split, item_source), logical_name in SOURCE_MAP.items()
    }
    version = str(config["normalization"]["stage_version"])
    output_root = (
        resolve_project_path(config, "artifact_root") / "normalized" / version
        if output_dir is None else Path(output_dir).resolve()
    )
    report_root = output_root.parent.parent / "reports" / "normalization" / version
    monitor = ResourceMonitor("normalization", country=country, source=source, shard=shard_id)
    resource_log = report_root / "resources.jsonl"
    resource_lock = threading.Lock()
    heartbeat_peak = 0
    def record_resource(sample: Any) -> None:
        nonlocal heartbeat_peak
        heartbeat_peak = max(heartbeat_peak, sample.rss_bytes)
        _append_jsonl(resource_log, sample.to_json(), resource_lock)
    heartbeat_stop, heartbeat_thread = monitor.start_heartbeat(
        float(config["runtime"]["heartbeat_seconds"]), record_resource,
    )
    started = time.monotonic()
    code_commit = _code_commit()
    results: list[dict[str, Any]] = []
    peak_rss = 0
    try:
        for current_split, current_source, logical_name in _selected_inputs(split, source):
            print(f"[NORMALIZATION] {current_split}/{current_source}", flush=True)
            result = _normalize_source(
                input_path=getattr(paths, logical_name), logical_name=logical_name,
                split=current_split, source=current_source, output_root=output_root,
                config=config, dependency_hash=dependency_hashes[(current_split, current_source)],
                country_filter=country,
                shard_filter=shard_filter, monitor=monitor, code_commit=code_commit,
            )
            peak_rss = max(peak_rss, int(result["peak_rss_bytes"]))
            results.append(result)
            sample = monitor.sample(processed_rows=int(result["rows_processed"]))
            record_resource(sample)
            print(
                f"[NORMALIZATION] complete {current_split}/{current_source} "
                f"written={result['shards_written']} skipped={result['shards_skipped']} "
                f"rss={sample.rss_bytes}",
                flush=True,
            )
    finally:
        heartbeat_stop.set()
        heartbeat_thread.join(timeout=10)
    peak_rss = max(peak_rss, heartbeat_peak)
    response: dict[str, Any] = {
        "status": "complete", "normalization_version": config["normalization"]["contract_version"],
        "dataset_fingerprint": dataset_fingerprint, "sources": results,
        "runtime_seconds": time.monotonic() - started, "peak_rss_bytes": peak_rss,
        "output_root": str(output_root),
    }
    if split is None and source is None and country is None and shard_filter is None:
        response["summary"] = _write_summary(
            output_root, report_root, config, dependency_hashes,
            response["runtime_seconds"], peak_rss,
        )
    return response
