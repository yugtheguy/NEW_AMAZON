"""Deterministic target-to-S1 exact and structured retrieval."""

from __future__ import annotations

import json
import os
import subprocess
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping
from urllib.parse import quote

from amazon_er.config import config_hash
from amazon_er.data.paths import resolve_dataset_paths
from amazon_er.infra.artifacts import validate_artifact
from amazon_er.infra.hashing import hash_file, hash_mapping
from amazon_er.infra.manifests import ShardManifest, load_manifest
from amazon_er.infra.resources import ResourceMonitor
from amazon_er.paths import project_root, resolve_project_path
from amazon_er.pipeline.normalization import validate_normalized_parquet


STAGE = "exact_structured"
SIGNALS = (
    (1, "exact_name_compact", "exact", "name_compact", "exact_name_max_bucket"),
    (2, "exact_name_token_sorted", "exact", "name_token_sorted", "exact_name_max_bucket"),
    (3, "exact_name_core", "exact", "name_core", "exact_name_max_bucket"),
    (4, "exact_address_compact", "exact", "address_compact", "exact_address_max_bucket"),
    (5, "structured_name_core_number", "structured", "name_core", "structured_max_bucket"),
    (6, "structured_name_compact_number", "structured", "name_compact", "structured_max_bucket"),
    (7, "structured_name_core_postal", "structured", "name_core", "structured_max_bucket"),
    (8, "structured_name_compact_postal", "structured", "name_compact", "structured_max_bucket"),
    (9, "structured_name_numeric_signature", "structured", "name_core", "structured_max_bucket"),
)
PROVENANCE = tuple(item[1] for item in SIGNALS)
CANDIDATE_COLUMNS = (
    "target_entity_id", "candidate_s1_entity_id", "target_source", "country",
    "exact_hit", "structured_hit", *PROVENANCE, "retriever_mask", "retriever_count",
)


class ExactStructuredError(RuntimeError):
    """Exact/structured retrieval or validation failed."""


@dataclass(frozen=True)
class NormalizedShard:
    path: Path
    manifest_path: Path
    split: str
    source: str
    country: str
    shard_id: str
    rows: int
    input_fingerprint: str
    artifact_hash: str
    normalization_commit: str


def _imports() -> tuple[Any, Any]:
    try:
        import duckdb
        import pyarrow as pa
    except ImportError as exc:
        raise RuntimeError("Retrieval requires the data extra: pip install -e '.[data]'") from exc
    return duckdb, pa


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _append_jsonl(path: Path, payload: str, lock: threading.Lock) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with lock, path.open("a", encoding="utf-8") as stream:
        stream.write(payload + "\n")


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


def _parquet_for_manifest(path: Path) -> Path:
    suffix = ".manifest.json"
    if not path.name.endswith(suffix):
        raise ExactStructuredError(f"Unexpected normalized manifest name: {path}")
    return path.with_name(path.name[:-len(suffix)] + ".parquet")


def _discover_normalized(root: Path) -> list[NormalizedShard]:
    if not root.is_dir():
        raise ExactStructuredError(f"Normalized root does not exist: {root}")
    shards: list[NormalizedShard] = []
    for manifest_path in sorted(root.rglob("part-*.manifest.json")):
        manifest = load_manifest(manifest_path)
        if manifest.stage != "normalization" or manifest.stage_version != "v1":
            continue
        if manifest.split not in {"train", "test"} or manifest.source not in {"source1", "source2", "source3"}:
            continue
        if not manifest.country or not manifest.input_fingerprint:
            raise ExactStructuredError(f"Incomplete normalized manifest identity: {manifest_path}")
        artifact = _parquet_for_manifest(manifest_path)
        if not artifact.is_file() or artifact.stat().st_size != manifest.artifact_size_bytes:
            raise ExactStructuredError(f"Missing or size-mismatched normalized artifact: {artifact}")
        actual_hash = hash_file(artifact)
        if actual_hash != manifest.artifact_hash:
            raise ExactStructuredError(f"Hash-mismatched normalized artifact: {artifact}")
        validate_normalized_parquet(
            artifact, expected_rows=manifest.output_rows, expected_country=manifest.country,
        )
        shards.append(NormalizedShard(
            artifact, manifest_path, manifest.split, manifest.source, manifest.country,
            manifest.shard_id, manifest.output_rows, manifest.input_fingerprint,
            manifest.artifact_hash, manifest.code_commit,
        ))
    if not shards:
        raise ExactStructuredError(f"No valid normalization_v1 shards found below {root}")
    identities = [(s.split, s.source, s.country, s.shard_id) for s in shards]
    if len(identities) != len(set(identities)):
        raise ExactStructuredError("Duplicate normalized shard identities found")
    return shards


def _q(value: str | Path) -> str:
    return str(value).replace("\\", "/").replace("'", "''")


def _paths_sql(paths: list[Path]) -> str:
    return "[" + ",".join(f"'{_q(path)}'" for path in paths) + "]"


def _signal_key(signal: int, alias: str) -> tuple[str, str]:
    if signal <= 4:
        column = SIGNALS[signal - 1][3]
        return f"{alias}.{column}", f"{alias}.{column} <> ''"
    if signal in {5, 6}:
        name = "name_core" if signal == 5 else "name_compact"
        return (
            f"{alias}.{name} || chr(31) || {alias}.primary_number",
            f"{alias}.{name} <> '' AND coalesce({alias}.primary_number, '') <> ''",
        )
    if signal in {7, 8}:
        name = "name_core" if signal == 7 else "name_compact"
        return (
            f"{alias}.{name} || chr(31) || {alias}.postal_like_tokens",
            f"{alias}.{name} <> '' AND {alias}.postal_like_tokens <> ''",
        )
    return (
        f"{alias}.name_core || chr(31) || {alias}.numeric_tokens",
        f"{alias}.name_core <> '' AND {alias}.numeric_tokens <> ''",
    )


def _long_union(alias: str, id_column: str, caps: Mapping[str, Any]) -> str:
    parts: list[str] = []
    for signal, _, family, _, cap_name in SIGNALS:
        key, valid = _signal_key(signal, alias)
        parts.append(
            f"SELECT {alias}.{id_column} AS entity_id, {signal}::UTINYINT AS signal, "
            f"'{family}'::VARCHAR AS family, ({key})::VARCHAR AS key, "
            f"{int(caps[cap_name])}::UINTEGER AS cap FROM {alias} WHERE {valid}"
        )
    return " UNION ALL ".join(parts)


def _configure_connection(connection: Any, config: Mapping[str, Any]) -> None:
    limit = str(config["audit"]["duckdb_memory_limit"]).replace("'", "")
    connection.execute(f"SET memory_limit='{limit}'")
    connection.execute(f"SET threads={int(config['audit']['duckdb_threads'])}")
    connection.execute("SET preserve_insertion_order=false")


def _prepare_s1(connection: Any, paths: list[Path], caps: Mapping[str, Any]) -> None:
    connection.execute(f"""
        CREATE OR REPLACE TEMP TABLE s1 AS
        SELECT entity_id, name_compact, name_token_sorted, name_core,
               address_compact, primary_number, postal_like_tokens, numeric_tokens
        FROM read_parquet({_paths_sql(paths)}, union_by_name=true, hive_partitioning=false)
    """)
    connection.execute(f"""
        CREATE OR REPLACE TEMP TABLE s1_long_unbucketed AS {_long_union('s1', 'entity_id', caps)}
    """)
    connection.execute("""
        CREATE OR REPLACE TEMP TABLE s1_long AS
        SELECT entity_id, signal, family, key, cap,
               count(*) OVER (PARTITION BY signal, key)::UINTEGER AS bucket_size
        FROM s1_long_unbucketed
    """)
    connection.execute("DROP TABLE s1_long_unbucketed")


def _prepare_target(connection: Any, path: Path, caps: Mapping[str, Any]) -> None:
    connection.execute(f"""
        CREATE OR REPLACE TEMP TABLE target AS
        SELECT entity_id, name_compact, name_token_sorted, name_core,
               address_compact, primary_number, postal_like_tokens, numeric_tokens
        FROM read_parquet('{_q(path)}', hive_partitioning=false)
    """)
    connection.execute(f"""
        CREATE OR REPLACE TEMP TABLE target_long AS {_long_union('target', 'entity_id', caps)}
    """)


def _suppression_stats(connection: Any) -> list[dict[str, Any]]:
    rows = connection.execute("""
        WITH buckets AS (
            SELECT signal, key, max(bucket_size) AS bucket_size, max(cap) AS cap
            FROM s1_long GROUP BY signal, key
        ), index_stats AS (
            SELECT signal, count(*) AS index_keys, max(bucket_size) AS largest_bucket,
                   approx_quantile(bucket_size, 0.95) AS p95_bucket_size
            FROM buckets GROUP BY signal
        ), query_stats AS (
            SELECT q.signal, count(*) AS queries_hitting_key,
                   count(*) FILTER (WHERE b.bucket_size > b.cap) AS high_frequency_suppressed
            FROM target_long q JOIN buckets b USING (signal, key) GROUP BY q.signal
        )
        SELECT i.signal, i.index_keys, coalesce(q.queries_hitting_key, 0),
               coalesce(q.high_frequency_suppressed, 0), i.largest_bucket, i.p95_bucket_size
        FROM index_stats i LEFT JOIN query_stats q USING (signal) ORDER BY i.signal
    """).fetchall()
    by_signal = {item[0]: item for item in rows}
    result: list[dict[str, Any]] = []
    for signal, name, family, _, cap_name in SIGNALS:
        row = by_signal.get(signal, (signal, 0, 0, 0, 0, None))
        result.append({
            "signal": name, "family": family, "cap_name": cap_name,
            "index_keys": int(row[1]), "queries_hitting_key": int(row[2]),
            "high_frequency_suppressed": int(row[3]), "largest_bucket": int(row[4]),
            "p95_bucket_size": None if row[5] is None else float(row[5]),
        })
    return result


def _candidate_select(target_source: str, country: str) -> str:
    flag_columns = ",\n".join(
        f"bool_or(signal = {signal}) AS {name}" for signal, name, _, _, _ in SIGNALS
    )
    return f"""
        WITH joined AS (
            SELECT q.entity_id AS target_entity_id, i.entity_id AS candidate_s1_entity_id, q.signal
            FROM target_long q JOIN s1_long i USING (signal, key)
            WHERE i.bucket_size <= i.cap
        ), combined AS (
            SELECT target_entity_id, candidate_s1_entity_id,
                   bool_or(signal BETWEEN 1 AND 4) AS exact_hit,
                   bool_or(signal BETWEEN 5 AND 9) AS structured_hit,
                   {flag_columns}
            FROM joined GROUP BY target_entity_id, candidate_s1_entity_id
        )
        SELECT target_entity_id::VARCHAR AS target_entity_id,
               candidate_s1_entity_id::VARCHAR AS candidate_s1_entity_id,
               '{target_source}'::VARCHAR AS target_source, '{_q(country)}'::VARCHAR AS country,
               exact_hit, structured_hit, {', '.join(PROVENANCE)},
               ((CASE WHEN exact_hit THEN 1 ELSE 0 END) |
                (CASE WHEN structured_hit THEN 2 ELSE 0 END))::USMALLINT AS retriever_mask,
               (CAST(exact_hit AS UTINYINT) + CAST(structured_hit AS UTINYINT))::UTINYINT AS retriever_count
        FROM combined ORDER BY target_entity_id, candidate_s1_entity_id
    """


def validate_candidate_parquet(
    path: Path, *, expected_country: str, expected_source: str, expected_rows: int,
) -> None:
    duckdb, pa = _imports()
    import pyarrow.parquet as pq
    parquet = pq.ParquetFile(path)
    if parquet.metadata.num_rows != expected_rows:
        raise ValueError(f"candidate row count mismatch: {parquet.metadata.num_rows} != {expected_rows}")
    if tuple(parquet.schema_arrow.names) != CANDIDATE_COLUMNS:
        raise ValueError(f"candidate_v1 column mismatch: {parquet.schema_arrow.names}")
    expected_types = {
        **{name: pa.string() for name in CANDIDATE_COLUMNS[:4]},
        **{name: pa.bool_() for name in ("exact_hit", "structured_hit", *PROVENANCE)},
        "retriever_mask": pa.uint16(), "retriever_count": pa.uint8(),
    }
    for field in parquet.schema_arrow:
        if field.type != expected_types[field.name]:
            raise ValueError(f"candidate_v1 type mismatch for {field.name}: {field.type}")
    connection = duckdb.connect(":memory:")
    try:
        row = connection.execute(f"""
            SELECT count(*) AS rows,
                   count(DISTINCT (target_entity_id, candidate_s1_entity_id, target_source)) AS unique_pairs,
                   count(*) FILTER (WHERE country <> ? OR target_source <> ?) AS wrong_partition
            FROM read_parquet('{_q(path)}', hive_partitioning=false)
        """, [expected_country, expected_source]).fetchone()
    finally:
        connection.close()
    if row != (expected_rows, expected_rows, 0):
        raise ValueError(f"candidate uniqueness/partition validation failed: {row}")


def _output_paths(root: Path, shard: NormalizedShard) -> tuple[Path, Path, Path]:
    directory = (
        root / shard.split / f"country={quote(shard.country, safe='')}" / f"source={shard.source}"
    )
    artifact = directory / f"part-{int(shard.shard_id):05d}.parquet"
    return artifact, artifact.with_suffix(".manifest.json"), artifact.with_suffix(".stats.json")


def _dependency_hash(
    s1_shards: list[NormalizedShard], target: NormalizedShard,
    config: Mapping[str, Any],
) -> str:
    return hash_mapping({
        "normalization_contract": config["normalization"]["contract_version"],
        "normalization_inputs": [
            {
                "identity": [s.split, s.source, s.country, s.shard_id],
                "input_fingerprint": s.input_fingerprint,
                "artifact_hash": s.artifact_hash,
                "code_commit": s.normalization_commit,
            }
            for s in [*s1_shards, target]
        ],
        "retrieval_version": config["retrieval"]["version"],
        "retrieval_config": config["retrieval"]["exact_structured"],
    })


def _valid_existing(
    artifact: Path, manifest_path: Path, stats_path: Path, *, dependency: str,
    config: Mapping[str, Any], code_commit: str, country: str, source: str,
) -> bool:
    if not manifest_path.exists() or not stats_path.exists():
        return False
    try:
        manifest = load_manifest(manifest_path)
        stats = json.loads(stats_path.read_text(encoding="utf-8"))
        if manifest.code_commit != code_commit or stats.get("dependency_hash") != dependency:
            return False
        validator = lambda path: validate_candidate_parquet(
            path, expected_country=country, expected_source=source,
            expected_rows=manifest.output_rows,
        )
        return validate_artifact(
            manifest_path, expected_config_hash=config_hash(config), expected_stage=STAGE,
            expected_stage_version=str(config["retrieval"]["version"]),
            expected_input_fingerprint=dependency, schema_validator=validator,
        ).valid and Path(manifest.artifact_path).resolve() == artifact.resolve()
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return False


def _write_shard(
    connection: Any, target: NormalizedShard, s1_shards: list[NormalizedShard],
    output_root: Path, config: Mapping[str, Any], code_commit: str,
) -> dict[str, Any]:
    artifact, manifest_path, stats_path = _output_paths(output_root, target)
    dependency = _dependency_hash(s1_shards, target, config)
    target_source = target.source.replace("source", "S")
    if _valid_existing(
        artifact, manifest_path, stats_path, dependency=dependency, config=config,
        code_commit=code_commit, country=target.country, source=target_source,
    ):
        manifest = load_manifest(manifest_path)
        return {
            "artifact": str(artifact), "rows": manifest.output_rows, "skipped": True,
            "suppression": json.loads(stats_path.read_text(encoding="utf-8"))["suppression"],
        }
    caps = config["retrieval"]["exact_structured"]
    _prepare_target(connection, target.path, caps)
    suppression = _suppression_stats(connection)
    artifact.parent.mkdir(parents=True, exist_ok=True)
    temporary = artifact.with_suffix(".parquet.tmp")
    if temporary.exists():
        temporary.unlink()
    started = time.monotonic()
    query = _candidate_select(target_source, target.country)
    connection.execute(f"COPY ({query}) TO '{_q(temporary)}' (FORMAT PARQUET, COMPRESSION ZSTD)")
    temporary.replace(artifact)
    rows = int(connection.execute(
        f"SELECT count(*) FROM read_parquet('{_q(artifact)}', hive_partitioning=false)"
    ).fetchone()[0])
    validate_candidate_parquet(
        artifact, expected_country=target.country, expected_source=target_source, expected_rows=rows,
    )
    peak_rss = ResourceMonitor(STAGE).sample().rss_bytes
    manifest = ShardManifest.complete(
        stage=STAGE, stage_version=str(config["retrieval"]["version"]),
        split=target.split, country=target.country, source=target_source, shard_id=target.shard_id,
        input_rows=target.rows, output_rows=rows, runtime_seconds=time.monotonic() - started,
        peak_ram_bytes=peak_rss, peak_vram_bytes=None, config_hash=config_hash(config),
        code_commit=code_commit, input_fingerprint=dependency,
        artifact_path=str(artifact.resolve()), artifact_size_bytes=artifact.stat().st_size,
        artifact_hash=hash_file(artifact),
    )
    manifest.write(manifest_path)
    _atomic_json(stats_path, {"dependency_hash": dependency, "suppression": suppression})
    return {"artifact": str(artifact), "rows": rows, "skipped": False, "suppression": suppression}


def _aggregate_suppression(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    for result in results:
        identity = (result["split"], result["country"], result["source"])
        for item in result["suppression"]:
            key = (*identity, item["signal"])
            current = grouped.setdefault(key, {
                "split": identity[0], "country": identity[1], "source": identity[2],
                "signal": item["signal"], "family": item["family"],
                "index_keys": 0, "queries_hitting_key": 0,
                "high_frequency_suppressed": 0, "largest_bucket": 0,
                "p95_bucket_size": item["p95_bucket_size"],
            })
            current["index_keys"] = max(current["index_keys"], item["index_keys"])
            current["queries_hitting_key"] += item["queries_hitting_key"]
            current["high_frequency_suppressed"] += item["high_frequency_suppressed"]
            current["largest_bucket"] = max(current["largest_bucket"], item["largest_bucket"])
    return [grouped[key] for key in sorted(grouped)]


def _candidate_volume(
    connection: Any, candidate_paths: list[Path], targets: list[NormalizedShard],
) -> dict[str, Any]:
    connection.execute(f"""
        CREATE OR REPLACE TEMP VIEW report_candidates AS
        SELECT * FROM read_parquet({_paths_sql(candidate_paths)}, union_by_name=true, hive_partitioning=false)
    """)
    target_parts = [
        f"SELECT entity_id AS target_entity_id, country, "
        f"'{item.source.replace('source', 'S')}'::VARCHAR AS target_source "
        f"FROM read_parquet('{_q(item.path)}', hive_partitioning=false)"
        for item in targets
    ]
    connection.execute(
        "CREATE OR REPLACE TEMP VIEW report_targets AS " + " UNION ALL ".join(target_parts)
    )
    total_pairs = int(connection.execute("SELECT count(*) FROM report_candidates").fetchone()[0])
    row = connection.execute("""
        WITH counts AS (
            SELECT t.target_entity_id, count(c.candidate_s1_entity_id) AS n
            FROM report_targets t LEFT JOIN report_candidates c
              USING (target_entity_id, country, target_source)
            GROUP BY t.target_entity_id, t.target_source
        )
        SELECT count(*), avg(n), quantile_cont(n, 0.5), quantile_cont(n, 0.9),
               quantile_cont(n, 0.95), quantile_cont(n, 0.99), max(n),
               count(*) FILTER (WHERE n=0), count(*) FILTER (WHERE n=1),
               count(*) FILTER (WHERE n BETWEEN 2 AND 5),
               count(*) FILTER (WHERE n BETWEEN 6 AND 10), count(*) FILTER (WHERE n>10)
        FROM counts
    """).fetchone()
    return {
        "candidate_pair_count": total_pairs, "target_count": int(row[0]),
        "avg_candidates_per_target": float(row[1] or 0), "p50": float(row[2] or 0),
        "p90": float(row[3] or 0), "p95": float(row[4] or 0), "p99": float(row[5] or 0),
        "max": int(row[6] or 0),
        "targets_with": {"0": int(row[7]), "1": int(row[8]), "2-5": int(row[9]),
                         "6-10": int(row[10]), ">10": int(row[11])},
    }


def _train_recall(connection: Any, gt_path: Path) -> dict[str, Any]:
    connection.execute(f"""
        CREATE OR REPLACE TEMP VIEW report_gt AS
        SELECT g.source1_entity_id::VARCHAR AS true_s1, trim(target_id)::VARCHAR AS target_entity_id,
               t.country, t.target_source
        FROM read_csv('{_q(gt_path)}', delim='\t', header=true, all_varchar=true) g,
             UNNEST(string_split(coalesce(g.matched_entity_ids, ''), ',')) ids(target_id)
        JOIN report_targets t ON t.target_entity_id = trim(target_id)
        WHERE trim(target_id) <> ''
        QUALIFY row_number() OVER (PARTITION BY true_s1, trim(target_id)) = 1
    """)
    total = int(connection.execute("SELECT count(*) FROM report_gt").fetchone()[0])
    recovered = connection.execute("""
        SELECT count(*) FILTER (WHERE c.exact_hit), count(*) FILTER (WHERE c.structured_hit),
               count(*) FILTER (WHERE c.target_entity_id IS NOT NULL),
               count(*) FILTER (WHERE c.exact_hit AND NOT c.structured_hit),
               count(*) FILTER (WHERE c.structured_hit AND NOT c.exact_hit)
        FROM report_gt g LEFT JOIN report_candidates c
          ON c.target_entity_id=g.target_entity_id AND c.candidate_s1_entity_id=g.true_s1
         AND c.country=g.country
    """).fetchone()
    def ratio(value: int) -> float:
        return float(value) / total if total else 0.0
    by_partition_rows = connection.execute("""
        SELECT g.country, g.target_source,
               count(*) AS gt_pairs,
               count(*) FILTER (WHERE c.target_entity_id IS NOT NULL) AS recovered
        FROM report_gt g LEFT JOIN report_candidates c
          ON c.target_entity_id=g.target_entity_id AND c.candidate_s1_entity_id=g.true_s1
         AND c.country=g.country
        GROUP BY 1,2 ORDER BY 1,2
    """).fetchall()
    return {
        "gt_pairs_total": total,
        "exact_only_pair_recall": ratio(int(recovered[0])),
        "structured_only_pair_recall": ratio(int(recovered[1])),
        "exact_union_structured_pair_recall": ratio(int(recovered[2])),
        "gt_target_hit_rate": ratio(int(recovered[2])),
        "gt_pairs_recovered_only_by_exact": int(recovered[3]),
        "gt_pairs_recovered_only_by_structured": int(recovered[4]),
        "by_country_source": [
            {"country": row[0], "target_source": row[1], "gt_pairs": int(row[2]),
             "recovered": int(row[3]), "pair_recall": float(row[3]) / row[2] if row[2] else 0.0}
            for row in by_partition_rows
        ],
    }


def run_exact_structured_retrieval(
    config: Mapping[str, Any], *, normalized_root: str | Path | None,
    data_root: str | Path | None = None, output_dir: str | Path | None = None,
    split: str | None = None, source: str | None = None,
    country: str | None = None, shard_id: str | None = None,
) -> dict[str, Any]:
    if normalized_root is None:
        raise ExactStructuredError("--normalized-root is required")
    if split not in {None, "train", "test"}:
        raise ExactStructuredError("split must be train or test")
    source_map = {None: None, "source2": "source2", "source3": "source3", "S2": "source2", "S3": "source3"}
    if source not in source_map:
        raise ExactStructuredError("source must be source2/source3 or S2/S3")
    selected_source = source_map[source]
    try:
        selected_shard = None if shard_id is None else str(int(shard_id))
    except ValueError as exc:
        raise ExactStructuredError("shard-id must be a non-negative integer") from exc
    if selected_shard is not None and int(selected_shard) < 0:
        raise ExactStructuredError("shard-id must be a non-negative integer")
    all_shards = _discover_normalized(Path(normalized_root).resolve())
    targets = [
        item for item in all_shards
        if item.source in {"source2", "source3"}
        and (split is None or item.split == split)
        and (selected_source is None or item.source == selected_source)
        and (country is None or item.country == country)
        and (selected_shard is None or item.shard_id == selected_shard)
    ]
    if not targets:
        raise ExactStructuredError("No normalized target shards match the requested filters")
    output_root = (
        resolve_project_path(config, "artifact_root") / "retrieval" / str(config["retrieval"]["version"]) / STAGE
        if output_dir is None else Path(output_dir).resolve()
    )
    report_root = resolve_project_path(config, "artifact_root") / "reports" / "retrieval" / str(config["retrieval"]["version"]) / STAGE
    report_root.mkdir(parents=True, exist_ok=True)
    resource_log = report_root / "resources.jsonl"
    lock = threading.Lock()
    monitor = ResourceMonitor(STAGE, country=country, source=source, shard=shard_id)
    stop, thread = monitor.start_heartbeat(
        float(config["runtime"]["heartbeat_seconds"]),
        lambda sample: _append_jsonl(resource_log, sample.to_json(), lock),
    )
    duckdb, _ = _imports()
    connection = duckdb.connect(":memory:")
    _configure_connection(connection, config)
    code_commit = _code_commit()
    results: list[dict[str, Any]] = []
    started = time.monotonic()
    try:
        partitions = sorted({(item.split, item.country) for item in targets})
        for current_split, current_country in partitions:
            s1_shards = [
                item for item in all_shards
                if item.split == current_split and item.source == "source1" and item.country == current_country
            ]
            if not s1_shards:
                raise ExactStructuredError(f"Missing S1 normalization shards for {current_split}/{current_country}")
            _prepare_s1(connection, [item.path for item in s1_shards], config["retrieval"]["exact_structured"])
            for target in [t for t in targets if (t.split, t.country) == (current_split, current_country)]:
                print(
                    f"[EXACT_STRUCTURED] split={target.split} country={target.country} "
                    f"source={target.source} shard={target.shard_id} target_rows={target.rows}", flush=True,
                )
                result = _write_shard(connection, target, s1_shards, output_root, config, code_commit)
                result.update(split=target.split, country=target.country, source=target.source, shard=target.shard_id)
                results.append(result)
                sample = monitor.sample(processed_rows=target.rows)
                _append_jsonl(resource_log, sample.to_json(), lock)
                state = monitor.memory_state(
                    float(config["runtime"]["warning_ram_fraction"]),
                    float(config["runtime"]["critical_ram_fraction"]),
                )
                print(
                    f"[EXACT_STRUCTURED] complete candidates={result['rows']} skipped={result['skipped']} "
                    f"rows_per_sec={sample.rows_per_second} rss={sample.rss_bytes} "
                    f"available_ram={sample.available_ram_bytes} elapsed={sample.elapsed_seconds:.1f}s",
                    flush=True,
                )
                if state == "critical":
                    raise ExactStructuredError("Critical RAM threshold reached after safe shard completion")
                if state == "warning":
                    print("[EXACT_STRUCTURED] WARNING RAM threshold reached", flush=True)
    finally:
        connection.close()
        stop.set()
        thread.join(timeout=10)
    candidate_paths = [Path(item["artifact"]) for item in results]
    target_lookup = {(item.split, item.source, item.country, item.shard_id): item for item in targets}
    report_targets = [target_lookup[(r["split"], r["source"], r["country"], r["shard"])] for r in results]
    report_connection = duckdb.connect(":memory:")
    _configure_connection(report_connection, config)
    try:
        volume = _candidate_volume(report_connection, candidate_paths, report_targets)
        recall = None
        if any(item["split"] == "train" for item in results):
            if data_root is None:
                raise ExactStructuredError("--data-root is required when train retrieval is selected")
            paths = resolve_dataset_paths(config, data_root)
            recall = _train_recall(report_connection, paths.train_ground_truth)
    finally:
        report_connection.close()
    summary = {
        "status": "complete", "stage": STAGE, "retrieval_version": config["retrieval"]["version"],
        "normalization_version": config["normalization"]["contract_version"],
        "runtime_seconds": time.monotonic() - started,
        "output_root": str(output_root), "shards": len(results),
        "shards_written": sum(not item["skipped"] for item in results),
        "shards_skipped": sum(item["skipped"] for item in results),
        "volume": volume, "train_recall": recall,
        "suppression": _aggregate_suppression(results),
    }
    _atomic_json(report_root / "summary.json", summary)
    return summary
