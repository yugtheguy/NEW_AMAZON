"""Memory-bounded dataset contract and invariant audit."""

from __future__ import annotations

import csv
import json
import math
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from amazon_er.data.fingerprint import combined_fingerprint, sampled_file_fingerprint
from amazon_er.data.paths import DatasetPaths, resolve_dataset_paths
from amazon_er.infra.resources import ResourceMonitor, ResourceSample
from amazon_er.paths import resolve_project_path


SOURCE_FILES = (
    "train_source1", "train_source2", "train_source3",
    "test_source1", "test_source2", "test_source3",
)
EXPECTED_SOURCE_COLUMNS = ["entity_id", "business_name", "business_address", "country"]
EXPECTED_GT_COLUMNS = ["source1_entity_id", "matched_entity_ids"]


class DatasetContractError(ValueError):
    """The physical data violates the schema needed for a trustworthy audit."""


def _sql_string(value: object) -> str:
    return str(value).replace("\\", "/").replace("'", "''")


def _query_dicts(connection: Any, sql: str) -> list[dict[str, Any]]:
    cursor = connection.execute(sql)
    names = [item[0] for item in cursor.description]
    return [dict(zip(names, row)) for row in cursor.fetchall()]


def _single(connection: Any, sql: str) -> dict[str, Any]:
    rows = _query_dicts(connection, sql)
    if len(rows) != 1:
        raise DatasetContractError(f"Expected one aggregate row, received {len(rows)}")
    return rows[0]


def _header(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        return next(csv.reader(stream, delimiter="\t"))


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    return value


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(_json_safe(payload), indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def _create_views(connection: Any, paths: DatasetPaths) -> None:
    for logical_name, path in paths.sources().items():
        connection.execute(
            f"CREATE VIEW {logical_name} AS SELECT * FROM read_csv("
            f"'{_sql_string(path)}', delim='\\t', header=true, all_varchar=true, "
            "nullstr='__AMAZON_ER_NULL_SENTINEL__', quote='', escape='', strict_mode=true)"
        )


def _file_contract(connection: Any, logical_name: str, path: Path) -> dict[str, Any]:
    columns = _header(path)
    expected = EXPECTED_GT_COLUMNS if logical_name == "train_ground_truth" else EXPECTED_SOURCE_COLUMNS
    if columns != expected:
        raise DatasetContractError(
            f"{logical_name} columns/order {columns!r} do not match discovered contract {expected!r}"
        )
    schema = _query_dicts(connection, f"DESCRIBE SELECT * FROM {logical_name}")
    return {
        "logical_name": logical_name,
        "filename": path.name,
        "resolved_path": str(path),
        "file_size_bytes": path.stat().st_size,
        "modified_time_utc": datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).isoformat(),
        "delimiter": "tab",
        "encoding": "UTF-8 (header decoded strictly; full file parsed by DuckDB)",
        "columns": columns,
        "column_order": columns,
        "inferred_dtypes": {row["column_name"]: row["column_type"] for row in schema},
        "row_count": int(connection.execute(f"SELECT count(*) FROM {logical_name}").fetchone()[0]),
    }


def _missing_stats(connection: Any, table: str, column: str, rows: int) -> dict[str, Any]:
    result = _single(connection, f"""
        SELECT count(*) FILTER (WHERE {column} IS NULL) AS null_count,
               count(*) FILTER (WHERE {column} = '') AS empty_count,
               count(*) FILTER (WHERE {column} <> '' AND trim({column}) = '') AS whitespace_only_count
        FROM {table}
    """)
    for key in ("null_count", "empty_count", "whitespace_only_count"):
        result[key] = int(result[key])
        result[key.replace("_count", "_fraction")] = result[key] / rows if rows else 0.0
    return result


def _text_stats(connection: Any, table: str, column: str) -> dict[str, Any]:
    row = _single(connection, f"""
        SELECT quantile_disc(length({column}), [0.5, 0.9, 0.95, 0.99]) AS length_quantiles,
               max(length({column})) AS max_length,
               quantile_disc(array_length(regexp_split_to_array(trim({column}), '\\s+')), [0.5, 0.9, 0.95, 0.99]) AS token_quantiles,
               max(array_length(regexp_split_to_array(trim({column}), '\\s+'))) AS max_tokens
        FROM {table} WHERE {column} IS NOT NULL AND trim({column}) <> ''
    """)
    keys = ("p50", "p90", "p95", "p99")
    return {
        "length": {**dict(zip(keys, row["length_quantiles"] or [])), "max": row["max_length"]},
        "tokens": {**dict(zip(keys, row["token_quantiles"] or [])), "max": row["max_tokens"]},
    }


def _source_audit(connection: Any, table: str, rows: int) -> dict[str, Any]:
    identity = _single(connection, f"""
        SELECT count(*) AS rows,
               count(*) FILTER (WHERE entity_id IS NULL) AS entity_id_null_count,
               count(DISTINCT entity_id) AS unique_entity_ids
        FROM {table}
    """)
    duplicate = _single(connection, f"""
        SELECT count(*) AS duplicate_id_values, coalesce(sum(n - 1), 0) AS duplicate_rows
        FROM (SELECT entity_id, count(*) AS n FROM {table} GROUP BY entity_id HAVING count(*) > 1)
    """)
    countries = _query_dicts(connection, f"""
        SELECT CASE WHEN country IS NULL THEN '<NULL>' WHEN country = '' THEN '<EMPTY>'
                    WHEN trim(country) = '' THEN '<WHITESPACE>' ELSE country END AS country,
               count(*) AS rows
        FROM {table} GROUP BY 1 ORDER BY 1
    """)
    return {
        "rows": int(identity["rows"]),
        "entity_id_null_count": int(identity["entity_id_null_count"]),
        "unique_entity_ids": int(identity["unique_entity_ids"]),
        "duplicate_entity_ids": int(duplicate["duplicate_id_values"]),
        "duplicate_rows": int(duplicate["duplicate_rows"]),
        "country_counts": {str(row["country"]): int(row["rows"]) for row in countries},
        "missingness": {
            column: _missing_stats(connection, table, column, rows)
            for column in ("business_name", "business_address", "country")
        },
        "text": {
            column: _text_stats(connection, table, column)
            for column in ("business_name", "business_address")
        },
    }


def _script_audit(connection: Any, table: str) -> list[dict[str, Any]]:
    rows = _query_dicts(connection, f"""
        WITH flags AS (
            SELECT country, business_name,
                regexp_matches(business_name, '[A-Za-z]') AS latin,
                regexp_matches(business_name, '[ऀ-ॿ]') AS devanagari,
                regexp_matches(business_name, '[ঀ-෿ༀ-႟]') AS other_indic,
                regexp_matches(business_name, '[^\\x00-\\x7F]') AS non_ascii
            FROM {table}
        ), classified AS (
            SELECT country, non_ascii,
                CASE WHEN business_name IS NULL OR trim(business_name) = '' THEN 'empty'
                     WHEN (CAST(latin AS INTEGER) + CAST(devanagari AS INTEGER) + CAST(other_indic AS INTEGER)
                           + CAST(non_ascii AND NOT devanagari AND NOT other_indic AS INTEGER)) > 1 THEN 'mixed'
                     WHEN devanagari THEN 'Devanagari'
                     WHEN other_indic THEN 'other Indic'
                     WHEN latin THEN 'Latin'
                     ELSE 'other/non-Latin' END AS script
            FROM flags
        )
        SELECT country, script, count(*) AS rows,
               count(*) FILTER (WHERE non_ascii) AS non_ascii_rows
        FROM classified GROUP BY country, script ORDER BY country, script
    """)
    return [
        {"country": row["country"], "script": row["script"], "rows": int(row["rows"]),
         "non_ascii_rows": int(row["non_ascii_rows"])}
        for row in rows
    ]


def _overlap(connection: Any, left: str, right: str) -> int:
    return int(connection.execute(
        f"SELECT count(*) FROM (SELECT entity_id FROM {left} INTERSECT SELECT entity_id FROM {right})"
    ).fetchone()[0])


def _prepare_gt(connection: Any) -> dict[str, Any]:
    connection.execute("""
        CREATE TEMP TABLE gt_pairs AS
        SELECT source1_entity_id AS s1_id, trim(target_id) AS target_id,
               CASE WHEN starts_with(trim(target_id), 'S2-') THEN 'S2'
                    WHEN starts_with(trim(target_id), 'S3-') THEN 'S3' ELSE 'INVALID' END AS target_source
        FROM train_ground_truth,
             UNNEST(string_split(coalesce(matched_entity_ids, ''), ',')) AS targets(target_id)
        WHERE trim(target_id) <> ''
    """)
    base = _single(connection, """
        SELECT count(*) AS pair_rows, count(DISTINCT (s1_id, target_id)) AS unique_pairs,
               count(*) FILTER (WHERE target_source='S2') AS s2_pairs,
               count(*) FILTER (WHERE target_source='S3') AS s3_pairs,
               count(*) FILTER (WHERE target_source='INVALID') AS invalid_target_prefix_pairs
        FROM gt_pairs
    """)
    gt_rows = _single(connection, """
        SELECT count(*) AS ground_truth_rows,
               count(DISTINCT source1_entity_id) AS unique_s1_rows,
               count(*) FILTER (WHERE matched_entity_ids='') AS empty_match_rows
        FROM train_ground_truth
    """)
    return {key: int(value) for key, value in {**base, **gt_rows}.items()}


def _gt_integrity(connection: Any) -> dict[str, int]:
    result = _single(connection, """
        SELECT
          (SELECT count(*) FROM gt_pairs g LEFT JOIN train_source1 s ON g.s1_id=s.entity_id WHERE s.entity_id IS NULL) AS orphan_s1,
          (SELECT count(*) FROM gt_pairs g LEFT JOIN train_source2 s ON g.target_id=s.entity_id WHERE g.target_source='S2' AND s.entity_id IS NULL) AS orphan_s2,
          (SELECT count(*) FROM gt_pairs g LEFT JOIN train_source3 s ON g.target_id=s.entity_id WHERE g.target_source='S3' AND s.entity_id IS NULL) AS orphan_s3
    """)
    return {key: int(value) for key, value in result.items()}


def _ownership(connection: Any, source: str) -> dict[str, Any]:
    rows = _query_dicts(connection, f"""
        WITH owners AS (
          SELECT target_id, count(DISTINCT s1_id) AS n FROM gt_pairs
          WHERE target_source='{source}' GROUP BY target_id
        )
        SELECT CASE WHEN n=1 THEN '1' WHEN n=2 THEN '2' ELSE '3+' END AS owners,
               count(*) AS targets, (SELECT coalesce(max(n), 0) FROM owners) AS max_owners
        FROM owners GROUP BY 1 ORDER BY 1
    """)
    distribution = {"1": 0, "2": 0, "3+": 0}
    max_owners = 0
    for row in rows:
        distribution[row["owners"]] = int(row["targets"])
        max_owners = max(max_owners, int(row["max_owners"]))
    return {
        "max_owners": max_owners,
        "targets_with_multiple_owners": distribution["2"] + distribution["3+"],
        "owner_distribution": distribution,
    }


def _country_consistency(connection: Any) -> dict[str, int]:
    s2 = int(connection.execute("""
        SELECT count(*) FROM gt_pairs g JOIN train_source1 a ON g.s1_id=a.entity_id
        JOIN train_source2 b ON g.target_id=b.entity_id
        WHERE g.target_source='S2' AND a.country IS DISTINCT FROM b.country
    """).fetchone()[0])
    s3 = int(connection.execute("""
        SELECT count(*) FROM gt_pairs g JOIN train_source1 a ON g.s1_id=a.entity_id
        JOIN train_source3 b ON g.target_id=b.entity_id
        WHERE g.target_source='S3' AND a.country IS DISTINCT FROM b.country
    """).fetchone()[0])
    return {"s1_s2_mismatches": s2, "s1_s3_mismatches": s3, "combined_mismatches": s2 + s3}


def _cardinality(connection: Any) -> dict[str, Any]:
    connection.execute("""
        CREATE TEMP VIEW s1_cardinality AS
        WITH positive AS (
          SELECT s1_id,
                 count(DISTINCT target_id) FILTER (WHERE target_source='S2') AS s2_matches,
                 count(DISTINCT target_id) FILTER (WHERE target_source='S3') AS s3_matches
          FROM gt_pairs GROUP BY s1_id
        )
        SELECT s.entity_id AS s1_id, s.country,
               coalesce(p.s2_matches, 0) AS s2_matches,
               coalesce(p.s3_matches, 0) AS s3_matches,
               coalesce(p.s2_matches, 0) + coalesce(p.s3_matches, 0) AS total_matches
        FROM train_source1 s LEFT JOIN positive p ON s.entity_id=p.s1_id
    """)
    def summarize(where: str = "") -> dict[str, Any]:
        row = _single(connection, f"""
            SELECT count(*) FILTER (WHERE total_matches=0) AS "0",
                   count(*) FILTER (WHERE total_matches=1) AS "1",
                   count(*) FILTER (WHERE total_matches=2) AS "2",
                   count(*) FILTER (WHERE total_matches=3) AS "3",
                   count(*) FILTER (WHERE total_matches=4) AS "4",
                   count(*) FILTER (WHERE total_matches>=5) AS "5+",
                   avg(total_matches) AS mean, median(total_matches) AS median,
                   quantile_disc(total_matches, 0.90) AS p90,
                   quantile_disc(total_matches, 0.95) AS p95,
                   quantile_disc(total_matches, 0.99) AS p99,
                   max(total_matches) AS max, count(*) AS s1_rows
            FROM s1_cardinality {where}
        """)
        integers = {"0", "1", "2", "3", "4", "5+", "p90", "p95", "p99", "max", "s1_rows"}
        return {key: int(value) if key in integers else float(value) for key, value in row.items()}
    overall = summarize()
    countries = [row[0] for row in connection.execute("SELECT DISTINCT country FROM s1_cardinality ORDER BY country").fetchall()]
    by_country = {
        country: summarize(f"WHERE country='{_sql_string(country)}'")
        for country in countries
    }
    overall["zero_match_fraction"] = overall["0"] / overall["s1_rows"] if overall["s1_rows"] else 0.0
    for item in by_country.values():
        item["zero_match_fraction"] = item["0"] / item["s1_rows"] if item["s1_rows"] else 0.0
    return {"overall": overall, "by_country": by_country}


def _distractors(connection: Any, table: str, source: str) -> dict[str, Any]:
    overall = _single(connection, f"""
        SELECT count(*) AS total,
               count(*) FILTER (WHERE g.target_id IS NOT NULL) AS matched,
               count(*) FILTER (WHERE g.target_id IS NULL) AS unmatched
        FROM {table} t LEFT JOIN (SELECT DISTINCT target_id FROM gt_pairs WHERE target_source='{source}') g
        ON t.entity_id=g.target_id
    """)
    by_country_rows = _query_dicts(connection, f"""
        SELECT t.country, count(*) AS total,
               count(*) FILTER (WHERE g.target_id IS NOT NULL) AS matched,
               count(*) FILTER (WHERE g.target_id IS NULL) AS unmatched
        FROM {table} t LEFT JOIN (SELECT DISTINCT target_id FROM gt_pairs WHERE target_source='{source}') g
        ON t.entity_id=g.target_id GROUP BY t.country ORDER BY t.country
    """)
    def clean(row: dict[str, Any]) -> dict[str, Any]:
        result = {key: int(value) for key, value in row.items() if key != "country"}
        result["distractor_fraction"] = result["unmatched"] / result["total"] if result["total"] else 0.0
        return result
    return {"overall": clean(overall), "by_country": {row["country"]: clean(row) for row in by_country_rows}}


def _source_distribution(connection: Any) -> dict[str, int]:
    row = _single(connection, """
        WITH sources AS (
          SELECT s1_id, bool_or(target_source='S2') AS has_s2, bool_or(target_source='S3') AS has_s3
          FROM gt_pairs GROUP BY s1_id
        )
        SELECT
          (SELECT count(*) FROM gt_pairs WHERE target_source='S2') AS s2_positive_pairs,
          (SELECT count(*) FROM gt_pairs WHERE target_source='S3') AS s3_positive_pairs,
          count(*) FILTER (WHERE has_s2) AS unique_s1_with_s2,
          count(*) FILTER (WHERE has_s3) AS unique_s1_with_s3,
          count(*) FILTER (WHERE has_s2 AND has_s3) AS s1_both,
          count(*) FILTER (WHERE has_s2 AND NOT has_s3) AS s1_only_s2,
          count(*) FILTER (WHERE has_s3 AND NOT has_s2) AS s1_only_s3
        FROM sources
    """)
    return {key: int(value) for key, value in row.items()}


def _metric_status(root: Path) -> dict[str, Any]:
    candidates = sorted(
        str(path.relative_to(root)) for path in root.rglob("*")
        if path.is_file() and any(token in path.name.lower() for token in ("metric", "scorer", "validate_submission"))
    )
    return {
        "status": "PROJECT CONTRACT IMPLEMENTED; OFFICIAL CODE COMPARISON UNVERIFIED",
        "organizer_files_found": candidates,
        "reason": "Organizer submission validator does not compute the competition score.",
    }


def _render_markdown(audit: Mapping[str, Any]) -> str:
    lines = ["# Data Audit", "", f"Dataset fingerprint: `{audit['dataset_fingerprint']}`", "", "## Files", ""]
    lines.extend(["| logical name | rows | unique IDs | duplicate IDs | countries |", "|---|---:|---:|---:|---|"])
    for name in SOURCE_FILES:
        item, source = audit["files"][name], audit["sources"][name]
        lines.append(
            f"| {name} | {item['row_count']} | {source['unique_entity_ids']} | "
            f"{source['duplicate_entity_ids']} | {', '.join(source['country_counts'])} |"
        )
    gt = audit["ground_truth"]
    integrity = gt["integrity"]
    ownership = gt["ownership"]
    consistency = gt["country_consistency"]
    cardinality = gt["s1_cardinality"]["overall"]
    distractors = gt["target_distractors"]
    distribution = gt["source_distribution"]
    lines.extend([
        "", "## Ground truth", "",
        f"- Positive pairs: {gt['pair_rows']:,} ({gt['unique_pairs']:,} unique)",
        f"- Orphan references: S1={integrity['orphan_s1']}, S2={integrity['orphan_s2']}, S3={integrity['orphan_s3']}",
        f"- Maximum owners: S2={ownership['S2']['max_owners']}, S3={ownership['S3']['max_owners']}",
        f"- Cross-country pairs: S1↔S2={consistency['s1_s2_mismatches']}, S1↔S3={consistency['s1_s3_mismatches']}",
        "", "## S1 cardinality", "",
        f"- 0={cardinality['0']:,}, 1={cardinality['1']:,}, 2={cardinality['2']:,}, 3={cardinality['3']:,}, 4={cardinality['4']:,}, 5+={cardinality['5+']:,}",
        f"- Mean={cardinality['mean']:.4f}, median={cardinality['median']:.0f}, p95={cardinality['p95']}, max={cardinality['max']}",
        f"- Zero-match fraction={cardinality['zero_match_fraction']:.6%}",
        "", "## Targets and sources", "",
        f"- S2 distractors: {distractors['S2']['overall']['unmatched']:,}/{distractors['S2']['overall']['total']:,} ({distractors['S2']['overall']['distractor_fraction']:.4%})",
        f"- S3 distractors: {distractors['S3']['overall']['unmatched']:,}/{distractors['S3']['overall']['total']:,} ({distractors['S3']['overall']['distractor_fraction']:.4%})",
        f"- Positive pairs: S2={distribution['s2_positive_pairs']:,}, S3={distribution['s3_positive_pairs']:,}",
        f"- S1 source coverage: both={distribution['s1_both']:,}, only S2={distribution['s1_only_s2']:,}, only S3={distribution['s1_only_s3']:,}",
    ])
    inv = audit["invariants"]
    lines.extend(["", "## Verified invariants", "",
                  f"- `TARGET_EXCLUSIVITY_VERIFIED = {str(inv['target_exclusivity_verified']).lower()}`",
                  f"- `STRICT_COUNTRY_BLOCKING_SAFE = {str(inv['strict_country_blocking_safe']).lower()}`", ""])
    return "\n".join(lines)


def run_data_audit(
    config: Mapping[str, Any], *, data_root: str | Path | None = None,
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Run the canonical one-time audit and emit versioned reports."""
    paths = resolve_dataset_paths(config, data_root)
    version = str(config["audit"]["version"])
    output = (
        resolve_project_path(config, "artifact_root") / "reports" / "data_audit" / version
        if output_dir is None else Path(output_dir).resolve()
    )
    output.mkdir(parents=True, exist_ok=True)
    sample_bytes = int(config["audit"]["fingerprint_sample_bytes"])
    fingerprints = {name: sampled_file_fingerprint(path, sample_bytes) for name, path in paths.sources().items()}
    dataset_fingerprint = combined_fingerprint(fingerprints)
    report_paths = [
        output / "audit.json", output / "audit.md", output / "dataset_contract.json",
        output / "input_fingerprints.json", output / "invariants.json",
    ]
    if any(path.exists() for path in report_paths):
        if not all(path.is_file() and path.stat().st_size > 0 for path in report_paths):
            raise DatasetContractError(
                f"Immutable audit version {version} is incomplete under {output}; repair it explicitly or bump audit.version"
            )
        try:
            existing = json.loads((output / "audit.json").read_text(encoding="utf-8"))
            existing_fingerprints = json.loads((output / "input_fingerprints.json").read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise DatasetContractError(f"Immutable audit version {version} is unreadable: {exc}") from exc
        if existing.get("dataset_fingerprint") != dataset_fingerprint or existing_fingerprints != fingerprints:
            raise DatasetContractError(
                f"Immutable audit version {version} belongs to different inputs; bump audit.version"
            )
        print("[DATA-AUDIT] existing versioned report verified; no overwrite", flush=True)
        return existing
    try:
        import duckdb
    except ImportError as exc:
        raise RuntimeError("Data audit requires the data extra: pip install -e '.[data]'") from exc
    temp_dir = output / "duckdb_tmp"
    temp_dir.mkdir(parents=True, exist_ok=True)
    monitor = ResourceMonitor("data-audit")
    peak_rss = 0
    def observe(sample: ResourceSample) -> None:
        nonlocal peak_rss
        peak_rss = max(peak_rss, sample.rss_bytes)
    stop, heartbeat = monitor.start_heartbeat(5.0, observe)
    started = time.monotonic()
    connection = duckdb.connect(":memory:")
    memory_limit = _sql_string(config["audit"]["duckdb_memory_limit"])
    connection.execute(f"SET memory_limit='{memory_limit}'")
    connection.execute(f"SET threads={int(config['audit']['duckdb_threads'])}")
    connection.execute("SET preserve_insertion_order=false")
    connection.execute(f"SET temp_directory='{_sql_string(temp_dir)}'")
    try:
        print("[DATA-AUDIT] input fingerprints verified", flush=True)
        _create_views(connection, paths)
        print("[DATA-AUDIT] source schemas and row counts", flush=True)
        files = {name: _file_contract(connection, name, path) for name, path in paths.sources().items()}
        print("[DATA-AUDIT] source integrity and text statistics", flush=True)
        sources = {name: _source_audit(connection, name, files[name]["row_count"]) for name in SOURCE_FILES}
        print("[DATA-AUDIT] script statistics", flush=True)
        scripts = {name: _script_audit(connection, name) for name in SOURCE_FILES}
        print("[DATA-AUDIT] ground-truth integrity", flush=True)
        gt = _prepare_gt(connection)
        gt["integrity"] = _gt_integrity(connection)
        gt["ownership"] = {"S2": _ownership(connection, "S2"), "S3": _ownership(connection, "S3")}
        print("[DATA-AUDIT] country consistency and cardinality", flush=True)
        gt["country_consistency"] = _country_consistency(connection)
        gt["s1_cardinality"] = _cardinality(connection)
        gt["target_distractors"] = {
            "S2": _distractors(connection, "train_source2", "S2"),
            "S3": _distractors(connection, "train_source3", "S3"),
        }
        gt["source_distribution"] = _source_distribution(connection)
        overlaps = {
            split: {
                "s1_s2": _overlap(connection, f"{split}_source1", f"{split}_source2"),
                "s1_s3": _overlap(connection, f"{split}_source1", f"{split}_source3"),
                "s2_s3": _overlap(connection, f"{split}_source2", f"{split}_source3"),
            } for split in ("train", "test")
        }
        train_countries = sorted(set().union(*(sources[name]["country_counts"] for name in SOURCE_FILES[:3])))
        test_countries = sorted(set().union(*(sources[name]["country_counts"] for name in SOURCE_FILES[3:])))
        ownership, consistency = gt["ownership"], gt["country_consistency"]
        invariants = {
            "target_exclusivity_verified": ownership["S2"]["max_owners"] <= 1 and ownership["S3"]["max_owners"] <= 1,
            "strict_country_blocking_safe": consistency["combined_mismatches"] == 0,
            "train_countries": train_countries,
            "test_countries": test_countries,
            "expected_train_country_pattern_verified": train_countries == ["India", "US"],
            "expected_test_country_pattern_verified": test_countries == ["France", "India", "US"],
            "zero_match_s1_fraction": gt["s1_cardinality"]["overall"]["zero_match_fraction"],
            "max_s2_owners": ownership["S2"]["max_owners"],
            "max_s3_owners": ownership["S3"]["max_owners"],
            "cross_country_gt_pairs": consistency["combined_mismatches"],
        }
        observe(monitor.sample())
        runtime = time.monotonic() - started
        audit = {
            "audit_version": version, "dataset_root": str(paths.root),
            "dataset_fingerprint": dataset_fingerprint,
            "files": files, "sources": sources, "id_overlaps": overlaps,
            "ground_truth": gt, "script_characteristics": scripts,
            "optional_diagnostics": "SKIPPED_FOR_RUNTIME",
            "metric_verification": _metric_status(paths.root), "invariants": invariants,
            "runtime_seconds": runtime, "peak_rss_bytes": peak_rss,
        }
        contract = {
            "audit_version": version, "dataset_fingerprint": audit["dataset_fingerprint"],
            "files": {
                name: {key: item[key] for key in (
                    "logical_name", "filename", "file_size_bytes", "delimiter", "encoding",
                    "columns", "column_order", "inferred_dtypes", "row_count",
                )} for name, item in files.items()
            },
            "ground_truth_structure": {
                "columns": files["train_ground_truth"]["columns"],
                "representation": "one S1 row with comma-separated S2/S3 IDs; empty string means zero matches",
            },
            "invariants": invariants,
        }
        _atomic_json(output / "audit.json", audit)
        _atomic_json(output / "dataset_contract.json", contract)
        _atomic_json(output / "input_fingerprints.json", fingerprints)
        _atomic_json(output / "invariants.json", invariants)
        (output / "audit.md").write_text(_render_markdown(audit), encoding="utf-8")
        print("[DATA-AUDIT] complete", flush=True)
        return audit
    finally:
        stop.set()
        heartbeat.join(timeout=10)
        connection.close()
