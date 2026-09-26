"""Read-only Phase 2A/2B union and blocker diagnostic audit."""
from __future__ import annotations

import json
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

from amazon_er.data.paths import resolve_dataset_paths
from amazon_er.infra.manifests import load_manifest
from amazon_er.paths import resolve_project_path
from amazon_er.pipeline.exact_structured import _discover_normalized, _paths_sql


STAGE = "phase2ab_union_audit"
K_VALUES = (1, 3, 5, 10, 20, 50)
VOLUME_K_VALUES = (5, 10, 20, 50)


class Phase2ABAuditError(RuntimeError):
    """The stored artifacts cannot support a trustworthy audit."""


@dataclass(frozen=True)
class CandidateShard:
    path: Path
    split: str
    country: str
    source: str
    shard_id: str
    rows: int


def _imports() -> tuple[Any, Any]:
    try:
        import duckdb
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError("Phase 2A/2B audit requires the data extra") from exc
    return duckdb, pq


def _parquet_for_manifest(path: Path) -> Path:
    suffix = ".manifest.json"
    if not path.name.endswith(suffix):
        raise Phase2ABAuditError(f"Unexpected manifest name: {path}")
    return path.with_name(path.name[:-len(suffix)] + ".parquet")


def _discover_candidates(root: Path, stage: str, required: Iterable[str]) -> list[CandidateShard]:
    _, pq = _imports()
    if not root.is_dir():
        raise Phase2ABAuditError(f"{stage} root does not exist: {root}")
    shards: list[CandidateShard] = []
    required_set = set(required)
    for manifest_path in sorted(root.rglob("part-*.manifest.json")):
        manifest = load_manifest(manifest_path)
        if manifest.stage != stage:
            continue
        if manifest.status != "complete" or manifest.split not in {"train", "test"}:
            raise Phase2ABAuditError(f"Incomplete {stage} manifest: {manifest_path}")
        if manifest.source not in {"S2", "S3"} or not manifest.country:
            raise Phase2ABAuditError(f"Invalid {stage} identity: {manifest_path}")
        artifact = _parquet_for_manifest(manifest_path)
        if not artifact.is_file() or artifact.stat().st_size != manifest.artifact_size_bytes:
            raise Phase2ABAuditError(f"Missing or size-mismatched artifact: {artifact}")
        parquet = pq.ParquetFile(artifact)
        if parquet.metadata.num_rows != manifest.output_rows:
            raise Phase2ABAuditError(f"Row-count mismatch: {artifact}")
        missing = required_set - set(parquet.schema_arrow.names)
        if missing:
            raise Phase2ABAuditError(f"Schema mismatch in {artifact}; missing {sorted(missing)}")
        shards.append(CandidateShard(
            artifact, manifest.split, manifest.country, manifest.source,
            manifest.shard_id, manifest.output_rows,
        ))
    if not shards:
        raise Phase2ABAuditError(f"No {stage} artifacts found below {root}")
    identities = [(x.split, x.country, x.source, x.shard_id) for x in shards]
    if len(identities) != len(set(identities)):
        raise Phase2ABAuditError(f"Duplicate {stage} shard identities")
    return shards


def _rows(connection: Any, sql: str) -> list[dict[str, Any]]:
    cursor = connection.execute(sql)
    names = [item[0] for item in cursor.description]
    return [dict(zip(names, row)) for row in cursor.fetchall()]


def _ratio(count: int, total: int) -> float | None:
    return count / total if total else None


def _metric_record(row: Mapping[str, Any]) -> dict[str, Any]:
    total = int(row["gt_pairs"])
    names = ("phase2a", "phase2b_k50", "intersection", "phase2a_only", "phase2b_only", "union")
    result: dict[str, Any] = {"gt_pairs": total}
    for name in names:
        count = int(row[f"{name}_count"])
        result[name] = {"count": count, "recall": _ratio(count, total)}
        result[f"{name}_count"] = count
        result[f"{name}_recall"] = _ratio(count, total)
    return result


def _atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def _normal_view(shards: list[Any], *, split: str, sources: set[str], source_alias: bool) -> str:
    parts = []
    for source in sorted(sources):
        paths = [x.path for x in shards if x.split == split and x.source == source]
        if not paths:
            continue
        alias = source.replace("source", "S")
        extra = f", '{alias}'::VARCHAR AS target_source" if source_alias else ""
        parts.append(f"SELECT *{extra} FROM read_parquet({_paths_sql(paths)}, union_by_name=true)")
    if not parts:
        raise Phase2ABAuditError(f"No normalized {split}/{sorted(sources)} shards")
    return " UNION ALL ".join(parts)


def _configure(connection: Any, cfg: Mapping[str, Any], temp_dir: Path) -> None:
    memory = str(cfg["audit"]["duckdb_memory_limit"]).replace("'", "")
    threads = int(cfg["audit"]["duckdb_threads"])
    temp = str(temp_dir).replace("\\", "/").replace("'", "''")
    connection.execute(f"SET memory_limit='{memory}'")
    connection.execute(f"SET threads={threads}")
    connection.execute(f"SET temp_directory='{temp}'")
    connection.execute("SET preserve_insertion_order=false")


def _create_base_views(connection: Any, normalized: list[Any], exact: list[CandidateShard], blocker: list[CandidateShard], truth: Path) -> None:
    s1_sql = _normal_view(normalized, split="train", sources={"source1"}, source_alias=False)
    target_train_sql = _normal_view(normalized, split="train", sources={"source2", "source3"}, source_alias=True)
    target_all_sql = " UNION ALL ".join([
        _normal_view(normalized, split=split, sources={"source2", "source3"}, source_alias=True)
        for split in ("train", "test")
    ])
    connection.execute(f"CREATE TEMP VIEW s1_train AS {s1_sql}")
    connection.execute(f"CREATE TEMP VIEW target_train AS {target_train_sql}")
    connection.execute(f"CREATE TEMP VIEW target_all AS {target_all_sql}")
    connection.execute(
        f"CREATE TEMP VIEW phase2a AS SELECT target_entity_id, candidate_s1_entity_id, "
        f"target_source, country FROM read_parquet({_paths_sql([x.path for x in exact])}, union_by_name=true)"
    )
    connection.execute(
        f"CREATE TEMP VIEW phase2b AS SELECT target_entity_id, candidate_s1_entity_id, target_source, "
        f"country, blocker_rank, block_translit_name_hit, block_translit_name_number_hit "
        f"FROM read_parquet({_paths_sql([x.path for x in blocker])}, union_by_name=true)"
    )
    truth_sql = str(truth).replace("\\", "/").replace("'", "''")
    connection.execute(
        "CREATE TEMP TABLE gt_raw AS "
        f"SELECT source1_entity_id, trim(target_id) AS target_entity_id FROM read_csv_auto('{truth_sql}', "
        "delim='\\t', header=true, all_varchar=true) "
        "CROSS JOIN UNNEST(string_split(coalesce(matched_entity_ids, ''), ',')) AS u(target_id) "
        "WHERE trim(target_id) <> ''"
    )
    duplicate_targets = connection.execute(
        "SELECT count(*) FROM (SELECT entity_id FROM target_train GROUP BY entity_id HAVING count(*) > 1)"
    ).fetchone()[0]
    if duplicate_targets:
        raise Phase2ABAuditError(f"Target entity IDs are not unique in normalized train data: {duplicate_targets}")
    connection.execute(
        "CREATE TEMP TABLE gt_pairs AS SELECT DISTINCT g.source1_entity_id, g.target_entity_id, "
        "t.target_source, t.country FROM gt_raw g JOIN target_train t ON t.entity_id=g.target_entity_id"
    )
    raw_count = connection.execute("SELECT count(*) FROM (SELECT DISTINCT * FROM gt_raw)").fetchone()[0]
    joined_count = connection.execute("SELECT count(*) FROM gt_pairs").fetchone()[0]
    if raw_count != joined_count:
        raise Phase2ABAuditError(f"GT target resolution mismatch: parsed={raw_count}, normalized={joined_count}")


def _union_metrics(connection: Any) -> tuple[dict[str, Any], dict[str, Any]]:
    connection.execute(
        "CREATE TEMP TABLE gt_flags AS WITH a AS (SELECT DISTINCT g.source1_entity_id,g.target_entity_id,g.target_source,g.country "
        "FROM gt_pairs g JOIN phase2a p ON p.target_entity_id=g.target_entity_id AND p.candidate_s1_entity_id=g.source1_entity_id "
        "AND p.target_source=g.target_source AND p.country=g.country), b AS (SELECT g.source1_entity_id,g.target_entity_id,g.target_source,g.country, "
        "min(p.blocker_rank) AS blocker_rank FROM gt_pairs g JOIN phase2b p ON p.target_entity_id=g.target_entity_id "
        "AND p.candidate_s1_entity_id=g.source1_entity_id AND p.target_source=g.target_source AND p.country=g.country GROUP BY ALL) "
        "SELECT g.*, (a.target_entity_id IS NOT NULL) AS a_hit, coalesce(b.blocker_rank, 255) AS b_rank "
        "FROM gt_pairs g LEFT JOIN a USING(source1_entity_id,target_entity_id,target_source,country) "
        "LEFT JOIN b USING(source1_entity_id,target_entity_id,target_source,country)"
    )
    select = (
        "count(*) AS gt_pairs, count(*) FILTER (a_hit) AS phase2a_count, "
        "count(*) FILTER (b_rank<=50) AS phase2b_k50_count, "
        "count(*) FILTER (a_hit AND b_rank<=50) AS intersection_count, "
        "count(*) FILTER (a_hit AND b_rank>50) AS phase2a_only_count, "
        "count(*) FILTER (NOT a_hit AND b_rank<=50) AS phase2b_only_count, "
        "count(*) FILTER (a_hit OR b_rank<=50) AS union_count"
    )
    overall = _metric_record(_rows(connection, f"SELECT {select} FROM gt_flags")[0])
    breakdown_rows = _rows(connection, f"SELECT country, target_source, {select} FROM gt_flags GROUP BY country,target_source ORDER BY 1,2")
    breakdown = {f"{x['country']} {x['target_source']}": _metric_record(x) for x in breakdown_rows}
    by_k = {}
    total = overall["gt_pairs"]
    for k in K_VALUES:
        union_count, unique_count = connection.execute(
            f"SELECT count(*) FILTER (a_hit OR b_rank<={k}), count(*) FILTER (NOT a_hit AND b_rank<={k}) FROM gt_flags"
        ).fetchone()
        by_k[str(k)] = {
            "union_count": int(union_count), "union_recall": _ratio(int(union_count), total),
            "unique_beyond_phase2a_count": int(unique_count),
            "unique_beyond_phase2a_recall": _ratio(int(unique_count), total),
        }
    return {"overall": overall, "by_country_source": breakdown}, by_k


def _candidate_volume(connection: Any, exact: list[CandidateShard], blocker: list[CandidateShard]) -> dict[str, Any]:
    exact_by_id = {(x.split, x.country, x.source, x.shard_id): x.path for x in exact}
    blocker_by_id = {(x.split, x.country, x.source, x.shard_id): x.path for x in blocker}
    parts = []
    counts = ", ".join(f"count(*) FILTER (is_a OR b_rank<={k})::BIGINT AS c{k}" for k in VOLUME_K_VALUES)
    for identity in sorted(exact_by_id):
        a_path, b_path = exact_by_id[identity], blocker_by_id[identity]
        parts.append(
            f"SELECT target_entity_id,target_source,country,{counts} FROM (SELECT target_entity_id,candidate_s1_entity_id,"
            "target_source,country,max(is_a) AS is_a,min(b_rank) AS b_rank FROM ("
            f"SELECT target_entity_id,candidate_s1_entity_id,target_source,country,true AS is_a,255::UTINYINT AS b_rank FROM read_parquet({_paths_sql([a_path])}) UNION ALL "
            f"SELECT target_entity_id,candidate_s1_entity_id,target_source,country,false,blocker_rank FROM read_parquet({_paths_sql([b_path])})"
            ") GROUP BY 1,2,3,4) GROUP BY 1,2,3"
        )
    connection.execute("CREATE TEMP TABLE candidate_counts AS " + " UNION ALL ".join(parts))
    universe_count = connection.execute("SELECT count(*) FROM target_all").fetchone()[0]
    result: dict[str, Any] = {}
    for k in VOLUME_K_VALUES:
        row = _rows(connection, f"WITH x AS (SELECT coalesce(c.c{k},0)::BIGINT AS n FROM target_all t LEFT JOIN "
            "candidate_counts c ON c.target_entity_id=t.entity_id AND c.target_source=t.target_source AND c.country=t.country) "
            "SELECT sum(n) AS total_candidate_pairs, avg(n) AS average_candidates_per_target, "
            "quantile_disc(n,0.50) AS p50, quantile_disc(n,0.90) AS p90, quantile_disc(n,0.95) AS p95, "
            "quantile_disc(n,0.99) AS p99, max(n) AS max FROM x")[0]
        result[str(k)] = {"target_count": int(universe_count), **{
            name: (float(value) if name == "average_candidates_per_target" else int(value))
            for name, value in row.items()
        }}
    return result


def _token_diagnostics(connection: Any, cfg: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    mk = cfg["retrieval"]["multikey"]
    min_name, min_address = int(mk["min_name_token_length"]), int(mk["min_address_token_length"])
    current_name, current_address = int(mk["rare_name_token_max_df"]), int(mk["rare_address_token_max_df"])
    max_name, max_address = int(mk["max_name_tokens_per_target"]), int(mk["max_address_tokens_per_target"])
    connection.execute("CREATE TEMP TABLE rules(rule VARCHAR,name_df INTEGER,address_df INTEGER,max_name INTEGER,max_address INTEGER)")
    connection.executemany("INSERT INTO rules VALUES (?,?,?,?,?)", [
        ("current", current_name, current_address, max_name, max_address),
        ("df100", 100, 100, 4, 4), ("df300", 300, 300, 4, 4), ("df1000", 1000, 1000, 4, 4),
    ])
    name_expr = "coalesce(nullif(name_core,''),name_tokens)"
    eligible = "(octet_length(encode(coalesce(name_raw,'')))<>length(coalesce(name_raw,'')) OR name_transliterated<>name_tokens)"
    for family, expression, minimum in (
        ("name", name_expr, min_name), ("address", "address_normalized", min_address),
        ("translit", "name_transliterated", min_name),
    ):
        connection.execute(
            f"CREATE TEMP TABLE {family}_df AS SELECT country,token,count(*)::INTEGER AS df FROM ("
            f"SELECT DISTINCT country,entity_id,token FROM s1_train CROSS JOIN UNNEST(string_split(trim({expression}),' ')) AS u(token) "
            f"WHERE length(token)>={minimum} AND token<>'') GROUP BY country,token"
        )
    connection.execute(
        "CREATE TEMP VIEW audit_rows AS "
        "SELECT DISTINCT 's1' AS kind,s.entity_id,s.country,NULL::VARCHAR AS target_source,s.name_raw,s.name_tokens,s.name_core,s.name_transliterated,s.address_normalized,s.numeric_tokens "
        "FROM s1_train s JOIN gt_pairs g ON g.source1_entity_id=s.entity_id AND g.country=s.country "
        "UNION ALL SELECT DISTINCT 'target',t.entity_id,t.country,t.target_source,t.name_raw,t.name_tokens,t.name_core,t.name_transliterated,t.address_normalized,t.numeric_tokens "
        "FROM target_train t JOIN gt_pairs g ON g.target_entity_id=t.entity_id AND g.country=t.country AND g.target_source=t.target_source"
    )
    selections = (
        ("name", "coalesce(nullif(a.name_core,''),a.name_tokens)", min_name, "r.name_df", "r.max_name", "true"),
        ("address", "a.address_normalized", min_address, "r.address_df", "r.max_address", "true"),
        ("translit", "a.name_transliterated", min_name, "r.name_df", "r.max_name", eligible),
    )
    for family, expression, minimum, threshold, maximum, condition in selections:
        connection.execute(
            f"CREATE TEMP TABLE selected_{family} AS SELECT kind,entity_id,country,target_source,rule,token,rn FROM ("
            f"SELECT b.kind,b.entity_id,b.country,b.target_source,r.rule,b.token,d.df,row_number() OVER ("
            f"PARTITION BY b.kind,b.entity_id,b.country,b.target_source,r.rule ORDER BY d.df,b.token) AS rn,r.{maximum.split('.')[1]} AS lim "
            f"FROM (SELECT DISTINCT a.kind,a.entity_id,a.country,a.target_source,u.token FROM audit_rows a "
            f"CROSS JOIN UNNEST(string_split(trim({expression}),' ')) AS u(token) WHERE length(u.token)>={minimum} "
            f"AND u.token<>'' AND {condition}) b CROSS JOIN rules r JOIN {family}_df d ON d.country=b.country AND d.token=b.token "
            f"WHERE d.df<={threshold}) WHERE rn<=lim"
        )
    connection.execute(
        "CREATE TEMP TABLE selected_number AS SELECT DISTINCT kind,entity_id,country,target_source,token FROM audit_rows "
        "CROSS JOIN UNNEST(list_slice(string_split(coalesce(numeric_tokens,''),'|'),1,2)) AS u(token) WHERE token<>''"
    )
    connection.execute(
        "CREATE TEMP TABLE address_signature AS SELECT kind,entity_id,country,target_source,rule,"
        "string_agg(token,chr(31) ORDER BY token) AS signature FROM selected_address WHERE rn<=2 GROUP BY 1,2,3,4,5 HAVING count(*)>=2"
    )
    hit_specs = {
        "name": "selected_name", "address": "selected_address", "translit": "selected_translit",
    }
    for label, table in hit_specs.items():
        connection.execute(
            f"CREATE TEMP TABLE hit_{label} AS SELECT DISTINCT g.source1_entity_id,g.target_entity_id,g.country,g.target_source,s.rule "
            f"FROM gt_pairs g JOIN {table} s ON s.kind='s1' AND s.entity_id=g.source1_entity_id AND s.country=g.country "
            f"JOIN {table} t ON t.kind='target' AND t.entity_id=g.target_entity_id AND t.country=g.country "
            f"AND t.target_source=g.target_source AND t.rule=s.rule AND t.token=s.token"
        )
    connection.execute(
        "CREATE TEMP TABLE hit_number AS SELECT DISTINCT g.source1_entity_id,g.target_entity_id,g.country,g.target_source "
        "FROM gt_pairs g JOIN selected_number s ON s.kind='s1' AND s.entity_id=g.source1_entity_id AND s.country=g.country "
        "JOIN selected_number t ON t.kind='target' AND t.entity_id=g.target_entity_id AND t.country=g.country "
        "AND t.target_source=g.target_source AND t.token=s.token"
    )
    connection.execute(
        "CREATE TEMP TABLE hit_address_pair AS SELECT DISTINCT g.source1_entity_id,g.target_entity_id,g.country,g.target_source,s.rule "
        "FROM gt_pairs g JOIN address_signature s ON s.kind='s1' AND s.entity_id=g.source1_entity_id AND s.country=g.country "
        "JOIN address_signature t ON t.kind='target' AND t.entity_id=g.target_entity_id AND t.country=g.country "
        "AND t.target_source=g.target_source AND t.rule=s.rule AND t.signature=s.signature"
    )
    connection.execute(
        "CREATE TEMP TABLE key_ceiling_pairs AS SELECT g.*,r.rule,(hn.target_entity_id IS NOT NULL) AS name_hit,"
        "(ha.target_entity_id IS NOT NULL) AS address_hit,(hnum.target_entity_id IS NOT NULL) AS number_hit,"
        "(hap.target_entity_id IS NOT NULL) AS address_pair_hit,(ht.target_entity_id IS NOT NULL) AS translit_hit "
        "FROM gt_pairs g CROSS JOIN rules r LEFT JOIN hit_name hn USING(source1_entity_id,target_entity_id,country,target_source,rule) "
        "LEFT JOIN hit_address ha USING(source1_entity_id,target_entity_id,country,target_source,rule) "
        "LEFT JOIN hit_number hnum USING(source1_entity_id,target_entity_id,country,target_source) "
        "LEFT JOIN hit_address_pair hap USING(source1_entity_id,target_entity_id,country,target_source,rule) "
        "LEFT JOIN hit_translit ht USING(source1_entity_id,target_entity_id,country,target_source,rule)"
    )
    expression = "(name_hit OR (number_hit AND address_hit) OR address_pair_hit OR translit_hit)"
    ceiling_rows = _rows(connection, f"SELECT rule,count(*) AS gt_pairs,count(*) FILTER ({expression}) AS covered FROM key_ceiling_pairs GROUP BY rule")
    ceiling = {x["rule"]: {"gt_pairs": int(x["gt_pairs"]), "covered_count": int(x["covered"]), "coverage": _ratio(int(x["covered"]), int(x["gt_pairs"]))} for x in ceiling_rows}
    detail_rows = _rows(connection, f"SELECT rule,country,target_source,count(*) AS gt_pairs,count(*) FILTER ({expression}) AS covered FROM key_ceiling_pairs GROUP BY 1,2,3 ORDER BY 1,2,3")
    for row in detail_rows:
        ceiling[row["rule"]].setdefault("by_country_source", {})[f"{row['country']} {row['target_source']}"] = {
            "gt_pairs": int(row["gt_pairs"]), "covered_count": int(row["covered"]),
            "coverage": _ratio(int(row["covered"]), int(row["gt_pairs"])),
        }
    target_eligible = connection.execute(f"SELECT count(*) FROM target_train WHERE {eligible}").fetchone()[0]
    targets_with_keys = connection.execute(
        f"SELECT count(DISTINCT (t.entity_id,t.country,t.target_source)) FROM target_train t CROSS JOIN UNNEST("
        "string_split(trim(t.name_transliterated),' ')) u(token) JOIN translit_df d ON d.country=t.country AND d.token=u.token "
        f"WHERE {eligible.replace('name_raw','t.name_raw').replace('name_transliterated','t.name_transliterated').replace('name_tokens','t.name_tokens')} "
        f"AND length(u.token)>={min_name} AND d.df<={current_name}"
    ).fetchone()[0]
    current = "rule='current'"
    shared, unique, already_name = connection.execute(
        f"SELECT count(*) FILTER (translit_hit), count(*) FILTER (translit_hit AND NOT (name_hit OR (number_hit AND address_hit) OR address_pair_hit)), "
        f"count(*) FILTER (translit_hit AND name_hit) FROM key_ceiling_pairs WHERE {current}"
    ).fetchone()
    cross_namespace, lost_cross_namespace = connection.execute(
        "SELECT count(DISTINCT (g.source1_entity_id,g.target_entity_id,g.country,g.target_source)), "
        "count(DISTINCT (g.source1_entity_id,g.target_entity_id,g.country,g.target_source)) FILTER (ht.target_entity_id IS NULL) FROM gt_pairs g "
        "JOIN selected_name s ON s.kind='s1' AND s.entity_id=g.source1_entity_id AND s.country=g.country AND s.rule='current' "
        "JOIN selected_translit t ON t.kind='target' AND t.entity_id=g.target_entity_id AND t.country=g.country "
        "AND t.target_source=g.target_source AND t.rule='current' AND t.token=s.token "
        "LEFT JOIN hit_translit ht ON ht.source1_entity_id=g.source1_entity_id AND ht.target_entity_id=g.target_entity_id "
        "AND ht.country=g.country AND ht.target_source=g.target_source AND ht.rule='current'"
    ).fetchone()
    stored_targets = connection.execute(
        "SELECT count(DISTINCT (b.target_entity_id,b.country,b.target_source)) FROM phase2b b JOIN target_train t "
        "ON t.entity_id=b.target_entity_id AND t.country=b.country AND t.target_source=b.target_source "
        "WHERE b.block_translit_name_hit OR b.block_translit_name_number_hit"
    ).fetchone()[0]
    bug = int(lost_cross_namespace) > 0
    transliteration = {
        "targets_eligible": int(target_eligible), "targets_with_translit_keys": int(targets_with_keys),
        "stored_targets_with_translit_candidate_hits": int(stored_targets),
        "gt_pairs_sharing_translit_key": int(shared), "gt_pairs_uniquely_recovered_via_translit": int(unique),
        "gt_pairs_sharing_translit_and_normal_name": int(already_name),
        "target_translit_to_s1_normal_gt_pairs": int(cross_namespace),
        "target_translit_to_s1_normal_without_shared_translit_key": int(lost_cross_namespace),
        "cross_namespace_correctness_issue": bug,
        "reason": (
            "CLEAR CORRECTNESS ISSUE: translit-family keys are gated independently on both rows; a meaningful target transliteration cannot match an ASCII S1 normal-name key because the key families differ."
            if bug else "No cross-namespace loss was observed; zero unique recovery is explained by absent shared translit keys or overlap with normal keys."
        ),
    }
    return ceiling, transliteration


def _markdown(summary: Mapping[str, Any]) -> str:
    o = summary["union_metrics"]["overall"]
    lines = ["# PHASE 2B.1 RESULT", "", "## PHASE2A", f"Recall: {o['phase2a']['recall']} ({o['phase2a']['count']}/{o['gt_pairs']})", "",
        "## PHASE2B K50", f"Recall: {o['phase2b_k50']['recall']} ({o['phase2b_k50']['count']}/{o['gt_pairs']})", "",
        "## OVERLAP", f"A ∩ B: {o['intersection']['count']} / {o['intersection']['recall']}", "", "## UNIQUE",
        f"A only: {o['phase2a_only']['count']} / {o['phase2a_only']['recall']}", f"B only: {o['phase2b_only']['count']} / {o['phase2b_only']['recall']}", "",
        "## UNION", f"A ∪ B: {o['union']['count']} / {o['union']['recall']}", "", "## UNION BY K", "", "| K | union count | recall | unique beyond A |", "|---:|---:|---:|---:|"]
    for k, item in summary["union_by_k"].items(): lines.append(f"| {k} | {item['union_count']} | {item['union_recall']} | {item['unique_beyond_phase2a_count']} |")
    lines += ["", "## COUNTRY/SOURCE", "", "| partition | GT | A recall | B K50 recall | union recall |", "|---|---:|---:|---:|---:|"]
    for label, item in summary["union_metrics"]["by_country_source"].items(): lines.append(f"| {label} | {item['gt_pairs']} | {item['phase2a']['recall']} | {item['phase2b_k50']['recall']} | {item['union']['recall']} |")
    lines += ["", "## CANDIDATE VOLUME BY K", "", "| K | pairs | avg | p50 | p90 | p95 | p99 | max |", "|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for k, item in summary["candidate_volume_by_k"].items(): lines.append(f"| {k} | {item['total_candidate_pairs']} | {item['average_candidates_per_target']} | {item['p50']} | {item['p90']} | {item['p95']} | {item['p99']} | {item['max']} |")
    lines += ["", "## RAW COVERAGE SEMANTICS", "", summary["raw_coverage_semantics"]["exact_definition"], "", "Label: **RESTRICTED_KEY_COVERAGE**", "", "## TRUE-CEILING DIAGNOSTIC", "", "| rule | covered | GT | coverage |", "|---|---:|---:|---:|"]
    for rule, item in summary["true_ceiling_diagnostic"].items(): lines.append(f"| {rule} | {item['covered_count']} | {item['gt_pairs']} | {item['coverage']} |")
    t = summary["transliteration"]
    lines += ["", "## TRANSLITERATION", "", f"Eligible targets: {t['targets_eligible']}", f"Targets with translit keys: {t['targets_with_translit_keys']}", f"Shared GT: {t['gt_pairs_sharing_translit_key']}", f"Unique GT: {t['gt_pairs_uniquely_recovered_via_translit']}", f"Reason: {t['reason']}", "", "## BLOCKERS"]
    lines += [f"- {x}" for x in summary["blockers"]] or ["- None"]
    lines += ["", "## NEXT", "", "Ready to choose Phase 2C retrieval based on the measured union ceiling.", ""]
    return "\n".join(lines)


def run_phase2ab_union_audit(cfg: Mapping[str, Any], *, normalized_root: str | None,
    exact_structured_root: str | None, multikey_root: str | None, data_root: str | None = None,
    output_dir: str | None = None) -> dict[str, Any]:
    if not normalized_root or not exact_structured_root or not multikey_root:
        raise Phase2ABAuditError("--normalized-root, --exact-structured-root, and --multikey-root are required")
    normalized = _discover_normalized(Path(normalized_root).resolve())
    exact = _discover_candidates(Path(exact_structured_root).resolve(), "exact_structured", {
        "target_entity_id", "candidate_s1_entity_id", "target_source", "country",
    })
    blocker = _discover_candidates(Path(multikey_root).resolve(), "multikey_blocker", {
        "target_entity_id", "candidate_s1_entity_id", "target_source", "country", "blocker_rank",
        "block_translit_name_hit", "block_translit_name_number_hit",
    })
    target_ids = {(x.split, x.country, x.source.replace("source", "S"), x.shard_id) for x in normalized if x.source in {"source2", "source3"}}
    for name, shards in (("Phase2A", exact), ("Phase2B", blocker)):
        identities = {(x.split, x.country, x.source, x.shard_id) for x in shards}
        if identities != target_ids:
            raise Phase2ABAuditError(f"{name} shard inventory differs from normalization: missing={len(target_ids-identities)}, extra={len(identities-target_ids)}")
    paths = resolve_dataset_paths(cfg, data_root)
    report_root = Path(output_dir).resolve() if output_dir else resolve_project_path(cfg, "artifact_root") / "reports" / "retrieval" / str(cfg["retrieval"]["version"]) / STAGE
    report_root.mkdir(parents=True, exist_ok=True)
    duckdb, _ = _imports()
    temporary = Path(tempfile.mkdtemp(prefix="phase2ab-audit-", dir=report_root.parent))
    connection = duckdb.connect(":memory:")
    try:
        _configure(connection, cfg, temporary)
        _create_base_views(connection, normalized, exact, blocker, paths.train_ground_truth)
        union_metrics, union_by_k = _union_metrics(connection)
        volume = _candidate_volume(connection, exact, blocker)
        ceiling, transliteration = _token_diagnostics(connection, cfg)
    finally:
        connection.close()
        shutil.rmtree(temporary, ignore_errors=True)
    blockers = []
    if transliteration["cross_namespace_correctness_issue"]:
        blockers.append("Transliteration namespace gating loses observed GT overlap; production retrieval was not changed by this audit.")
    summary = {
        "status": "complete", "stage": STAGE,
        "inputs": {"normalized_shards": len(normalized), "phase2a_shards": len(exact), "phase2b_shards": len(blocker)},
        "union_metrics": union_metrics, "union_by_k": union_by_k, "candidate_volume_by_k": volume,
        "raw_coverage_semantics": {
            "label": "RESTRICTED_KEY_COVERAGE", "before_bucket_cap_pruning": True,
            "after_df_eligibility": True, "after_token_selection_limits": True,
            "after_minimum_token_lengths": True, "after_key_validity_rules": True,
            "exact_definition": "The stored raw_key_coverage tests overlap between _row_keys(target) and index.entity_keys[true]. Both key sets have already applied minimum lengths, S1-derived DF eligibility, rare-token ordering, per-row token limits, numeric limits, transliteration eligibility, and key construction. It is measured before bucket-cap suppression and top-K truncation; it is not an unlimited key ceiling.",
        },
        "true_ceiling_diagnostic": ceiling, "transliteration": transliteration,
        "blockers": blockers,
        "next": "Ready to choose Phase 2C retrieval based on union ceiling.",
    }
    _atomic(report_root / "summary.json", json.dumps(summary, indent=2, sort_keys=True) + "\n")
    _atomic(report_root / "summary.md", _markdown(summary))
    return summary
