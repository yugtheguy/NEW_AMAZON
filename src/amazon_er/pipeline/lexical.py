"""Batch-bounded target-to-S1 lexical retrieval for Phase 2B."""

from __future__ import annotations

import json
import os
import subprocess
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

from amazon_er.config import config_hash
from amazon_er.data.paths import resolve_dataset_paths
from amazon_er.infra.artifacts import validate_artifact
from amazon_er.infra.hashing import hash_file, hash_mapping
from amazon_er.infra.manifests import ShardManifest, load_manifest
from amazon_er.infra.resources import ResourceMonitor
from amazon_er.paths import project_root, resolve_project_path
from amazon_er.pipeline.exact_structured import NormalizedShard, _discover_normalized


STAGE = "lexical"
FAMILIES = ("name_word", "address_word", "transliteration", "rare_token", "numeric")
SCHEMAS = {
    "name_word": (("word_name_present", "bool"), ("word_name_score", "float32"), ("word_name_rank", "uint8")),
    "address_word": (("address_present", "bool"), ("address_score", "float32"), ("address_rank", "uint8")),
    "transliteration": (("translit_hit", "bool"), ("translit_score", "float32"), ("translit_rank", "uint8")),
    "rare_token": (("rare_token_hit", "bool"), ("rare_token_min_df", "uint32"), ("rare_token_overlap_count", "uint8")),
    "numeric": (("numeric_hit", "bool"), ("numeric_overlap_count", "uint8")),
}
IDENTITY = ("target_entity_id", "candidate_s1_entity_id", "target_source", "country")


class LexicalError(RuntimeError):
    pass


def _imports() -> tuple[Any, Any, Any]:
    try:
        import numpy as np
        import pyarrow as pa
        from sklearn.feature_extraction.text import TfidfVectorizer
    except ImportError as exc:
        raise RuntimeError("Lexical retrieval requires: pip install -e '.[data,retrieval]'") from exc
    return np, pa, TfidfVectorizer


def _code_commit() -> str:
    root = project_root()
    try:
        result = subprocess.run(
            ["git", "-c", f"safe.directory={root}", "rev-parse", "HEAD"], cwd=root,
            capture_output=True, text=True, timeout=5, check=True,
        )
        return result.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return os.environ.get("CODE_COMMIT", "UNAVAILABLE")


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _pa_schema(family: str) -> Any:
    _, pa, _ = _imports()
    types = {"bool": pa.bool_(), "float32": pa.float32(), "uint8": pa.uint8(), "uint32": pa.uint32()}
    return pa.schema([
        *(pa.field(name, pa.string()) for name in IDENTITY),
        *(pa.field(name, types[kind]) for name, kind in SCHEMAS[family]),
    ])


def _tokens(value: str | None) -> set[str]:
    return set((value or "").split())


def _numeric_keys(row: Mapping[str, Any]) -> set[str]:
    keys: set[str] = set()
    primary = row.get("primary_number")
    if primary:
        keys.add("p:" + str(primary))
    for value in str(row.get("numeric_tokens") or "").split("|"):
        if value:
            keys.add("n:" + value)
    for value in str(row.get("postal_like_tokens") or "").split("|"):
        if value:
            keys.add("z:" + value)
    return keys


def _bounded_postings(documents: Iterable[set[str]], max_df: int) -> tuple[dict[str, list[int]], dict[str, int]]:
    cached = list(documents)
    counts: Counter[str] = Counter()
    for values in cached:
        counts.update(values)
    eligible = {token for token, count in counts.items() if count <= max_df}
    postings: dict[str, list[int]] = defaultdict(list)
    for index, values in enumerate(cached):
        for token in values & eligible:
            postings[token].append(index)
    return dict(postings), {token: counts[token] for token in eligible}


@dataclass
class CountryIndex:
    ids: list[str]
    name_vectorizer: Any
    name_matrix: Any
    address_vectorizer: Any
    address_matrix: Any
    rare_postings: dict[str, list[int]]
    rare_df: dict[str, int]
    numeric_postings: dict[str, list[int]]


def build_country_index(paths: list[Path], behavior: Mapping[str, Any]) -> CountryIndex:
    _, pa, TfidfVectorizer = _imports()
    import pyarrow.parquet as pq
    columns = [
        "entity_id", "name_core", "address_normalized", "name_tokens",
        "primary_number", "numeric_tokens", "postal_like_tokens",
    ]
    table = pa.concat_tables([pq.ParquetFile(path).read(columns=columns) for path in paths])
    values = {name: table[name].to_pylist() for name in columns}
    import numpy as np
    kwargs = dict(
        ngram_range=(int(behavior["word_ngram_min"]), int(behavior["word_ngram_max"])),
        min_df=int(behavior["min_df"]), dtype=np.float32, norm="l2", lowercase=False,
        token_pattern=r"(?u)\b\w+\b",
    )
    name_vectorizer = TfidfVectorizer(**kwargs)
    address_vectorizer = TfidfVectorizer(**kwargs)
    name_matrix = name_vectorizer.fit_transform(values["name_core"])
    address_matrix = address_vectorizer.fit_transform(values["address_normalized"])
    rare_docs = (
        _tokens(name) | _tokens(address)
        for name, address in zip(values["name_tokens"], values["address_normalized"])
    )
    rare_postings, rare_df = _bounded_postings(rare_docs, int(behavior["rare_token_max_df"]))
    numeric_docs = (
        _numeric_keys({name: values[name][i] for name in ("primary_number", "numeric_tokens", "postal_like_tokens")})
        for i in range(table.num_rows)
    )
    numeric_postings, _ = _bounded_postings(numeric_docs, int(behavior["numeric_max_df"]))
    return CountryIndex(
        [str(value) for value in values["entity_id"]], name_vectorizer, name_matrix,
        address_vectorizer, address_matrix, rare_postings, rare_df, numeric_postings,
    )


def _topk(matrix: Any, ids: list[str], *, k: int, min_score: float) -> list[list[tuple[int, float, int]]]:
    np, _, _ = _imports()
    result: list[list[tuple[int, float, int]]] = []
    matrix = matrix.tocsr()
    for row_index in range(matrix.shape[0]):
        start, end = matrix.indptr[row_index], matrix.indptr[row_index + 1]
        indices, scores = matrix.indices[start:end], matrix.data[start:end]
        keep = scores >= min_score
        indices, scores = indices[keep], scores[keep]
        if len(scores) > k:
            chosen = np.argpartition(scores, -k)[-k:]
            indices, scores = indices[chosen], scores[chosen]
        ordered = sorted(zip(indices.tolist(), scores.tolist()), key=lambda item: (-item[1], ids[item[0]]))
        result.append([(index, float(score), rank) for rank, (index, score) in enumerate(ordered, 1)])
    return result


def retrieve_batch(index: CountryIndex, rows: list[dict[str, Any]], behavior: Mapping[str, Any]) -> dict[str, list[dict[str, Any]]]:
    target_ids = [str(row["entity_id"]) for row in rows]
    name_query = index.name_vectorizer.transform([row["name_core"] or "" for row in rows])
    address_query = index.address_vectorizer.transform([row["address_normalized"] or "" for row in rows])
    name_hits = _topk(name_query @ index.name_matrix.T, index.ids, k=int(behavior["kmax"]), min_score=float(behavior["name_min_score"]))
    address_hits = _topk(address_query @ index.address_matrix.T, index.ids, k=int(behavior["kmax"]), min_score=float(behavior["address_min_score"]))
    meaningful = [
        any(ord(char) > 127 for char in (row.get("name_raw") or ""))
        or (row.get("name_transliterated") or "") != (row.get("name_tokens") or "")
        for row in rows
    ]
    translit_query = index.name_vectorizer.transform([
        (row["name_transliterated"] or "") if use else "" for row, use in zip(rows, meaningful)
    ])
    translit_hits = _topk(
        translit_query @ index.name_matrix.T, index.ids,
        k=int(behavior["transliteration_k"]), min_score=float(behavior["transliteration_min_score"]),
    )
    output = {family: [] for family in FAMILIES}
    for row_number, target_id in enumerate(target_ids):
        for index_id, score, rank in name_hits[row_number]:
            output["name_word"].append({"target_entity_id": target_id, "candidate_s1_entity_id": index.ids[index_id], "word_name_present": True, "word_name_score": score, "word_name_rank": rank})
        for index_id, score, rank in address_hits[row_number]:
            output["address_word"].append({"target_entity_id": target_id, "candidate_s1_entity_id": index.ids[index_id], "address_present": True, "address_score": score, "address_rank": rank})
        for index_id, score, rank in translit_hits[row_number]:
            output["transliteration"].append({"target_entity_id": target_id, "candidate_s1_entity_id": index.ids[index_id], "translit_hit": True, "translit_score": score, "translit_rank": rank})
        rare_tokens = sorted(
            (_tokens(rows[row_number].get("name_tokens")) | _tokens(rows[row_number].get("address_normalized"))) & index.rare_postings.keys(),
            key=lambda token: (index.rare_df[token], token),
        )[:int(behavior["rare_tokens_per_query"])]
        rare_candidates: dict[int, list[int]] = defaultdict(list)
        for token in rare_tokens:
            for candidate in index.rare_postings[token]:
                rare_candidates[candidate].append(index.rare_df[token])
        ordered_rare = sorted(
            rare_candidates.items(), key=lambda item: (-len(item[1]), min(item[1]), index.ids[item[0]]),
        )[:int(behavior["rare_candidate_limit"])]
        for candidate, dfs in ordered_rare:
            output["rare_token"].append({"target_entity_id": target_id, "candidate_s1_entity_id": index.ids[candidate], "rare_token_hit": True, "rare_token_min_df": min(dfs), "rare_token_overlap_count": len(dfs)})
        numeric_candidates: Counter[int] = Counter()
        for key in _numeric_keys(rows[row_number]):
            numeric_candidates.update(index.numeric_postings.get(key, ()))
        ordered_numeric = sorted(
            numeric_candidates.items(), key=lambda item: (-item[1], index.ids[item[0]]),
        )[:int(behavior["numeric_candidate_limit"])]
        for candidate, overlap in ordered_numeric:
            output["numeric"].append({"target_entity_id": target_id, "candidate_s1_entity_id": index.ids[candidate], "numeric_hit": True, "numeric_overlap_count": overlap})
    return output


def _paths(root: Path, family: str, target: NormalizedShard) -> tuple[Path, Path]:
    directory = root / family / target.split / f"country={target.country}" / f"source={target.source}"
    artifact = directory / f"part-{int(target.shard_id):05d}.parquet"
    return artifact, artifact.with_suffix(".manifest.json")


def validate_lexical_parquet(path: Path, family: str, country: str, source: str, rows: int) -> None:
    import pyarrow.parquet as pq
    parquet = pq.ParquetFile(path)
    if parquet.metadata.num_rows != rows or parquet.schema_arrow != _pa_schema(family):
        raise ValueError("lexical row count/schema mismatch")
    seen: set[tuple[str, str, str]] = set()
    for batch in parquet.iter_batches(columns=list(IDENTITY)):
        for item in batch.to_pylist():
            if item["country"] != country or item["target_source"] != source:
                raise ValueError("lexical artifact has wrong partition identity")
            key = (item["target_entity_id"], item["candidate_s1_entity_id"], item["target_source"])
            if key in seen:
                raise ValueError("duplicate lexical candidate")
            seen.add(key)


def _dependency(s1: list[NormalizedShard], target: NormalizedShard, config: Mapping[str, Any], family: str) -> str:
    return hash_mapping({
        "normalization_version": config["normalization"]["contract_version"],
        "inputs": [(s.artifact_hash, s.input_fingerprint) for s in [*s1, target]],
        "family": family, "version": config["retrieval"]["version"],
        "config": config["retrieval"]["lexical"],
    })


def _valid(path: Path, manifest_path: Path, family: str, target: NormalizedShard, dependency: str, config: Mapping[str, Any], commit: str) -> bool:
    if not manifest_path.exists():
        return False
    try:
        manifest = load_manifest(manifest_path)
        if manifest.code_commit != commit or Path(manifest.artifact_path).resolve() != path.resolve():
            return False
        source = target.source.replace("source", "S")
        validator = lambda value: validate_lexical_parquet(value, family, target.country, source, manifest.output_rows)
        return validate_artifact(
            manifest_path, expected_config_hash=config_hash(config), expected_stage=f"{STAGE}:{family}",
            expected_stage_version=str(config["retrieval"]["version"]),
            expected_input_fingerprint=dependency, schema_validator=validator,
        ).valid
    except (OSError, ValueError):
        return False


def _write_target(index: CountryIndex, target: NormalizedShard, s1: list[NormalizedShard], root: Path, config: Mapping[str, Any], commit: str) -> dict[str, Any]:
    _, pa, _ = _imports()
    import pyarrow.parquet as pq
    behavior = config["retrieval"]["lexical"]
    dependencies = {family: _dependency(s1, target, config, family) for family in FAMILIES}
    locations = {family: _paths(root, family, target) for family in FAMILIES}
    valid = {family: _valid(*locations[family], family, target, dependencies[family], config, commit) for family in FAMILIES}
    if all(valid.values()):
        return {"skipped": True, "rows": {family: load_manifest(locations[family][1]).output_rows for family in FAMILIES}}
    writers: dict[str, Any] = {}
    temporaries: dict[str, Path] = {}
    counts = {family: 0 for family in FAMILIES}
    started = time.monotonic()
    try:
        for family in FAMILIES:
            if valid[family]:
                counts[family] = load_manifest(locations[family][1]).output_rows
                continue
            artifact, _ = locations[family]
            artifact.parent.mkdir(parents=True, exist_ok=True)
            temporary = artifact.with_suffix(".parquet.tmp")
            if temporary.exists(): temporary.unlink()
            temporaries[family] = temporary
            writers[family] = pq.ParquetWriter(temporary, _pa_schema(family), compression="zstd")
        columns = ["entity_id", "name_raw", "name_core", "name_tokens", "name_transliterated", "address_normalized", "primary_number", "numeric_tokens", "postal_like_tokens"]
        parquet = pq.ParquetFile(target.path)
        for batch in parquet.iter_batches(batch_size=int(behavior["query_batch_size"]), columns=columns):
            rows = batch.to_pylist()
            retrieved = retrieve_batch(index, rows, behavior)
            for family, values in retrieved.items():
                if family not in writers or not values: continue
                source = target.source.replace("source", "S")
                for value in values:
                    value["target_source"], value["country"] = source, target.country
                table = pa.Table.from_pylist(values, schema=_pa_schema(family))
                writers[family].write_table(table)
                counts[family] += table.num_rows
    finally:
        for writer in writers.values(): writer.close()
    sample = ResourceMonitor(STAGE).sample(processed_rows=target.rows)
    for family, writer in writers.items():
        artifact, manifest_path = locations[family]
        temporaries[family].replace(artifact)
        validate_lexical_parquet(artifact, family, target.country, target.source.replace("source", "S"), counts[family])
        manifest = ShardManifest.complete(
            stage=f"{STAGE}:{family}", stage_version=str(config["retrieval"]["version"]),
            split=target.split, country=target.country, source=target.source.replace("source", "S"), shard_id=target.shard_id,
            input_rows=target.rows, output_rows=counts[family], runtime_seconds=time.monotonic()-started,
            peak_ram_bytes=sample.rss_bytes, peak_vram_bytes=None, config_hash=config_hash(config), code_commit=commit,
            input_fingerprint=dependencies[family], artifact_path=str(artifact.resolve()),
            artifact_size_bytes=artifact.stat().st_size, artifact_hash=hash_file(artifact),
        )
        manifest.write(manifest_path)
    return {"skipped": False, "rows": counts, "resource": json.loads(sample.to_json())}


def run_lexical_retrieval(
    config: Mapping[str, Any], *, normalized_root: str | Path | None,
    exact_structured_root: str | Path | None, data_root: str | Path | None = None,
    output_dir: str | Path | None = None, split: str | None = None, source: str | None = None,
    country: str | None = None, shard_id: str | None = None,
) -> dict[str, Any]:
    if normalized_root is None: raise LexicalError("--normalized-root is required")
    source_map = {None: None, "source2": "source2", "source3": "source3", "S2": "source2", "S3": "source3"}
    if split not in {None, "train", "test"} or source not in source_map:
        raise LexicalError("invalid split/source filter")
    selected_shard = None if shard_id is None else str(int(shard_id))
    inventory = _discover_normalized(Path(normalized_root).resolve())
    targets = [s for s in inventory if s.source in {"source2", "source3"} and (split is None or s.split==split) and (source_map[source] is None or s.source==source_map[source]) and (country is None or s.country==country) and (selected_shard is None or s.shard_id==selected_shard)]
    if not targets: raise LexicalError("no normalized target shards match filters")
    root = Path(output_dir).resolve() if output_dir else resolve_project_path(config, "artifact_root") / "retrieval" / str(config["retrieval"]["version"])
    commit, results = _code_commit(), []
    started = time.monotonic()
    report_root = resolve_project_path(config,"artifact_root") / "reports" / "retrieval" / str(config["retrieval"]["version"]) / STAGE
    resource_log = report_root / "resources.jsonl"
    report_root.mkdir(parents=True, exist_ok=True)
    for current_split, current_country in sorted({(s.split, s.country) for s in targets}):
        s1 = [s for s in inventory if s.split==current_split and s.source=="source1" and s.country==current_country]
        print(f"[LEXICAL] building S1 indices split={current_split} country={current_country}", flush=True)
        index = build_country_index([s.path for s in s1], config["retrieval"]["lexical"])
        for target in [s for s in targets if (s.split,s.country)==(current_split,current_country)]:
            print(f"[LEXICAL] split={target.split} country={target.country} source={target.source} shard={target.shard_id} rows={target.rows}", flush=True)
            result = _write_target(index, target, s1, root, config, commit)
            results.append({"split":target.split,"country":target.country,"source":target.source,"shard":target.shard_id,**result})
            sample = result.get("resource") or json.loads(ResourceMonitor(STAGE, country=target.country, source=target.source, shard=target.shard_id).sample(target.rows).to_json())
            with resource_log.open("a", encoding="utf-8") as stream: stream.write(json.dumps(sample, sort_keys=True)+"\n")
            print(f"[LEXICAL] complete skipped={result['skipped']} rows={result['rows']} rss={sample['rss_bytes']} available_ram={sample['available_ram_bytes']} elapsed={sample['elapsed_seconds']:.1f}s", flush=True)
            fraction = float(sample["ram_fraction"])
            if fraction >= float(config["runtime"]["critical_ram_fraction"]): raise LexicalError("critical RAM threshold reached after safe shard completion")
            if fraction >= float(config["runtime"]["warning_ram_fraction"]): print("[LEXICAL] WARNING RAM threshold reached", flush=True)
        del index
    response = {"status":"complete","stage":STAGE,"runtime_seconds":time.monotonic()-started,"output_root":str(root),"shards":len(results),"shards_skipped":sum(r["skipped"] for r in results),"results":results}
    _atomic_json(report_root / "run_summary.json", response)
    # The production audit requires unfiltered train+test output and Phase 2A input.
    if split is None and source is None and country is None and selected_shard is None:
        if exact_structured_root is None or data_root is None:
            raise LexicalError("full lexical run requires --exact-structured-root and --data-root for the K audit")
        response["audit"] = run_lexical_audit(config, root, Path(exact_structured_root), Path(normalized_root), Path(data_root), report_root)
    return response


def run_lexical_audit(config: Mapping[str, Any], root: Path, exact_root: Path, normalized_root: Path, data_root: Path, report_root: Path) -> dict[str, Any]:
    """Offline rank-filtered train audit; expensive retrieval is never rerun for K."""
    import duckdb
    paths = resolve_dataset_paths(config, data_root)
    connection = duckdb.connect(":memory:")
    behavior = config["retrieval"]["lexical"]
    connection.execute(f"SET memory_limit='{str(behavior['report_memory_limit']).replace(chr(39),'')}'")
    connection.execute(f"SET threads={int(behavior['report_threads'])}")
    connection.execute("SET preserve_insertion_order=false")
    def pattern(value: Path) -> str: return str(value.resolve()).replace("\\","/").replace("'","''")
    views = {"phase2a": exact_root, **{family: root/family for family in FAMILIES}}
    for name, value in views.items():
        files = list((value/"train").rglob("part-*.parquet"))
        if not files: raise LexicalError(f"missing train audit input for {name}: {value}")
        literal = "["+",".join("'"+pattern(p)+"'" for p in files)+"]"
        connection.execute(f"CREATE VIEW {name} AS SELECT * FROM read_parquet({literal}, union_by_name=true, hive_partitioning=false)")
    target_files = [s.path for s in _discover_normalized(normalized_root) if s.split=="train" and s.source in {"source2","source3"}]
    literal = "["+",".join("'"+pattern(p)+"'" for p in target_files)+"]"
    gt = pattern(paths.train_ground_truth)
    connection.execute(f"CREATE VIEW targets AS SELECT entity_id target_entity_id,country FROM read_parquet({literal},union_by_name=true,hive_partitioning=false)")
    connection.execute(f"CREATE TABLE gt AS SELECT source1_entity_id::VARCHAR true_s1,trim(x)::VARCHAR target_entity_id,t.country FROM read_csv('{gt}',delim='\\t',header=true,all_varchar=true),UNNEST(string_split(coalesce(matched_entity_ids,''),',')) u(x) JOIN targets t ON t.target_entity_id=trim(x) WHERE trim(x)<>''")
    total = connection.execute("SELECT count(*) FROM gt").fetchone()[0]
    audits=[]
    for k in (3,5,8,10,15):
        conditions = {
            "phase2a":"EXISTS(SELECT 1 FROM phase2a c WHERE c.target_entity_id=g.target_entity_id AND c.candidate_s1_entity_id=g.true_s1 AND c.country=g.country)",
            "name_word":f"EXISTS(SELECT 1 FROM name_word c WHERE c.target_entity_id=g.target_entity_id AND c.candidate_s1_entity_id=g.true_s1 AND c.country=g.country AND c.word_name_rank<={k})",
            "address_word":f"EXISTS(SELECT 1 FROM address_word c WHERE c.target_entity_id=g.target_entity_id AND c.candidate_s1_entity_id=g.true_s1 AND c.country=g.country AND c.address_rank<={k})",
            "transliteration":"EXISTS(SELECT 1 FROM transliteration c WHERE c.target_entity_id=g.target_entity_id AND c.candidate_s1_entity_id=g.true_s1 AND c.country=g.country)",
            "rare_token":"EXISTS(SELECT 1 FROM rare_token c WHERE c.target_entity_id=g.target_entity_id AND c.candidate_s1_entity_id=g.true_s1 AND c.country=g.country)",
            "numeric":"EXISTS(SELECT 1 FROM numeric c WHERE c.target_entity_id=g.target_entity_id AND c.candidate_s1_entity_id=g.true_s1 AND c.country=g.country)",
        }
        quoted = lambda name: f'"{name}"'
        select=",".join(f"CAST({condition} AS BOOLEAN) {quoted(name)}" for name,condition in conditions.items())
        connection.execute(f"CREATE OR REPLACE TEMP TABLE flags AS SELECT g.*,{select} FROM gt g")
        names=list(conditions)
        union_expression=" OR ".join(quoted(name) for name in names)
        row=connection.execute("SELECT "+",".join(f"sum(CAST({quoted(n)} AS BIGINT))" for n in names)+",sum(CAST("+union_expression+" AS BIGINT)) FROM flags").fetchone()
        recovered=dict(zip(names,map(int,row[:-1])))
        recall={name:value/total for name,value in recovered.items()}
        union=int(row[-1])
        unique_beyond_2a={n:int(connection.execute(f"SELECT count(*) FROM flags WHERE {quoted(n)} AND NOT {quoted('phase2a')}").fetchone()[0]) for n in names[1:]}
        earlier=quoted("phase2a"); unique_earlier={}
        for n in names[1:]:
            unique_earlier[n]=int(connection.execute(f"SELECT count(*) FROM flags WHERE {quoted(n)} AND NOT ({earlier})").fetchone()[0]); earlier+=f" OR {quoted(n)}"
        by_partition=[{"country":r[0],"target_source":r[1],"gt_pairs":int(r[2]),"recovered":int(r[3]),"recall":float(r[3])/r[2]} for r in connection.execute("SELECT country,CASE WHEN starts_with(target_entity_id,'S2') THEN 'S2' ELSE 'S3' END,count(*),sum(CAST("+union_expression+" AS BIGINT)) FROM flags GROUP BY 1,2 ORDER BY 1,2").fetchall()]
        connection.execute(f"""CREATE OR REPLACE TEMP TABLE candidate_counts AS
            WITH pairs AS (
                SELECT target_entity_id,candidate_s1_entity_id FROM phase2a
                UNION SELECT target_entity_id,candidate_s1_entity_id FROM name_word WHERE word_name_rank<={k}
                UNION SELECT target_entity_id,candidate_s1_entity_id FROM address_word WHERE address_rank<={k}
                UNION SELECT target_entity_id,candidate_s1_entity_id FROM transliteration
                UNION SELECT target_entity_id,candidate_s1_entity_id FROM rare_token
                UNION SELECT target_entity_id,candidate_s1_entity_id FROM numeric
            ) SELECT target_entity_id,count(*) n FROM pairs GROUP BY target_entity_id""")
        distribution=connection.execute("SELECT avg(coalesce(c.n,0)),quantile_cont(coalesce(c.n,0),0.95),quantile_cont(coalesce(c.n,0),0.99) FROM targets t LEFT JOIN candidate_counts c USING(target_entity_id)").fetchone()
        audits.append({"k":k,"gt_pairs_total":int(total),"recovered":recovered,"pair_recall":recall,"union_recovered":union,"union_pair_recall":union/total,"gt_target_hit_rate":union/total,"avg_candidates_per_target":float(distribution[0]),"p95_candidates_per_target":float(distribution[1]),"p99_candidates_per_target":float(distribution[2]),"unique_beyond_phase2a":unique_beyond_2a,"unique_beyond_earlier":unique_earlier,"by_country_source":by_partition})
    connection.close()
    result={"k_sweep":audits}
    _atomic_json(report_root/"audit.json",result)
    return result
