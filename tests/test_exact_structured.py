from pathlib import Path

from amazon_er.config import load_config
from amazon_er.pipeline.exact_structured import run_exact_structured_retrieval
from amazon_er.pipeline.normalization import run_normalization


HEADER = "entity_id\tbusiness_name\tbusiness_address\tcountry\n"


def _write_dataset(root: Path) -> Path:
    train, test = root / "train", root / "test"
    train.mkdir(parents=True)
    test.mkdir(parents=True)
    train_rows = {
        "source1": [
            "s1a\tAcme Corp\t10 Main 12345\tUS",
            "s1c1\tCommon Inc\t1 Alpha 11111\tUS",
            "s1c2\tCommon Inc\t2 Beta 22222\tUS",
            "s1c3\tCommon Inc\t3 Gamma 33333\tUS",
            "s1c4\tCommon Inc\t4 Delta 44444\tUS",
            "s1addr\tDifferent Name\t88 Exact Road 88000\tUS",
            "s1multi\tMulti Company\t55 Match Street 55555\tUS",
            "s1sorted\tAlpha Beta\t61 Sorted Road 61000\tUS",
            "s1core\tCore Business LLC\t62 Core Road 62000\tUS",
            "s1p1\tPostal Inc\t101 A Road 77771\tUS",
            "s1p2\tPostal Inc\t102 B Road 77772\tUS",
            "s1p3\tPostal Inc\t103 C Road 77773\tUS",
            "s1p4\tPostal Inc\t104 D Road 77774\tUS",
            "s1india\tCross Name\t90 India Road 90000\tIndia",
        ],
        "source2": [
            "S2-t1\tAcme Corp\t10 Main 12345\tUS",
            "S2-t2\tCommon LLC\t2 Somewhere 22222\tUS",
            "S2-t3\tUnrelated\t88 Exact Road 88000\tUS",
            "S2-zero\tNo Match\t999 Nowhere 99999\tUS",
            "S2-cross\tCross Name\t90 India Road 90000\tUS",
            "S2-sorted\tBeta Alpha\t600 Other Road 60000\tUS",
            "S2-core\tCore Business Inc\t601 Other Road 60001\tUS",
            "S2-postal\tPostal LLC\t999 Other Road 77771\tUS",
            "S2-india\tCross Name\t90 India Road 90000\tIndia",
        ],
        "source3": [
            "S3-t4\tMulti Company\t55 Match Street 55555\tUS",
        ],
    }
    for source, rows in train_rows.items():
        (train / f"train_{source}.tsv").write_text(HEADER + "\n".join(rows) + "\n", encoding="utf-8")
    test_rows = {
        "source1": ["ts1\tTest Company\t7 Test Road 70000\tUS"],
        "source2": ["TS2\tTest Company\t7 Test Road 70000\tUS"],
        "source3": ["TS3\tAbsent\t8 Other Road 80000\tUS"],
    }
    for source, rows in test_rows.items():
        (test / f"test_{source}.tsv").write_text(HEADER + "\n".join(rows) + "\n", encoding="utf-8")
    (train / "train_ground_truth.tsv").write_text(
        "source1_entity_id\tmatched_entity_ids\n"
        "s1a\tS2-t1\n"
        "s1c1\t\n"
        "s1c2\tS2-t2\n"
        "s1c3\t\n"
        "s1c4\t\n"
        "s1addr\tS2-t3\n"
        "s1multi\tS3-t4\n"
        "s1sorted\tS2-sorted\n"
        "s1core\tS2-core\n"
        "s1p1\tS2-postal\n"
        "s1india\tS2-india\n",
        encoding="utf-8",
    )
    return root


def test_exact_structured_pipeline_pairs_safety_recall_and_resume(tmp_path):
    config = load_config("configs/smoke.yaml")
    data_root = _write_dataset(tmp_path / "dataset")
    normalized = tmp_path / "normalized" / "v1"
    run_normalization(config, data_root=data_root, output_dir=normalized)
    output = tmp_path / "retrieval" / "v1" / "exact_structured"

    first = run_exact_structured_retrieval(
        config, normalized_root=normalized, data_root=data_root, output_dir=output,
        split="train",
    )
    assert first["shards_written"] > 0
    assert first["train_recall"]["gt_pairs_total"] == 8
    assert first["train_recall"]["exact_union_structured_pair_recall"] == 1.0
    assert first["volume"]["target_count"] == 10
    assert first["volume"]["candidate_pair_count"] == 8
    assert first["volume"]["targets_with"]["0"] >= 1
    assert any(item["high_frequency_suppressed"] for item in first["suppression"])

    import duckdb
    paths = sorted(output.rglob("part-*.parquet"))
    rows = duckdb.connect(":memory:").execute(
        "SELECT target_entity_id, candidate_s1_entity_id, target_source, exact_hit, "
        "structured_hit, retriever_mask, retriever_count "
        "FROM read_parquet(?, union_by_name=true) ORDER BY 1,2",
        [[str(path) for path in paths]],
    ).fetchall()
    pairs = {(row[0], row[1], row[2]) for row in rows}
    assert pairs == {
        ("S2-india", "s1india", "S2"),
        ("S2-core", "s1core", "S2"),
        ("S2-postal", "s1p1", "S2"),
        ("S2-sorted", "s1sorted", "S2"),
        ("S2-t1", "s1a", "S2"),
        ("S2-t2", "s1c2", "S2"),
        ("S2-t3", "s1addr", "S2"),
        ("S3-t4", "s1multi", "S3"),
    }
    common = next(row for row in rows if row[0] == "S2-t2")
    assert common[3:] == (False, True, 2, 1)
    multi = next(row for row in rows if row[0] == "S3-t4")
    assert multi[3:] == (True, True, 3, 2)
    assert not any(pair[0] == "S2-cross" for pair in pairs)

    import pyarrow.parquet as pq
    evidence = {}
    for path in paths:
        for row in pq.ParquetFile(path).read().to_pylist():
            evidence[row["target_entity_id"]] = row
    assert evidence["S2-sorted"]["exact_name_token_sorted"]
    assert not evidence["S2-sorted"]["exact_name_compact"]
    assert evidence["S2-core"]["exact_name_core"]
    assert evidence["S2-t3"]["exact_address_compact"]
    assert evidence["S2-t2"]["structured_name_core_number"]
    assert evidence["S2-postal"]["structured_name_core_postal"]

    second = run_exact_structured_retrieval(
        config, normalized_root=normalized, data_root=data_root, output_dir=output,
        split="train",
    )
    assert second["shards_written"] == 0
    assert second["shards_skipped"] == first["shards"]

    paths[0].write_bytes(b"corrupt")
    third = run_exact_structured_retrieval(
        config, normalized_root=normalized, data_root=data_root, output_dir=output,
        split="train",
    )
    assert third["shards_written"] == 1
    assert third["train_recall"]["exact_union_structured_pair_recall"] == 1.0
