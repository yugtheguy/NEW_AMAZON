from pathlib import Path

import pyarrow.parquet as pq

from amazon_er.config import load_config
from amazon_er.pipeline.lexical import FAMILIES, run_lexical_retrieval
from amazon_er.pipeline.exact_structured import run_exact_structured_retrieval
from amazon_er.pipeline.normalization import run_normalization


HEADER = "entity_id\tbusiness_name\tbusiness_address\tcountry\n"


def _dataset(root: Path) -> Path:
    train, test = root / "train", root / "test"
    train.mkdir(parents=True); test.mkdir(parents=True)
    train_rows = {
        "source1": [
            "u1\tAcme Tools LLC\t10 Main Street 55555\tUS",
            "u2\tAcme Services\t99 West Road 77777\tUS",
            "u3\tCafé Unique\t44 Rue Special 44444\tUS",
            "u4\tZephyrare Shop\t73 North Avenue 33333\tUS",
            "i1\tAcme Tools LLC\t10 Main Street 55555\tIndia",
        ],
        "source2": [
            "S2-name\tAcme Tools Extra\tNo Address\tUS",
            "S2-address\tUnrelated Name\t10 Main Street 55555\tUS",
            "S2-translit\tCafé Unique\tElsewhere\tUS",
            "S2-rare\tZephyrare Different\tElsewhere\tUS",
            "S2-numeric\tNo Similarity\t10 Unknown\tUS",
            "S2-india\tAcme Tools LLC\t10 Main Street 55555\tIndia",
        ],
        "source3": ["S3-zero\tNothing Matching\tNo Place\tUS"],
    }
    for source, rows in train_rows.items():
        (train / f"train_{source}.tsv").write_text(HEADER + "\n".join(rows) + "\n", encoding="utf-8")
    for source in ("source1", "source2", "source3"):
        (test / f"test_{source}.tsv").write_text(
            HEADER + f"T-{source}\tTest {source}\t1 Test Road\tUS\n", encoding="utf-8"
        )
    (train / "train_ground_truth.tsv").write_text(
        "source1_entity_id\tmatched_entity_ids\n"
        "u1\tS2-name,S2-address,S2-numeric\n"
        "u2\t\n"
        "u3\tS2-translit\n"
        "u4\tS2-rare\n"
        "i1\tS2-india\n",
        encoding="utf-8",
    )
    return root


def _rows(root: Path, family: str):
    result = []
    for path in sorted((root / family).rglob("part-*.parquet")):
        result.extend(pq.ParquetFile(path).read().to_pylist())
    return result


def test_lexical_channels_direction_ranking_country_and_resume(tmp_path):
    config = load_config("configs/smoke.yaml")
    data = _dataset(tmp_path / "data")
    normalized = tmp_path / "normalized" / "v1"
    run_normalization(config, data_root=data, output_dir=normalized)
    output = tmp_path / "retrieval" / "v1"
    first = run_lexical_retrieval(
        config, normalized_root=normalized, exact_structured_root=None,
        data_root=data, output_dir=output, split="train", country="US", source="source2",
    )
    assert first["shards"] == 2
    for family in FAMILIES:
        rows = _rows(output, family)
        keys = [(r["target_entity_id"], r["candidate_s1_entity_id"], r["target_source"]) for r in rows]
        assert len(keys) == len(set(keys))
        assert all(r["country"] == "US" and r["candidate_s1_entity_id"] != "i1" for r in rows)
    name = _rows(output, "name_word")
    assert next(r for r in name if r["target_entity_id"] == "S2-name" and r["word_name_rank"] == 1)["candidate_s1_entity_id"] == "u1"
    address = _rows(output, "address_word")
    assert next(r for r in address if r["target_entity_id"] == "S2-address" and r["address_rank"] == 1)["candidate_s1_entity_id"] == "u1"
    assert any(r["target_entity_id"] == "S2-translit" and r["candidate_s1_entity_id"] == "u3" for r in _rows(output, "transliteration"))
    assert any(r["target_entity_id"] == "S2-rare" and r["candidate_s1_entity_id"] == "u4" for r in _rows(output, "rare_token"))
    assert any(r["target_entity_id"] == "S2-numeric" and r["candidate_s1_entity_id"] == "u1" for r in _rows(output, "numeric"))
    second = run_lexical_retrieval(
        config, normalized_root=normalized, exact_structured_root=None,
        data_root=data, output_dir=output, split="train", country="US", source="source2",
    )
    assert second["shards_skipped"] == second["shards"]


def test_lexical_offline_k_audit(tmp_path):
    config = load_config("configs/smoke.yaml")
    data = _dataset(tmp_path / "data")
    normalized = tmp_path / "normalized" / "v1"
    run_normalization(config, data_root=data, output_dir=normalized)
    exact = tmp_path / "retrieval" / "v1" / "exact_structured"
    run_exact_structured_retrieval(config, normalized_root=normalized, data_root=data, output_dir=exact)
    lexical_root = tmp_path / "retrieval" / "v1"
    result = run_lexical_retrieval(
        config, normalized_root=normalized, exact_structured_root=exact,
        data_root=data, output_dir=lexical_root,
    )
    sweep = result["audit"]["k_sweep"]
    assert [item["k"] for item in sweep] == [3, 5, 8, 10, 15]
    assert all(item["gt_pairs_total"] == 6 for item in sweep)
    assert all("India" in {row["country"] for row in item["by_country_source"]} for item in sweep)
    assert all(item["union_pair_recall"] >= item["pair_recall"]["phase2a"] for item in sweep)
