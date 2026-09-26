from pathlib import Path

import pytest

from amazon_er.config import load_config
from amazon_er.data.normalize import build_address_views, build_name_views, normalize_record
from amazon_er.pipeline.normalization import NORMALIZED_COLUMNS, run_normalization


SUFFIXES = ["ltd", "limited", "pvt", "private", "llc", "inc", "corp", "company", "sarl", "sa"]


def name(value):
    return build_name_views(value, legal_suffixes=SUFFIXES)


def test_name_unicode_views_and_suffix_boundaries():
    views = name("Ｓｏｃｉéｔé Générale S.A.")
    assert views["name_nfkc"].startswith("Société")
    assert views["name_casefold"].startswith("société")
    assert views["name_accent_fold"].startswith("societe")
    assert views["name_compact"] == "societegeneralesa"
    assert views["name_tokens"] == "societe generale s a"
    assert views["name_token_sorted"] == "a generale s societe"
    assert views["name_core"] == "societe generale"

    assert name("Royal Coffee Company")["name_core"] == "royal coffee"
    assert name("Salt")["name_core"] == "salt"
    assert name("Company Road")["name_core"] == "company road"


def test_unicode_compact_and_deterministic_transliteration():
    views = name("मॉडर्न फाइनेंस प्राइवेट लिमिटेड")
    assert "म" in views["name_compact"]
    assert views["name_raw"] == "मॉडर्न फाइनेंस प्राइवेट लिमिटेड"
    assert views["name_transliterated"]
    assert views == name("मॉडर्न फाइनेंस प्राइवेट लिमिटेड")


def test_punctuation_tokenization_and_sorting():
    views = name("The Coffee-Shop Pvt. Ltd.")
    assert views["name_tokens"] == "the coffee shop pvt ltd"
    assert views["name_token_sorted"] == "coffee ltd pvt shop the"
    assert views["name_core"] == "the coffee shop"


def test_address_numeric_and_postal_views():
    views = build_address_views("12-B Main Street, 400001 / 75", postal_like_min_digits=4, postal_like_max_digits=8)
    assert views["address_normalized"] == "12 b main street 400001 75"
    assert views["numeric_tokens"] == "12|400001|75"
    assert views["primary_number"] == "12"
    assert views["postal_like_tokens"] == "400001"
    assert views["address_compact"] == "12bmainstreet40000175"


@pytest.mark.parametrize("value", [None, "", "   "])
def test_missing_name_policy(value):
    views = name(value)
    assert views["name_missing"]
    assert views["name_raw"] is value
    for key, result in views.items():
        if key not in {"name_raw", "name_missing"}:
            assert result not in {"nan", "none", "null"}


@pytest.mark.parametrize("value", [None, "", "   "])
def test_missing_address_policy(value):
    views = build_address_views(value)
    assert views["address_missing"]
    assert views["address_raw"] is value
    assert views["primary_number"] is None
    for key, result in views.items():
        if key not in {"address_raw", "address_missing", "primary_number"}:
            assert result not in {"nan", "none", "null"}


def test_train_and_test_share_one_record_function():
    arguments = dict(
        entity_id="S2-1", business_name="Café LLC", business_address="10 Rue, 75001",
        country="France", legal_suffixes=SUFFIXES, postal_like_min_digits=4,
        postal_like_max_digits=8,
    )
    first = normalize_record(**arguments)
    second = normalize_record(**arguments)
    assert first == second
    assert tuple(first) == NORMALIZED_COLUMNS


def _write_dataset(root: Path) -> Path:
    train, test = root / "train", root / "test"
    train.mkdir(parents=True)
    test.mkdir(parents=True)
    header = "entity_id\tbusiness_name\tbusiness_address\tcountry\n"
    source_rows = {
        "source1": ["S1-1\tCafé Company\t12 Main 400001\tUS", "S1-2\tमॉडर्न फाइनेंस\t\tIndia"],
        "source2": ["S2-1\tCoffee-Shop LLC\t10 Rue 75001\tUS", "S2-2\tभारत निजी\t77 Road 560001\tIndia", "S2-3\tMissing\t\tUS", "S2-4\tFour\t4 Road\tUS"],
        "source3": ["S3-1\tAlpha Inc\t1 Road\tUS", "S3-2\tBeta Pvt Ltd\t2 Road\tIndia"],
    }
    for source, rows in source_rows.items():
        (train / f"train_{source}.tsv").write_text(header + "\n".join(rows) + "\n", encoding="utf-8")
    test_rows = {
        "source1": ["S1-t1\tSociété Générale S.A.\t5 Rue 75001\tFrance"],
        "source2": ["S2-t1\tGamma SARL\t6 Rue 75002\tFrance", "S2-t2\tDelta\t7 Road\tUS"],
        "source3": ["S3-t1\tदेवनागरी\t8 Road 110001\tIndia"],
    }
    for source, rows in test_rows.items():
        (test / f"test_{source}.tsv").write_text(header + "\n".join(rows) + "\n", encoding="utf-8")
    (train / "train_ground_truth.tsv").write_text(
        "source1_entity_id\tmatched_entity_ids\nS1-1\tS2-1,S3-1\nS1-2\tS2-2,S3-2\n",
        encoding="utf-8",
    )
    return root


def test_full_smoke_resume_and_corrupt_recompute(tmp_path):
    data_root = _write_dataset(tmp_path / "dataset")
    output_root = tmp_path / "artifacts" / "normalized" / "v1"
    config = load_config("configs/smoke.yaml")

    first = run_normalization(config, data_root=data_root, output_dir=output_root)
    assert first["summary"]["cardinality_verified"]
    artifacts = sorted(output_root.rglob("part-*.parquet"))
    manifests = sorted(output_root.rglob("part-*.manifest.json"))
    assert artifacts and len(artifacts) == len(manifests)
    report_root = tmp_path / "artifacts" / "reports" / "normalization" / "v1"
    for filename in ("summary.json", "summary.md", "samples.json", "samples.md", "resources.jsonl"):
        assert (report_root / filename).exists()

    second = run_normalization(config, data_root=data_root, output_dir=output_root)
    assert sum(item["shards_written"] for item in second["sources"]) == 0
    assert sum(item["shards_skipped"] for item in second["sources"]) == len(artifacts)

    corrupted = artifacts[0]
    corrupted.write_bytes(b"corrupt smoke artifact")
    third = run_normalization(config, data_root=data_root, output_dir=output_root)
    assert sum(item["shards_written"] for item in third["sources"]) == 1
    assert third["summary"]["cardinality_verified"]
