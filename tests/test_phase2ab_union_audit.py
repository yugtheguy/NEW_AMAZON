from pathlib import Path

from amazon_er.config import load_config
from amazon_er.pipeline.exact_structured import run_exact_structured_retrieval
from amazon_er.pipeline.multikey import run_multikey_retrieval
from amazon_er.pipeline.normalization import run_normalization
from amazon_er.pipeline.phase2ab_union_audit import run_phase2ab_union_audit


HEADER = "entity_id\tbusiness_name\tbusiness_address\tcountry\n"


def _dataset(root: Path) -> Path:
    train, test = root / "train", root / "test"
    train.mkdir(parents=True)
    test.mkdir()
    rows = {
        "source1": [
            "a\tAB\tOne Lane 111\tUS",
            "b\tReliance Smart\tTwo Lane 222\tUS",
            "both\tExact Company\tThree Lane 333\tUS",
            "none\tLost Reference\tFour Lane 444\tUS",
            "cafe\tKafe\tFive Lane 555\tUS",
        ],
        "source2": [
            "S2-a\tAB\tDifferent 999\tUS",
            "S2-b\tReliance Retail\tDifferent 999\tUS",
            "S2-both\tExact Company\tDifferent 999\tUS",
            "S2-none\tUnknown Target\tDifferent 999\tUS",
            "S2-cafe\tКафе\tDifferent 999\tUS",
        ],
        "source3": ["S3-empty\tNo Match\tNowhere 000\tUS"],
    }
    for source, values in rows.items():
        (train / f"train_{source}.tsv").write_text(HEADER + "\n".join(values) + "\n", encoding="utf-8")
    for source in ("source1", "source2", "source3"):
        (test / f"test_{source}.tsv").write_text(
            HEADER + f"T-{source}\tTest {source}\tTest Road 123\tUS\n", encoding="utf-8",
        )
    (train / "train_ground_truth.tsv").write_text(
        "source1_entity_id\tmatched_entity_ids\n"
        "a\tS2-a\n"
        "b\tS2-b\n"
        "both\tS2-both\n"
        "none\tS2-none\n"
        "cafe\tS2-cafe\n",
        encoding="utf-8",
    )
    return root


def test_phase2ab_union_and_diagnostics(tmp_path):
    cfg = load_config("configs/smoke.yaml")
    data = _dataset(tmp_path / "data")
    normalized = tmp_path / "normalized" / "v1"
    exact = tmp_path / "exact"
    blocker = tmp_path / "blocker"
    reports = tmp_path / "reports"
    run_normalization(cfg, data_root=data, output_dir=normalized)
    run_exact_structured_retrieval(cfg, normalized_root=normalized, data_root=data, output_dir=exact)
    run_multikey_retrieval(cfg, normalized_root=normalized, data_root=data, output_dir=blocker)

    result = run_phase2ab_union_audit(
        cfg, normalized_root=str(normalized), exact_structured_root=str(exact),
        multikey_root=str(blocker), data_root=str(data), output_dir=str(reports),
    )

    overall = result["union_metrics"]["overall"]
    assert overall["gt_pairs"] == 5
    assert overall["phase2a"]["count"] == 2
    assert overall["phase2b_k50"]["count"] == 2
    assert overall["intersection"]["count"] == 1
    assert overall["phase2a_only"]["count"] == 1
    assert overall["phase2b_only"]["count"] == 1
    assert overall["union"]["count"] == 3
    assert result["union_by_k"]["5"]["union_count"] == 3
    oracle = result["oracle_macro_f0_5_ceiling"]
    assert oracle["evaluation_s1_entities"] == 5
    assert oracle["phase2a"] == 0.4
    assert oracle["phase2b_k50"] == 0.4
    assert oracle["union_by_k"]["5"] == 0.6
    assert result["candidate_volume_by_k"]["5"]["target_count"] == 8
    assert result["raw_coverage_semantics"]["label"] == "RESTRICTED_KEY_COVERAGE"
    assert result["true_ceiling_diagnostic"]["current"]["covered_count"] == 2
    assert result["transliteration"]["target_translit_to_s1_normal_gt_pairs"] >= 1
    assert result["transliteration"]["cross_namespace_correctness_issue"]
    assert (reports / "summary.json").is_file()
    assert (reports / "summary.md").is_file()
