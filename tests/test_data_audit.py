from pathlib import Path

from amazon_er.config import load_config
from amazon_er.data.audit import run_data_audit
from amazon_er.data.paths import EXPECTED_FILES, resolve_dataset_paths


HEADER = "entity_id\tbusiness_name\tbusiness_address\tcountry\n"


def write_dataset(root: Path, *, anomalous: bool) -> Path:
    train, test = root / "dataset" / "train", root / "dataset" / "test"
    train.mkdir(parents=True)
    test.mkdir(parents=True)
    train_s1 = [
        "S1-0\tZero\t0 Road\tUS",
        "S1-1\tOne\t1 Road\tUS",
        "S1-2\tबहु\t2 Road\tIndia",
        "S1-3\tOwner\t3 Road\tIndia",
    ]
    train_s2 = [
        "S2-a\tOne\t1 Road\tUS",
        "S2-b\tMulti\t2 Road\tIndia",
        "S2-d\tDistractor\t9 Road\tUS",
    ]
    if anomalous:
        train_s2 += ["S2-dup\tDuplicate\t4 Road\tIndia", "S2-dup\tDuplicate Again\t5 Road\tIndia"]
    train_s3 = ["S3-a\tMulti Three\t2 Road\tIndia", "S3-d\tDistractor Three\t8 Road\tIndia"]
    gt = [
        "S1-0\t",
        "S1-1\tS2-a",
        "S1-2\tS2-b,S3-a",
    ]
    if anomalous:
        train_s3[0] = "S3-a\tMulti Three\t2 Road\tUS"
        gt += ["S1-3\tS2-b", "S1-orphan\tS2-dup,S2-missing,S3-missing"]
    else:
        gt += ["S1-3\t"]
    files = {
        train / "train_source1.tsv": train_s1,
        train / "train_source2.tsv": train_s2,
        train / "train_source3.tsv": train_s3,
        train / "train_ground_truth.tsv": gt,
        test / "test_source1.tsv": ["S1-t\tTest\tRoad\tFrance"],
        test / "test_source2.tsv": ["S2-t\tTest\tRoad\tFrance"],
        test / "test_source3.tsv": ["S3-t\tTest\tRoad\tFrance"],
    }
    for path, rows in files.items():
        header = "source1_entity_id\tmatched_entity_ids\n" if "ground_truth" in path.name else HEADER
        path.write_text(header + "\n".join(rows) + "\n", encoding="utf-8")
    return root


def test_dataset_discovery_finds_nested_files(tmp_path):
    root = write_dataset(tmp_path / "input", anomalous=False)
    paths = resolve_dataset_paths(load_config("configs/smoke.yaml"), root)
    assert set(paths.sources()) == set(EXPECTED_FILES)
    assert paths.train_ground_truth.name == "train_ground_truth.tsv"


def test_audit_detects_contract_violations(tmp_path):
    root = write_dataset(tmp_path / "input", anomalous=True)
    audit = run_data_audit(load_config("configs/smoke.yaml"), data_root=root, output_dir=tmp_path / "report")
    gt = audit["ground_truth"]
    assert audit["sources"]["train_source2"]["duplicate_entity_ids"] == 1
    assert gt["integrity"] == {"orphan_s1": 3, "orphan_s2": 1, "orphan_s3": 1}
    assert gt["s1_cardinality"]["overall"]["0"] == 1
    assert gt["s1_cardinality"]["overall"]["2"] == 1
    assert not audit["invariants"]["target_exclusivity_verified"]
    assert not audit["invariants"]["strict_country_blocking_safe"]
    assert gt["country_consistency"]["s1_s3_mismatches"] == 1
    assert gt["target_distractors"]["S2"]["overall"]["unmatched"] == 1
    assert (tmp_path / "report" / "invariants.json").exists()


def test_clean_audit_verifies_invariants(tmp_path):
    root = write_dataset(tmp_path / "input", anomalous=False)
    config = load_config("configs/smoke.yaml")
    audit = run_data_audit(config, data_root=root, output_dir=tmp_path / "report")
    assert audit["ground_truth"]["integrity"] == {"orphan_s1": 0, "orphan_s2": 0, "orphan_s3": 0}
    assert audit["invariants"]["target_exclusivity_verified"]
    assert audit["invariants"]["strict_country_blocking_safe"]
    assert audit["ground_truth"]["target_distractors"]["S3"]["overall"]["unmatched"] == 1
    resumed = run_data_audit(config, data_root=root, output_dir=tmp_path / "report")
    assert resumed["dataset_fingerprint"] == audit["dataset_fingerprint"]
    assert resumed["runtime_seconds"] == audit["runtime_seconds"]
