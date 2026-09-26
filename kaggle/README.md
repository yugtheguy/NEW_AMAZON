# Kaggle Execution

Kaggle notebooks are thin launchers, not homes for production algorithms. A run should clone this repository, check out an explicit commit, install required dependency extras, attach competition data read-only, and invoke the package CLI.

```bash
python -m pip install -e .
python -m pip install -e ".[data]"
python -m amazon_er.cli run --stage data-audit --config configs/prod.yaml --data-root /kaggle/input/<dataset>
```

Phase 1 production normalization is a full-corpus CPU/I/O job and should run on Kaggle rather than a development machine:

```bash
python -m amazon_er.cli run \
  --stage normalize \
  --config configs/prod.yaml \
  --data-root /kaggle/input/<dataset>
```

Expected outputs are country-partitioned Parquet shards and manifests under `artifacts/normalized/v1/`, plus validation and sample reports under `artifacts/reports/normalization/v1/`. Re-running the command verifies and skips valid shards.
