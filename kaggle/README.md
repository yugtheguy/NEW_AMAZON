# Kaggle Execution

Kaggle notebooks are thin launchers, not homes for production algorithms. A run should clone this repository, check out an explicit commit, install required dependency extras, attach competition data read-only, and invoke the package CLI.

```bash
python -m pip install -e .
python -m amazon_er.cli run --stage data_audit --config configs/prod.yaml
```

`data_audit` is a planned stage and is not registered in Phase 0A; use `scripts/data_audit.py` for explicit files until Phase 0B locks the dataset contract.
