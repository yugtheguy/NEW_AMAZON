# Kaggle Execution

Kaggle notebooks are thin launchers, not homes for production algorithms. A run should clone this repository, check out an explicit commit, install required dependency extras, attach competition data read-only, and invoke the package CLI.

```bash
python -m pip install -e .
python -m pip install -e ".[data]"
python -m amazon_er.cli run --stage data-audit --config configs/prod.yaml --data-root /kaggle/input/<dataset>
```
