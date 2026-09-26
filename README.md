# Amazon ML Challenge 2026 — Business Entity Resolution

Production-oriented, target-centric entity resolution for assigning noisy Source 2/Source 3 business records to a deduplicated Source 1 reference entity or `NULL`.

The frozen architecture is a multi-stage cascade: sparse retrieval first, learned pair and collective scoring next, then conditional multilingual dense rescue and cross-encoder reranking for ambiguous cases only. Development happens in Git with Codex; heavy execution will use thin Kaggle launchers around repository code.

**Status:** Phase 0A repository bootstrap. No dataset results or model metrics have been measured.

## Setup

```bash
python -m venv .venv
python -m pip install -e ".[dev]"
```

## Checks

```bash
python scripts/smoke_test.py
pytest
python -m amazon_er.cli run --stage healthcheck --config configs/smoke.yaml
python -m amazon_er.cli run --stage data-audit --config configs/prod.yaml --data-root "<DATASET_ROOT>"
```

See [the architecture](docs/ARCHITECTURE.md), [implementation plan](docs/IMPLEMENTATION_PLAN.md), and [project rules](docs/PROJECT_RULES.md).
