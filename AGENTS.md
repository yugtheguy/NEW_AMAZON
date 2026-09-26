# Repository Instructions for Coding Agents

This repository implements the optimized Amazon ML Challenge 2026 business entity resolution pipeline. Read `docs/PROJECT_RULES.md` before changing production code.

## Scope and architecture

- Keep the target-centric orientation: S2/S3 target to S1 candidate, then S1 or NULL.
- Preserve the frozen sparse-first cascade in `docs/ARCHITECTURE.md`.
- Do not introduce graph clustering, transitive closure, ColBERT, generative LLM matching, extra cross-encoders, France pseudo-labeling, or external business data.
- Share core transformations across train, validation, and test.

## Required engineering behavior

- Label claims as VERIFIED, ASSUMED, INFERRED, or UNVERIFIED.
- Never claim code, artifacts, GPU use, metrics, or dataset properties without checking.
- Never concatenate all full-corpus shards in memory. Process, write, validate, release.
- Make large work country/source/shard aware, deterministic where practical, and resumable.
- A complete manifest never substitutes for validating the artifact itself.
- Treat versioned expensive artifacts as immutable; never overwrite them silently.
- Keep important constants in configuration and use compact data types.
- Validation starts from every S1 entity, including entities with zero candidates.
- The final metric is entity-level macro-F0.5, not pair accuracy.
- Verify CUDA, VRAM, and utilization before claiming GPU acceleration.
- Do not use destructive Git commands or force push.
- Do not encode target exclusivity or same-country assumptions before data audit proves them.
- Add complexity only for a measured failure mode.

## Development checks

Run `pytest`, `python scripts/smoke_test.py`, and the CLI healthcheck for foundation changes. Do not run competition-scale jobs unless explicitly requested.
