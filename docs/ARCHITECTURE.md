# Frozen Architecture

The primary orientation is **S2/S3 target record → candidate S1 reference entity → S1 or NULL**. Accepted assignments are inverted and aggregated by S1 for entity-level evaluation and submission.

1. Preserve raw values and build non-destructive normalization views.
2. Partition by observed country values without hardcoding a closed country list.
3. Build a sparse retrieval backbone: exact/composite, character-name, token-sorted character-name, word-name, address, structured numeric/postal, and reverse sparse rescue.
4. Union candidates with provenance, compact ranks and scores; choose the cap through a candidate-oracle sweep.
5. Compute shared pair/context features and train Stage-1 LightGBM.
6. Generate OOF Stage-1 predictions, add collective/sibling features, and train Stage-2 LightGBM.
7. Bypass neural models for high-confidence cases; apply multilingual E5 dense rescue conditionally.
8. Apply `BAAI/bge-reranker-v2-m3` only to the top one or two ambiguous candidates.
9. Fit OOF meta-calibration.
10. Assign each target to S1 or NULL; enforce exclusivity only if ground truth proves it.
11. Aggregate accepted targets by S1.
12. Optimize a source-aware, metric-aware entity macro-F0.5 decision policy.

**Sparse first. Dense rescue second. Cross-encoder last. No graph clustering by default.**
