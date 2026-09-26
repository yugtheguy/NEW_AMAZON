# Project Rules

1. **Do not lie.** Distinguish VERIFIED, ASSUMED, INFERRED, and UNVERIFIED. Test code, inspect artifacts, measure metrics, and verify GPU activity before reporting them.
2. **One canonical implementation.** Train, validation, and test call shared transformations.
3. **No full-data in-memory concatenation.** Read one shard, process, write, validate, release, continue.
4. **Shard large operations.** Use country, source, and shard ID where applicable.
5. **Resume safely.** Require a manifest, artifact verification, schema validation, config hash, and input/version dependencies. A manifest alone proves nothing.
6. **Versioned artifacts are immutable.** Never silently overwrite expensive outputs.
7. **Configuration before constants.** Avoid scattered magic values.
8. **Memory is first-class.** Use compact dtypes, avoid repeated text in candidate rows, monitor RSS, and avoid uncontrolled copies.
9. **GPU must actually be used.** Verify CUDA/device/VRAM/utilization for GPU-beneficial stages.
10. **Do not starve the GPU.** Future transformer work must allow batched tokenization, prefetching, workers, pinned memory, and suitable mixed precision.
11. **No destructive Git operations.** Never use `git clean -fd`, `git reset --hard`, or force push without explicit authorization.
12. **Prove data invariants.** Do not assume unique target ownership or same-country links before checking ground truth.
13. **Score zero-candidate S1 entities.** The evaluation universe is all supplied S1 reference IDs, never the candidate table.
14. **Use entity macro-F0.5.** Pair accuracy is not the final objective.
15. **Keep the pipeline simple.** New complexity must solve a measured failure mode.
