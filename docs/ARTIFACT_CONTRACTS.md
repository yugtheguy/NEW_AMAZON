# Artifact Contracts

## Versioned hierarchy

The intended hierarchy is `normalized/v1`, `indices/v1`, retrieval `v1` subtrees (`exact`, `structured`, `char_name`, `char_sorted`, `word_name`, `address`, `reverse`, `dense`), `candidates/candidate_v1`, `features/feature_v1`, model directories (`lgb_stage1`, `lgb_stage2`, `cross_encoder`, `calibrator`), OOF directories (`stage1`, `stage2`, `ce`), report directories (`data_audit`, `retrieval`, `validation`, `resources`), and `submissions`. These live below the configured artifact root and are created only when used.

## Candidate contract: `candidate_v1`

Required identity fields are `target_entity_id`, `candidate_s1_entity_id`, `target_source`, and `country`. Pair uniqueness is (`target_entity_id`, `candidate_s1_entity_id`, `target_source`); country is consistent metadata.

Optional evidence fields include `exact_hit`, `structured_hit`, each retriever's score/rank (`char_name`, `char_sorted`, `word_name`, `address`, `reverse`, `dense`), `retriever_mask`, `retriever_count`, and `fusion_score`. Candidate artifacts contain IDs and numeric metadata, not duplicated business text.

Optional retrieval absence uses an explicit `<retriever>_present = false` flag with null/NaN score and rank. Magic sentinels such as `-999` are forbidden. A present score or rank requires the corresponding presence flag to be true.

## Shard manifest and resume

The manifest records stage/version, country/source/shard, input/output rows, runtime, peak RAM/VRAM, config hash, code commit, input fingerprint, artifact path/size/hash, status, and creation time. A shard is skippable only if the manifest is complete, expected stage/version/config/input metadata matches, the artifact exists and is non-empty, size/hash agree, the artifact opens, and its schema validator succeeds. Otherwise it must be recomputed without silently overwriting another version.
