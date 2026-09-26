# Artifact Contracts

## Versioned hierarchy

The intended hierarchy is `normalized/v1`, `indices/v1`, retrieval `v1` subtrees (`exact`, `structured`, `char_name`, `char_sorted`, `word_name`, `address`, `reverse`, `dense`), `candidates/candidate_v1`, `features/feature_v1`, model directories (`lgb_stage1`, `lgb_stage2`, `cross_encoder`, `calibrator`), OOF directories (`stage1`, `stage2`, `ce`), report directories (`data_audit`, `retrieval`, `validation`, `resources`), and `submissions`. These live below the configured artifact root and are created only when used.

Normalized `v1` artifacts implement the logical `normalization_v1` contract documented in `NORMALIZATION.md`. They are partitioned by split, source, and country and sharded by deterministic per-country input order.

## Candidate contract: `candidate_v1`

Required identity fields are `target_entity_id`, `candidate_s1_entity_id`, `target_source`, and `country`. Pair uniqueness is (`target_entity_id`, `candidate_s1_entity_id`, `target_source`); country is consistent metadata.

Optional evidence fields include `exact_hit`, `structured_hit`, each retriever's score/rank (`char_name`, `char_sorted`, `word_name`, `address`, `reverse`, `dense`), `retriever_mask`, `retriever_count`, and `fusion_score`. Candidate artifacts contain IDs and numeric metadata, not duplicated business text.

Phase 2A adds boolean signal fields `exact_name_compact`, `exact_name_token_sorted`, `exact_name_core`, `exact_address_compact`, `structured_name_core_number`, `structured_name_compact_number`, `structured_name_core_postal`, `structured_name_compact_postal`, and `structured_name_numeric_signature`. Multiple signals for one identity key are merged into one row. `retriever_count` is the number of retrieval families present (the population count of the defined family bits), not the number of individual signal flags.

Stable `uint16` retriever-mask assignments for `candidate_v1` are: bit 0 exact, bit 1 structured, bit 2 character name, bit 3 token-sorted character name, bit 4 word name, bit 5 address, bit 6 reverse, and bit 7 dense. Changing these assignments requires a candidate contract version change.

Phase 2B writes retriever-specific `candidate_v1` shards below `retrieval/v1/{name_word,address_word,transliteration,rare_token,numeric}`. Name-word and address-word rows contain a float32 score and uint8 rank. Transliteration rows contain `translit_hit`, float32 `translit_score`, and uint8 `translit_rank`; rare-token rows contain `rare_token_hit`, uint32 `rare_token_min_df`, and uint8 `rare_token_overlap_count`; numeric rows contain `numeric_hit` and uint8 `numeric_overlap_count`. Text features are never duplicated into candidate rows.

Rare-token and numeric channels use deterministic per-channel explosion limits after ranking by evidence strength. These are operational safety limits, not the final cross-retriever candidate cap.

Optional retrieval absence uses an explicit `<retriever>_present = false` flag with null/NaN score and rank. Magic sentinels such as `-999` are forbidden. A present score or rank requires the corresponding presence flag to be true.

## Shard manifest and resume

The manifest records stage/version, split/country/source/shard, input/output rows, runtime, peak RAM/VRAM, config hash, code commit, input fingerprint, artifact path/size/hash, status, and creation time. A shard is skippable only if the manifest is complete, expected stage/version/config/input metadata matches, the artifact exists and is non-empty, size/hash agree, the artifact opens, and its schema validator succeeds. Otherwise it must be recomputed without silently overwriting another version.
