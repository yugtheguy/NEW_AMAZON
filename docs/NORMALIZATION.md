# Normalization Contract — `normalization_v1`

Normalization creates retrieval views and never replaces authoritative raw Unicode. The same `normalize_record` implementation serves train, validation, and test.

## Null policy

Raw text remains `null` when input is null. Empty and whitespace-only raw strings remain distinguishable in raw columns. Every derived textual view uses `""` for missing input; it never creates literal `nan`, `none`, or `null`. Boolean `name_missing` and `address_missing` flags cover null, empty, and whitespace-only input.

## Name views

- `name_nfkc`: Unicode NFKC.
- `name_casefold`: NFKC plus Unicode casefold.
- `name_accent_fold`: combining diacritics removed from the casefolded view.
- `name_compact`: accent-folded Unicode alphanumeric characters only.
- `name_tokens`: accent-folded Unicode alphanumeric tokens joined by single spaces.
- `name_token_sorted`: the same tokens sorted lexicographically.
- `name_core`: trailing legal suffix tokens removed conservatively while retaining at least one token.
- `name_transliterated`: Unidecode retrieval view, then casefold/accent-fold/token normalization. Unknown characters are preserved rather than causing failure.

The exact suffix list is configuration-owned: `ltd`, `limited`, `pvt`, `private`, `llp`, `llc`, `inc`, `incorporated`, `corp`, `corporation`, `co`, `company`, `sa`, `sas`, `sarl`, `sasu`, `eurl`, and `sci`.

## Address views

Address views preserve component order. `address_normalized` is accent-folded Unicode alphanumeric tokens joined with spaces; `address_compact` removes boundaries; `address_token_sorted` sorts tokens. `numeric_tokens` preserves digit-group order using `|`. `primary_number` is the first digit group and is not claimed to be a house number. `postal_like_tokens` retains digit groups whose lengths are inclusively 4–8.

## Storage and resume

Outputs are Zstandard-compressed Parquet under `artifacts/normalized/v1/<split>/<source>/country=<country>/`. Default shards contain 300,000 rows per split/source/country. Each shard is written through a temporary file, opened and validated, hashed, then accompanied by a complete manifest. Resume uses the shared Phase 0 artifact validator and requires matching stage/version, config hash, dataset/source dependency fingerprint, checksum, size, schema, row count, non-null ID/country, and country partition.

Full runs additionally verify Phase 0B country counts and global source-level ID uniqueness, then write summary and deterministic sample reports under `artifacts/reports/normalization/v1/`.
