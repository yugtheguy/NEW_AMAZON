# Dataset Contract

The facts below were verified by the Phase 0B `v1` audit. The combined sampled-input fingerprint is `86bb7ec55a54734cf1990db8673e202114040d86441b516515c673779b4bf0eb`.

Fingerprints use SHA-256 over the file size, header, and 1 MiB blocks sampled at the start, middle, and end of each file. Per-file fingerprints are stored in the runtime audit report.

## Files

| Logical file | Rows | Bytes |
|---|---:|---:|
| `train_source1.tsv` | 2,206,821 | 210,069,713 |
| `train_source2.tsv` | 5,034,616 | 489,301,488 |
| `train_source3.tsv` | 5,285,603 | 503,705,637 |
| `train_ground_truth.tsv` | 2,206,821 | 127,015,583 |
| `test_source1.tsv` | 1,732,544 | 175,022,086 |
| `test_source2.tsv` | 4,887,273 | 509,456,422 |
| `test_source3.tsv` | 5,082,316 | 506,002,772 |

All files are UTF-8 tab-separated text.

## Schemas

All six source files have this exact ordered schema, with all fields parsed as strings:

1. `entity_id`
2. `business_name`
3. `business_address`
4. `country`

The ground-truth schema is exactly:

1. `source1_entity_id`
2. `matched_entity_ids`

## Countries

| File | India | US | France |
|---|---:|---:|---:|
| train S1 | 883,188 | 1,323,633 | 0 |
| train S2 | 2,017,799 | 3,016,817 | 0 |
| train S3 | 2,115,547 | 3,170,056 | 0 |
| test S1 | 809,986 | 663,106 | 259,452 |
| test S2 | 2,312,565 | 1,871,330 | 703,378 |
| test S3 | 2,405,000 | 1,945,701 | 731,615 |

Verified train countries are exactly `India` and `US`. Verified test countries are exactly `France`, `India`, and `US`.

## ID Integrity

Every source file has zero null entity IDs, zero duplicate entity IDs, and `unique_entity_ids == rows`. Within both train and test, S1/S2, S1/S3, and S2/S3 ID overlap counts are all zero.

## Ground Truth Structure

Ground truth contains one row per train S1 entity. `matched_entity_ids` is a comma-separated list of S2/S3 IDs; an empty string means zero matches.

- GT rows: 2,206,821
- Unique S1 rows: 2,206,821
- Positive pair rows: 7,638,365
- Unique positive pairs: 7,638,365
- Invalid target prefixes: 0
- Missing S1 references: 0
- Missing S2 references: 0
- Missing S3 references: 0

## S1 Cardinality

| Matches | S1 entities |
|---:|---:|
| 0 | 123,247 |
| 1 | 119,157 |
| 2 | 375,212 |
| 3 | 530,841 |
| 4 | 484,115 |
| 5+ | 574,249 |

Mean matches are 3.4613, median 3, p90 6, p95 6, p99 8, and maximum 11. The verified zero-match rate is 5.5848% overall, 5.5878% for India, and 5.5828% for US.

## Target Ownership

- Maximum distinct S1 owners per matched S2 target: 1
- Maximum distinct S1 owners per matched S3 target: 1
- S2 targets with multiple owners: 0
- S3 targets with multiple owners: 0

## Country Consistency

- Cross-country S1↔S2 GT pairs: 0
- Cross-country S1↔S3 GT pairs: 0
- Combined cross-country GT pairs: 0

## Missingness

Business names and countries have zero null, empty, or whitespace-only values in all source files. Business addresses have no null or whitespace-only values, but empty strings occur in target sources:

| File | Empty addresses | Rate |
|---|---:|---:|
| train S2 | 168,967 | 3.3561% |
| train S3 | 175,916 | 3.3282% |
| test S2 | 129,408 | 2.6479% |
| test S3 | 136,098 | 2.6779% |

Train/test S1 business addresses have zero empty values.

## Script Characteristics

The lightweight classifier found material non-ASCII/script variation in noisy target sources. Aggregate non-ASCII name rates across S1/S2/S3 are approximately 19.01% for train India, 5.56% for train US, 19.48% for test India, 5.39% for test US, and 22.93% for test France. India contains Devanagari and other Indic scripts; France contains accented Latin text. These measurements support preserving Unicode and creating accent-folded/transliterated views without destroying raw text.

## Verified Architecture Invariants

`TARGET_EXCLUSIVITY_VERIFIED = true`

`STRICT_COUNTRY_BLOCKING_SAFE = true`
