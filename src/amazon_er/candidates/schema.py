"""Logical `candidate_v1` validation without tying the contract to one dataframe library."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from numbers import Integral, Real
from typing import Any

from amazon_er.constants import CANDIDATE_SCHEMA_VERSION, TARGET_SOURCES

REQUIRED_FIELDS = ("target_entity_id", "candidate_s1_entity_id", "target_source", "country")
PAIR_KEY = ("target_entity_id", "candidate_s1_entity_id", "target_source")
RETRIEVERS = ("char_name", "char_sorted", "word_name", "address", "reverse", "dense")
OPTIONAL_FIELDS = frozenset(
    {"exact_hit", "structured_hit", "retriever_mask", "retriever_count", "fusion_score"}
    | {f"{name}_{suffix}" for name in RETRIEVERS for suffix in ("present", "score", "rank")}
)


@dataclass(frozen=True)
class CandidateValidation:
    valid: bool
    errors: tuple[str, ...]
    row_count: int


def validate_candidates(rows: Iterable[Mapping[str, Any]], *, require_known_source: bool = True) -> CandidateValidation:
    errors: list[str] = []
    seen: dict[tuple[Any, ...], Any] = {}
    count = 0
    for index, row in enumerate(rows):
        count += 1
        missing = [name for name in REQUIRED_FIELDS if name not in row or row[name] is None]
        if missing:
            errors.append(f"row {index}: missing required fields {missing}")
            continue
        if require_known_source and row["target_source"] not in TARGET_SOURCES:
            errors.append(f"row {index}: target_source must be S2 or S3")
        if not isinstance(row["target_entity_id"], (str, Integral)) or isinstance(row["target_entity_id"], bool):
            errors.append(f"row {index}: target_entity_id must be str or integer")
        if not isinstance(row["candidate_s1_entity_id"], (str, Integral)) or isinstance(row["candidate_s1_entity_id"], bool):
            errors.append(f"row {index}: candidate_s1_entity_id must be str or integer")
        if not isinstance(row["country"], str) or not row["country"].strip():
            errors.append(f"row {index}: country must be a non-empty string")
        key = tuple(row[name] for name in PAIR_KEY)
        if key in seen:
            if seen[key] != row["country"]:
                errors.append(f"row {index}: duplicate pair has inconsistent country")
            else:
                errors.append(f"row {index}: duplicate candidate pair {key}")
        else:
            seen[key] = row["country"]
        for flag in ("exact_hit", "structured_hit"):
            if flag in row and row[flag] is not None and not isinstance(row[flag], (bool, Integral)):
                errors.append(f"row {index}: {flag} must be boolean/integer")
        for retriever in RETRIEVERS:
            present = row.get(f"{retriever}_present")
            score, rank = row.get(f"{retriever}_score"), row.get(f"{retriever}_rank")
            if present is False and (score is not None or rank is not None):
                errors.append(f"row {index}: absent {retriever} evidence must have null score and rank")
            if present is not True and (score is not None or rank is not None):
                errors.append(f"row {index}: {retriever}_present must be true when evidence exists")
            if score is not None and (not isinstance(score, Real) or isinstance(score, bool)):
                errors.append(f"row {index}: {retriever}_score must be numeric")
            if rank is not None and (not isinstance(rank, Integral) or isinstance(rank, bool) or rank < 1):
                errors.append(f"row {index}: {retriever}_rank must be a positive integer")
    return CandidateValidation(not errors, tuple(errors), count)
