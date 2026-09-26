"""Canonical non-destructive normalization shared by every data split."""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Iterable
from typing import Any


TOKEN_PATTERN = re.compile(r"[^\W_]+", flags=re.UNICODE)
NUMBER_PATTERN = re.compile(r"\d+", flags=re.UNICODE)
NORMALIZED_COLUMNS = (
    "entity_id", "country",
    "name_raw", "name_nfkc", "name_casefold", "name_accent_fold", "name_compact",
    "name_tokens", "name_token_sorted", "name_core", "name_transliterated",
    "address_raw", "address_nfkc", "address_casefold", "address_accent_fold",
    "address_normalized", "address_compact", "address_token_sorted",
    "numeric_tokens", "primary_number", "postal_like_tokens",
    "name_missing", "address_missing",
)


def _raw_text(value: object) -> str | None:
    return None if value is None else str(value)


def _view_text(value: object) -> str:
    raw = _raw_text(value)
    return "" if raw is None else raw


def nfkc(value: object) -> str:
    return unicodedata.normalize("NFKC", _view_text(value))


def casefold_text(value: object) -> str:
    return nfkc(value).casefold()


def accent_fold(value: object) -> str:
    decomposed = unicodedata.normalize("NFKD", _view_text(value))
    return "".join(character for character in decomposed if not unicodedata.combining(character))


def canonical_tokens(value: object) -> tuple[str, ...]:
    return tuple(TOKEN_PATTERN.findall(accent_fold(casefold_text(value))))


def joined_tokens(value: object) -> str:
    return " ".join(canonical_tokens(value))


def compact_text(value: object) -> str:
    return "".join(character for character in accent_fold(casefold_text(value)) if character.isalnum())


def _transliterate(value: str) -> str:
    try:
        from unidecode import unidecode
    except ImportError as exc:
        raise RuntimeError("Transliteration requires the data extra: pip install -e '.[data]'") from exc
    try:
        return unidecode(value, errors="preserve")
    except (TypeError, ValueError):
        return value


def remove_trailing_suffixes(tokens: Iterable[str], legal_suffixes: Iterable[str]) -> tuple[str, ...]:
    result = list(tokens)
    suffixes = {suffix.casefold() for suffix in legal_suffixes}
    while len(result) > 1:
        if result[-1] in suffixes:
            result.pop()
            continue
        removed_abbreviation = False
        for width in range(2, min(4, len(result) - 1) + 1):
            suffix_tokens = result[-width:]
            if all(len(token) == 1 for token in suffix_tokens) and "".join(suffix_tokens) in suffixes:
                del result[-width:]
                removed_abbreviation = True
                break
        if not removed_abbreviation:
            break
    return tuple(result)


def build_name_views(value: object, *, legal_suffixes: Iterable[str]) -> dict[str, Any]:
    raw = _raw_text(value)
    normalized_nfkc = nfkc(value)
    folded = normalized_nfkc.casefold()
    accent = accent_fold(folded)
    tokens = tuple(TOKEN_PATTERN.findall(accent))
    token_string = " ".join(tokens)
    transliterated = " ".join(TOKEN_PATTERN.findall(accent_fold(_transliterate(folded).casefold())))
    core = " ".join(remove_trailing_suffixes(tokens, legal_suffixes))
    return {
        "name_raw": raw,
        "name_nfkc": normalized_nfkc,
        "name_casefold": folded,
        "name_accent_fold": accent,
        "name_compact": "".join(character for character in accent if character.isalnum()),
        "name_tokens": token_string,
        "name_token_sorted": " ".join(sorted(tokens)),
        "name_core": core,
        "name_transliterated": transliterated,
        "name_missing": raw is None or not raw.strip(),
    }


def build_address_views(
    value: object, *, postal_like_min_digits: int = 4, postal_like_max_digits: int = 8,
) -> dict[str, Any]:
    if not 1 <= postal_like_min_digits <= postal_like_max_digits:
        raise ValueError("postal-like digit lengths must satisfy 1 <= min <= max")
    raw = _raw_text(value)
    normalized_nfkc = nfkc(value)
    folded = normalized_nfkc.casefold()
    accent = accent_fold(folded)
    tokens = tuple(TOKEN_PATTERN.findall(accent))
    normalized = " ".join(tokens)
    numbers = tuple(NUMBER_PATTERN.findall(normalized))
    postal = tuple(
        number for number in numbers
        if postal_like_min_digits <= len(number) <= postal_like_max_digits
    )
    return {
        "address_raw": raw,
        "address_nfkc": normalized_nfkc,
        "address_casefold": folded,
        "address_accent_fold": accent,
        "address_normalized": normalized,
        "address_compact": "".join(character for character in accent if character.isalnum()),
        "address_token_sorted": " ".join(sorted(tokens)),
        "numeric_tokens": "|".join(numbers),
        "primary_number": numbers[0] if numbers else None,
        "postal_like_tokens": "|".join(postal),
        "address_missing": raw is None or not raw.strip(),
    }


def normalize_record(
    *, entity_id: object, business_name: object, business_address: object, country: object,
    legal_suffixes: Iterable[str], postal_like_min_digits: int, postal_like_max_digits: int,
) -> dict[str, Any]:
    """The single canonical record transformation for train, validation, and test."""
    if entity_id is None or not str(entity_id):
        raise ValueError("entity_id must be non-null and non-empty")
    if country is None or not str(country):
        raise ValueError("country must be non-null and non-empty")
    result = {
        "entity_id": str(entity_id),
        "country": str(country),
        **build_name_views(business_name, legal_suffixes=legal_suffixes),
        **build_address_views(
            business_address,
            postal_like_min_digits=postal_like_min_digits,
            postal_like_max_digits=postal_like_max_digits,
        ),
    }
    return {name: result[name] for name in NORMALIZED_COLUMNS}
