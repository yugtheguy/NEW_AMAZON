"""Shared non-destructive normalization interfaces for later implementation."""

from __future__ import annotations

import re
import unicodedata


def _text(value: object) -> str:
    return "" if value is None else str(value)


def normalize_name(value: object) -> str:
    return " ".join(unicodedata.normalize("NFKC", _text(value)).casefold().split())


def normalize_address(value: object) -> str:
    return " ".join(unicodedata.normalize("NFKC", _text(value)).casefold().split())


def _accent_fold(value: str) -> str:
    return "".join(char for char in unicodedata.normalize("NFKD", value) if not unicodedata.combining(char))


def _compact(value: str) -> str:
    return "".join(char for char in value if char.isalnum())


def build_name_views(value: object) -> dict[str, object]:
    raw = _text(value)
    nfkc = unicodedata.normalize("NFKC", raw)
    folded = " ".join(nfkc.casefold().split())
    accent = _accent_fold(folded)
    tokens = folded.split()
    return {
        "name_raw": raw, "name_nfkc": nfkc, "name_casefold": folded,
        "name_accent_folded": accent, "name_compact": _compact(accent),
        "name_token_sorted": " ".join(sorted(tokens)), "name_core": folded,
        "name_transliterated": None,
    }


def build_address_views(value: object) -> dict[str, object]:
    raw = _text(value)
    nfkc = unicodedata.normalize("NFKC", raw)
    folded = " ".join(nfkc.casefold().split())
    accent = _accent_fold(folded)
    tokens = folded.split()
    numeric = re.findall(r"\d+", folded)
    postal = [token for token in tokens if any(char.isdigit() for char in token)]
    return {
        "address_raw": raw, "address_nfkc": nfkc, "address_casefold": folded,
        "address_accent_folded": accent, "address_compact": _compact(accent),
        "address_token_sorted": " ".join(sorted(tokens)), "numeric_tokens": numeric,
        "primary_number": numeric[0] if numeric else None, "postal_like_tokens": postal,
    }
