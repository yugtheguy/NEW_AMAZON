"""Thin convenience wrapper for the canonical package data-audit stage."""

from __future__ import annotations

import sys

from amazon_er.cli import main


if __name__ == "__main__":
    raise SystemExit(main(["run", "--stage", "data-audit", *sys.argv[1:]]))
