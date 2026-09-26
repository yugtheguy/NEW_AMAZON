"""Thin Kaggle-compatible pass-through to the repository CLI."""

from amazon_er.cli import main


if __name__ == "__main__":
    raise SystemExit(main())
