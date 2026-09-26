"""Command-line entry point."""

from __future__ import annotations

import argparse
import json

from amazon_er.config import load_config
from amazon_er.infra.logging_utils import configure_logging
from amazon_er.pipeline.runner import run_stage


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="amazon-er")
    subparsers = parser.add_subparsers(dest="command", required=True)
    run = subparsers.add_parser("run")
    run.add_argument("--stage", required=True)
    run.add_argument("--config", required=True)
    run.add_argument("--country")
    run.add_argument("--source", choices=("S2", "S3"))
    run.add_argument("--shard-id")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    configure_logging()
    config = load_config(args.config)
    result = run_stage(args.stage, config, country=args.country, source=args.source, shard_id=args.shard_id)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
