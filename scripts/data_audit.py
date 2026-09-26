"""CLI for small, explicit dataset audits. Dataset-specific GT joins arrive in Phase 0B."""

from __future__ import annotations

import argparse
import json

from amazon_er.data.audit import audit_ground_truth, audit_table


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="+")
    parser.add_argument("--entity-id-column", default="entity_id")
    parser.add_argument("--gt")
    parser.add_argument("--gt-s1-column")
    parser.add_argument("--gt-target-column")
    parser.add_argument("--gt-source-column")
    parser.add_argument("--gt-s1-country-column")
    parser.add_argument("--gt-target-country-column")
    args = parser.parse_args()
    try:
        reports = [audit_table(path, entity_id_column=args.entity_id_column) for path in args.paths]
        result = {"tables": reports}
        if args.gt:
            if not args.gt_s1_column or not args.gt_target_column:
                parser.error("--gt requires --gt-s1-column and --gt-target-column")
            result["ground_truth"] = audit_ground_truth(
                args.gt,
                s1_column=args.gt_s1_column,
                target_column=args.gt_target_column,
                source_column=args.gt_source_column,
                s1_country_column=args.gt_s1_country_column,
                target_country_column=args.gt_target_country_column,
            )
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        parser.error(str(exc))
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
