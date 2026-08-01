#!/usr/bin/env python3
"""List sigmoid-benchmark publication cases without scientific execution."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from one_d.publication_experiments import (  # noqa: E402
    DEFAULT_CATALOG_PATH,
    load_publication_catalog,
    publication_case_summary,
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", default=str(DEFAULT_CATALOG_PATH))
    parser.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    catalog = load_publication_catalog(args.catalog)
    ordered_cases = sorted(
        catalog.cases,
        key=lambda case: (int(case.figure.rsplit(" ", 1)[-1]), case.case_id),
    )
    summaries = [publication_case_summary(case) for case in ordered_cases]
    if args.json:
        print(
            json.dumps(
                {
                    "catalog_checksum": catalog.checksum(),
                    "case_count": len(summaries),
                    "cases": summaries,
                    "assembles_operators": False,
                    "solves": False,
                    "writes_files": False,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    print(f"Catalog checksum: {catalog.checksum()}")
    print(f"Cases: {len(summaries)} (read-only; legacy_sigmoid benchmark)")
    print("case_id | figure | model | operators | N_r | N_q | status | ready")
    for item in summaries:
        values = (
            item["case_id"],
            item["figure"],
            item["model"],
            item["operators"] or "n/a",
            item["N_r"] if item["N_r"] is not None else "n/a",
            item["N_q"] if item["N_q"] is not None else "n/a",
            item["status"],
            "yes" if item["execution_ready"] else "no",
        )
        print(" | ".join(str(value) for value in values))
        if item["missing_author_inputs"]:
            print("  missing: " + "; ".join(item["missing_author_inputs"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
