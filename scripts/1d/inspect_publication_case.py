#!/usr/bin/env python3
"""Inspect one resolved publication case without assembly, solving, or writes."""

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
    inspect_publication_case,
    load_publication_catalog,
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("case_id")
    parser.add_argument("--catalog", default=str(DEFAULT_CATALOG_PATH))
    parser.add_argument("--snapshot", help="optional snapshot path to inspect")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    catalog = load_publication_catalog(args.catalog)
    report = inspect_publication_case(
        catalog,
        args.case_id,
        snapshot_path=args.snapshot,
    )
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
