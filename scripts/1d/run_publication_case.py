#!/usr/bin/env python3
"""Dry-run by default; execute one fully specified publication case explicitly."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from one_d.publication_artifacts import (  # noqa: E402
    PublicationExecutionRefused,
    execute_publication_case,
)
from one_d.publication_experiments import (  # noqa: E402
    DEFAULT_CATALOG_PATH,
    dry_run_publication_case,
    load_publication_catalog,
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("case_id")
    parser.add_argument("--catalog", default=str(DEFAULT_CATALOG_PATH))
    parser.add_argument("--snapshot", help="input sigmoid snapshot path")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--execute", action="store_true")
    parser.add_argument("--output-root", default="results/1d/publication")
    parser.add_argument("--run-directory")
    parser.add_argument("--run-id")
    parser.add_argument(
        "--shared-offline",
        help="validated shared derivative/POD artifact directory",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="explicit overwrite authorization; completed publication runs remain immutable",
    )
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    catalog = load_publication_catalog(args.catalog)
    case = catalog.get(args.case_id)
    snapshot = args.snapshot or case.required_input_snapshot
    report = dry_run_publication_case(catalog, case, snapshot_path=snapshot)
    if not args.execute:
        print(json.dumps(report.to_dict(), indent=2, sort_keys=True, allow_nan=False))
        return 0
    if report.action != "would_execute":
        print(json.dumps(report.to_dict(), indent=2, sort_keys=True, allow_nan=False))
        return 2
    if args.run_directory and Path(args.run_directory).exists():
        message = (
            "completed publication run directories are immutable; choose a new --run-directory"
            if args.overwrite
            else "output exists; --overwrite authorization would be required"
        )
        print(message, file=sys.stderr)
        return 2
    try:
        run = execute_publication_case(
            catalog,
            case,
            input_snapshot=snapshot,
            execute=True,
            output_root=args.output_root,
            run_directory=args.run_directory,
            run_id=args.run_id,
            shared_offline_directory=args.shared_offline,
        )
    except (PublicationExecutionRefused, FileExistsError, FileNotFoundError, ValueError) as error:
        print(str(error), file=sys.stderr)
        return 2
    print(json.dumps({"action": "executed", "run_directory": str(run.root)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
