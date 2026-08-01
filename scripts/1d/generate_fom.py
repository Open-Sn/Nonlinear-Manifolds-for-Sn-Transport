#!/usr/bin/env python3
"""Safely inspect, generate, or reuse one configured 1-D FOM snapshot."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from one_d.config import load_config
from one_d.workflows import dry_run_fom, execute_fom_workflow


DEFAULT_CONFIG = REPOSITORY_ROOT / "configs" / "1d" / "legacy_production.json"


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--snapshot", type=Path, help="existing snapshot to inspect/reuse")
    parser.add_argument(
        "--output-dir", type=Path, help="explicit provenance run directory"
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument(
        "--execute",
        action="store_true",
        help="explicitly authorize assembly, integration, and output writes",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--hash-snapshot", action="store_true")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    config = load_config(args.config)
    if not args.execute:
        report = dry_run_fom(
            config,
            snapshot_path=args.snapshot,
            output_directory=args.output_dir,
            overwrite=args.overwrite,
        )
        print(json.dumps(report.to_dict(), indent=2, sort_keys=True))
        return 0

    outcome = execute_fom_workflow(
        config,
        execute=True,
        run_directory=args.output_dir,
        config_source=args.config,
        existing_snapshot=args.snapshot,
        overwrite=args.overwrite,
        hash_snapshot=args.hash_snapshot,
    )
    print(f"action: {outcome['action']}")
    print(f"snapshot: {outcome['snapshot_path']}")
    print(f"run directory: {outcome['run'].root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
