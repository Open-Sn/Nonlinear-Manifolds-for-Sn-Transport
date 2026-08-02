#!/usr/bin/env python3
"""Build deterministic archives from completed one-dimensional artifacts only."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from one_d.reproduction_archive import build_archives, load_archive_spec


def _run_overrides(values: list[str]) -> dict[str, str]:
    result: dict[str, str] = {}
    for value in values:
        key, separator, run_id = value.partition("=")
        if not separator or not key or not run_id:
            raise argparse.ArgumentTypeError(
                f"run override must have KEY=RUN_ID form: {value}"
            )
        if key in result:
            raise argparse.ArgumentTypeError(f"duplicate run override: {key}")
        result[key] = run_id
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Validate and package existing completed 1D reproducibility assets; "
            "this command never launches scientific execution."
        )
    )
    parser.add_argument(
        "--spec",
        type=Path,
        default=REPOSITORY_ROOT / "configs/1d/publication/archive_spec.json",
        help="archive inclusion and checksum specification",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPOSITORY_ROOT / "dist/1d",
        help="new directory for generated archives",
    )
    parser.add_argument(
        "--kind", choices=("core", "audit", "both"), default="both"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="validate and report exact inclusions without writing anything",
    )
    parser.add_argument(
        "--run-id",
        action="append",
        default=[],
        metavar="KEY=RUN_ID",
        help="explicit authoritative run override (repeatable)",
    )
    parser.add_argument(
        "--allow-unknown-run-id",
        action="store_true",
        help="accept explicitly supplied run IDs that differ from the specification",
    )
    parser.add_argument(
        "--doi",
        help=(
            "reserved unpublished dataset DOI to embed in README, citation, "
            "provenance, and archive metadata"
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    arguments = parser.parse_args(argv)
    try:
        overrides = _run_overrides(arguments.run_id)
        report = build_archives(
            load_archive_spec(arguments.spec),
            kind=arguments.kind,
            repository_root=REPOSITORY_ROOT,
            output_directory=arguments.output_dir,
            dry_run=arguments.dry_run,
            run_overrides=overrides,
            allow_unknown_run_ids=arguments.allow_unknown_run_id,
            doi=arguments.doi,
        )
    except (OSError, ValueError, RuntimeError) as error:
        parser.exit(2, f"archive build failed: {error}\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
