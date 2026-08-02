#!/usr/bin/env python3
"""Safely verify a one-dimensional reproduction archive."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from one_d.reproduction_archive import verify_archive


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Verify checksum, safe paths, file checksums, inventory, and important "
            "scientific checksums without importing solver entry points."
        )
    )
    parser.add_argument("archive", type=Path)
    parser.add_argument(
        "--expected-sha256",
        help="expected archive checksum (otherwise use a neighboring .sha256 sidecar)",
    )
    parser.add_argument(
        "--extract-to",
        type=Path,
        help="retain a complete safe extraction in a new directory",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    arguments = parser.parse_args(argv)
    try:
        report = verify_archive(
            arguments.archive,
            expected_sha256=arguments.expected_sha256,
            extract_to=arguments.extract_to,
        )
    except (OSError, ValueError, RuntimeError) as error:
        parser.exit(2, f"archive verification failed: {error}\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
