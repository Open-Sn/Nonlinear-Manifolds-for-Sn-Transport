#!/usr/bin/env python3
"""Build or validate a compact regenerated-sigmoid Figure 4 bundle."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from one_d.figure4 import build_figure4_bundle  # noqa: E402
from one_d.figure4_plotting import validate_figure4_bundle  # noqa: E402


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--output-root", default="results/1d/publication")
    parser.add_argument("--allow-partial", action="store_true")
    parser.add_argument("--build", action="store_true")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    if not args.build:
        print(
            json.dumps(
                {
                    "action": "would_build_figure4_bundle",
                    "run_id": args.run_id,
                    "output_directory": args.output_dir,
                    "allow_partial": args.allow_partial,
                    "writes_files": False,
                    "launches_scientific_execution": False,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    root = build_figure4_bundle(
        run_id=args.run_id,
        output_directory=args.output_dir,
        output_root=args.output_root,
        allow_partial=args.allow_partial,
    )
    metadata, arrays = validate_figure4_bundle(
        root, require_complete=not args.allow_partial
    )
    print(
        json.dumps(
            {
                "action": "built_figure4_bundle",
                "path": str(root),
                "case_set_status": metadata["case_set_status"],
                "array_count": len(arrays),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
