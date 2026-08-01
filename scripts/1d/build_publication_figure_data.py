#!/usr/bin/env python3
"""Build figure-ready arrays only from completed, validated result artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from one_d.publication_artifacts import (  # noqa: E402
    build_figure_data_bundle,
    validate_publication_artifact,
)
from one_d.publication_experiments import (  # noqa: E402
    DEFAULT_CATALOG_PATH,
    load_publication_catalog,
)


def _figure_name(value: str) -> str:
    normalized = value.lower().replace("figure", "").strip()
    if normalized not in {"1", "2", "3", "4", "5"}:
        raise argparse.ArgumentTypeError("figure must be 1, 2, 3, 4, or 5")
    return f"Figure {normalized}"


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--figure", required=True, type=_figure_name)
    parser.add_argument("--artifact", action="append", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--catalog", default=str(DEFAULT_CATALOG_PATH))
    parser.add_argument("--build", action="store_true", help="write the requested bundle")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    catalog = load_publication_catalog(args.catalog)
    matching = []
    for root in args.artifact:
        manifest = validate_publication_artifact(root, catalog=catalog)
        if manifest["figure"] == args.figure:
            matching.append(manifest["case_id"])
    report = {
        "figure": args.figure,
        "benchmark_variant": "legacy_sigmoid",
        "matching_case_ids": sorted(matching),
        "output_directory": args.output_dir,
        "action": "would_build" if matching else "refuse_missing_cases",
        "launches_scientific_execution": False,
        "writes_files": bool(args.build and matching),
    }
    if not args.build or not matching:
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0 if matching else 2
    bundle = build_figure_data_bundle(
        args.figure,
        args.artifact,
        catalog=catalog,
        output_directory=args.output_dir,
    )
    print(
        json.dumps(
            {
                **report,
                "action": "built",
                "case_set_complete": bundle.complete,
                "complete_publication_reproduction": False,
                "metadata_path": str(bundle.metadata_path),
                "data_path": str(bundle.data_path),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
