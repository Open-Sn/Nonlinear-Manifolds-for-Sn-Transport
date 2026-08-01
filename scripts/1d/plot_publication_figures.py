#!/usr/bin/env python3
"""Plot prebuilt publication figure data without importing solver entry points."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from one_d.publication_artifacts import plot_figure_data_bundle  # noqa: E402


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle_directory")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--plot", action="store_true", help="write plot files")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    bundle = Path(args.bundle_directory)
    metadata_path = bundle / "figure_data.json"
    data_path = bundle / "figure_data.npz"
    if not metadata_path.is_file() or not data_path.is_file():
        print("missing figure_data.json or figure_data.npz", file=sys.stderr)
        return 2
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("benchmark_variant") != "legacy_sigmoid":
        print("figure bundle is not marked legacy_sigmoid", file=sys.stderr)
        return 2
    if not args.plot:
        print(
            json.dumps(
                {
                    "action": "would_plot",
                    "figure": metadata.get("figure"),
                    "benchmark_variant": metadata["benchmark_variant"],
                    "complete_publication_reproduction": metadata.get(
                        "complete_publication_reproduction", False
                    ),
                    "output_directory": args.output_dir,
                    "launches_scientific_execution": False,
                    "writes_files": False,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    paths = plot_figure_data_bundle(bundle, output_directory=args.output_dir)
    print(json.dumps({"action": "plotted", "paths": [str(path) for path in paths]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
