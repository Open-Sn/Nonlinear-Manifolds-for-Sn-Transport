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
from one_d.publication_plotting import (  # noqa: E402
    manuscript_plot_plan,
    plot_manuscript_figure,
    validate_manuscript_figure_bundle,
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "bundle_directory",
        nargs="?",
        help="prebuilt bundle (legacy diagnostic-mode spelling)",
    )
    parser.add_argument(
        "--source-bundle",
        help="validated, complete Figure 1--3 figure-data bundle",
    )
    parser.add_argument("--figure", choices=("1", "2", "3"))
    parser.add_argument(
        "--layout",
        choices=("diagnostic", "manuscript"),
        default="diagnostic",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--plot",
        action="store_true",
        help="write diagnostic plots (retained for backward compatibility)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="validate and report manuscript inputs/outputs without writing",
    )
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    selected_bundle = args.source_bundle or args.bundle_directory
    if selected_bundle is None:
        print("a bundle directory or --source-bundle is required", file=sys.stderr)
        return 2
    if (
        args.source_bundle is not None
        and args.bundle_directory is not None
        and Path(args.source_bundle) != Path(args.bundle_directory)
    ):
        print("bundle_directory and --source-bundle disagree", file=sys.stderr)
        return 2
    bundle = Path(selected_bundle)

    if args.layout == "manuscript":
        if args.figure is None:
            print("--figure is required for manuscript layout", file=sys.stderr)
            return 2
        if args.plot and args.dry_run:
            print("--plot and --dry-run cannot be combined", file=sys.stderr)
            return 2
        try:
            validated = validate_manuscript_figure_bundle(
                bundle,
                expected_figure=f"Figure {args.figure}",
            )
            if args.dry_run:
                print(
                    json.dumps(
                        manuscript_plot_plan(
                            validated,
                            output_directory=args.output_dir,
                        ),
                        indent=2,
                        sort_keys=True,
                    )
                )
                return 0
            paths = plot_manuscript_figure(
                bundle,
                output_directory=args.output_dir,
                expected_figure=f"Figure {args.figure}",
            )
        except (FileExistsError, KeyError, OSError, ValueError) as error:
            print(str(error), file=sys.stderr)
            return 2
        print(
            json.dumps(
                {"action": "plotted_manuscript", "paths": [str(path) for path in paths]},
                indent=2,
            )
        )
        return 0

    if args.source_bundle is not None or args.figure is not None or args.dry_run:
        print(
            "--source-bundle, --figure, and --dry-run require --layout manuscript",
            file=sys.stderr,
        )
        return 2
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
    print(
        json.dumps(
            {"action": "plotted", "paths": [str(path) for path in paths]},
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
