#!/usr/bin/env python3
"""Plot regenerated manuscript Figure 4 from a validated bundle only."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from one_d.figure4_plotting import (  # noqa: E402
    figure4_plot_plan,
    plot_figure4_bundle,
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-bundle", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    if args.dry_run:
        print(
            json.dumps(
                figure4_plot_plan(
                    args.source_bundle,
                    output_directory=args.output_dir,
                ),
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    try:
        paths = plot_figure4_bundle(
            args.source_bundle,
            output_directory=args.output_dir,
        )
    except (FileExistsError, KeyError, OSError, ValueError) as error:
        print(str(error), file=sys.stderr)
        return 2
    print(
        json.dumps(
            {"action": "plotted_figure4", "paths": [str(path) for path in paths]},
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
