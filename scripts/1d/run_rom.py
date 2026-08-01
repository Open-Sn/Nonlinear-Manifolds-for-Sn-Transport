#!/usr/bin/env python3
"""Safely inspect or execute one explicitly selected configured 1-D ROM."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from one_d.config import load_config
from one_d.workflows import dry_run_rom, execute_rom_workflow


DEFAULT_CONFIG = REPOSITORY_ROOT / "configs" / "1d" / "legacy_production.json"


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--model",
        choices=("linear", "element-wise", "elementwise", "tensorial"),
        default="elementwise",
    )
    parser.add_argument(
        "--operators", choices=("projected", "inferred"), default="inferred"
    )
    parser.add_argument("--input-snapshot", type=Path)
    parser.add_argument(
        "--output-dir", type=Path, help="explicit provenance run directory"
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument(
        "--execute",
        action="store_true",
        help="explicitly authorize assembly, ROM solution, and output writes",
    )
    parser.add_argument(
        "--historical-sequence",
        action="store_true",
        help="report the six-case legacy sequence; execute it through the root script",
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    config = load_config(args.config)
    input_snapshot = args.input_snapshot or Path(config.output.snapshot_filename)
    if not args.execute:
        report = dry_run_rom(
            config,
            model=args.model,
            operators=args.operators,
            input_snapshot=input_snapshot,
            historical_full_sequence=args.historical_sequence,
        )
        print(json.dumps(report.to_dict(), indent=2, sort_keys=True))
        return 0

    if args.historical_sequence:
        if args.input_snapshot is not None or args.output_dir is not None:
            raise SystemExit(
                "historical sequence uses the root script's canonical path and output behavior"
            )
        import Nonlinear_Manifold_ROM

        Nonlinear_Manifold_ROM.main()
        return 0

    outcome = execute_rom_workflow(
        config,
        model=args.model,
        operators=args.operators,
        input_snapshot=input_snapshot,
        execute=True,
        run_directory=args.output_dir,
        config_source=args.config,
    )
    print(f"run directory: {outcome['run'].root}")
    print(json.dumps(outcome["metrics"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
