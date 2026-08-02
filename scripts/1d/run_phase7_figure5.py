#!/usr/bin/env python3
"""Plan or resume exactly the author-approved sigmoid Figure 5 study."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from one_d.figure5 import (  # noqa: E402
    execute_figure5_plan,
    figure5_execution_plan,
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--snapshot", required=True)
    parser.add_argument("--fom-manifest", required=True)
    parser.add_argument("--shared-offline", required=True)
    parser.add_argument("--output-root", default="results/1d/publication")
    parser.add_argument("--config", default="configs/1d/legacy_production.json")
    parser.add_argument(
        "--catalog",
        default="configs/1d/publication/experiments.json",
    )
    authorization = parser.add_mutually_exclusive_group()
    authorization.add_argument("--execute", action="store_true")
    authorization.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    if not args.execute:
        plan = figure5_execution_plan(
            run_id=args.run_id,
            output_root=args.output_root,
        )
        plan["action"] = "would_execute_or_resume_figure5"
        plan["writes_files"] = False
        print(json.dumps(plan, indent=2, sort_keys=True, allow_nan=False))
        return 0
    plan = execute_figure5_plan(
        run_id=args.run_id,
        snapshot_path=args.snapshot,
        fom_manifest_path=args.fom_manifest,
        shared_offline_directory=args.shared_offline,
        output_root=args.output_root,
        config_path=args.config,
        catalog_path=args.catalog,
    )
    print(json.dumps(plan, indent=2, sort_keys=True, allow_nan=False))
    return 0 if plan["status"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
