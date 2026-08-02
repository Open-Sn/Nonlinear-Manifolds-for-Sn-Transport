#!/usr/bin/env python3
"""Plan or resume the author-approved regenerated sigmoid Figure 4 study."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from one_d.figure4 import execute_phase8, phase8_dry_run_plan  # noqa: E402


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--snapshot", required=True)
    parser.add_argument("--fom-manifest", required=True)
    parser.add_argument("--shared-offline", required=True)
    parser.add_argument("--shared-metric-inputs", required=True)
    parser.add_argument("--figure5-bundle", required=True)
    parser.add_argument("--output-root", default="results/1d/publication")
    authorization = parser.add_mutually_exclusive_group()
    authorization.add_argument("--execute", action="store_true")
    authorization.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    if not args.execute:
        plan = phase8_dry_run_plan(run_id=args.run_id, output_root=args.output_root)
        print(json.dumps(plan, indent=2, sort_keys=True, allow_nan=False))
        return 0
    result = execute_phase8(
        run_id=args.run_id,
        snapshot_path=args.snapshot,
        fom_manifest_path=args.fom_manifest,
        shared_offline_directory=args.shared_offline,
        shared_metric_inputs_path=args.shared_metric_inputs,
        figure5_bundle_path=args.figure5_bundle,
        output_root=args.output_root,
    )
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0 if result["status"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
