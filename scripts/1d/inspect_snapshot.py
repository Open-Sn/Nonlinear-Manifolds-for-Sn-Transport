#!/usr/bin/env python3
"""Report configured compatibility and metadata without modifying a snapshot."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from one_d.config import load_config
from one_d.fom import inspect_snapshot


DEFAULT_CONFIG = REPOSITORY_ROOT / "configs" / "1d" / "legacy_production.json"


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("snapshot", type=Path)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--sha256", action="store_true")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    config = load_config(args.config)
    inspection = inspect_snapshot(
        args.snapshot, config, include_sha256=args.sha256
    )
    print(json.dumps(inspection.to_dict(), indent=2, sort_keys=True))
    return 0 if inspection.compatible else 1


if __name__ == "__main__":
    raise SystemExit(main())
