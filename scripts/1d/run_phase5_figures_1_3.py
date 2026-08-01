#!/usr/bin/env python3
"""Validate a production FOM and execute only the seven Phase-5 Figure 1-3 cases."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Any


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from one_d.config import load_config  # noqa: E402
from one_d.publication_artifacts import (  # noqa: E402
    build_figure_data_bundle,
    execute_publication_case,
    plot_figure_data_bundle,
    sha256_file,
    validate_publication_artifact,
)
from one_d.publication_experiments import (  # noqa: E402
    DEFAULT_CATALOG_PATH,
    load_publication_catalog,
    publication_case_summary,
)
from one_d.shared_offline import (  # noqa: E402
    build_shared_offline_artifacts,
    create_dataset_summary,
    load_shared_offline_artifacts,
)


PHASE5_CASES = (
    "fig1_pod_reducibility",
    "fig2_linear_projected",
    "fig2_elementwise_projected",
    "fig2_tensorial_projected",
    "fig3_linear_inferred",
    "fig3_elementwise_inferred",
    "fig3_tensorial_inferred",
)
FIGURE_CASES = {
    "Figure 1": PHASE5_CASES[:1],
    "Figure 2": PHASE5_CASES[1:4],
    "Figure 3": PHASE5_CASES[4:7],
}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", required=True)
    parser.add_argument("--fom-run-directory", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--config", default="configs/1d/legacy_production.json")
    parser.add_argument("--catalog", default=str(DEFAULT_CATALOG_PATH))
    parser.add_argument("--output-root", default="results/1d/publication")
    parser.add_argument(
        "--execute",
        action="store_true",
        help="authorize shared offline work and exactly the seven Figure 1-3 cases",
    )
    parser.add_argument("--skip-plots", action="store_true")
    return parser.parse_args(argv)


def _write_json(path: Path, content: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(content, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _case_table(catalog) -> list[dict[str, Any]]:
    table = []
    for case_id in PHASE5_CASES:
        case = catalog.get(case_id)
        if not case.fully_specified or not case.execution_allowed:
            raise ValueError(f"Phase-5 case is no longer fully specified: {case_id}")
        summary = publication_case_summary(case)
        if summary["benchmark_variant"] != "legacy_sigmoid":
            raise ValueError(f"Phase-5 case has wrong benchmark variant: {case_id}")
        table.append(summary)
    return table


def _comparison_report(
    output_path: Path,
    completed: dict[str, Path],
    failures: dict[str, str],
) -> None:
    rows = []
    for case_id in PHASE5_CASES[1:]:
        root = completed.get(case_id)
        if root is None:
            rows.append(f"| {case_id} | failed/incomplete | — | — | — | — |")
            continue
        metrics = json.loads((root / "metrics.json").read_text(encoding="utf-8"))[
            "metrics"
        ]
        diagnostics = json.loads(
            (root / "diagnostics.json").read_text(encoding="utf-8")
        )["diagnostics"]
        inference = diagnostics.get("inference") or {}
        rows.append(
            "| {} | {:.8e} | {:.8e} | {:.8e} | {:.8e} | {} |".format(
                case_id,
                metrics["mean_instantaneous_error_summary"],
                metrics["maximum_instantaneous_error"],
                metrics["mean_training_error"],
                metrics["mean_extrapolation_error"],
                inference.get("iteration_count", "not applicable"),
            )
        )
    pod_text = "POD spectrum unavailable."
    pod_root = completed.get("fig1_pod_reducibility")
    if pod_root is not None:
        pod = json.loads((pod_root / "metrics.json").read_text(encoding="utf-8"))[
            "metrics"
        ]
        pod_text = (
            "The sigmoid dataset retains {:.8%} of POD energy at N_r=16 "
            "and {:.12%} at total dimension 564.".format(
                pod["retained_energy_at_N_r"],
                pod["retained_energy_at_total_dimension"],
            )
        )
    nonlinear_complete = any(
        case_id in completed
        for case_id in (
            "fig2_elementwise_projected",
            "fig2_tensorial_projected",
            "fig3_elementwise_inferred",
            "fig3_tensorial_inferred",
        )
    )
    trend_text = (
        "At least one nonlinear case completed, so ordering must be assessed from its artifact."
        if nonlinear_complete
        else (
            "No nonlinear case completed, so linear/elementwise/tensorial error ordering, "
            "nonlinear extrapolation stability, and nonlinear projected-versus-inferred "
            "trends cannot be established from this run."
        )
    )
    text = "\n".join(
        [
            "# Phase 5 Figures 1–3 comparison",
            "",
            "These results use the preserved `legacy_sigmoid` benchmark. They are not an exact",
            "reproduction of the manuscript figures, which state a zero initial angular flux.",
            "Hardware, BLAS/LAPACK, corrected derivative/time-grid behavior, the original source",
            "commit, and an original publication dataset checksum also limit direct comparison.",
            "",
            pod_text,
            "",
            "| Case | Mean M-error | Maximum M-error | Training mean | Extrapolation mean | Inference iterations |",
            "|---|---:|---:|---:|---:|---:|",
            *rows,
            "",
            trend_text,
            "The two completed linear projected/inferred histories are numerically almost identical;",
            "that is evidence only for this preserved dataset and implementation, not manuscript agreement.",
            "Interpret POD decay and completed-case training/extrapolation behavior only as qualitative",
            "evidence. Numerical differences under the",
            "preserved sigmoid benchmark require original publication provenance for attribution.",
            "",
            "Failures: " + (json.dumps(failures, sort_keys=True) if failures else "none"),
            "",
        ]
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(text, encoding="utf-8")


def main(argv=None) -> int:
    args = parse_args(argv)
    config = load_config(args.config)
    catalog = load_publication_catalog(args.catalog)
    case_table = _case_table(catalog)
    report: dict[str, Any] = {
        "schema_version": "1.0.0",
        "run_id": args.run_id,
        "creation_time_utc": datetime.now(timezone.utc).isoformat(),
        "authorized_case_ids": list(PHASE5_CASES),
        "forbidden_cases_executed": [],
        "regularization_sweeps_executed": False,
        "resolved_cases": case_table,
        "execution_authorized": bool(args.execute),
    }
    if not args.execute:
        report["action"] = "would_execute_phase5_figures_1_3"
        print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
        return 0

    snapshot = Path(args.snapshot)
    fom_run = Path(args.fom_run_directory)
    output_root = Path(args.output_root)
    if not snapshot.is_file():
        raise FileNotFoundError(f"production snapshot does not exist: {snapshot}")
    if snapshot.name != config.output.snapshot_filename:
        raise ValueError("production snapshot does not use the canonical filename")
    if not (fom_run / "manifest.json").is_file():
        raise FileNotFoundError("FOM provenance manifest is missing")
    dataset_sha256 = sha256_file(snapshot)
    report["dataset_sha256"] = dataset_sha256

    summary_path = fom_run / "metrics" / "dataset_summary.json"
    if summary_path.exists():
        dataset_summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if dataset_summary.get("snapshot", {}).get("sha256") != dataset_sha256:
            raise ValueError("existing dataset summary checksum does not match the snapshot")
    else:
        dataset_summary = create_dataset_summary(
            config,
            snapshot,
            summary_path,
            dataset_sha256=dataset_sha256,
        )
    report["dataset_summary"] = str(summary_path)

    shared_root = output_root / "shared_offline" / dataset_sha256 / args.run_id
    if shared_root.exists():
        load_shared_offline_artifacts(
            shared_root, config, dataset_sha256=dataset_sha256
        )
    else:
        build_shared_offline_artifacts(
            config,
            snapshot,
            shared_root,
            dataset_sha256=dataset_sha256,
            retained_dimension=564,
        )
    report["shared_offline_directory"] = str(shared_root)

    completed: dict[str, Path] = {}
    failures: dict[str, str] = {}
    for case_id in PHASE5_CASES:
        case = catalog.get(case_id)
        expected_root = output_root / case_id / args.run_id
        try:
            if expected_root.exists():
                manifest = validate_publication_artifact(
                    expected_root, catalog=catalog
                )
                if manifest["solver"]["success"] is not True:
                    raise ValueError("existing case artifact is not successful")
                run_root = expected_root
            else:
                run = execute_publication_case(
                    catalog,
                    case,
                    input_snapshot=snapshot,
                    execute=True,
                    output_root=output_root,
                    run_id=args.run_id,
                    shared_offline_directory=shared_root,
                )
                validate_publication_artifact(run.root, catalog=catalog)
                run_root = run.root
            completed[case_id] = run_root
        except Exception as error:
            failures[case_id] = f"{type(error).__name__}: {error}"

    bundles: dict[str, str] = {}
    plots: dict[str, list[str]] = {}
    for figure, expected_cases in FIGURE_CASES.items():
        roots = [completed[case_id] for case_id in expected_cases if case_id in completed]
        if not roots:
            continue
        number = figure.split()[-1]
        bundle_root = output_root / "figure_data" / f"figure{number}" / args.run_id
        if bundle_root.exists():
            bundle = type(
                "ExistingBundle",
                (),
                {"root": bundle_root, "metadata_path": bundle_root / "figure_data.json"},
            )()
        else:
            bundle = build_figure_data_bundle(
                figure,
                roots,
                catalog=catalog,
                output_directory=bundle_root,
            )
        bundles[figure] = str(bundle.root)
        if not args.skip_plots:
            plot_root = output_root / "plots" / f"figure{number}" / args.run_id
            if plot_root.exists():
                plot_paths = sorted(plot_root.glob("*.png"))
            else:
                plot_paths = plot_figure_data_bundle(
                    bundle.root,
                    output_directory=plot_root,
                )
            plots[figure] = [str(path) for path in plot_paths]

    phase5_root = output_root / "phase5_runs" / args.run_id
    comparison_path = phase5_root / "comparison_report.md"
    _comparison_report(comparison_path, completed, failures)
    report.update(
        {
            "action": "completed" if not failures else "partially_completed",
            "completed_case_directories": {
                case_id: str(path) for case_id, path in completed.items()
            },
            "failures": failures,
            "figure_data_bundles": bundles,
            "plots": plots,
            "comparison_report": str(comparison_path),
            "complete_publication_reproduction": False,
            "completion_time_utc": datetime.now(timezone.utc).isoformat(),
        }
    )
    report_path = phase5_root / "execution_report.json"
    _write_json(report_path, report)
    print(json.dumps({**report, "execution_report": str(report_path)}, indent=2, sort_keys=True))
    return 0 if not failures else 2


if __name__ == "__main__":
    raise SystemExit(main())
