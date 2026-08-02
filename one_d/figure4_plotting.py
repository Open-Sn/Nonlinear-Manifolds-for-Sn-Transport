"""Validate and plot regenerated-sigmoid Figure 4 bundles only."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any

import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
RANKS = (8, 16, 24, 32, 40, 48, 56, 64)
MODELS = ("linear", "elementwise", "tensorial")
OPERATORS = ("projected", "inferred")
RESULT_LABEL = "regenerated_sigmoid_benchmark"
SELECTION_PROVENANCE = "regenerated_sigmoid_search"
METRIC_ID = "relative_space_time_l2_error_v1"
TIMING_ID = "rom_solve_ivp_only_v1"
OVERALL_TITLE = "Convergence and computational efficiency of low-rank models"
PANEL_TITLES = {
    "projected": "Projected streaming operator",
    "inferred": "Inferred streaming operator",
}
SPEEDUP_TICKS = (50, 100, 200, 500, 1000, 2000)
SERIES_LABELS = {
    "linear": "Linear",
    "elementwise": "Polynomial (elementwise)",
    "tensorial": "Tensorial",
}
SERIES_STYLES = {
    "linear": {"color": "#333333", "marker": "o", "linestyle": "--"},
    "elementwise": {"color": "#0072B2", "marker": "s", "linestyle": "-"},
    "tensorial": {"color": "#D55E00", "marker": "^", "linestyle": "-"},
}


def _write_json(path: Path, content: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(content, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object in {path}")
    return value


def _sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _expected_array_names() -> set[str]:
    names = {"n_r", "n_q"}
    for operators in OPERATORS:
        for model in MODELS:
            prefix = f"{operators}__{model}"
            names.update(
                {
                    prefix + "__error",
                    prefix + "__online_speedup",
                    prefix + "__online_seconds",
                }
            )
            if model != "linear":
                names.add(prefix + "__gamma")
                if operators == "inferred":
                    names.add(prefix + "__lambda_Q")
    return names


def validate_figure4_bundle(
    bundle_directory: str | Path,
    *,
    require_complete: bool = True,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Validate the compact Figure 4 bundle without importing execution code."""
    root = Path(bundle_directory)
    metadata = _read_json(root / "figure4_data.json")
    data_path = root / "figure4_data.npz"
    if metadata.get("figure") != "Figure 4":
        raise ValueError("bundle is not Figure 4")
    if metadata.get("result_label") != RESULT_LABEL:
        raise ValueError("Figure 4 result label is invalid")
    if metadata.get("selection_provenance") != SELECTION_PROVENANCE:
        raise ValueError("Figure 4 selection provenance is invalid")
    if metadata.get("complete_publication_reproduction") is not False:
        raise ValueError("regenerated Figure 4 cannot claim complete reproduction")
    if require_complete and metadata.get("case_set_status") != "complete":
        raise ValueError("Figure 4 bundle is partial")
    if metadata.get("metric_definition", {}).get("metric_id") != METRIC_ID:
        raise ValueError("Figure 4 metric definition changed")
    if metadata.get("timing_definition", {}).get("online_timing_id") != TIMING_ID:
        raise ValueError("Figure 4 timing definition changed")
    if metadata.get("N_r_values") != list(RANKS):
        raise ValueError("Figure 4 rank values changed")
    if metadata.get("N_q_values") != [564 - rank for rank in RANKS]:
        raise ValueError("Figure 4 nonlinear dimensions changed")
    if metadata.get("data_file", {}).get("sha256") != _sha256(data_path):
        raise ValueError("Figure 4 data checksum mismatch")
    if not metadata.get("search_definition", {}).get("content_checksum_sha256"):
        raise ValueError("Figure 4 search checksum is missing")
    if not metadata.get("selected_parameters", {}).get("content_checksum_sha256"):
        raise ValueError("Figure 4 selected-parameter checksum is missing")

    with np.load(data_path, allow_pickle=False) as archive:
        if set(archive.files) != _expected_array_names():
            raise ValueError("Figure 4 bundle arrays are incomplete or unexpected")
        arrays = {name: np.asarray(archive[name]) for name in archive.files}
    if not np.array_equal(arrays["n_r"], np.asarray(RANKS)):
        raise ValueError("Figure 4 n_r array changed")
    if not np.array_equal(arrays["n_q"], np.asarray([564 - rank for rank in RANKS])):
        raise ValueError("Figure 4 n_q array changed")
    for name, array in arrays.items():
        if array.shape != (8,):
            raise ValueError(f"Figure 4 array {name} must have shape (8,)")
        if require_complete and not np.all(np.isfinite(array)):
            raise ValueError(f"complete Figure 4 array {name} is not finite")
        if name not in {"n_r", "n_q"}:
            finite = array[np.isfinite(array)]
            if finite.size and np.any(finite <= 0.0):
                raise ValueError(f"Figure 4 array {name} must be positive")
    records = metadata.get("case_records", {})
    if require_complete and len(records) != 48:
        raise ValueError("complete Figure 4 bundle requires 48 case records")
    for operators in OPERATORS:
        for model in MODELS:
            for index, rank in enumerate(RANKS):
                case_id = f"fig4_{model}_{operators}_nr{rank}"
                if case_id not in records:
                    if require_complete:
                        raise ValueError(f"Figure 4 case record is missing: {case_id}")
                    continue
                record = records[case_id]
                case = record["case"]
                metrics = record["metrics"]
                if case["N_r"] != rank or case["operators"] != operators:
                    raise ValueError(f"Figure 4 case dimensions changed: {case_id}")
                if model == "linear":
                    if case["N_q"] is not None:
                        raise ValueError("linear Figure 4 N_q must be null")
                elif case["N_q"] != 564 - rank:
                    raise ValueError("nonlinear Figure 4 dimensions do not sum to 564")
                prefix = f"{operators}__{model}"
                if arrays[prefix + "__error"][index] != metrics[
                    "relative_space_time_l2_error"
                ]:
                    raise ValueError(f"Figure 4 error array disagrees with {case_id}")
                if arrays[prefix + "__online_speedup"][index] != metrics["online_speedup"]:
                    raise ValueError(f"Figure 4 speed-up array disagrees with {case_id}")
    return metadata, arrays


def _git_metadata() -> dict[str, Any]:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPOSITORY_ROOT,
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=REPOSITORY_ROOT,
                check=True,
                capture_output=True,
                text=True,
                timeout=10,
            ).stdout.strip()
        )
        return {"commit": commit, "dirty": dirty}
    except (OSError, subprocess.SubprocessError):
        return {"commit": None, "dirty": None}


def figure4_plot_plan(
    bundle_directory: str | Path,
    *,
    output_directory: str | Path,
) -> dict[str, Any]:
    metadata, _ = validate_figure4_bundle(bundle_directory)
    output = Path(output_directory)
    return {
        "action": "would_plot_manuscript_figure4",
        "source_bundle": str(bundle_directory),
        "source_data_checksum_sha256": metadata["data_file"]["sha256"],
        "case_set_status": metadata["case_set_status"],
        "output_directory": str(output),
        "output_directory_exists": output.exists(),
        "outputs": [
            str(output / "figure4_manuscript.png"),
            str(output / "figure4_manuscript.pdf"),
            str(output / "plot_metadata.json"),
            str(output / "figure4_caption.md"),
        ],
        "launches_scientific_execution": False,
        "writes_files": False,
    }


def _caption(metadata: dict[str, Any], render_run_id: str) -> str:
    return "\n".join(
        [
            "# Figure 4 caption",
            "",
            "Relative space-time error (top) and machine-specific online speed-up",
            "(bottom) versus latent dimension for projected (left) and inferred",
            "(right) linear, polynomial (elementwise), and tensorial reduced models.",
            "Errors use the author-approved full-$[0,10]$ trapezoidal relative",
            "space-time $M$-norm. Speed-ups divide validated FOM integration time",
            "by wall-clock time inside the reduced `solve_ivp` call only.",
            "",
            "This is the author-confirmed regenerated sigmoid benchmark. Nonlinear",
            "regularization",
            "parameters were selected by the Phase 8 regenerated search and are not",
            "recovered historical Figure 4 values.",
            "This presentation-only rerender uses the unchanged validated bundle; no",
            "curve values, metrics, timing values, or selected parameters were changed.",
            "",
            f"Bundle run: `{metadata['run_id']}`.",
            f"Presentation render run: `{render_run_id}`.",
            "",
        ]
    )


def plot_figure4_bundle(
    bundle_directory: str | Path,
    *,
    output_directory: str | Path,
) -> list[Path]:
    """Generate the 2-by-2 manuscript composite from a bundle only."""
    metadata, arrays = validate_figure4_bundle(bundle_directory)
    output = Path(output_directory)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite Figure 4 plot output: {output}")
    output.mkdir(parents=True, exist_ok=False)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 11,
            "legend.fontsize": 9,
            "pdf.fonttype": 42,
        }
    )
    n_r = arrays["n_r"]
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 9.2), constrained_layout=True)
    plot_limits: dict[str, Any] = {}
    for column, operators in enumerate(OPERATORS):
        error_axis = axes[0, column]
        speed_axis = axes[1, column]
        for model in MODELS:
            style = SERIES_STYLES[model]
            prefix = f"{operators}__{model}"
            error_axis.plot(
                n_r,
                arrays[prefix + "__error"],
                label=SERIES_LABELS[model],
                linewidth=1.5,
                markersize=5,
                **style,
            )
            speed_axis.plot(
                n_r,
                arrays[prefix + "__online_speedup"],
                label=SERIES_LABELS[model],
                linewidth=1.5,
                markersize=5,
                **style,
            )
        error_axis.set_title(PANEL_TITLES[operators])
        error_axis.set_yscale("log")
        error_axis.set_xticks(n_r)
        error_axis.set_xlabel(r"Latent dimension $N_r$")
        error_axis.set_ylabel(r"Relative space-time error $E_{rel}$")
        error_axis.grid(True, which="both", alpha=0.25, linewidth=0.5)
        error_axis.legend(frameon=False)

        speed_axis.set_xticks(n_r)
        speed_axis.set_yscale("log")
        speed_axis.set_yticks(
            SPEEDUP_TICKS,
            labels=[str(value) for value in SPEEDUP_TICKS],
        )
        speed_axis.set_xlabel(r"Latent dimension $N_r$")
        speed_axis.set_ylabel("Online speed-up")
        speed_axis.grid(True, which="both", alpha=0.25, linewidth=0.5)
        speed_axis.legend(frameon=False)
        plot_limits[operators] = {
            "error_y_limits": list(error_axis.get_ylim()),
            "speedup_y_limits": list(speed_axis.get_ylim()),
            "n_r_limits": list(error_axis.get_xlim()),
        }
    fig.suptitle(OVERALL_TITLE, fontsize=15)
    png = output / "figure4_manuscript.png"
    pdf = output / "figure4_manuscript.pdf"
    fig.savefig(png, dpi=300, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)

    caption = output / "figure4_caption.md"
    caption.write_text(_caption(metadata, output.name), encoding="utf-8")
    plot_metadata = {
        "schema_version": "1.0.0",
        "figure": "Figure 4",
        "layout": "manuscript_2_by_2",
        "render_run_id": output.name,
        "source_bundle": {
            "path": str(bundle_directory),
            "metadata_sha256": _sha256(Path(bundle_directory) / "figure4_data.json"),
            "data_sha256": metadata["data_file"]["sha256"],
        },
        "case_set_status": metadata["case_set_status"],
        "complete_publication_reproduction": False,
        "result_label": RESULT_LABEL,
        "benchmark_presentation_label": "regenerated sigmoid benchmark",
        "selection_provenance": SELECTION_PROVENANCE,
        "selected_parameter_checksum_sha256": metadata["selected_parameters"][
            "content_checksum_sha256"
        ],
        "benchmark_variant": metadata["benchmark_variant"],
        "initial_condition_provenance": metadata["initial_condition_provenance"],
        "metric_definition": metadata["metric_definition"],
        "timing_definition": metadata["timing_definition"],
        "series_labels": SERIES_LABELS,
        "N_r_values": list(RANKS),
        "titles": {
            "overall": OVERALL_TITLE,
            "panels": PANEL_TITLES,
        },
        "axis_scales": {
            "N_r": "linear",
            "error": "logarithmic",
            "speedup": "logarithmic",
        },
        "axis_ticks": {"speedup": list(SPEEDUP_TICKS)},
        "plot_limits": plot_limits,
        "rendering_provenance": {
            "classification": "presentation_only_rerender",
            "source_bundle_validated_and_unchanged": True,
            "scientific_values_changed": False,
            "description": (
                "Presentation-only rerender from the unchanged validated Figure 4 bundle."
            ),
        },
        "plotting_source": _git_metadata(),
        "scientific_execution": {
            "solver_run": False,
            "rom_run": False,
            "inference_run": False,
            "parameter_search_run": False,
            "fom_run": False,
            "derivatives_recomputed": False,
            "pod_svd_recomputed": False,
            "bundle_reconstructed": False,
            "metrics_recomputed": False,
            "figure5_rerun": False,
        },
        "outputs": {
            "png": {"path": str(png), "sha256": _sha256(png)},
            "pdf": {"path": str(pdf), "sha256": _sha256(pdf)},
            "caption": {"path": str(caption), "sha256": _sha256(caption)},
        },
    }
    metadata_path = output / "plot_metadata.json"
    _write_json(metadata_path, plot_metadata)
    return [png, pdf, metadata_path, caption]
