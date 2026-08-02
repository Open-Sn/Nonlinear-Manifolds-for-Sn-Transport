"""Bundle-only manuscript plotting for sigmoid-benchmark Figure 5."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any

import numpy as np

from .figure5 import NQ_VALUES, SERIES, validate_figure5_bundle


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
OVERALL_TITLE = "Effect of lifting dimension on accuracy and online efficiency"
PANEL_TITLES = {
    "projected": "Projected streaming operator",
    "inferred": "Inferred streaming operator",
}
SPEEDUP_TICKS = (50, 100, 200, 300, 500)
SERIES_LABELS = {
    "fixed_linear": r"Linear, $N_r=32$",
    "elementwise": r"Polynomial (elementwise), $N_r=32$",
    "tensorial": r"Tensorial, $N_r=32$",
    "enlarged_linear": r"Enlarged linear, $32+N_q$",
    "best_projection": r"Best $M$-projection, $32+N_q$",
}
SERIES_STYLES = {
    "fixed_linear": {"color": "#333333", "marker": "o", "linestyle": "--"},
    "elementwise": {"color": "#0072B2", "marker": "s", "linestyle": "-"},
    "tensorial": {"color": "#D55E00", "marker": "^", "linestyle": "-"},
    "enlarged_linear": {"color": "#009E73", "marker": "D", "linestyle": "-"},
    "best_projection": {"color": "#CC79A7", "marker": "v", "linestyle": ":"},
}


def _write_json(path: Path, content: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(content, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


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


def figure5_plot_plan(
    bundle_directory: str | Path,
    *,
    output_directory: str | Path,
) -> dict[str, Any]:
    metadata, _arrays = validate_figure5_bundle(bundle_directory)
    output = Path(output_directory)
    return {
        "action": "would_plot_manuscript_figure5",
        "source_bundle": str(bundle_directory),
        "source_data_checksum_sha256": metadata["data_file"]["sha256"],
        "case_set_status": metadata["case_set_status"],
        "benchmark_variant": metadata["benchmark_variant"],
        "output_directory": str(output),
        "output_directory_exists": output.exists(),
        "outputs": [
            str(output / "figure5_manuscript.png"),
            str(output / "figure5_manuscript.pdf"),
            str(output / "plot_metadata.json"),
            str(output / "figure5_caption.md"),
        ],
        "launches_scientific_execution": False,
        "writes_files": False,
    }


def _caption(metadata: dict[str, Any], render_run_id: str) -> str:
    return "\n".join(
        [
            "# Figure 5 caption",
            "",
            "Relative space-time error (top) and machine-specific online speed-up",
            "(bottom) versus lifting dimension for projected (left) and inferred",
            "(right) reduced models for the regenerated sigmoid benchmark. All",
            "nonlinear models use",
            "$N_r=32$; Polynomial denotes the elementwise quadratic lifting, while",
            "the enlarged linear and best $M$-projection dimensions are $32+N_q$.",
            "The projection benchmark has no speed-up because it is not an integrated",
            "dynamical model. Errors use the author-approved full-$[0,10]$ trapezoidal",
            "relative space-time $M$-norm definition. Speed-ups divide the validated",
            "production FOM integration time by wall-clock time inside the reduced",
            "`solve_ivp` call only.",
            "",
            "The manuscript text states zero initial angular flux; the authors confirm",
            "that the one-dimensional figure calculations used the localized sigmoid",
            "initialization preserved in `configs/1d/legacy_production.json`.",
            "This presentation-only rerender uses the unchanged validated bundle; no",
            "curve values, metrics, or timing values were changed.",
            "",
            f"Bundle run: `{metadata['run_id']}`.",
            f"Presentation render run: `{render_run_id}`.",
            "",
        ]
    )


def plot_figure5_bundle(
    bundle_directory: str | Path,
    *,
    output_directory: str | Path,
) -> list[Path]:
    """Generate the 2-by-2 manuscript composite without importing solvers."""
    metadata, arrays = validate_figure5_bundle(bundle_directory)
    output = Path(output_directory)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite Figure 5 plot output: {output}")
    output.mkdir(parents=True, exist_ok=False)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 11,
            "legend.fontsize": 8,
            "pdf.fonttype": 42,
        }
    )
    n_q = arrays["n_q"]
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 9.2), constrained_layout=True)
    plot_limits: dict[str, Any] = {}
    for column, operators in enumerate(("projected", "inferred")):
        error_axis = axes[0, column]
        speed_axis = axes[1, column]
        for series in SERIES:
            style = SERIES_STYLES[series]
            error_axis.plot(
                n_q,
                arrays[f"{operators}__{series}__error"],
                label=SERIES_LABELS[series],
                linewidth=1.5,
                markersize=5,
                **style,
            )
            if series != "best_projection":
                speed_axis.plot(
                    n_q,
                    arrays[f"{operators}__{series}__online_speedup"],
                    label=SERIES_LABELS[series],
                    linewidth=1.5,
                    markersize=5,
                    **style,
                )
        error_axis.set_title(PANEL_TITLES[operators])
        error_axis.set_xscale("log", base=2)
        error_axis.set_yscale("log")
        error_axis.set_xticks(n_q, labels=[str(value) for value in n_q])
        error_axis.set_xlabel(r"Lifting dimension $N_q$")
        error_axis.set_ylabel(r"Relative space-time error $E_{rel}$")
        error_axis.grid(True, which="both", alpha=0.25, linewidth=0.5)
        error_axis.legend(frameon=False)

        speed_axis.set_xscale("log", base=2)
        speed_axis.set_yscale("log")
        speed_axis.set_yticks(
            SPEEDUP_TICKS,
            labels=[str(value) for value in SPEEDUP_TICKS],
        )
        speed_axis.set_xticks(n_q, labels=[str(value) for value in n_q])
        speed_axis.set_xlabel(r"Lifting dimension $N_q$")
        speed_axis.set_ylabel("Online speed-up")
        speed_axis.grid(True, which="both", alpha=0.25, linewidth=0.5)
        speed_axis.legend(frameon=False)
        plot_limits[operators] = {
            "error_y_limits": list(error_axis.get_ylim()),
            "speedup_y_limits": list(speed_axis.get_ylim()),
            "n_q_limits": list(error_axis.get_xlim()),
        }
    fig.suptitle(OVERALL_TITLE, fontsize=15)
    png = output / "figure5_manuscript.png"
    pdf = output / "figure5_manuscript.pdf"
    fig.savefig(png, dpi=300, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)

    caption = output / "figure5_caption.md"
    caption.write_text(_caption(metadata, output.name), encoding="utf-8")
    plot_metadata = {
        "schema_version": "1.0.0",
        "figure": "Figure 5",
        "layout": "manuscript_2_by_2",
        "render_run_id": output.name,
        "source_bundle": {
            "path": str(bundle_directory),
            "metadata_sha256": _sha256(Path(bundle_directory) / "figure5_data.json"),
            "data_sha256": metadata["data_file"]["sha256"],
        },
        "case_set_status": metadata["case_set_status"],
        "complete_publication_reproduction": False,
        "benchmark_variant": metadata["benchmark_variant"],
        "benchmark_presentation_label": "regenerated sigmoid benchmark",
        "initial_condition_provenance": metadata["initial_condition_provenance"],
        "metric_definition": metadata["metric_definition"],
        "timing_definition": metadata["timing_definition"],
        "series_labels": SERIES_LABELS,
        "N_q_values": list(NQ_VALUES),
        "projection_benchmark_in_speedup_panels": False,
        "titles": {
            "overall": OVERALL_TITLE,
            "panels": PANEL_TITLES,
        },
        "axis_scales": {
            "N_q": "logarithmic_base_2",
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
                "Presentation-only rerender from the unchanged validated Figure 5 bundle."
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
