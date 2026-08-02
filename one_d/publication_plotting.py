"""Bundle-only manuscript plotting for validated one-dimensional results."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any

import numpy as np

from .config import load_config
from .publication_experiments import (
    AUTHOR_CONFIRMED_SIGMOID_PROVENANCE,
    BENCHMARK_VARIANT,
    EXPECTED_DEVIATION,
    LEGACY_CONFIG_CHECKSUM,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
LEGACY_CONFIG_PATH = REPOSITORY_ROOT / "configs" / "1d" / "legacy_production.json"
FIELD_TIME = 2.5
TRAINING_BOUNDARY = 7.5
FIGURE_EXPECTED_CASES = {
    "Figure 1": {"fig1_pod_reducibility"},
    "Figure 2": {
        "fig2_linear_projected",
        "fig2_elementwise_projected",
        "fig2_tensorial_projected",
    },
    "Figure 3": {
        "fig3_linear_inferred",
        "fig3_elementwise_inferred",
        "fig3_tensorial_inferred",
    },
}
MODEL_LABELS = {
    "linear": "Linear",
    "elementwise": "Polynomial (elementwise)",
    "tensorial": "Tensorial",
}
MODEL_COLORS = {
    "linear": "#333333",
    "elementwise": "#1f77b4",
    "tensorial": "#d62728",
}
DIRECTION_COLORS = ("#0072B2", "#E69F00", "#009E73", "#CC79A7")


@dataclass(frozen=True)
class PhaseSpacePlotData:
    spatial_coordinates: np.ndarray
    ordinates: np.ndarray
    direction_blocks: np.ndarray


@dataclass(frozen=True)
class ValidatedFigureBundle:
    figure: str
    root: Path
    metadata: dict[str, Any]
    arrays: dict[str, np.ndarray]
    bundle_checksum_sha256: str
    file_checksums_sha256: dict[str, str]
    source_manifests: tuple[dict[str, Any], ...]


def sha256_file(path: str | Path) -> str:
    hasher = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def bundle_checksum(root: str | Path) -> tuple[str, dict[str, str]]:
    """Hash the two immutable bundle files with names and length framing."""
    root = Path(root)
    files = ("figure_data.json", "figure_data.npz")
    checksums = {name: sha256_file(root / name) for name in files}
    hasher = hashlib.sha256()
    for name in files:
        encoded = name.encode("utf-8")
        digest = bytes.fromhex(checksums[name])
        hasher.update(len(encoded).to_bytes(8, "little"))
        hasher.update(encoded)
        hasher.update(len(digest).to_bytes(8, "little"))
        hasher.update(digest)
    return hasher.hexdigest(), checksums


def production_spatial_dof_coordinates(
    config_path: str | Path = LEGACY_CONFIG_PATH,
) -> np.ndarray:
    """Return left/right endpoint coordinates in exact cell-major DG order."""
    config = load_config(config_path)
    interfaces: list[float] = []
    left = 0.0
    for width, count in zip(
        config.problem.region_widths,
        config.problem.cells_per_region,
    ):
        local = np.linspace(left, left + width, int(count) + 1)
        if interfaces:
            interfaces.extend(local[1:].tolist())
        else:
            interfaces.extend(local.tolist())
        left += width
    boundaries = np.asarray(interfaces, dtype=float)
    coordinates = np.empty(2 * (boundaries.size - 1), dtype=float)
    coordinates[0::2] = boundaries[:-1]
    coordinates[1::2] = boundaries[1:]
    return coordinates


def production_ordinates(
    config_path: str | Path = LEGACY_CONFIG_PATH,
) -> np.ndarray:
    """Return ascending Gauss--Legendre ordinates without visual reordering."""
    config = load_config(config_path)
    ordinates, _weights = np.polynomial.legendre.leggauss(
        config.problem.angular_ordinates
    )
    return ordinates


def split_direction_major_state(
    state: np.ndarray,
    *,
    spatial_coordinates: np.ndarray | None = None,
    ordinates: np.ndarray | None = None,
) -> PhaseSpacePlotData:
    """Split a flat direction-major state while preserving DG endpoint values."""
    coordinates = np.asarray(
        production_spatial_dof_coordinates()
        if spatial_coordinates is None
        else spatial_coordinates,
        dtype=float,
    )
    directions = np.asarray(
        production_ordinates() if ordinates is None else ordinates,
        dtype=float,
    )
    values = np.asarray(state, dtype=float)
    if values.ndim != 1:
        raise ValueError("phase-space plotting state must be one-dimensional")
    expected = directions.size * coordinates.size
    if values.size != expected:
        raise ValueError(
            f"phase-space plotting state must have {expected} entries; "
            f"received {values.size}"
        )
    if not np.all(np.isfinite(values)):
        raise ValueError("phase-space plotting state must be finite")
    return PhaseSpacePlotData(
        spatial_coordinates=coordinates,
        ordinates=directions,
        direction_blocks=values.reshape(directions.size, coordinates.size),
    )


def _read_json(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"expected a JSON object in {path}")
    return data


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(data, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _resolve_repository_path(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else REPOSITORY_ROOT / candidate


def _source_manifests(metadata: dict[str, Any], figure: str) -> tuple[dict[str, Any], ...]:
    sources = metadata.get("sources")
    if not isinstance(sources, list):
        raise ValueError("figure bundle requires source artifact metadata")
    expected = FIGURE_EXPECTED_CASES[figure]
    if {source.get("case_id") for source in sources} != expected:
        raise ValueError("figure bundle source cases are incomplete")
    collected: list[dict[str, Any]] = []
    for source in sources:
        root = _resolve_repository_path(source.get("artifact_root", ""))
        manifest_path = root / "manifest.json"
        if not manifest_path.is_file():
            raise ValueError(f"missing source case manifest: {manifest_path}")
        manifest = _read_json(manifest_path)
        required = {
            "case_id": source["case_id"],
            "figure": figure,
            "benchmark_variant": BENCHMARK_VARIANT,
            "experiment_catalog_checksum": metadata["catalog_checksum"],
        }
        for key, value in required.items():
            if manifest.get(key) != value:
                raise ValueError(f"source case manifest has invalid {key}")
        if manifest.get("base_configuration_checksum") != LEGACY_CONFIG_CHECKSUM:
            raise ValueError("source case manifest has invalid base configuration checksum")
        if manifest.get("manuscript_deviation") != EXPECTED_DEVIATION:
            raise ValueError("source case manifest has invalid manuscript deviation")
        snapshot = manifest.get("input_snapshot", {})
        if snapshot.get("checksum_sha256") != source.get("input_snapshot_checksum"):
            raise ValueError("source case snapshot checksum does not match bundle metadata")
        collected.append(
            {
                "case_id": source["case_id"],
                "artifact_root": str(source["artifact_root"]),
                "manifest_path": str(manifest_path),
                "manifest_sha256": sha256_file(manifest_path),
                "resolved_configuration_checksum_sha256": manifest.get(
                    "resolved_configuration_checksum_sha256"
                ),
                "input_snapshot_checksum_sha256": snapshot.get("checksum_sha256"),
                "model": manifest.get("model"),
            }
        )
    return tuple(collected)


def _validate_array_metadata(
    metadata: dict[str, Any], arrays: dict[str, np.ndarray]
) -> None:
    records = metadata.get("array_metadata")
    if not isinstance(records, list):
        raise ValueError("figure bundle lacks array metadata")
    indexed: dict[str, tuple[tuple[int, ...], str]] = {}
    for record in records:
        if set(record) != {"array_name", "artifact_path", "dtype", "shape"}:
            raise ValueError("figure bundle array metadata has invalid fields")
        if record["artifact_path"] != "figure_data.npz":
            raise ValueError("figure bundle arrays must reside in figure_data.npz")
        indexed[record["array_name"]] = (
            tuple(record["shape"]),
            record["dtype"],
        )
    if set(indexed) != set(arrays):
        raise ValueError("figure bundle arrays do not match array metadata")
    for name, array in arrays.items():
        shape, dtype = indexed[name]
        if array.shape != shape or str(array.dtype) != dtype:
            raise ValueError(f"figure bundle metadata mismatch for {name}")
        if array.dtype == object or not np.all(np.isfinite(array)):
            raise ValueError(f"figure bundle array {name} is nonnumeric or non-finite")


def _validate_figure1(arrays: dict[str, np.ndarray]) -> None:
    prefix = "fig1_pod_reducibility__"
    expected = {
        prefix + name
        for name in (
            "basis_dimensions",
            "highlighted_latent_dimension",
            "highlighted_lifting_dimension",
            "highlighted_total_dimension",
            "pod_eigenvalues",
            "retained_energy_fraction",
            "unresolved_energy_fraction",
        )
    }
    if set(arrays) != expected:
        raise ValueError("Figure 1 bundle lacks the complete POD-energy arrays")
    dimensions = arrays[prefix + "basis_dimensions"]
    unresolved = arrays[prefix + "unresolved_energy_fraction"]
    retained = arrays[prefix + "retained_energy_fraction"]
    if dimensions.ndim != 1 or dimensions.size < 564:
        raise ValueError("Figure 1 dimensions must cover at least dimension 564")
    if not np.array_equal(dimensions, np.arange(1, dimensions.size + 1)):
        raise ValueError("Figure 1 basis dimensions are not consecutive")
    if unresolved.shape != dimensions.shape or retained.shape != dimensions.shape:
        raise ValueError("Figure 1 POD-energy arrays have incompatible shapes")
    if np.any(unresolved <= 0.0) or np.any(retained < 0.0):
        raise ValueError("Figure 1 POD-energy fractions are outside plotting range")
    highlights = {
        "latent": int(arrays[prefix + "highlighted_latent_dimension"][0]),
        "lifting": int(arrays[prefix + "highlighted_lifting_dimension"][0]),
        "total": int(arrays[prefix + "highlighted_total_dimension"][0]),
    }
    if highlights != {"latent": 16, "lifting": 548, "total": 564}:
        raise ValueError("Figure 1 highlighted dimensions are incompatible")


def _validate_fields_and_errors(
    figure: str, arrays: dict[str, np.ndarray], metadata: dict[str, Any]
) -> None:
    expected_cases = FIGURE_EXPECTED_CASES[figure]
    suffixes = {
        "discrepancy",
        "dof_index",
        "field_time",
        "field_time_index",
        "fom_angular_flux",
        "instantaneous_normalized_mass_error",
        "rom_angular_flux",
        "time",
        "training_end_time",
    }
    expected_arrays = {
        case_id + "__" + suffix
        for case_id in expected_cases
        for suffix in suffixes
    }
    if set(arrays) != expected_arrays or len(arrays) != 27:
        raise ValueError(f"{figure} bundle must contain all 27 expected arrays")
    if set(metadata.get("series_membership", [])) != {
        "linear",
        "elementwise",
        "tensorial",
    }:
        raise ValueError(f"{figure} bundle lacks a required model series")
    coordinates = production_spatial_dof_coordinates()
    reference: np.ndarray | None = None
    for case_id in sorted(expected_cases):
        prefix = case_id + "__"
        dof_index = arrays[prefix + "dof_index"]
        fom = arrays[prefix + "fom_angular_flux"]
        rom = arrays[prefix + "rom_angular_flux"]
        discrepancy = arrays[prefix + "discrepancy"]
        time = arrays[prefix + "time"]
        errors = arrays[prefix + "instantaneous_normalized_mass_error"]
        if not np.array_equal(dof_index, np.arange(6000)):
            raise ValueError(f"{case_id} does not use exact direction-major DOF indices")
        if any(value.shape != (6000,) for value in (fom, rom, discrepancy)):
            raise ValueError(f"{case_id} field arrays must contain 6,000 entries")
        split_direction_major_state(fom, spatial_coordinates=coordinates)
        if not np.array_equal(discrepancy, fom - rom):
            raise ValueError(f"{case_id} discrepancy is not reference minus reconstruction")
        if reference is None:
            reference = fom
        elif not np.array_equal(reference, fom):
            raise ValueError(f"{figure} cases do not share an identical FOM reference field")
        if float(arrays[prefix + "field_time"][0]) != FIELD_TIME:
            raise ValueError(f"{case_id} field time must be exactly 2.5")
        if int(arrays[prefix + "field_time_index"][0]) != 2500:
            raise ValueError(f"{case_id} field time index must be exactly 2500")
        if float(arrays[prefix + "training_end_time"][0]) != TRAINING_BOUNDARY:
            raise ValueError(f"{case_id} training boundary must be exactly 7.5")
        if (
            time.shape != (10001,)
            or errors.shape != time.shape
            or time[0] != 0.0
            or time[-1] != 10.0
            or np.any(np.diff(time) <= 0.0)
            or np.any(errors <= 0.0)
        ):
            raise ValueError(f"{case_id} error history is incompatible")


def validate_manuscript_figure_bundle(
    bundle_directory: str | Path,
    *,
    expected_figure: str | None = None,
) -> ValidatedFigureBundle:
    """Validate a complete immutable Figure 1--3 data bundle without solving."""
    root = Path(bundle_directory)
    metadata_path = root / "figure_data.json"
    data_path = root / "figure_data.npz"
    if not metadata_path.is_file() or not data_path.is_file():
        raise ValueError("manuscript plotting requires figure_data.json and figure_data.npz")
    metadata = _read_json(metadata_path)
    figure = metadata.get("figure")
    if figure not in FIGURE_EXPECTED_CASES:
        raise ValueError("manuscript layout supports only Figure 1, Figure 2, or Figure 3")
    if expected_figure is not None and figure != expected_figure:
        raise ValueError(f"source bundle is {figure}, not requested {expected_figure}")
    if metadata.get("benchmark_variant") != BENCHMARK_VARIANT:
        raise ValueError("source bundle is not the legacy_sigmoid benchmark")
    if metadata.get("manuscript_deviation") != EXPECTED_DEVIATION:
        raise ValueError("source bundle lacks transparent manuscript discrepancy metadata")
    expected_cases = FIGURE_EXPECTED_CASES[figure]
    if (
        not metadata.get("case_set_complete")
        or metadata.get("status") != "complete_input_set"
        or set(metadata.get("case_ids", [])) != expected_cases
        or set(metadata.get("expected_case_ids", [])) != expected_cases
    ):
        raise ValueError(f"{figure} source bundle is incomplete")
    with np.load(data_path, allow_pickle=False) as archive:
        arrays = {name: np.asarray(archive[name]) for name in archive.files}
    _validate_array_metadata(metadata, arrays)
    if figure == "Figure 1":
        _validate_figure1(arrays)
    else:
        _validate_fields_and_errors(figure, arrays, metadata)
    sources = _source_manifests(metadata, figure)
    checksum, checksums = bundle_checksum(root)
    return ValidatedFigureBundle(
        figure=figure,
        root=root,
        metadata=metadata,
        arrays=arrays,
        bundle_checksum_sha256=checksum,
        file_checksums_sha256=checksums,
        source_manifests=sources,
    )


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
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=REPOSITORY_ROOT,
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        ).stdout
        return {"commit": commit, "dirty": bool(status.strip())}
    except (OSError, subprocess.SubprocessError):
        return {"commit": None, "dirty": None}


def manuscript_plot_plan(
    validated: ValidatedFigureBundle,
    *,
    output_directory: str | Path,
) -> dict[str, Any]:
    """Describe a validated manuscript plot without writing anything."""
    output = Path(output_directory)
    figure_number = validated.figure.split()[-1]
    return {
        "action": "would_plot_manuscript_layout",
        "figure": validated.figure,
        "source_bundle": str(validated.root),
        "source_bundle_checksum_sha256": validated.bundle_checksum_sha256,
        "benchmark_variant": BENCHMARK_VARIANT,
        "provenance_status": AUTHOR_CONFIRMED_SIGMOID_PROVENANCE[
            "provenance_status"
        ],
        "complete_case_set": True,
        "field_time": FIELD_TIME if validated.figure != "Figure 1" else None,
        "training_boundary": (
            TRAINING_BOUNDARY if validated.figure != "Figure 1" else None
        ),
        "output_directory": str(output),
        "outputs": [
            str(output / f"figure{figure_number}_manuscript.png"),
            str(output / f"figure{figure_number}_manuscript.pdf"),
            str(output / "plot_metadata.json"),
        ],
        "output_directory_exists": output.exists(),
        "launches_scientific_execution": False,
        "writes_files": False,
    }


def _plot_direction_blocks(axis: Any, state: np.ndarray) -> None:
    layout = split_direction_major_state(state)
    for index, (ordinate, values) in enumerate(
        zip(layout.ordinates, layout.direction_blocks)
    ):
        axis.plot(
            layout.spatial_coordinates,
            values,
            color=DIRECTION_COLORS[index],
            linewidth=1.15,
            label=rf"$\mu_{{{index + 1}}}={ordinate:.6f}$",
        )
    axis.set_xlim(0.0, 3.0)
    axis.grid(True, alpha=0.2, linewidth=0.5)


def _base_metadata(validated: ValidatedFigureBundle) -> dict[str, Any]:
    datasets = sorted(
        {
            source["input_snapshot_checksum_sha256"]
            for source in validated.source_manifests
        }
    )
    configurations = {
        source["case_id"]: source["resolved_configuration_checksum_sha256"]
        for source in validated.source_manifests
    }
    return {
        "schema_version": "1.0.0",
        "figure": validated.figure,
        "layout": "manuscript",
        "benchmark_variant": BENCHMARK_VARIANT,
        "manuscript_deviation": EXPECTED_DEVIATION,
        "initial_condition_provenance": AUTHOR_CONFIRMED_SIGMOID_PROVENANCE,
        "complete_case_set": True,
        "exact_pixel_reproduction": False,
        "scientific_execution": {
            "fom_run": False,
            "rom_run": False,
            "derivatives_recomputed": False,
            "pod_svd_recomputed": False,
            "parameter_search_run": False,
        },
        "source_bundle": {
            "path": str(validated.root),
            "checksum_algorithm": "sha256-framed-figure-bundle-files-v1",
            "checksum_sha256": validated.bundle_checksum_sha256,
            "file_checksums_sha256": validated.file_checksums_sha256,
        },
        "source_case_manifests": list(validated.source_manifests),
        "dataset_checksums_sha256": datasets,
        "configuration_checksums_sha256": configurations,
        "catalog_checksum": validated.metadata["catalog_checksum"],
        "case_ids": sorted(validated.metadata["case_ids"]),
        "plotting_source": _git_metadata(),
    }


def _plot_figure1_manuscript(
    validated: ValidatedFigureBundle, output: Path, plt: Any
) -> tuple[Path, Path, dict[str, Any]]:
    prefix = "fig1_pod_reducibility__"
    dimensions = validated.arrays[prefix + "basis_dimensions"]
    unresolved = validated.arrays[prefix + "unresolved_energy_fraction"]
    latent = int(validated.arrays[prefix + "highlighted_latent_dimension"][0])
    lifting = int(validated.arrays[prefix + "highlighted_lifting_dimension"][0])
    total = int(validated.arrays[prefix + "highlighted_total_dimension"][0])
    value_latent = float(unresolved[latent - 1])
    value_total = float(unresolved[total - 1])
    x_limits = [0.0, 700.0]
    y_limits = [1.0e-16, 1.1]
    mask = dimensions <= x_limits[1]

    fig, axis = plt.subplots(figsize=(8.4, 5.2), constrained_layout=True)
    axis.axvspan(0.0, latent, color="#56B4E9", alpha=0.22, label=r"Latent modes $N_r=16$")
    axis.axvspan(
        latent,
        total,
        color="#E69F00",
        alpha=0.18,
        label=r"Lifting modes $N_q=548$",
    )
    axis.semilogy(
        dimensions[mask],
        unresolved[mask],
        color="black",
        linewidth=1.6,
        label="Unresolved POD energy",
    )
    axis.axvline(latent, color="#0072B2", linestyle="--", linewidth=1.2)
    axis.axvline(total, color="#D55E00", linestyle="--", linewidth=1.2)
    axis.scatter(
        [latent, total],
        [value_latent, value_total],
        color=["#0072B2", "#D55E00"],
        zorder=4,
    )
    axis.annotate(
        rf"$E_{{unres}}(16)={value_latent:.3e}$",
        xy=(latent, value_latent),
        xytext=(55, 0.025),
        arrowprops={"arrowstyle": "->", "color": "#0072B2"},
    )
    axis.annotate(
        rf"$E_{{unres}}(564)={value_total:.3e}$",
        xy=(total, value_total),
        xytext=(360, 3.0e-14),
        arrowprops={"arrowstyle": "->", "color": "#D55E00"},
    )
    axis.set(
        xlim=x_limits,
        ylim=y_limits,
        xlabel="Retained / reconstruction dimension",
        ylabel="Relative unresolved POD energy",
    )
    axis.grid(True, which="both", alpha=0.22, linewidth=0.5)
    axis.legend(loc="upper right", frameon=False)

    png = output / "figure1_manuscript.png"
    pdf = output / "figure1_manuscript.pdf"
    fig.savefig(png, dpi=300, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    details = {
        "displayed_x_limits": x_limits,
        "displayed_y_limits": y_limits,
        "y_scale": "logarithmic",
        "highlighted_dimensions": {
            "N_r": latent,
            "N_q": lifting,
            "N_r_plus_N_q": total,
        },
        "relative_unresolved_energy": {
            "dimension_16": value_latent,
            "dimension_564": value_total,
        },
        "displayed_curve_through_dimension": int(dimensions[mask][-1]),
    }
    return png, pdf, details


def _model_from_case(case_id: str) -> str:
    for model in MODEL_LABELS:
        if f"_{model}_" in case_id:
            return model
    raise ValueError(f"cannot classify model for {case_id}")


def _inference_metadata(validated: ValidatedFigureBundle) -> dict[str, Any]:
    if validated.figure != "Figure 3":
        return {}
    records: dict[str, Any] = {}
    for source in validated.source_manifests:
        diagnostics_path = _resolve_repository_path(source["artifact_root"]) / "diagnostics.json"
        diagnostics = _read_json(diagnostics_path).get("diagnostics", {})
        inference = diagnostics.get("inference")
        coefficients = source["model"]["regularization"]
        applied = diagnostics.get("regularization")
        if applied is None:
            training_count = int(diagnostics.get("training_snapshot_count", 7501))
            applied = {
                "coefficient_scale": float(training_count),
                "lifting_gamma_coefficient": coefficients["gamma"],
                "lifting_gram_ridge_actual": (
                    None
                    if coefficients["gamma"] is None
                    else float(coefficients["gamma"] * training_count)
                ),
                "linear_lambda_coefficient": coefficients["lambda_L"],
                "linear_inference_gram_ridge_actual": (
                    None
                    if coefficients["lambda_L"] is None
                    else float(coefficients["lambda_L"] * training_count)
                ),
                "quadratic_lambda_coefficient": coefficients["lambda_Q"],
                "quadratic_inference_gram_ridge_actual": (
                    None
                    if coefficients["lambda_Q"] is None
                    else float(coefficients["lambda_Q"] * training_count)
                ),
                "derivation": "coefficient_times_training_snapshot_count",
            }
        records[source["case_id"]] = {
            "diagnostics_path": str(diagnostics_path),
            "diagnostics_sha256": sha256_file(diagnostics_path),
            "inference": (
                inference
                if inference is not None
                else {"method": "closed_form_linear_inference", "iteration_count": None}
            ),
            "regularization_coefficients_from_manifest": coefficients,
            "applied_regularization": applied,
        }
    return records


def _plot_fields_and_errors_manuscript(
    validated: ValidatedFigureBundle, output: Path, plt: Any
) -> tuple[Path, Path, dict[str, Any]]:
    cases = {
        _model_from_case(case_id): case_id
        for case_id in validated.metadata["case_ids"]
    }
    reference = validated.arrays[cases["linear"] + "__fom_angular_flux"]
    field_values = [reference] + [
        validated.arrays[cases[model] + "__rom_angular_flux"]
        for model in ("linear", "elementwise", "tensorial")
    ]
    field_min = float(min(np.min(values) for values in field_values))
    field_max = float(max(np.max(values) for values in field_values))
    field_padding = max(1.0e-12, 0.04 * (field_max - field_min))
    field_limits = [field_min - field_padding, field_max + field_padding]

    fig = plt.figure(figsize=(16.0, 10.8), constrained_layout=True)
    grid = fig.add_gridspec(3, 4, height_ratios=(1.0, 1.0, 0.9))
    top_axes = [fig.add_subplot(grid[0, column]) for column in range(4)]
    middle_axes = [fig.add_subplot(grid[1, column]) for column in range(4)]
    error_axis = fig.add_subplot(grid[2, :])
    titles = [
        "Full-order reference",
        MODEL_LABELS["linear"],
        MODEL_LABELS["elementwise"],
        MODEL_LABELS["tensorial"],
    ]
    for axis, values, title in zip(top_axes, field_values, titles):
        _plot_direction_blocks(axis, values)
        axis.set_ylim(field_limits)
        axis.set_title(title)
        axis.set_xlabel(r"$x$")
    top_axes[0].set_ylabel(r"Angular flux $\psi_d(x,t=2.5)$")
    top_axes[0].legend(loc="best", fontsize=7, frameon=False)

    middle_axes[0].axis("off")
    middle_axes[0].text(
        0.5,
        0.5,
        "Signed discrepancy\nreference − reconstruction",
        ha="center",
        va="center",
        fontsize=12,
    )
    discrepancy_limits: dict[str, list[float]] = {}
    for axis, model in zip(middle_axes[1:], ("linear", "elementwise", "tensorial")):
        case_id = cases[model]
        discrepancy = validated.arrays[case_id + "__discrepancy"]
        _plot_direction_blocks(axis, discrepancy)
        bound = float(np.max(np.abs(discrepancy)))
        bound = max(bound * 1.06, 1.0e-14)
        limits = [-bound, bound]
        discrepancy_limits[case_id] = limits
        axis.set_ylim(limits)
        axis.axhline(0.0, color="0.35", linewidth=0.6)
        axis.set_title(MODEL_LABELS[model])
        axis.set_xlabel(r"$x$")
    middle_axes[1].set_ylabel(r"$\psi_{FOM}-\psi_{ROM}$")

    representative_errors: dict[str, dict[str, float]] = {}
    for model in ("linear", "elementwise", "tensorial"):
        case_id = cases[model]
        time = validated.arrays[case_id + "__time"]
        errors = validated.arrays[
            case_id + "__instantaneous_normalized_mass_error"
        ]
        error_axis.semilogy(
            time,
            errors,
            color=MODEL_COLORS[model],
            linewidth=1.5,
            label=MODEL_LABELS[model],
        )
        representative_errors[case_id] = {
            "t_2_5": float(errors[2500]),
            "t_7_5": float(errors[7500]),
            "t_10": float(errors[-1]),
        }
    error_axis.axvline(
        FIELD_TIME,
        color="black",
        linestyle="--",
        linewidth=1.2,
        label=r"Field time $t=2.5$",
    )
    error_axis.axvline(
        TRAINING_BOUNDARY,
        color="red",
        linestyle="--",
        linewidth=1.2,
        label=r"Training / extrapolation boundary $t=7.5$",
    )
    error_axis.set(
        xlim=(0.0, 10.0),
        xlabel=r"Time $t$",
        ylabel="Instantaneous steady-state-normalized $M$-error",
    )
    error_axis.grid(True, which="both", alpha=0.22, linewidth=0.5)
    error_axis.legend(ncol=5, fontsize=9, loc="upper right", frameon=False)
    operator_label = "Projected" if validated.figure == "Figure 2" else "Inferred"
    fig.suptitle(f"{operator_label} reduced-model comparison", fontsize=15)

    figure_number = validated.figure.split()[-1]
    png = output / f"figure{figure_number}_manuscript.png"
    pdf = output / f"figure{figure_number}_manuscript.pdf"
    fig.savefig(png, dpi=300, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)

    ordinates = production_ordinates()
    details = {
        "field_time": FIELD_TIME,
        "training_extrapolation_boundary": TRAINING_BOUNDARY,
        "field_time_marker": {"time": FIELD_TIME, "color": "black", "style": "dashed"},
        "training_boundary_marker": {
            "time": TRAINING_BOUNDARY,
            "color": "red",
            "style": "dashed",
        },
        "discrepancy_definition": "reference_minus_reconstruction",
        "physical_x_limits": [0.0, 3.0],
        "field_y_limits_shared": field_limits,
        "discrepancy_y_limits_by_case": discrepancy_limits,
        "error_history": {
            "time_limits": [0.0, 10.0],
            "y_scale": "logarithmic",
            "metric_id": "instantaneous_steady_state_normalized_mass_error",
            "representative_values": representative_errors,
        },
        "phase_space_mapping": {
            "ordering": "direction_major",
            "angular_directions": 4,
            "ordinates_in_stored_order": ordinates.tolist(),
            "spatial_dofs_per_direction": 1500,
            "cells": 750,
            "dofs_per_cell": 2,
            "spatial_ordering": "cell-major left/right linear-DG endpoints",
            "interface_treatment": "duplicate DG endpoint values preserved; no averaging",
        },
        "series_labels": MODEL_LABELS,
        "inference_details": _inference_metadata(validated),
    }
    return png, pdf, details


def plot_manuscript_figure(
    bundle_directory: str | Path,
    *,
    output_directory: str | Path,
    expected_figure: str | None = None,
) -> list[Path]:
    """Write manuscript PNG/PDF/metadata from an already validated bundle."""
    validated = validate_manuscript_figure_bundle(
        bundle_directory,
        expected_figure=expected_figure,
    )
    output = Path(output_directory)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite manuscript output: {output}")
    output.mkdir(parents=True, exist_ok=False)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "legend.fontsize": 9,
            "pdf.fonttype": 42,
        }
    )
    if validated.figure == "Figure 1":
        png, pdf, details = _plot_figure1_manuscript(validated, output, plt)
    else:
        png, pdf, details = _plot_fields_and_errors_manuscript(
            validated, output, plt
        )
    metadata = _base_metadata(validated)
    metadata["plot_details"] = details
    metadata["outputs"] = {
        "png": {"path": str(png), "sha256": sha256_file(png)},
        "pdf": {"path": str(pdf), "sha256": sha256_file(pdf)},
    }
    metadata_path = output / "plot_metadata.json"
    _write_json(metadata_path, metadata)
    return [png, pdf, metadata_path]
