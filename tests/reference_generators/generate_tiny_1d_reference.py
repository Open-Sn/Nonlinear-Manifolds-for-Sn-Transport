#!/usr/bin/env python3
"""Generate independently constructed golden data for the tiny 1-D problem.

This script deliberately depends only on the standard library, NumPy, and
SciPy.  The DG operators below are assembled directly from the endpoint basis
weak form; no project implementation or pytest fixture is imported.
"""

from __future__ import annotations

import argparse
from datetime import date
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import scipy
from scipy import linalg


ARTIFACT_NAME = "tiny_1d_reference.npz"
MANIFEST_NAME = "tiny_1d_manifest.json"
GENERATOR_PATH = "tests/reference_generators/generate_tiny_1d_reference.py"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parents[1] / "golden"

# Tabulated Gauss--Legendre order-four data. Conventional weights sum to two;
# the transport convention in this repository normalizes them to sum to one.
GL4_NODES = np.array(
    [
        -0.8611363115940526,
        -0.3399810435848563,
        0.3399810435848563,
        0.8611363115940526,
    ],
    dtype=np.float64,
)
GL4_WEIGHTS = 0.5 * np.array(
    [
        0.3478548451374539,
        0.6521451548625461,
        0.6521451548625461,
        0.3478548451374539,
    ],
    dtype=np.float64,
)

TOLERANCES = {
    "analytic": {"rtol": 0.0, "atol": 2.0e-15},
    "operators": {"rtol": 0.0, "atol": 1.0e-13},
    "steady_state": {"rtol": 1.0e-10, "atol": 1.0e-12},
    "transient_state": {"rtol": 1.0e-8, "atol": 1.0e-10},
    "pod_spectrum": {"rtol": 1.0e-8, "atol": 1.0e-12},
    "pod_projector": {"rtol": 1.0e-8, "atol": 1.0e-10},
    "pod_projection_error": {"rtol": 1.0e-8, "atol": 1.0e-12},
}

ARRAY_METADATA = {
    "mu": (
        "analytic",
        "analytic",
        "Tabulated order-four Gauss--Legendre ordinates in ascending order.",
    ),
    "weights": (
        "analytic",
        "analytic",
        "Tabulated Gauss--Legendre weights divided by two so they sum to one.",
    ),
    "cell_edges": (
        "analytic",
        "analytic",
        "Six uniform half-width cells spanning [0, 3].",
    ),
    "cell_material_ids": (
        "analytic",
        "analytic",
        "Zero-based material identifier for each cell from left to right.",
    ),
    "mass_matrix": (
        "independent_numerical",
        "operators",
        "Direction-major DG phase-space mass matrix M.",
    ),
    "streaming_matrix": (
        "independent_numerical",
        "operators",
        "Upwind DG streaming matrix G.",
    ),
    "total_interaction_matrix": (
        "independent_numerical",
        "operators",
        "Total-interaction matrix A from sigma_t.",
    ),
    "scattering_matrix": (
        "independent_numerical",
        "operators",
        "Isotropic scattering matrix B with input-direction quadrature weights.",
    ),
    "system_matrix": (
        "independent_numerical",
        "operators",
        "Transport system matrix F = G + A - B.",
    ),
    "boundary_inflow_matrix": (
        "independent_numerical",
        "operators",
        "Map from four angular inflow values to the phase-space source.",
    ),
    "boundary_source": (
        "independent_numerical",
        "operators",
        "Source b for unit left inflow in the most-normal positive ordinate.",
    ),
    "steady_state": (
        "independent_numerical",
        "steady_state",
        "Dense-solve solution of F psi_inf = b.",
    ),
    "time": (
        "analytic",
        "analytic",
        "Six prescribed output times from 0.00 through 0.05.",
    ),
    "transient_state": (
        "independent_numerical",
        "transient_state",
        "Zero-initial-condition solution from an augmented matrix exponential.",
    ),
    "pod_eigenvalues": (
        "independent_numerical",
        "pod_spectrum",
        "Descending eigenvalues of C = S.T @ M @ S without a 1/Ns factor.",
    ),
    "pod_retained_energy": (
        "independent_numerical",
        "pod_spectrum",
        "Cumulative fraction of correlation energy retained by each rank.",
    ),
    "pod_unresolved_energy": (
        "independent_numerical",
        "pod_spectrum",
        "Fraction of correlation energy unresolved after each rank.",
    ),
    "pod_projector_rank3": (
        "independent_numerical",
        "pod_projector",
        "Sign-invariant rank-three M-orthogonal projector V3 @ V3.T @ M.",
    ),
    "pod_projection_error_rank3": (
        "independent_numerical",
        "pod_projection_error",
        "Absolute M-norm best-projection error for each centered snapshot.",
    ),
}


def _update_length_prefixed(hasher: "hashlib._Hash", payload: bytes) -> None:
    """Hash one field with an unsigned 64-bit big-endian length prefix."""
    hasher.update(len(payload).to_bytes(8, byteorder="big", signed=False))
    hasher.update(payload)


def canonical_content_checksum(arrays: dict[str, np.ndarray]) -> str:
    """Return the deterministic SHA-256 checksum of numerical array content.

    Arrays are visited in sorted-name order. For each array, four
    length-prefixed fields are hashed: UTF-8 name, NumPy ``dtype.str``, compact
    JSON shape, and contiguous C-order bytes. ZIP/NPZ container metadata is not
    part of this checksum.
    """
    hasher = hashlib.sha256()
    for name in sorted(arrays):
        array = np.ascontiguousarray(np.asarray(arrays[name]))
        _update_length_prefixed(hasher, name.encode("utf-8"))
        _update_length_prefixed(hasher, array.dtype.str.encode("ascii"))
        shape = json.dumps(list(array.shape), separators=(",", ":")).encode("ascii")
        _update_length_prefixed(hasher, shape)
        _update_length_prefixed(hasher, array.tobytes(order="C"))
    return hasher.hexdigest()


def assemble_independent_operators() -> dict[str, np.ndarray]:
    """Assemble the tiny dense DG operators directly from the weak form.

    The spatial basis on each cell consists of the two endpoint Lagrange
    functions. For cell width h, direct integration gives

        M_e = h/6 [[2, 1], [1, 2]].

    Integrating the streaming derivative by parts gives the reference-element
    volume block 0.5 [[1, 1], [-1, -1]]. Upwind numerical fluxes add the
    outflow diagonal and couple only to the immediately upwind neighbor. The
    resulting phase-space ordering is direction, cell, local endpoint.
    """
    mu = GL4_NODES.copy()
    weights = GL4_WEIGHTS.copy()
    cell_edges = np.linspace(0.0, 3.0, 7, dtype=np.float64)
    cell_material_ids = np.repeat(np.arange(3, dtype=np.int64), 2)
    sigma_t = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    sigma_s = np.array([0.0, 0.99, 0.0], dtype=np.float64)

    n_directions = mu.size
    n_cells = cell_material_ids.size
    n_spatial = 2 * n_cells
    n_phase = n_directions * n_spatial

    spatial_mass = np.zeros((n_spatial, n_spatial), dtype=np.float64)
    spatial_total = np.zeros_like(spatial_mass)
    spatial_scattering = np.zeros_like(spatial_mass)
    for cell, material in enumerate(cell_material_ids):
        width = cell_edges[cell + 1] - cell_edges[cell]
        local_mass = (width / 6.0) * np.array(
            [[2.0, 1.0], [1.0, 2.0]], dtype=np.float64
        )
        dofs = slice(2 * cell, 2 * cell + 2)
        spatial_mass[dofs, dofs] = local_mass
        spatial_total[dofs, dofs] = sigma_t[material] * local_mass
        spatial_scattering[dofs, dofs] = sigma_s[material] * local_mass

    mass = np.kron(np.eye(n_directions, dtype=np.float64), spatial_mass)
    total = np.kron(np.eye(n_directions, dtype=np.float64), spatial_total)
    # Every output direction receives the scalar-flux sum. Each input-direction
    # block is therefore weighted by that input ordinate's normalized weight.
    angular_scattering = np.tile(weights, (n_directions, 1))
    scattering = np.kron(angular_scattering, spatial_scattering)

    streaming = np.zeros((n_phase, n_phase), dtype=np.float64)
    boundary_map = np.zeros((n_phase, n_directions), dtype=np.float64)
    weak_gradient = 0.5 * np.array(
        [[1.0, 1.0], [-1.0, -1.0]], dtype=np.float64
    )
    for direction, ordinate in enumerate(mu):
        offset = direction * n_spatial
        for cell in range(n_cells):
            left = offset + 2 * cell
            right = left + 1
            local_streaming = ordinate * weak_gradient
            if ordinate > 0.0:
                local_streaming = local_streaming.copy()
                local_streaming[1, 1] += ordinate
                if cell == 0:
                    boundary_map[left, direction] = ordinate
                else:
                    streaming[left, left - 1] = -ordinate
            else:
                local_streaming = local_streaming.copy()
                local_streaming[0, 0] -= ordinate
                if cell == n_cells - 1:
                    boundary_map[right, direction] = -ordinate
                else:
                    streaming[right, right + 1] = ordinate
            streaming[left : right + 1, left : right + 1] = local_streaming

    angular_inflow = np.zeros(n_directions, dtype=np.float64)
    angular_inflow[-1] = 1.0  # physical most-normal positive ordinate at x=0
    boundary_source = boundary_map @ angular_inflow
    system = streaming + total - scattering

    return {
        "mu": mu,
        "weights": weights,
        "cell_edges": cell_edges,
        "cell_material_ids": cell_material_ids,
        "mass_matrix": mass,
        "streaming_matrix": streaming,
        "total_interaction_matrix": total,
        "scattering_matrix": scattering,
        "system_matrix": system,
        "boundary_inflow_matrix": boundary_map,
        "boundary_source": boundary_source,
    }


def _mass_norm(vector: np.ndarray, mass: np.ndarray) -> float:
    squared_norm = float(vector @ mass @ vector)
    return float(np.sqrt(max(0.0, squared_norm)))


def build_reference_arrays() -> tuple[dict[str, np.ndarray], dict[str, float]]:
    """Construct and internally validate every golden array in memory."""
    arrays = assemble_independent_operators()
    mass = arrays["mass_matrix"]
    system = arrays["system_matrix"]
    source = arrays["boundary_source"]
    n_phase = mass.shape[0]

    steady = linalg.solve(system, source, assume_a="gen", check_finite=True)
    source_norm = np.linalg.norm(source)
    steady_residual = np.linalg.norm(system @ steady - source) / source_norm

    # For constant forcing, augment psi with a constant scalar. Exponentiating
    # this homogeneous system is independent of the production Radau path.
    inverse_mass_system = linalg.solve(
        mass, system, assume_a="pos", check_finite=True
    )
    inverse_mass_source = linalg.solve(
        mass, source, assume_a="pos", check_finite=True
    )
    augmented = np.zeros((n_phase + 1, n_phase + 1), dtype=np.float64)
    augmented[:n_phase, :n_phase] = -inverse_mass_system
    augmented[:n_phase, -1] = inverse_mass_source
    initial_augmented = np.zeros(n_phase + 1, dtype=np.float64)
    initial_augmented[-1] = 1.0
    time = np.arange(6, dtype=np.float64) * 0.01
    transient = np.column_stack(
        [(linalg.expm(current_time * augmented) @ initial_augmented)[:n_phase] for current_time in time]
    )

    centered = transient - steady[:, None]
    correlation = centered.T @ mass @ centered  # deliberately no 1/Ns factor
    eigenvalues, eigenvectors = linalg.eigh(correlation, check_finite=True)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]
    negative_tolerance = (
        100.0
        * np.finfo(np.float64).eps
        * max(1.0, float(np.max(np.abs(eigenvalues))))
    )
    if np.min(eigenvalues) < -negative_tolerance:
        raise RuntimeError("correlation matrix has a materially negative eigenvalue")
    eigenvalues = np.maximum(eigenvalues, 0.0)
    positive_threshold = (
        correlation.shape[0]
        * np.finfo(np.float64).eps
        * max(1.0, float(eigenvalues[0]))
    )
    positive = eigenvalues > positive_threshold
    if np.count_nonzero(positive) < 3:
        raise RuntimeError("fewer than three non-negligible POD eigenvalues")

    # Snapshot-method POD vectors are formed only for positive eigenvalues.
    pod_basis = (
        centered @ eigenvectors[:, positive]
    ) / np.sqrt(eigenvalues[positive])[None, :]
    basis_rank3 = pod_basis[:, :3]
    projector_rank3 = basis_rank3 @ basis_rank3.T @ mass
    projection_residual = centered - projector_rank3 @ centered
    projection_errors = np.sqrt(
        np.maximum(
            0.0,
            np.einsum("ij,ij->j", projection_residual, mass @ projection_residual),
        )
    )
    total_energy = float(np.sum(eigenvalues))
    retained_energy = np.cumsum(eigenvalues) / total_energy
    unresolved_energy = np.maximum(0.0, 1.0 - retained_energy)

    arrays.update(
        {
            "steady_state": np.asarray(steady, dtype=np.float64),
            "time": time,
            "transient_state": np.asarray(transient, dtype=np.float64),
            "pod_eigenvalues": np.asarray(eigenvalues, dtype=np.float64),
            "pod_retained_energy": np.asarray(retained_energy, dtype=np.float64),
            "pod_unresolved_energy": np.asarray(unresolved_energy, dtype=np.float64),
            "pod_projector_rank3": np.asarray(projector_rank3, dtype=np.float64),
            "pod_projection_error_rank3": np.asarray(
                projection_errors, dtype=np.float64
            ),
        }
    )

    initial_distance = _mass_norm(-steady, mass)
    final_distance = _mass_norm(transient[:, -1] - steady, mass)
    projector_idempotence = np.linalg.norm(
        projector_rank3 @ projector_rank3 - projector_rank3
    )
    projector_mass_symmetry = np.linalg.norm(
        projector_rank3.T @ mass - mass @ projector_rank3
    )
    basis_orthogonality = np.linalg.norm(
        basis_rank3.T @ mass @ basis_rank3 - np.eye(3)
    )

    if mass.shape != (48, 48) or system.shape != (48, 48):
        raise RuntimeError("tiny operator dimensions are not 48 by 48")
    if source.shape != (48,) or transient.shape != (48, 6):
        raise RuntimeError("tiny source or transient dimensions are incorrect")
    if steady_residual >= 1.0e-12:
        raise RuntimeError(f"steady residual is too large: {steady_residual:.3e}")
    if not all(np.all(np.isfinite(array)) for array in arrays.values()):
        raise RuntimeError("reference construction produced non-finite values")
    if not np.array_equal(transient[:, 0], np.zeros(n_phase)):
        raise RuntimeError("matrix-exponential solution does not preserve psi(0)=0")
    if not final_distance < initial_distance:
        raise RuntimeError("transient did not move toward the steady solution")
    if projector_idempotence >= 1.0e-10 or projector_mass_symmetry >= 1.0e-10:
        raise RuntimeError("rank-three projector invariants failed")
    if basis_orthogonality >= 1.0e-10:
        raise RuntimeError("rank-three POD basis is not M-orthonormal")

    invariants = {
        "steady_relative_residual": float(steady_residual),
        "initial_state_max_abs": float(np.max(np.abs(transient[:, 0]))),
        "initial_distance_to_steady_mass_norm": float(initial_distance),
        "final_distance_to_steady_mass_norm": float(final_distance),
        "rank3_projector_idempotence_norm": float(projector_idempotence),
        "rank3_projector_mass_symmetry_norm": float(projector_mass_symmetry),
        "rank3_basis_mass_orthogonality_norm": float(basis_orthogonality),
        "normalized_weight_sum": float(np.sum(arrays["weights"])),
    }
    return arrays, invariants


def build_manifest(
    arrays: dict[str, np.ndarray], invariants: dict[str, float]
) -> dict[str, object]:
    """Create the provenance manifest for one independently built artifact."""
    array_entries = {}
    for name in sorted(arrays):
        authority, tolerance_group, description = ARRAY_METADATA[name]
        array = np.asarray(arrays[name])
        array_entries[name] = {
            "authority": authority,
            "description": description,
            "shape": list(array.shape),
            "dtype": array.dtype.name,
            "checksum_dtype_string": array.dtype.str,
            "tolerance_group": tolerance_group,
        }

    return {
        "schema_version": "1.0.0",
        "artifact_name": ARTIFACT_NAME,
        "authority": {
            "artifact": "independent_numerical",
            "permitted_classifications": ["analytic", "independent_numerical"],
            "regression_only_content": False,
        },
        "publication_reference": False,
        "scope_statement": (
            "Tiny verification data only; it is not authoritative for the "
            "published production calculation."
        ),
        "problem": {
            "spatial_domain": [0.0, 3.0],
            "regions": [
                {"interval": [0.0, 1.0], "material": "void", "cells": 2},
                {
                    "interval": [1.0, 2.0],
                    "material": "scattering_slab",
                    "cells": 2,
                },
                {"interval": [2.0, 3.0], "material": "void", "cells": 2},
            ],
            "spatial_dg_dofs": 12,
            "angular_ordinates": 4,
            "phase_space_dofs": 48,
            "sigma_t": [0.0, 1.0, 0.0],
            "sigma_s": [0.0, 0.99, 0.0],
            "particle_velocity": 1.0,
            "left_inflow": {
                "ordinate": "physical most-normal positive",
                "angular_index": 3,
                "amplitude": 1.0,
            },
            "right_inflow": "vacuum",
            "initial_condition": "zero angular flux (tiny reference only)",
            "production_sigmoid_represented": False,
            "time_grid": [0.0, 0.01, 0.02, 0.03, 0.04, 0.05],
        },
        "state_ordering": (
            "direction-major; within each direction, cells run left-to-right; "
            "within each cell, the left then right endpoint DG degree of freedom"
        ),
        "matrix_definitions": {
            "M": "direction-major phase-space DG mass matrix",
            "G": "upwind DG streaming matrix",
            "A": "total-interaction matrix from sigma_t",
            "B": "isotropic scattering matrix weighted by input ordinate",
            "F": "G + A - B",
            "equation": "M dpsi/dt + F psi = b",
        },
        "generator": {
            "path": GENERATOR_PATH,
            "method": (
                "Independent dense endpoint-DG assembly; scipy.linalg.solve for "
                "the steady state; scipy.linalg.expm on an augmented homogeneous "
                "system for the transient; scipy.linalg.eigh of S.T @ M @ S for POD."
            ),
            "imports_production_modules": False,
            "numpy_version": np.__version__,
            "scipy_version": scipy.__version__,
            "generation_date": date.today().isoformat(),
        },
        "pod_definition": {
            "centering": "transient_state - steady_state[:, None]",
            "correlation": "C = S.T @ M @ S",
            "normalization_factor": 1.0,
            "eigensolver": "dense symmetric scipy.linalg.eigh",
            "rank3_projector": "V3 @ V3.T @ M",
            "projection_error": "sqrt((S-P3S).T @ M @ (S-P3S)) per snapshot",
        },
        "arrays": array_entries,
        "expected_invariants": invariants,
        "recommended_tolerances": TOLERANCES,
        "content_checksum": {
            "algorithm": "sha256-canonical-array-content-v1",
            "description": (
                "Sort array names; for each array hash four length-prefixed fields: "
                "UTF-8 name, NumPy dtype.str, compact-JSON shape, and contiguous "
                "C-order bytes. NPZ ZIP bytes and metadata are excluded."
            ),
            "sha256": canonical_content_checksum(arrays),
        },
    }


def write_reference(output_dir: Path) -> None:
    arrays, invariants = build_reference_arrays()
    manifest = build_manifest(arrays, invariants)
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_dir / ARTIFACT_NAME, **arrays)
    (output_dir / MANIFEST_NAME).write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"wrote {output_dir / ARTIFACT_NAME}")
    print(f"wrote {output_dir / MANIFEST_NAME}")
    print(f"content checksum: {manifest['content_checksum']['sha256']}")


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {name: archive[name] for name in archive.files}


def check_reference(output_dir: Path) -> int:
    """Compare independently regenerated arrays with committed artifact content."""
    artifact_path = output_dir / ARTIFACT_NAME
    manifest_path = output_dir / MANIFEST_NAME
    if not artifact_path.is_file() or not manifest_path.is_file():
        print(
            f"missing reference artifact or manifest in {output_dir}", file=sys.stderr
        )
        return 1

    regenerated, _ = build_reference_arrays()
    committed = _load_npz(artifact_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    errors: list[str] = []

    expected_names = set(manifest.get("arrays", {}))
    if set(committed) != expected_names:
        errors.append(
            f"committed array names differ: expected {sorted(expected_names)}, "
            f"received {sorted(committed)}"
        )
    if set(regenerated) != expected_names:
        errors.append(
            f"regenerated array names differ: expected {sorted(expected_names)}, "
            f"received {sorted(regenerated)}"
        )

    maximum_absolute_difference = 0.0
    for name in sorted(expected_names & set(committed) & set(regenerated)):
        metadata = manifest["arrays"][name]
        stored = committed[name]
        fresh = regenerated[name]
        expected_shape = tuple(metadata["shape"])
        expected_dtype = np.dtype(metadata["dtype"])
        if stored.shape != expected_shape or fresh.shape != expected_shape:
            errors.append(
                f"{name}: expected shape {expected_shape}, stored {stored.shape}, "
                f"regenerated {fresh.shape}"
            )
            continue
        if stored.dtype != expected_dtype or fresh.dtype != expected_dtype:
            errors.append(
                f"{name}: expected dtype {expected_dtype}, stored {stored.dtype}, "
                f"regenerated {fresh.dtype}"
            )
            continue
        tolerance = manifest["recommended_tolerances"][metadata["tolerance_group"]]
        if np.issubdtype(stored.dtype, np.integer):
            agrees = np.array_equal(stored, fresh)
            difference = 0.0 if agrees else float("inf")
        else:
            difference = float(np.max(np.abs(stored - fresh), initial=0.0))
            agrees = np.allclose(
                stored,
                fresh,
                rtol=float(tolerance["rtol"]),
                atol=float(tolerance["atol"]),
            )
        maximum_absolute_difference = max(maximum_absolute_difference, difference)
        if not agrees:
            errors.append(
                f"{name}: numerical disagreement; max_abs={difference:.6e}, "
                f"rtol={tolerance['rtol']}, atol={tolerance['atol']}"
            )

    stored_checksum = canonical_content_checksum(committed)
    expected_checksum = manifest.get("content_checksum", {}).get("sha256")
    if stored_checksum != expected_checksum:
        errors.append(
            f"content checksum mismatch: expected {expected_checksum}, "
            f"received {stored_checksum}"
        )

    if errors:
        print("reference check failed:", file=sys.stderr)
        for error in errors:
            print(f"  - {error}", file=sys.stderr)
        return 1

    print(f"reference check passed: {len(committed)} arrays")
    print(f"content checksum: {stored_checksum}")
    print(f"maximum regenerated absolute difference: {maximum_absolute_difference:.6e}")
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="compare regenerated arrays with existing files without rewriting them",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="directory to write or check (default: tests/golden)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    output_dir = args.output_dir.resolve()
    if args.check:
        return check_reference(output_dir)
    write_reference(output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
