import time

import numpy as np
import scipy as sp
import scipy.sparse as sparse

import Nonlinear_Manifold_ROM as rom
from Nonlinear_Manifold_ROM import NonlinearManifoldReducedModel
from Transport_Driver_Benchmark_1D import validate_solve_ivp_result


def run_tiny_end_to_end(operators, monkeypatch, pod_rank=6):
    """Run the deterministic six-cell verification case and return metrics."""
    n_phase = operators.mass.shape[0]
    evaluation_times = np.linspace(0.0, 0.05, 6)
    boundary_values = np.zeros(operators.quadrature.ndir)
    # GL nodes are ascending, so the final ordinate is physical most-normal +mu.
    boundary_values[-1] = 1.0
    boundary_source = operators.boundary @ boundary_values

    fom_start = time.perf_counter()
    rhs = lambda current_time, state: sparse.linalg.spsolve(
        operators.mass, boundary_source - operators.operator @ state
    )
    fom_result = sp.integrate.solve_ivp(
        rhs,
        (evaluation_times[0], evaluation_times[-1]),
        np.zeros(n_phase),
        method="Radau",
        t_eval=evaluation_times,
        atol=1e-10,
        rtol=1e-8,
    )
    validate_solve_ivp_result(
        fom_result, evaluation_times, n_phase, "tiny full-order solve"
    )
    fom_runtime = time.perf_counter() - fom_start

    steady = sparse.linalg.spsolve(operators.operator, boundary_source)
    steady_residual = np.linalg.norm(
        operators.operator @ steady - boundary_source
    ) / np.linalg.norm(boundary_source)
    centered_snapshots = fom_result.y - steady[:, None]

    dense_mass = operators.mass.toarray()
    dense_mass_sqrt = np.real_if_close(sp.linalg.sqrtm(dense_mass))
    monkeypatch.setattr(rom, "globalMM", operators.mass)
    monkeypatch.setattr(
        rom, "globalMMsqrt", sparse.csc_matrix(dense_mass_sqrt)
    )
    monkeypatch.setattr(rom, "globalAbsorption", operators.total)
    monkeypatch.setattr(rom, "globalScattering", operators.scattering)
    monkeypatch.setattr(rom, "globalStreaming", operators.streaming)

    model = NonlinearManifoldReducedModel(None)
    model.solutionDG1 = fom_result.y
    model.solutionInf = steady
    model.global_training_set = centered_snapshots
    model.global_derivative_set = np.column_stack(
        [rhs(t, fom_result.y[:, index]) for index, t in enumerate(evaluation_times)]
    )
    model.TT = evaluation_times[-1]
    model.dt = evaluation_times[1] - evaluation_times[0]
    model.time_steps = evaluation_times
    model.train_size = evaluation_times.size
    model.n_dofs = n_phase
    model.compute_pod(size_R=pod_rank, size_Q=0)

    pod_residual = np.linalg.norm(
        model.pod_linear_basis.T
        @ dense_mass
        @ model.pod_linear_basis
        - np.eye(pod_rank)
    )
    model.compute_projected_operators()
    model.compute_initial_conditions()

    rom_start = time.perf_counter()
    reconstruction = model.solve(intrusive=True)
    rom_runtime = time.perf_counter() - rom_start
    errors = model.compute_errors(reconstruction)

    return {
        "fom": fom_result.y,
        "reconstruction": reconstruction,
        "steady_residual": steady_residual,
        "pod_residual": pod_residual,
        "maximum_error": float(np.max(errors)),
        "mean_error": float(np.mean(errors)),
        "fom_runtime": fom_runtime,
        "rom_runtime": rom_runtime,
    }


def test_tiny_deterministic_fom_to_projected_rom(
    tiny_transport_operators, monkeypatch
):
    metrics = run_tiny_end_to_end(tiny_transport_operators, monkeypatch)

    assert metrics["fom"].shape == (48, 6)
    assert metrics["reconstruction"].shape == (48, 6)
    assert np.all(np.isfinite(metrics["fom"]))
    assert np.all(np.isfinite(metrics["reconstruction"]))
    assert metrics["steady_residual"] < 1e-12
    assert metrics["pod_residual"] < 2e-11
    assert np.isfinite(metrics["maximum_error"])
    # This is a loose integration invariant, not a golden trajectory.
    assert metrics["maximum_error"] < 0.1

    print(
        "tiny metrics:",
        f"steady_residual={metrics['steady_residual']:.6e}",
        f"pod_residual={metrics['pod_residual']:.6e}",
        f"maximum_error={metrics['maximum_error']:.6e}",
        f"mean_error={metrics['mean_error']:.6e}",
        f"fom_runtime={metrics['fom_runtime']:.6f}s",
        f"rom_runtime={metrics['rom_runtime']:.6f}s",
    )


def test_truncated_rank_tiny_rom_is_finite_and_not_better_than_full_rank(
    tiny_transport_operators, monkeypatch
):
    full_rank = run_tiny_end_to_end(
        tiny_transport_operators, monkeypatch, pod_rank=6
    )
    truncated = run_tiny_end_to_end(
        tiny_transport_operators, monkeypatch, pod_rank=3
    )

    assert truncated["reconstruction"].shape == (48, 6)
    assert np.all(np.isfinite(truncated["reconstruction"]))
    assert np.isfinite(truncated["maximum_error"])
    assert np.isfinite(truncated["mean_error"])
    assert truncated["maximum_error"] >= 0.0
    assert truncated["mean_error"] >= 0.0
    # The rank-six basis spans this six-snapshot smoke trajectory essentially
    # exactly. Truncating to rank three must not improve that projection result.
    assert truncated["maximum_error"] + 1.0e-12 >= full_rank["maximum_error"]
