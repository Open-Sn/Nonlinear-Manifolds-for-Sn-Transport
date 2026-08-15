"""Plotting utilities for the nonlinear-manifold ROM examples."""

import matplotlib.pyplot as plt
import numpy as np


def plot_rom_comparison(
    x_coordinates,
    n_directions,
    reference_solution,
    rom_solutions,
    errors,
    time_values,
    train_size,
    frame=2499,
    frame_marker_time=2.5,
    training_boundary_time=None,
    output_path=None,
    show=False,
):
    """Compare linear, polynomial, and tensorial ROM solutions with a reference.

    The top row shows the four angular components of each solution at one
    snapshot, the middle row shows the corresponding discrepancies, and the
    bottom row shows the normalized errors over time.

    Parameters
    ----------
    x_coordinates : array_like
        Spatial coordinates for one angular component.
    n_directions : int
        Number of angular directions stored in each solution snapshot.
    reference_solution : array_like, shape (n_dofs, n_times)
        Full-order reference solution.
    rom_solutions : sequence of array_like
        Linear, polynomial, and tensorial reconstructed solutions.
    errors : sequence of array_like
        Normalized errors for the three reconstructed solutions.
    time_values : array_like
        Times corresponding to the solution snapshots and errors.
    train_size : int
        Number of snapshots in the training interval.
    frame : int, optional
        Snapshot index displayed in the solution and discrepancy panels.
    frame_marker_time : float, optional
        Time marked by the vertical black dashed line in the error panel.
    training_boundary_time : float, optional
        Time separating training and extrapolation. When omitted, the first
        extrapolation snapshot time is used.
    output_path : path-like, optional
        Destination for the figure. The figure is not saved when omitted.
    show : bool, optional
        Display the figure when ``True``.

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure containing all comparison panels.
    axes : tuple
        The solution axes, discrepancy axes, and error axis.
    """
    x_coordinates = np.asarray(x_coordinates)
    reference_solution = np.asarray(reference_solution)
    rom_solutions = tuple(np.asarray(solution) for solution in rom_solutions)
    errors = tuple(np.asarray(error) for error in errors)
    time_values = np.asarray(time_values)

    if x_coordinates.ndim != 1 or x_coordinates.size == 0:
        raise ValueError("x_coordinates must be a non-empty one-dimensional array.")
    if not isinstance(n_directions, (int, np.integer)) or n_directions <= 0:
        raise ValueError("n_directions must be a positive integer.")
    if reference_solution.ndim != 2:
        raise ValueError("reference_solution must be a two-dimensional array.")
    if len(rom_solutions) != 3:
        raise ValueError("rom_solutions must contain linear, polynomial, and tensorial solutions.")
    if any(solution.shape != reference_solution.shape for solution in rom_solutions):
        raise ValueError("Every ROM solution must have the same shape as reference_solution.")
    if reference_solution.shape[0] != n_directions * x_coordinates.size:
        raise ValueError("The spatial coordinates and direction count do not match the solution size.")

    n_times = reference_solution.shape[1]
    if time_values.ndim != 1 or time_values.size != n_times:
        raise ValueError("time_values must contain one value per solution snapshot.")
    if len(errors) != 3 or any(error.ndim != 1 or error.size != n_times for error in errors):
        raise ValueError("errors must contain three one-dimensional arrays with one value per snapshot.")
    if not isinstance(train_size, (int, np.integer)) or not 0 < train_size <= n_times:
        raise ValueError("train_size must be an integer between 1 and the number of snapshots.")
    if not isinstance(frame, (int, np.integer)) or not 0 <= frame < n_times:
        raise ValueError("frame is outside the available snapshot range.")

    titles = ("Reference", "Linear", "Polynomial", "Tensorial")
    solutions = (reference_solution,) + rom_solutions
    direction_labels = [rf"$\mu_{index}$" for index in range(1, n_directions + 1)]

    fig, grid_axes = plt.subplots(
        nrows=3,
        ncols=4,
        figsize=(7.8, 5.0),
        gridspec_kw={"height_ratios": [0.9, 0.9 * 0.64 / 1.3, 0.64]},
    )

    solution_axes = tuple(grid_axes[0, :])
    discrepancy_axes = tuple(grid_axes[1, :])
    reference_frame = reference_solution[:, frame]

    for index, (title, solution) in enumerate(zip(titles, solutions)):
        solution_frame = solution[:, frame]
        reshaped_solution = solution_frame.reshape((n_directions, -1)).T
        reshaped_discrepancy = (solution_frame - reference_frame).reshape((n_directions, -1)).T

        solution_ax = solution_axes[index]
        discrepancy_ax = discrepancy_axes[index]
        solution_ax.plot(x_coordinates, reshaped_solution, label=direction_labels)
        solution_ax.legend(fontsize=8)
        discrepancy_ax.plot(x_coordinates, reshaped_discrepancy)

        for ax in (solution_ax, discrepancy_ax):
            ax.set_xlim((0, 3))
            ax.grid(True)

        solution_ax.set_ylim((-0.15, 1.15))
        solution_ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
        solution_ax.set_title(title)
        solution_ax.tick_params(labelbottom=False)

        discrepancy_ax.set_ylim((-0.32, 0.32))
        discrepancy_ax.set_yticks([-0.2, 0, 0.2])
        discrepancy_ax.set_xlabel("x")

        if index != 0:
            solution_ax.tick_params(labelleft=False)
            discrepancy_ax.tick_params(labelleft=False)
            if frame >= train_size:
                solution_ax.text(
                    1.3,
                    1.075,
                    "Extrapolating",
                    color="r",
                    fontsize=10,
                    fontweight="bold",
                )

    solution_axes[0].set_ylabel("Solution", labelpad=12)
    discrepancy_axes[0].set_ylabel("Discrepancy")

    error_gridspec = grid_axes[2, 2].get_gridspec()
    for ax in grid_axes[2, :]:
        ax.remove()
    error_ax = fig.add_subplot(error_gridspec[2, :])

    colors = ("C0", "C1", "C2")
    labels = ("Linear", "Polynomial", "Tensorial")
    for error, color, label in zip(errors, colors, labels):
        error_ax.semilogy(
            time_values[:train_size],
            error[:train_size],
            "-",
            color=color,
            label=label,
        )
        if train_size < n_times:
            error_ax.semilogy(
                time_values[train_size:],
                error[train_size:],
                "-.",
                color=color,
            )

    if train_size < n_times:
        extrapolation_time = (
            time_values[train_size]
            if training_boundary_time is None
            else training_boundary_time
        )
        error_ax.semilogy([extrapolation_time, extrapolation_time], [1e-6, 1e0], "r--")
        error_ax.text(extrapolation_time + 0.55, 0.048, "Extrapolation", color="r", fontsize=10)

    plot_end_time = np.ceil(time_values[-1])
    error_ax.plot([frame_marker_time, frame_marker_time], [1e-6, 1e6], "--", color="k")
    error_ax.set_ylim((0.5e-3, 2e-1))
    error_ax.set_xlim((0, plot_end_time))
    error_ax.set_xticks(np.arange(0, plot_end_time + 1, 1))
    error_ax.grid(True)
    error_ax.set_xlabel("t")
    error_ax.set_ylabel("Normalized Error")
    error_ax.legend(fontsize=8.5)

    fig.tight_layout()
    fig.subplots_adjust(hspace=0.2)
    box = error_ax.get_position()
    box.y0 -= 0.045
    box.y1 -= 0.045
    error_ax.set_position(box)

    if output_path is not None:
        fig.savefig(output_path, bbox_inches="tight")
    if show:
        plt.show()

    return fig, (solution_axes, discrepancy_axes, error_ax)


def plot_relative_unresolved_energy(
    svd_val,
    size_R=16,
    size_Q=548,
    output_path=None,
    show=False,
):
    """Plot the relative POD energy unresolved by each reduced-basis size.

    Parameters
    ----------
    svd_val : array_like
        Singular values of the mass-matrix-weighted training snapshots.
    size_R : int, optional
        Number of linear POD modes to highlight.
    size_Q : int, optional
        Number of nonlinear-closure POD modes to highlight.
    output_path : path-like, optional
        Destination for the figure. The figure is not saved when omitted.
    show : bool, optional
        Display the figure when ``True``.

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure containing the relative unresolved energy plot.
    ax : matplotlib.axes.Axes
        Axes containing the plot.
    relative_energy : numpy.ndarray
        Relative unresolved energy associated with each plotted basis size.
    """
    svd_val = np.asarray(svd_val, dtype=float)
    if svd_val.ndim != 1 or svd_val.size == 0:
        raise ValueError("svd_val must be a non-empty one-dimensional array.")
    if not np.all(np.isfinite(svd_val)) or np.any(svd_val < 0.0):
        raise ValueError("svd_val must contain finite, non-negative values.")
    if not isinstance(size_R, (int, np.integer)) or size_R < 0:
        raise ValueError("size_R must be a non-negative integer.")
    if not isinstance(size_Q, (int, np.integer)) or size_Q < 0:
        raise ValueError("size_Q must be a non-negative integer.")
    if size_R + size_Q > svd_val.size:
        raise ValueError("size_R + size_Q cannot exceed the number of singular values.")

    squared_singular_values = np.square(svd_val)
    total_energy = np.sum(squared_singular_values)
    if not np.isfinite(total_energy) or total_energy <= 0.0:
        raise ValueError("svd_val must have positive, finite total energy.")

    tail_energy = np.cumsum(squared_singular_values[::-1])[::-1]
    relative_energy = np.sqrt(tail_energy / total_energy)
    rb_size = np.arange(1, relative_energy.size + 1)

    fig, ax = plt.subplots(figsize=(5, 3.6))
    ax.semilogy(rb_size, relative_energy, "k")

    if size_R:
        linear_modes = np.arange(1, size_R + 1)
        ax.fill_between(
            linear_modes,
            0.9 * relative_energy[:size_R],
            alpha=0.4,
        )

    if size_Q:
        closure_modes = np.arange(size_R + 1, size_R + size_Q + 1)
        ax.fill_between(
            closure_modes,
            0.9 * relative_energy[size_R : size_R + size_Q],
            alpha=0.7,
            color="pink",
        )

    ax.set_xlim((0, 700))
    ax.set_ylim((1e-16, 1e0))
    ax.set_ylabel(r"$\rho_{\mathrm{miss}}(N_\varepsilon)$")
    ax.set_xlabel(r"$N_{\varepsilon}$")
    ax.set_title("Relative Unresolved Energy")
    ax.grid()
    fig.tight_layout()

    if output_path is not None:
        fig.savefig(output_path)
    if show:
        plt.show()

    return fig, ax, relative_energy


def _validate_rom_sweep_inputs(
    dimensions,
    projected_errors,
    projected_times,
    inferred_errors,
    inferred_times,
    error_series_count,
    time_series_count,
    reference_time,
    allow_zero_dimensions=False,
):
    """Validate and normalize data shared by the ROM sweep plots."""
    dimensions = np.asarray(dimensions, dtype=float)
    if dimensions.ndim != 1 or dimensions.size == 0:
        raise ValueError("dimensions must be a non-empty one-dimensional array.")
    if not np.all(np.isfinite(dimensions)):
        raise ValueError("dimensions must contain only finite values.")
    if allow_zero_dimensions:
        if np.any(dimensions < 0.0):
            raise ValueError("dimensions must contain non-negative values.")
    elif np.any(dimensions <= 0.0):
        raise ValueError("dimensions must contain positive values.")

    def validate_series(values, expected_count, name):
        try:
            series = tuple(np.asarray(value, dtype=float) for value in values)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name} must be a sequence of numeric arrays.") from exc

        if len(series) != expected_count:
            raise ValueError(f"{name} must contain exactly {expected_count} series.")
        if any(value.ndim != 1 or value.size != dimensions.size for value in series):
            raise ValueError(
                f"Every series in {name} must be one-dimensional and match dimensions."
            )
        if any(not np.all(np.isfinite(value)) for value in series):
            raise ValueError(f"{name} must contain only finite values.")
        if any(np.any(value <= 0.0) for value in series):
            raise ValueError(f"{name} must contain only positive values.")
        return series

    projected_errors = validate_series(
        projected_errors, error_series_count, "projected_errors"
    )
    projected_times = validate_series(
        projected_times, time_series_count, "projected_times"
    )
    inferred_errors = validate_series(
        inferred_errors, error_series_count, "inferred_errors"
    )
    inferred_times = validate_series(
        inferred_times, time_series_count, "inferred_times"
    )

    if not np.isscalar(reference_time):
        raise ValueError("reference_time must be a positive finite scalar.")
    try:
        reference_time = float(reference_time)
    except (TypeError, ValueError) as exc:
        raise ValueError("reference_time must be a positive finite scalar.") from exc
    if not np.isfinite(reference_time) or reference_time <= 0.0:
        raise ValueError("reference_time must be a positive finite scalar.")

    return (
        dimensions,
        projected_errors,
        projected_times,
        inferred_errors,
        inferred_times,
        reference_time,
    )


def plot_rom_dimension_sweep(
    reduced_dimensions,
    projected_errors,
    projected_times,
    inferred_errors,
    inferred_times,
    reference_time=541.1463527679443,
    output_path=None,
    show=False,
):
    """Plot error and online speed-up while varying the linear ROM dimension.

    The error and timing inputs each contain three series ordered as tensorial,
    polynomial, and linear. The left column displays projected streaming
    operators and the right column displays inferred streaming operators.

    Parameters
    ----------
    reduced_dimensions : array_like
        Linear reduced dimensions, :math:`N_r`, used in the sweep.
    projected_errors, inferred_errors : sequence of array_like
        Relative time-average errors for tensorial, polynomial, and linear ROMs.
    projected_times, inferred_times : sequence of array_like
        Online solution times for tensorial, polynomial, and linear ROMs.
    reference_time : float, optional
        Full-order online time used to compute each speed-up.
    output_path : path-like, optional
        Destination for the figure. The figure is not saved when omitted.
    show : bool, optional
        Display the figure when ``True``.

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure containing the four sweep panels.
    axs : numpy.ndarray
        Two-by-two array of axes.
    """
    (
        reduced_dimensions,
        projected_errors,
        projected_times,
        inferred_errors,
        inferred_times,
        reference_time,
    ) = _validate_rom_sweep_inputs(
        reduced_dimensions,
        projected_errors,
        projected_times,
        inferred_errors,
        inferred_times,
        error_series_count=3,
        time_series_count=3,
        reference_time=reference_time,
    )

    labels = ("Tensorial", "Polynomial", "Linear")
    styles = ("-<", "->", "-v")
    colors = ("C2", "C1", "C0")
    xticks = [0, 16, 32, 48, 64, 80]
    fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(7.0, 5.2))

    for errors, times, column in (
        (projected_errors, projected_times, 0),
        (inferred_errors, inferred_times, 1),
    ):
        for error, timing, style, color, label in zip(
            errors, times, styles, colors, labels
        ):
            axs[0, column].semilogy(
                reduced_dimensions, error, style, color=color, label=label
            )
            axs[1, column].semilogy(
                reduced_dimensions,
                reference_time / timing,
                style,
                color=color,
                label=label,
            )

        axs[0, column].set_xticks(xticks)
        axs[1, column].set_xticks(xticks)
        axs[0, column].tick_params(labelbottom=False)
        axs[0, column].set_xlim((0, 80))
        axs[1, column].set_xlim((0, 80))
        axs[0, column].set_ylim((1e-3, 1e-1))
        axs[1, column].set_ylim((1e1, 3e3))
        axs[0, column].grid(which="both")
        axs[1, column].grid(which="both")
        axs[0, column].legend(loc="lower left")
        axs[1, column].legend()
        axs[1, column].set_xlabel(r"$N_r$")

    axs[0, 0].set_title("Projected Streaming Operator")
    axs[0, 1].set_title("Inferred Streaming Operator")
    axs[0, 0].set_ylabel("Relative Time-Average Error")
    axs[1, 0].set_ylabel("Online Speed-Up")
    axs[0, 1].tick_params(labelleft=False)
    axs[1, 1].tick_params(labelleft=False)

    fig.tight_layout()
    if output_path is not None:
        fig.savefig(output_path)
    if show:
        plt.show()

    return fig, axs


def plot_closure_dimension_sweep(
    closure_dimensions,
    projected_errors,
    projected_times,
    inferred_errors,
    inferred_times,
    reference_time=541.1463527679443,
    output_path=None,
    show=False,
):
    """Plot error and online speed-up while varying the closure dimension.

    Error inputs contain five series ordered as tensorial, polynomial, fixed
    linear, expanded linear, and projection. Timing inputs contain the same
    first four model series and omit projection. A zero-dimensional entry is
    accepted for compatibility with the review sweep and omitted from the
    logarithmic plots.

    Parameters
    ----------
    closure_dimensions : array_like
        Nonlinear closure dimensions, :math:`N_q`, used in the sweep.
    projected_errors, inferred_errors : sequence of array_like
        Five relative time-average error series in the order described above.
    projected_times, inferred_times : sequence of array_like
        Four online solution-time series, excluding projection.
    reference_time : float, optional
        Full-order online time used to compute each speed-up.
    output_path : path-like, optional
        Destination for the figure. The figure is not saved when omitted.
    show : bool, optional
        Display the figure when ``True``.

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure containing the four sweep panels.
    axs : numpy.ndarray
        Two-by-two array of axes.
    """
    (
        closure_dimensions,
        projected_errors,
        projected_times,
        inferred_errors,
        inferred_times,
        reference_time,
    ) = _validate_rom_sweep_inputs(
        closure_dimensions,
        projected_errors,
        projected_times,
        inferred_errors,
        inferred_times,
        error_series_count=5,
        time_series_count=4,
        reference_time=reference_time,
        allow_zero_dimensions=True,
    )

    positive_dimensions = closure_dimensions > 0.0
    if not np.any(positive_dimensions):
        raise ValueError("closure_dimensions must contain at least one positive value.")
    closure_dimensions = closure_dimensions[positive_dimensions]
    projected_errors = tuple(value[positive_dimensions] for value in projected_errors)
    projected_times = tuple(value[positive_dimensions] for value in projected_times)
    inferred_errors = tuple(value[positive_dimensions] for value in inferred_errors)
    inferred_times = tuple(value[positive_dimensions] for value in inferred_times)

    error_labels = (
        r"Tensorial    ($N_r = 32$)",
        r"Polynomial ($N_r = 32$)",
        r"Linear ($N_r = 32$)",
        r"Linear ($N_r = 32+N_q$)",
        "Projection",
    )
    styles = ("-<", "->", "-v", "-^", "-o")
    colors = ("C2", "C1", "C0", "C3", "C4")

    fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(7.0, 5.2))

    for errors, times, column in (
        (projected_errors, projected_times, 0),
        (inferred_errors, inferred_times, 1),
    ):
        for error, style, color, label in zip(
            errors, styles, colors, error_labels
        ):
            axs[0, column].loglog(
                closure_dimensions, error, style, color=color, label=label
            )

        for timing, style, color, label in zip(
            times, styles[:4], colors[:4], error_labels[:4]
        ):
            axs[1, column].loglog(
                closure_dimensions,
                reference_time / timing,
                style,
                color=color,
                label=label,
            )

        axs[0, column].tick_params(labelbottom=False)
        axs[0, column].set_ylim((1e-3 / 8, 6e-2))
        axs[1, column].set_ylim((1e1, 6e2))
        axs[0, column].grid(which="both")
        axs[1, column].grid(which="both")
        axs[0, column].legend(loc="lower left")
        axs[1, column].legend(loc="lower left")
        axs[1, column].set_xlabel(r"$N_q$")

    axs[0, 0].set_title("Projected Streaming Operator")
    axs[0, 1].set_title("Inferred Streaming Operator")
    axs[0, 0].set_ylabel("Relative Time-Average Error")
    axs[1, 0].set_ylabel("Online Speed-Up")
    axs[0, 1].tick_params(labelleft=False)
    axs[1, 1].tick_params(labelleft=False)

    fig.tight_layout()
    if output_path is not None:
        fig.savefig(output_path)
    if show:
        plt.show()

    return fig, axs
