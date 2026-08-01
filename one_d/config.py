"""Typed, validated configuration for the existing one-dimensional workflow."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any


_SNAPSHOT_METADATA = re.compile(
    r"_A(?P<angles>\d+)_T(?P<final>[0-9]+(?:\.[0-9]+)?)_"
    r"Nt(?P<times>\d+)_Nx(?P<cells>\d+)_"
)
_MODELS = {"linear", "elementwise", "tensorial"}
_OPERATOR_CHOICES = {"projected", "inferred"}
_HISTORICAL_CASES = {
    f"{model}:{operators}"
    for model in _MODELS
    for operators in _OPERATOR_CHOICES
}


def _finite(value: float, label: str) -> None:
    if not math.isfinite(float(value)):
        raise ValueError(f"{label} must be finite")


@dataclass(frozen=True)
class InitialConditionConfig:
    kind: str
    transition_location: float | None = None
    steepness: float | None = None
    amplitude: float = 1.0
    angular_block: str | int = "final"

    def __post_init__(self) -> None:
        if self.kind not in {"zero", "localized_sigmoid"}:
            raise ValueError("initial condition must be 'zero' or 'localized_sigmoid'")
        _finite(self.amplitude, "initial-condition amplitude")
        if self.amplitude < 0.0:
            raise ValueError("initial-condition amplitude must be nonnegative")
        if self.kind == "localized_sigmoid":
            if self.transition_location is None or self.steepness is None:
                raise ValueError(
                    "localized_sigmoid requires transition_location and steepness"
                )
            _finite(self.transition_location, "sigmoid transition location")
            _finite(self.steepness, "sigmoid steepness")
            if self.steepness <= 0.0:
                raise ValueError("sigmoid steepness must be positive")
        if not (
            self.angular_block == "final"
            or isinstance(self.angular_block, int)
            and self.angular_block >= 0
        ):
            raise ValueError("angular_block must be 'final' or a nonnegative integer")


@dataclass(frozen=True)
class ProblemConfig:
    region_widths: tuple[float, ...]
    cells_per_region: tuple[int, ...]
    sigma_t: tuple[float, ...]
    sigma_s: tuple[float, ...]
    angular_ordinates: int
    inflow_boundary: str
    inflow_direction: str
    inflow_amplitude: float
    initial_condition: InitialConditionConfig
    particle_velocity: float

    def __post_init__(self) -> None:
        region_count = len(self.region_widths)
        if region_count == 0:
            raise ValueError("at least one spatial region is required")
        if not (
            len(self.cells_per_region)
            == len(self.sigma_t)
            == len(self.sigma_s)
            == region_count
        ):
            raise ValueError(
                "region widths, cell counts, sigma_t, and sigma_s must have matching lengths"
            )
        for width in self.region_widths:
            _finite(width, "region width")
            if width <= 0.0:
                raise ValueError("region widths must be positive")
        if any(not isinstance(count, int) or count <= 0 for count in self.cells_per_region):
            raise ValueError("cell counts must be positive integers")
        for name, values in (("sigma_t", self.sigma_t), ("sigma_s", self.sigma_s)):
            for value in values:
                _finite(value, name)
                if value < 0.0:
                    raise ValueError(f"{name} values must be nonnegative")
        if not isinstance(self.angular_ordinates, int) or self.angular_ordinates <= 0:
            raise ValueError("angular_ordinates must be a positive integer")
        if self.angular_ordinates % 2:
            raise ValueError("angular_ordinates must be even")
        if self.inflow_boundary not in {"left", "right"}:
            raise ValueError("inflow_boundary must be 'left' or 'right'")
        if self.inflow_direction not in {
            "most_normal",
            "most_grazing",
            "isotropic",
        }:
            raise ValueError(
                "inflow_direction must be 'most_normal', 'most_grazing', or 'isotropic'"
            )
        _finite(self.inflow_amplitude, "inflow amplitude")
        if self.inflow_amplitude < 0.0:
            raise ValueError("inflow amplitude must be nonnegative")
        _finite(self.particle_velocity, "particle velocity")
        if self.particle_velocity <= 0.0:
            raise ValueError("particle velocity must be positive")
        if (
            isinstance(self.initial_condition.angular_block, int)
            and self.initial_condition.angular_block >= self.angular_ordinates
        ):
            raise ValueError("initial-condition angular block is out of range")

    @property
    def cell_count(self) -> int:
        return sum(self.cells_per_region)

    @property
    def spatial_dofs(self) -> int:
        return 2 * self.cell_count

    @property
    def phase_space_dofs(self) -> int:
        return self.angular_ordinates * self.spatial_dofs


@dataclass(frozen=True)
class TimeIntegrationConfig:
    initial_time: float
    final_time: float
    output_spacing: float
    fom_method: str
    fom_absolute_tolerance: float
    fom_relative_tolerance: float
    rom_method: str
    rom_absolute_tolerance: float
    rom_relative_tolerance: float
    training_end_time: float

    def __post_init__(self) -> None:
        for name in (
            "initial_time",
            "final_time",
            "output_spacing",
            "fom_absolute_tolerance",
            "fom_relative_tolerance",
            "rom_absolute_tolerance",
            "rom_relative_tolerance",
            "training_end_time",
        ):
            _finite(getattr(self, name), name.replace("_", " "))
        if self.final_time <= self.initial_time:
            raise ValueError("final_time must be greater than initial_time")
        if self.output_spacing <= 0.0:
            raise ValueError("output_spacing must be positive")
        for name in (
            "fom_absolute_tolerance",
            "fom_relative_tolerance",
            "rom_absolute_tolerance",
            "rom_relative_tolerance",
        ):
            if getattr(self, name) <= 0.0:
                raise ValueError(f"{name} must be positive")
        if not self.fom_method or not self.rom_method:
            raise ValueError("FOM and ROM methods must be nonempty")
        if not self.initial_time <= self.training_end_time <= self.final_time:
            raise ValueError("training_end_time must lie in the time interval")
        _ = self.output_count

    @property
    def output_count(self) -> int:
        interval = self.final_time - self.initial_time
        intervals = int(round(interval / self.output_spacing))
        if intervals < 1 or not math.isclose(
            intervals * self.output_spacing,
            interval,
            rel_tol=0.0,
            abs_tol=1.0e-12,
        ):
            raise ValueError(
                "time interval must be an integer multiple of output_spacing"
            )
        return intervals + 1


@dataclass(frozen=True)
class RomConfig:
    latent_dimension: int
    lifting_dimension: int
    embedding_type: str
    streaming_operators: str
    lifting_regularization: float
    linear_inference_regularization: float
    quadratic_inference_regularization_linear: float
    quadratic_inference_regularization_elementwise: float
    quadratic_inference_regularization_tensorial: float
    nonlinear_inference_tolerance: float
    nonlinear_inference_maximum_iterations: int
    historical_sequence: tuple[str, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.latent_dimension, int) or self.latent_dimension <= 0:
            raise ValueError("latent_dimension must be a positive integer")
        if not isinstance(self.lifting_dimension, int) or self.lifting_dimension < 0:
            raise ValueError("lifting_dimension must be a nonnegative integer")
        if self.embedding_type not in _MODELS:
            raise ValueError(
                "embedding_type must be 'linear', 'elementwise', or 'tensorial'"
            )
        if self.streaming_operators not in _OPERATOR_CHOICES:
            raise ValueError("streaming_operators must be 'projected' or 'inferred'")
        for name in (
            "lifting_regularization",
            "linear_inference_regularization",
            "quadratic_inference_regularization_linear",
            "quadratic_inference_regularization_elementwise",
            "quadratic_inference_regularization_tensorial",
        ):
            value = getattr(self, name)
            _finite(value, name.replace("_", " "))
            if value < 0.0:
                raise ValueError(f"{name} must be nonnegative")
        _finite(
            self.nonlinear_inference_tolerance,
            "nonlinear inference tolerance",
        )
        if self.nonlinear_inference_tolerance <= 0.0:
            raise ValueError("nonlinear inference tolerance must be positive")
        if (
            not isinstance(self.nonlinear_inference_maximum_iterations, int)
            or self.nonlinear_inference_maximum_iterations < 1
        ):
            raise ValueError(
                "nonlinear_inference_maximum_iterations must be a positive integer"
            )
        if not self.historical_sequence or any(
            case not in _HISTORICAL_CASES for case in self.historical_sequence
        ):
            raise ValueError("historical_sequence contains an invalid ROM case")

    def quadratic_regularization_for(self, model: str) -> float:
        if model == "elementwise":
            return self.quadratic_inference_regularization_elementwise
        if model == "tensorial":
            return self.quadratic_inference_regularization_tensorial
        return self.quadratic_inference_regularization_linear


@dataclass(frozen=True)
class OutputConfig:
    snapshot_filename: str
    output_root: str
    reuse_existing_snapshot: bool
    allow_overwrite: bool

    def __post_init__(self) -> None:
        if not self.snapshot_filename or Path(self.snapshot_filename).name != self.snapshot_filename:
            raise ValueError("snapshot_filename must be a nonempty base filename")
        if not self.output_root:
            raise ValueError("output_root must be nonempty")
        if not isinstance(self.reuse_existing_snapshot, bool):
            raise ValueError("reuse_existing_snapshot must be boolean")
        if not isinstance(self.allow_overwrite, bool):
            raise ValueError("allow_overwrite must be boolean")


@dataclass(frozen=True)
class OneDConfig:
    schema_version: str
    name: str
    description: str
    problem: ProblemConfig
    time: TimeIntegrationConfig
    rom: RomConfig
    output: OutputConfig

    def __post_init__(self) -> None:
        if self.schema_version != "1.0.0":
            raise ValueError("unsupported 1D configuration schema_version")
        if not self.name:
            raise ValueError("configuration name must be nonempty")
        metadata = _SNAPSHOT_METADATA.search(self.output.snapshot_filename)
        if metadata is None:
            raise ValueError(
                "snapshot filename must encode A, T, Nt, and Nx metadata"
            )
        if int(metadata.group("angles")) != self.problem.angular_ordinates:
            raise ValueError("snapshot filename angular metadata is inconsistent")
        if int(metadata.group("times")) != self.time.output_count:
            raise ValueError("snapshot filename time-count metadata is inconsistent")
        if int(metadata.group("cells")) != self.problem.cell_count:
            raise ValueError("snapshot filename cell-count metadata is inconsistent")
        if not math.isclose(
            float(metadata.group("final")),
            self.time.final_time,
            rel_tol=0.0,
            abs_tol=1.0e-12,
        ):
            raise ValueError("snapshot filename final-time metadata is inconsistent")

    @property
    def expected_snapshot_shape(self) -> tuple[int, int]:
        return (self.problem.phase_space_dofs, self.time.output_count)

    @property
    def expected_snapshot_bytes_float64(self) -> int:
        rows, columns = self.expected_snapshot_shape
        return rows * columns * 8

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def canonical_json(self) -> str:
        return json.dumps(
            self.to_dict(), sort_keys=True, separators=(",", ":"), allow_nan=False
        )

    def checksum(self) -> str:
        return hashlib.sha256(self.canonical_json().encode("utf-8")).hexdigest()

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "OneDConfig":
        problem_data = dict(data["problem"])
        problem_data["region_widths"] = tuple(problem_data["region_widths"])
        problem_data["cells_per_region"] = tuple(problem_data["cells_per_region"])
        problem_data["sigma_t"] = tuple(problem_data["sigma_t"])
        problem_data["sigma_s"] = tuple(problem_data["sigma_s"])
        problem_data["initial_condition"] = InitialConditionConfig(
            **problem_data["initial_condition"]
        )
        rom_data = dict(data["rom"])
        rom_data["historical_sequence"] = tuple(rom_data["historical_sequence"])
        return cls(
            schema_version=data["schema_version"],
            name=data["name"],
            description=data.get("description", ""),
            problem=ProblemConfig(**problem_data),
            time=TimeIntegrationConfig(**data["time"]),
            rom=RomConfig(**rom_data),
            output=OutputConfig(**data["output"]),
        )

    @classmethod
    def from_json(cls, path: str | Path) -> "OneDConfig":
        path = Path(path)
        return cls.from_dict(json.loads(path.read_text(encoding="utf-8")))


def load_config(path: str | Path) -> OneDConfig:
    """Load and validate a 1-D JSON configuration."""
    return OneDConfig.from_json(path)
