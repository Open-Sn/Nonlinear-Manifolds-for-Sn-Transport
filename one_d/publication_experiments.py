"""Typed catalog for sigmoid-based publication-oriented 1-D experiments."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass, replace
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any, Iterable

from .config import OneDConfig, load_config


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CATALOG_PATH = REPOSITORY_ROOT / "configs" / "1d" / "publication" / "experiments.json"
DEFAULT_FIGURE4_SELECTED_PARAMETERS_RELATIVE_PATH = (
    "configs/1d/publication/figure4_selected_parameters.json"
)
LEGACY_CONFIG_RELATIVE_PATH = "configs/1d/legacy_production.json"
LEGACY_CONFIG_CHECKSUM = "cc442174134332f4b722cfa65ef179e1abc350c3e27e342a8bfeb184aa1b2759"
GOLDEN_CONTENT_CHECKSUM = "91c84e813e5cbfabd0bf0c5be436afc19e64152b7f06c9f1a572a76038108238"
KNOWN_HISTORICAL_CATALOG_CHECKSUMS = {
    "c21db43a79d8343581862479289ed62780db9e74ff2b30f0129bf04204473c92",
    "59788662d7f3a40b8366f0c91f6ff757ae41f6357eca16f6e7e54b419d0127ed",
}
BENCHMARK_VARIANT = "legacy_sigmoid"
SIGMOID_FORMULA = "1 - 1 / (1 + exp(-100 * (x - 0.1)))"
CANONICAL_SNAPSHOT_FILENAME = "solutionDG1_A4_T10_Nt10001_Nx750_continuous_bis.npy"
SPECIFICATION_STATUSES = {
    "fully_specified",
    "partially_specified",
    "requires_author_input",
}
MODEL_TYPES = {"pod_analysis", "linear", "elementwise", "tensorial", "comparison_study"}
OPERATOR_TYPES = {None, "projected", "inferred"}
CASE_ID_PATTERN = re.compile(r"^[a-z0-9][a-z0-9_]*$")
FIGURE4_RANKS = (8, 16, 24, 32, 40, 48, 56, 64)
FIGURE4_NONLINEAR_MODELS = ("elementwise", "tensorial")
FIGURE4_OPERATOR_TYPES = ("projected", "inferred")
FIGURE4_TOTAL_NONLINEAR_DIMENSION = 564
FIGURE4_TRAINING_SNAPSHOT_COUNT = 7501
FIGURE4_PARAMETER_PROVENANCE = "regenerated_sigmoid_search"
FIGURE4_SELECTION_METRIC_ID = "relative_space_time_l2_error_v1"
FIGURE4_TIMING_POLICY_ID = "rom_solve_ivp_only_v1"
FIGURE4_EXPECTED_SOURCE_METADATA = {
    "search_run_id": "phase8-20260802T130106Z",
    "search_definition_checksum_sha256": (
        "db6741a48f446fd3bc72ed450efe24b59b8562c77f747be7b19ffbcfd9e8309b"
    ),
    "source_selected_parameter_content_checksum_sha256": (
        "c4f567bf3c7390ebe79f73727bee9b42edbc68bc367863ab97b65a2c72b7f366"
    ),
    "source_selected_parameter_file_checksum_sha256": (
        "8ba6e39760425accf40688ceddfb8aaf90e97723d4de1f14fa8b14931dd1be2f"
    ),
    "dataset_checksum_sha256": (
        "a3885dc5a071f67afb514e3d130d15cd993737a174313084f7e1ed0911cef6b3"
    ),
    "configuration_checksum_sha256": LEGACY_CONFIG_CHECKSUM,
    "search_catalog_checksum_sha256": (
        "59788662d7f3a40b8366f0c91f6ff757ae41f6357eca16f6e7e54b419d0127ed"
    ),
    "source_state_checksum_sha256": (
        "6868f3c71323977320be6c6dd9142ef9c2cfc9cc4bb8fa39d47a60a00dc92f38"
    ),
}

EXPECTED_DEVIATION = {
    "initial_condition": {
        "paper": "zero angular flux",
        "repository": "localized sigmoid in final angular block",
        "status": "intentional_repository_deviation",
    }
}

AUTHOR_CONFIRMED_SIGMOID_PROVENANCE = {
    "benchmark_variant": BENCHMARK_VARIANT,
    "manuscript_text_initial_condition": "zero angular flux",
    "author_confirmed_figure_initial_condition": (
        "localized sigmoid in final positive angular block"
    ),
    "provenance_status": "author_confirmed_figure_generation_configuration",
    "author_confirmation_scope": [
        "Figure 1",
        "Figure 2",
        "Figure 3",
        "Figure 4",
        "Figure 5",
    ],
    "repository_reproduction_scope": [
        "Figure 1",
        "Figure 2",
        "Figure 3",
        "Figure 4",
        "Figure 5",
    ],
    "authoritative_configuration": LEGACY_CONFIG_RELATIVE_PATH,
    "authoritative_configuration_checksum": LEGACY_CONFIG_CHECKSUM,
    "initial_condition_formula": SIGMOID_FORMULA,
}


@dataclass(frozen=True)
class PublicationCase:
    case_id: str
    figure: str
    purpose: str
    title: str
    base_configuration_path: str
    base_configuration_checksum: str
    benchmark_variant: str
    manuscript_deviation: dict[str, Any]
    model_type: str
    operator_construction: str | None
    latent_dimension: int | None
    lifting_dimension: int | None
    lifting_regularization_gamma: float | None
    lambda_L: float | None
    lambda_Q: float | None
    inference_tolerance: float | None
    maximum_iterations: int | None
    training_interval: tuple[float, float]
    steady_state_centering: bool
    parameter_sweep: dict[str, Any] | None
    required_input_snapshot: str
    requested_outputs: tuple[str, ...]
    expected_artifact_names: tuple[str, ...]
    specification_status: str
    missing_information: tuple[str, ...]
    execution_allowed: bool
    reported_manuscript_metadata: dict[str, Any]
    notes: tuple[str, ...]
    provenance: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return json.loads(
            json.dumps(asdict(self), sort_keys=True, allow_nan=False)
        )

    @property
    def fully_specified(self) -> bool:
        return self.specification_status == "fully_specified"


@dataclass(frozen=True)
class PublicationCatalog:
    schema_version: str
    source_path: Path
    source_data: dict[str, Any]
    cases: tuple[PublicationCase, ...]

    def canonical_json(self) -> str:
        return json.dumps(
            self.source_data,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )

    def checksum(self) -> str:
        return hashlib.sha256(self.canonical_json().encode("utf-8")).hexdigest()

    def get(self, case_id: str) -> PublicationCase:
        for case in self.cases:
            if case.case_id == case_id:
                return case
        raise KeyError(f"unknown publication case: {case_id}")


@dataclass(frozen=True)
class PublicationDryRunReport:
    case_id: str
    figure: str
    title: str
    catalog_checksum: str
    base_configuration_path: str
    base_configuration_checksum: str
    benchmark_variant: str
    sigmoid_initial_condition: dict[str, Any]
    manuscript_deviation: dict[str, Any]
    model_type: str
    operator_construction: str | None
    latent_dimension: int | None
    lifting_dimension: int | None
    lifting_regularization_gamma: float | None
    lambda_L: float | None
    lambda_Q: float | None
    inference_tolerance: float | None
    maximum_iterations: int | None
    N_s: int
    applied_ridges: dict[str, float]
    parameter_provenance: str
    selected_parameter_file: str | None
    selection_metric_id: str
    timing_policy_id: str
    complete_publication_reproduction: bool
    expected_snapshot_path: str
    expected_snapshot_shape: tuple[int, int]
    expected_training_snapshot_count: int
    estimated_snapshot_bytes: int
    estimated_snapshot_mib: float
    estimated_training_snapshot_bytes: int
    estimated_training_snapshot_mib: float
    expected_artifact_names: tuple[str, ...]
    specification_status: str
    missing_information: tuple[str, ...]
    execution_allowed: bool
    snapshot_exists: bool
    snapshot_compatible: bool | None
    action: str
    assembles_operators: bool = False
    solves: bool = False
    writes_files: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _load_json(path: Path) -> dict[str, Any]:
    def reject_constant(value: str) -> None:
        raise ValueError(f"non-finite JSON value is not allowed: {value}")

    data = json.loads(path.read_text(encoding="utf-8"), parse_constant=reject_constant)
    if not isinstance(data, dict):
        raise ValueError("publication catalog root must be a JSON object")
    return data


def _canonical_checksum(value: Any) -> str:
    canonical = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _is_portable_relative_path(value: str) -> bool:
    return bool(value) and not Path(value).is_absolute() and not re.match(
        r"^[A-Za-z]:[\\/]", value
    )


def _expected_figure4_nonlinear_case_ids() -> set[str]:
    return {
        f"fig4_{model}_{operators}_nr{rank}"
        for model in FIGURE4_NONLINEAR_MODELS
        for operators in FIGURE4_OPERATOR_TYPES
        for rank in FIGURE4_RANKS
    }


def _validate_positive_coefficient(
    value: Any,
    *,
    lower: float,
    upper: float,
    label: str,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a finite numeric coefficient")
    coefficient = float(value)
    if not math.isfinite(coefficient) or not lower <= coefficient <= upper:
        raise ValueError(f"{label} is outside the approved Figure 4 search range")
    return coefficient


def load_figure4_selected_parameters(path: str | Path) -> dict[str, Any]:
    """Load and strictly validate the portable regenerated Figure 4 selections."""
    selected_path = Path(path)
    if not selected_path.is_file():
        raise FileNotFoundError(
            f"tracked Figure 4 selected-parameter file is missing: {selected_path}"
        )
    value = _load_json(selected_path)
    payload = dict(value)
    checksum = payload.pop("content_checksum_sha256", None)
    if checksum != _canonical_checksum(payload):
        raise ValueError("tracked Figure 4 selected-parameter checksum mismatch")

    expected_top_level = {
        "schema_version",
        "artifact_type",
        "benchmark_variant",
        "result_label",
        "parameter_provenance",
        "specification_status",
        "execution_ready",
        "complete_publication_reproduction",
        "historical_parameter_recovery",
        "provenance_statements",
        "selection_metric_id",
        "timing_policy_id",
        "search_run_id",
        "relative_result_root_hint",
        "search_definition_checksum_sha256",
        "source_selected_parameter_content_checksum_sha256",
        "source_selected_parameter_file_checksum_sha256",
        "dataset_checksum_sha256",
        "configuration_checksum_sha256",
        "search_catalog_checksum_sha256",
        "source_state_checksum_sha256",
        "training_snapshot_count",
        "regularization_scaling",
        "cases",
        "content_checksum_sha256",
    }
    if set(value) != expected_top_level:
        raise ValueError("tracked Figure 4 selected-parameter fields are incomplete or unexpected")
    expected_constants = {
        "schema_version": "1.0.0",
        "artifact_type": "tracked_figure4_selected_parameters",
        "benchmark_variant": BENCHMARK_VARIANT,
        "result_label": "regenerated_sigmoid_benchmark",
        "parameter_provenance": FIGURE4_PARAMETER_PROVENANCE,
        "specification_status": "fully_specified",
        "execution_ready": True,
        "complete_publication_reproduction": False,
        "historical_parameter_recovery": False,
        "selection_metric_id": FIGURE4_SELECTION_METRIC_ID,
        "timing_policy_id": FIGURE4_TIMING_POLICY_ID,
        "training_snapshot_count": FIGURE4_TRAINING_SNAPSHOT_COUNT,
    }
    for key, expected in expected_constants.items():
        if value.get(key) != expected:
            raise ValueError(f"tracked Figure 4 selected parameters have invalid {key}")
    for key, expected in FIGURE4_EXPECTED_SOURCE_METADATA.items():
        if value.get(key) != expected:
            raise ValueError(f"tracked Figure 4 selected parameters have invalid {key}")
    if not _is_portable_relative_path(value["relative_result_root_hint"]):
        raise ValueError("tracked Figure 4 result-root hint must be a portable relative path")
    statements = value.get("provenance_statements")
    if not isinstance(statements, list) or not all(
        isinstance(statement, str) and statement for statement in statements
    ):
        raise ValueError("tracked Figure 4 provenance statements are invalid")
    normalized_statements = " ".join(statements).lower()
    for required_phrase in ("not recovered historical", "sigmoid", "exactly once"):
        if required_phrase not in normalized_statements:
            raise ValueError(
                f"tracked Figure 4 provenance must state {required_phrase!r}"
            )

    scaling = value.get("regularization_scaling")
    expected_scaling = {
        "gamma_coefficient_range": [7.0e-10, 5.0e-5],
        "lambda_Q_coefficient_range": [6.0e-9, 2.0e-4],
        "lambda_L": 0.0,
        "coefficient_application": "applied ridge = coefficient * N_s exactly once",
        "inferred_gamma_policy": "reuse corresponding projected-selected gamma",
    }
    if scaling != expected_scaling:
        raise ValueError("tracked Figure 4 regularization-scaling policy is invalid")

    cases = value.get("cases")
    if not isinstance(cases, dict) or set(cases) != _expected_figure4_nonlinear_case_ids():
        raise ValueError("tracked Figure 4 selections require the exact 32 nonlinear cases")
    combinations: set[tuple[str, str, int]] = set()
    for case_id, case in cases.items():
        if not isinstance(case, dict):
            raise ValueError(f"tracked Figure 4 case {case_id} must be an object")
        model = case.get("model")
        operators = case.get("operator_type")
        rank = case.get("N_r")
        expected_id = f"fig4_{model}_{operators}_nr{rank}"
        if (
            model not in FIGURE4_NONLINEAR_MODELS
            or operators not in FIGURE4_OPERATOR_TYPES
            or rank not in FIGURE4_RANKS
            or case_id != expected_id
        ):
            raise ValueError(f"tracked Figure 4 case identity mismatch: {case_id}")
        combination = (model, operators, rank)
        if combination in combinations:
            raise ValueError("tracked Figure 4 model/operator/rank combinations are not unique")
        combinations.add(combination)
        if case.get("N_q") != FIGURE4_TOTAL_NONLINEAR_DIMENSION - rank:
            raise ValueError(f"tracked Figure 4 dimensions do not sum to 564: {case_id}")
        gamma = _validate_positive_coefficient(
            case.get("gamma"), lower=7.0e-10, upper=5.0e-5, label=f"{case_id} gamma"
        )
        if case.get("origin") not in {"coarse", "refined"}:
            raise ValueError(f"tracked Figure 4 selection origin is invalid: {case_id}")
        tie = case.get("tie_policy_result")
        if not isinstance(tie, dict) or set(tie) != {
            "larger_regularization_chosen",
            "tied_candidate_ids",
        }:
            raise ValueError(f"tracked Figure 4 tie-policy result is invalid: {case_id}")
        if not isinstance(tie["larger_regularization_chosen"], bool):
            raise ValueError(f"tracked Figure 4 tie-policy flag is invalid: {case_id}")
        tied_ids = tie["tied_candidate_ids"]
        if (
            not isinstance(tied_ids, list)
            or not tied_ids
            or len(tied_ids) != len(set(tied_ids))
            or not all(isinstance(candidate_id, str) and candidate_id for candidate_id in tied_ids)
        ):
            raise ValueError(f"tracked Figure 4 tied candidate IDs are invalid: {case_id}")

        ridges = case.get("applied_ridges")
        if not isinstance(ridges, dict):
            raise ValueError(f"tracked Figure 4 applied ridges are invalid: {case_id}")
        if ridges.get("gamma") != gamma * FIGURE4_TRAINING_SNAPSHOT_COUNT:
            raise ValueError(f"tracked Figure 4 gamma ridge is not coefficient*N_s: {case_id}")
        common_fields = {
            "model",
            "operator_type",
            "N_r",
            "N_q",
            "gamma",
            "applied_ridges",
            "origin",
            "tie_policy_result",
        }
        if operators == "projected":
            if set(case) != common_fields or set(ridges) != {"gamma"}:
                raise ValueError(
                    f"projected Figure 4 selection carries inapplicable fields: {case_id}"
                )
        else:
            expected_fields = common_fields | {
                "lambda_L",
                "lambda_Q",
                "gamma_source_case_id",
            }
            if set(case) != expected_fields or set(ridges) != {
                "gamma",
                "lambda_L",
                "lambda_Q",
            }:
                raise ValueError(f"inferred Figure 4 selection fields are invalid: {case_id}")
            lambda_q = _validate_positive_coefficient(
                case.get("lambda_Q"),
                lower=6.0e-9,
                upper=2.0e-4,
                label=f"{case_id} lambda_Q",
            )
            if case.get("lambda_L") != 0.0 or ridges.get("lambda_L") != 0.0:
                raise ValueError(f"inferred Figure 4 lambda_L must be zero: {case_id}")
            if ridges.get("lambda_Q") != lambda_q * FIGURE4_TRAINING_SNAPSHOT_COUNT:
                raise ValueError(
                    f"tracked Figure 4 lambda_Q ridge is not coefficient*N_s: {case_id}"
                )
            projected_id = f"fig4_{model}_projected_nr{rank}"
            if case.get("gamma_source_case_id") != projected_id:
                raise ValueError(f"inferred Figure 4 gamma source is invalid: {case_id}")

    for model in FIGURE4_NONLINEAR_MODELS:
        for rank in FIGURE4_RANKS:
            projected = cases[f"fig4_{model}_projected_nr{rank}"]
            inferred = cases[f"fig4_{model}_inferred_nr{rank}"]
            if inferred["gamma"] != projected["gamma"]:
                raise ValueError(
                    f"inferred Figure 4 gamma does not reuse projected selection: {model}/N_r={rank}"
                )
    return value


def _reject_initial_condition_overrides(raw: dict[str, Any]) -> None:
    def walk(value: Any, path: tuple[str, ...]) -> None:
        if isinstance(value, dict):
            for key, nested in value.items():
                lowered = str(key).lower().replace("-", "_")
                if "initial_condition" in lowered:
                    raise ValueError(
                        "publication cases may not override the legacy sigmoid initial condition"
                    )
                walk(nested, path + (str(key),))
        elif isinstance(value, list):
            for index, nested in enumerate(value):
                walk(nested, path + (str(index),))

    for key, value in raw.items():
        if key != "manuscript_deviation":
            if "initial_condition" in key.lower().replace("-", "_"):
                raise ValueError(
                    "publication cases may not override the legacy sigmoid initial condition"
                )
            walk(value, (key,))


def _finite_optional(value: float | None, label: str) -> None:
    if value is not None and not math.isfinite(float(value)):
        raise ValueError(f"{label} must be null or finite")


def _validate_case(case: PublicationCase) -> None:
    if not CASE_ID_PATTERN.fullmatch(case.case_id):
        raise ValueError(f"invalid publication case ID: {case.case_id!r}")
    if case.base_configuration_path != LEGACY_CONFIG_RELATIVE_PATH:
        raise ValueError("every publication case must use the legacy production configuration")
    if case.base_configuration_checksum != LEGACY_CONFIG_CHECKSUM:
        raise ValueError("publication case has the wrong legacy configuration checksum")
    if case.benchmark_variant != BENCHMARK_VARIANT:
        raise ValueError("publication case benchmark_variant must be legacy_sigmoid")
    if case.manuscript_deviation != EXPECTED_DEVIATION:
        raise ValueError("publication case lacks the intentional initial-condition deviation metadata")
    if case.model_type not in MODEL_TYPES:
        raise ValueError(f"unsupported publication model type: {case.model_type}")
    if case.operator_construction not in OPERATOR_TYPES:
        raise ValueError("operator construction must be projected, inferred, or null")
    if case.specification_status not in SPECIFICATION_STATUSES:
        raise ValueError(f"invalid specification status: {case.specification_status}")
    if case.training_interval != (0.0, 7.5):
        raise ValueError("publication cases must use the training interval [0, 7.5]")
    if not case.steady_state_centering:
        raise ValueError("publication cases must retain steady-state centering")
    if case.required_input_snapshot != CANONICAL_SNAPSHOT_FILENAME:
        raise ValueError("publication case must use the canonical legacy snapshot filename")
    if not case.requested_outputs or not case.expected_artifact_names:
        raise ValueError("publication cases must declare outputs and artifact names")
    if case.latent_dimension is not None and case.latent_dimension <= 0:
        raise ValueError("latent_dimension must be null or positive")
    if case.lifting_dimension is not None and case.lifting_dimension < 0:
        raise ValueError("lifting_dimension must be null or nonnegative")
    for field, value in (
        ("lifting_regularization_gamma", case.lifting_regularization_gamma),
        ("lambda_L", case.lambda_L),
        ("lambda_Q", case.lambda_Q),
        ("inference_tolerance", case.inference_tolerance),
    ):
        _finite_optional(value, field)
        if value is not None and value < 0.0:
            raise ValueError(f"{field} must be nonnegative")
    if case.maximum_iterations is not None and case.maximum_iterations <= 0:
        raise ValueError("maximum_iterations must be null or positive")
    if case.operator_construction == "projected" and (
        case.lambda_L is not None or case.lambda_Q is not None
    ):
        raise ValueError("projected cases may not carry inference regularization")
    if case.model_type == "linear":
        if case.lifting_dimension not in {None, 0}:
            raise ValueError("linear cases must mark N_q not applicable")
        if case.lifting_regularization_gamma is not None or case.lambda_Q is not None:
            raise ValueError("linear cases may not carry nonlinear regularization")
    if case.model_type in {"elementwise", "tensorial"} and case.fully_specified:
        if case.lifting_dimension is None or case.lifting_regularization_gamma is None:
            raise ValueError("fully specified nonlinear cases require N_q and gamma")
        if case.operator_construction == "inferred" and case.lambda_Q is None:
            raise ValueError("fully specified inferred nonlinear cases require lambda_Q")
    if case.fully_specified:
        if case.missing_information or not case.execution_allowed:
            raise ValueError("fully specified cases must have no missing inputs and allow execution")
    elif case.execution_allowed:
        raise ValueError("under-specified publication cases must refuse execution")
    if case.specification_status == "requires_author_input" and not case.missing_information:
        raise ValueError("requires_author_input cases must list missing information")


def _case_from_dict(raw: dict[str, Any]) -> PublicationCase:
    _reject_initial_condition_overrides(raw)
    required = {
        "case_id",
        "figure",
        "purpose",
        "title",
        "base_configuration_path",
        "base_configuration_checksum",
        "benchmark_variant",
        "manuscript_deviation",
        "model_type",
        "operator_construction",
        "latent_dimension",
        "lifting_dimension",
        "lifting_regularization_gamma",
        "lambda_L",
        "lambda_Q",
        "inference_tolerance",
        "maximum_iterations",
        "training_interval",
        "steady_state_centering",
        "parameter_sweep",
        "required_input_snapshot",
        "requested_outputs",
        "expected_artifact_names",
        "specification_status",
        "missing_information",
        "execution_allowed",
        "reported_manuscript_metadata",
        "notes",
        "provenance",
    }
    missing = sorted(required.difference(raw))
    extra = sorted(set(raw).difference(required))
    if missing or extra:
        details = []
        if missing:
            details.append("missing " + ", ".join(missing))
        if extra:
            details.append("unexpected " + ", ".join(extra))
        raise ValueError("invalid publication case fields: " + "; ".join(details))
    case = PublicationCase(
        **{
            **raw,
            "training_interval": tuple(float(value) for value in raw["training_interval"]),
            "requested_outputs": tuple(raw["requested_outputs"]),
            "expected_artifact_names": tuple(raw["expected_artifact_names"]),
            "missing_information": tuple(raw["missing_information"]),
            "notes": tuple(raw["notes"]),
        }
    )
    _validate_case(case)
    return case


def _expand_figure4_family(
    raw: dict[str, Any],
    selected_parameters: dict[str, Any],
) -> Iterable[PublicationCase]:
    _reject_initial_condition_overrides(raw)
    dimensions = raw["latent_dimensions"]
    total_dimension = int(raw["total_dimension"])
    for model in raw["model_types"]:
        for operators in raw["operator_constructions"]:
            for latent in dimensions:
                latent = int(latent)
                linear = model == "linear"
                case_id = f"fig4_{model}_{operators}_nr{latent}"
                selected = None if linear else selected_parameters["cases"][case_id]
                if linear:
                    parameter_sweep = {
                        "latent_dimensions": dimensions,
                        "model_dimension_constraint": "linear model dimension = N_r",
                        "nonlinear_total_dimension_constraint_applicable": False,
                    }
                    reported_metadata = {
                        "figure": 4,
                        "linear_model_dimension": "N_r",
                        "nonlinear_regularization_applicable": False,
                    }
                    notes = [
                        "The linear dynamical model is fully parameterized.",
                        "The author-approved aggregate metric and online timing policy apply.",
                    ]
                    provenance = {
                        **raw["provenance"],
                        "parameter_provenance": "catalog_direct_linear",
                        "complete_publication_reproduction": False,
                    }
                else:
                    parameter_sweep = {
                        "latent_dimensions": dimensions,
                        "total_dimension_constraint": "N_r + N_q = 564",
                        "lifting_dimensions": [total_dimension - int(value) for value in dimensions],
                        "gamma_candidate_range": raw["gamma_candidate_range"],
                        "lambda_Q_candidate_range": raw["lambda_Q_candidate_range"],
                        "lambda_L": 0.0,
                        "selected_parameter_file_schema": raw["selected_parameter_file_schema"],
                        "selected_parameter_file": raw["selected_parameter_file"],
                        "selected_parameter_content_checksum_sha256": selected_parameters[
                            "content_checksum_sha256"
                        ],
                        "training_snapshot_count": selected_parameters[
                            "training_snapshot_count"
                        ],
                        "applied_ridges": selected["applied_ridges"],
                        "selection_origin": selected["origin"],
                        "tie_policy_result": selected["tie_policy_result"],
                        "selection_policy": (
                            "reviewed regenerated Phase 8 selection; search is not invoked during execution"
                        ),
                    }
                    reported_metadata = {
                        **raw["reported_manuscript_metadata"],
                        "selected_parameter_provenance": FIGURE4_PARAMETER_PROVENANCE,
                        "historical_parameter_recovery": False,
                    }
                    notes = raw["notes"]
                    provenance = {
                        **raw["provenance"],
                        "parameter_provenance": FIGURE4_PARAMETER_PROVENANCE,
                        "selected_parameter_file": raw["selected_parameter_file"],
                        "selected_parameter_content_checksum_sha256": selected_parameters[
                            "content_checksum_sha256"
                        ],
                        "source_selected_parameter_content_checksum_sha256": (
                            selected_parameters[
                                "source_selected_parameter_content_checksum_sha256"
                            ]
                        ),
                        "search_run_id": selected_parameters["search_run_id"],
                        "search_definition_checksum_sha256": selected_parameters[
                            "search_definition_checksum_sha256"
                        ],
                        "dataset_checksum_sha256": selected_parameters[
                            "dataset_checksum_sha256"
                        ],
                        "source_state_checksum_sha256": selected_parameters[
                            "source_state_checksum_sha256"
                        ],
                        "training_snapshot_count": selected_parameters[
                            "training_snapshot_count"
                        ],
                        "applied_ridges": selected["applied_ridges"],
                        "selection_origin": selected["origin"],
                        "tie_policy_result": selected["tie_policy_result"],
                        "gamma_source_case_id": selected.get("gamma_source_case_id"),
                        "historical_parameter_recovery": False,
                        "complete_publication_reproduction": False,
                    }
                case_raw = {
                    "case_id": case_id,
                    "figure": "Figure 4",
                    "purpose": raw["purpose"],
                    "title": f"Figure 4 {model} {operators} case at N_r={latent}",
                    "base_configuration_path": raw["base_configuration_path"],
                    "base_configuration_checksum": raw["base_configuration_checksum"],
                    "benchmark_variant": raw["benchmark_variant"],
                    "manuscript_deviation": raw["manuscript_deviation"],
                    "model_type": model,
                    "operator_construction": operators,
                    "latent_dimension": latent,
                    "lifting_dimension": None if linear else total_dimension - latent,
                    "lifting_regularization_gamma": (
                        None if linear else selected["gamma"]
                    ),
                    "lambda_L": 0.0 if operators == "inferred" else None,
                    "lambda_Q": (
                        selected["lambda_Q"]
                        if not linear and operators == "inferred"
                        else None
                    ),
                    "inference_tolerance": (
                        1.0e-6 if operators == "inferred" and not linear else None
                    ),
                    "maximum_iterations": (
                        100000 if operators == "inferred" and not linear else None
                    ),
                    "training_interval": raw["training_interval"],
                    "steady_state_centering": True,
                    "parameter_sweep": parameter_sweep,
                    "required_input_snapshot": raw["required_input_snapshot"],
                    "requested_outputs": raw["requested_outputs"],
                    "expected_artifact_names": raw["expected_artifact_names"],
                    "specification_status": "fully_specified",
                    "missing_information": [],
                    "execution_allowed": True,
                    "reported_manuscript_metadata": reported_metadata,
                    "notes": notes,
                    "provenance": provenance,
                }
                yield _case_from_dict(case_raw)


def _validate_legacy_policy(config: OneDConfig) -> None:
    initial = config.problem.initial_condition
    if config.checksum() != LEGACY_CONFIG_CHECKSUM:
        raise ValueError("legacy production configuration checksum has changed")
    if config.output.snapshot_filename != CANONICAL_SNAPSHOT_FILENAME:
        raise ValueError("legacy production snapshot filename has changed")
    if not (
        initial.kind == "localized_sigmoid"
        and initial.transition_location == 0.1
        and initial.steepness == 100.0
        and initial.amplitude == 1.0
        and initial.angular_block == "final"
    ):
        raise ValueError("legacy publication benchmark is not the protected sigmoid variant")


def load_publication_catalog(
    path: str | Path = DEFAULT_CATALOG_PATH,
    *,
    repository_root: str | Path = REPOSITORY_ROOT,
) -> PublicationCatalog:
    source_path = Path(path)
    data = _load_json(source_path)
    if data.get("schema_version") != "1.0.0":
        raise ValueError("unsupported publication catalog schema version")
    policy = data.get("benchmark_policy", {})
    expected_policy = {
        "benchmark_variant": BENCHMARK_VARIANT,
        "base_configuration_path": LEGACY_CONFIG_RELATIVE_PATH,
        "base_configuration_checksum": LEGACY_CONFIG_CHECKSUM,
        "required_input_snapshot": CANONICAL_SNAPSHOT_FILENAME,
        "sigmoid_formula": SIGMOID_FORMULA,
        "manuscript_deviation": EXPECTED_DEVIATION,
    }
    if policy != expected_policy:
        raise ValueError("publication catalog benchmark policy does not match the protected legacy sigmoid")
    root = Path(repository_root)
    legacy = load_config(root / LEGACY_CONFIG_RELATIVE_PATH)
    _validate_legacy_policy(legacy)
    cases = [_case_from_dict(raw) for raw in data.get("cases", [])]
    family = data.get("figure4_case_family")
    if family is not None:
        selected_relative_path = family.get("selected_parameter_file")
        if not isinstance(selected_relative_path, str) or not _is_portable_relative_path(
            selected_relative_path
        ):
            raise ValueError(
                "Figure 4 selected_parameter_file must be a portable relative path"
            )
        selected_parameters = load_figure4_selected_parameters(
            root / selected_relative_path
        )
        if (
            family.get("selected_parameter_content_checksum_sha256")
            != selected_parameters["content_checksum_sha256"]
        ):
            raise ValueError(
                "Figure 4 catalog and tracked selected-parameter checksums do not match"
            )
        if family.get("parameter_provenance") != FIGURE4_PARAMETER_PROVENANCE:
            raise ValueError("Figure 4 catalog has invalid regenerated parameter provenance")
        if (
            family.get("specification_status") != "fully_specified"
            or family.get("execution_ready") is not True
            or family.get("complete_publication_reproduction") is not False
        ):
            raise ValueError("Figure 4 catalog readiness metadata is invalid")
        cases.extend(_expand_figure4_family(family, selected_parameters))
    identifiers = [case.case_id for case in cases]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("publication catalog case IDs must be unique")
    if not cases:
        raise ValueError("publication catalog must contain at least one case")
    return PublicationCatalog(
        schema_version=data["schema_version"],
        source_path=source_path,
        source_data=deepcopy(data),
        cases=tuple(cases),
    )


def resolve_base_configuration(
    case: PublicationCase,
    *,
    repository_root: str | Path = REPOSITORY_ROOT,
) -> OneDConfig:
    config = load_config(Path(repository_root) / case.base_configuration_path)
    _validate_legacy_policy(config)
    if config.checksum() != case.base_configuration_checksum:
        raise ValueError("case and base configuration checksums do not match")
    return config


def resolve_case_configuration(
    case: PublicationCase,
    *,
    repository_root: str | Path = REPOSITORY_ROOT,
) -> OneDConfig:
    """Apply only allowed ROM-stage overrides to the legacy benchmark."""
    config = resolve_base_configuration(case, repository_root=repository_root)
    if case.model_type not in {"pod_analysis", "linear", "elementwise", "tensorial"}:
        raise ValueError("comparison-study cases do not resolve to one executable ROM")
    embedding = "linear" if case.model_type == "pod_analysis" else case.model_type
    rom = config.rom
    kwargs: dict[str, Any] = {
        "latent_dimension": int(case.latent_dimension),
        "lifting_dimension": int(case.lifting_dimension or 0),
        "embedding_type": embedding,
    }
    if case.operator_construction is not None:
        kwargs["streaming_operators"] = case.operator_construction
    if case.lifting_regularization_gamma is not None:
        kwargs["lifting_regularization"] = case.lifting_regularization_gamma
    if case.lambda_L is not None:
        kwargs["linear_inference_regularization"] = case.lambda_L
    if case.lambda_Q is not None:
        kwargs[f"quadratic_inference_regularization_{embedding}"] = case.lambda_Q
    if case.inference_tolerance is not None:
        kwargs["nonlinear_inference_tolerance"] = case.inference_tolerance
    if case.maximum_iterations is not None:
        kwargs["nonlinear_inference_maximum_iterations"] = case.maximum_iterations
    return replace(config, rom=replace(rom, **kwargs))


def publication_case_summary(case: PublicationCase) -> dict[str, Any]:
    return {
        "case_id": case.case_id,
        "figure": case.figure,
        "benchmark_variant": case.benchmark_variant,
        "model": case.model_type,
        "operators": case.operator_construction,
        "N_r": case.latent_dimension,
        "N_q": case.lifting_dimension,
        "gamma": case.lifting_regularization_gamma,
        "lambda_L": case.lambda_L,
        "lambda_Q": case.lambda_Q,
        "N_s": case.provenance.get("training_snapshot_count"),
        "applied_ridges": case.provenance.get("applied_ridges", {}),
        "parameter_provenance": case.provenance.get("parameter_provenance"),
        "selected_parameter_file": case.provenance.get("selected_parameter_file"),
        "selection_metric_id": case.provenance.get("metric_id"),
        "timing_policy_id": case.provenance.get("online_timing_id"),
        "status": case.specification_status,
        "execution_ready": case.execution_allowed,
        "missing_author_inputs": list(case.missing_information),
    }


def inspect_publication_case(
    catalog: PublicationCatalog,
    case_id: str,
    *,
    snapshot_path: str | Path | None = None,
    repository_root: str | Path = REPOSITORY_ROOT,
) -> dict[str, Any]:
    case = catalog.get(case_id)
    config = resolve_base_configuration(case, repository_root=repository_root)
    dry_run = dry_run_publication_case(
        catalog,
        case,
        snapshot_path=snapshot_path,
        repository_root=repository_root,
    )
    return {
        "case": case.to_dict(),
        "catalog_checksum": catalog.checksum(),
        "legacy_configuration_checksum": config.checksum(),
        "sigmoid_initial_condition": dry_run.sigmoid_initial_condition,
        "manuscript_deviation_note": (
            "The repository intentionally uses the localized sigmoid instead of the manuscript's zero angular flux."
        ),
        "dry_run": dry_run.to_dict(),
    }


def dry_run_publication_case(
    catalog: PublicationCatalog,
    case: PublicationCase,
    *,
    snapshot_path: str | Path | None = None,
    repository_root: str | Path = REPOSITORY_ROOT,
) -> PublicationDryRunReport:
    config = resolve_base_configuration(case, repository_root=repository_root)
    path = Path(snapshot_path or case.required_input_snapshot)
    from .fom import inspect_snapshot

    inspection = inspect_snapshot(path, config) if path.exists() else None
    if not case.execution_allowed:
        action = "refuse_under_specified"
    elif inspection is None:
        action = "refuse_missing_snapshot"
    elif not inspection.compatible:
        action = "refuse_incompatible_snapshot"
    else:
        action = "would_execute"
    training_count = int(
        round((config.time.training_end_time - config.time.initial_time) / config.time.output_spacing)
    ) + 1
    training_bytes = config.problem.phase_space_dofs * training_count * 8
    initial = config.problem.initial_condition
    return PublicationDryRunReport(
        case_id=case.case_id,
        figure=case.figure,
        title=case.title,
        catalog_checksum=catalog.checksum(),
        base_configuration_path=case.base_configuration_path,
        base_configuration_checksum=case.base_configuration_checksum,
        benchmark_variant=case.benchmark_variant,
        sigmoid_initial_condition={
            "kind": initial.kind,
            "formula": SIGMOID_FORMULA,
            "transition_location": initial.transition_location,
            "steepness": initial.steepness,
            "amplitude": initial.amplitude,
            "angular_block": initial.angular_block,
        },
        manuscript_deviation=case.manuscript_deviation,
        model_type=case.model_type,
        operator_construction=case.operator_construction,
        latent_dimension=case.latent_dimension,
        lifting_dimension=case.lifting_dimension,
        lifting_regularization_gamma=case.lifting_regularization_gamma,
        lambda_L=case.lambda_L,
        lambda_Q=case.lambda_Q,
        inference_tolerance=case.inference_tolerance,
        maximum_iterations=case.maximum_iterations,
        N_s=training_count,
        applied_ridges=dict(case.provenance.get("applied_ridges", {})),
        parameter_provenance=case.provenance.get(
            "parameter_provenance", "catalog_direct"
        ),
        selected_parameter_file=case.provenance.get("selected_parameter_file"),
        selection_metric_id=case.provenance.get(
            "metric_id", FIGURE4_SELECTION_METRIC_ID
        ),
        timing_policy_id=case.provenance.get(
            "online_timing_id", FIGURE4_TIMING_POLICY_ID
        ),
        complete_publication_reproduction=case.provenance.get(
            "complete_publication_reproduction", False
        ),
        expected_snapshot_path=str(path),
        expected_snapshot_shape=config.expected_snapshot_shape,
        expected_training_snapshot_count=training_count,
        estimated_snapshot_bytes=config.expected_snapshot_bytes_float64,
        estimated_snapshot_mib=config.expected_snapshot_bytes_float64 / (1024.0**2),
        estimated_training_snapshot_bytes=training_bytes,
        estimated_training_snapshot_mib=training_bytes / (1024.0**2),
        expected_artifact_names=case.expected_artifact_names,
        specification_status=case.specification_status,
        missing_information=case.missing_information,
        execution_allowed=case.execution_allowed,
        snapshot_exists=path.exists(),
        snapshot_compatible=None if inspection is None else inspection.compatible,
        action=action,
    )
