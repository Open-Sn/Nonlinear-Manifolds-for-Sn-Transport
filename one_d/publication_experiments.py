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
LEGACY_CONFIG_RELATIVE_PATH = "configs/1d/legacy_production.json"
LEGACY_CONFIG_CHECKSUM = "cc442174134332f4b722cfa65ef179e1abc350c3e27e342a8bfeb184aa1b2759"
GOLDEN_CONTENT_CHECKSUM = "91c84e813e5cbfabd0bf0c5be436afc19e64152b7f06c9f1a572a76038108238"
KNOWN_HISTORICAL_CATALOG_CHECKSUMS = {
    "c21db43a79d8343581862479289ed62780db9e74ff2b30f0129bf04204473c92",
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
    "author_confirmation_scope": ["Figure 1", "Figure 2", "Figure 3"],
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


def _expand_figure4_family(raw: dict[str, Any]) -> Iterable[PublicationCase]:
    _reject_initial_condition_overrides(raw)
    dimensions = raw["latent_dimensions"]
    total_dimension = int(raw["total_dimension"])
    for model in raw["model_types"]:
        for operators in raw["operator_constructions"]:
            for latent in dimensions:
                latent = int(latent)
                linear = model == "linear"
                missing: list[str] = []
                if not linear:
                    missing.append(
                        f"author-selected gamma for {model}/{operators}/N_r={latent}",
                    )
                if not linear and operators == "inferred":
                    missing.append(
                        f"author-selected lambda_Q for {model}/{operators}/N_r={latent}",
                    )
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
                else:
                    parameter_sweep = {
                        "latent_dimensions": dimensions,
                        "total_dimension_constraint": "N_r + N_q = 564",
                        "lifting_dimensions": [total_dimension - int(value) for value in dimensions],
                        "gamma_candidate_range": raw["gamma_candidate_range"],
                        "lambda_Q_candidate_range": raw["lambda_Q_candidate_range"],
                        "lambda_L": 0.0,
                        "selected_parameter_file_schema": raw["selected_parameter_file_schema"],
                        "selection_policy": "no selection without an explicit objective, search result, and author-approved provenance",
                    }
                    reported_metadata = raw["reported_manuscript_metadata"]
                    notes = raw["notes"]
                case_raw = {
                    "case_id": f"fig4_{model}_{operators}_nr{latent}",
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
                    "lifting_regularization_gamma": None,
                    "lambda_L": 0.0 if operators == "inferred" else None,
                    "lambda_Q": None,
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
                    "specification_status": (
                        "fully_specified" if linear else "requires_author_input"
                    ),
                    "missing_information": missing,
                    "execution_allowed": linear,
                    "reported_manuscript_metadata": reported_metadata,
                    "notes": notes,
                    "provenance": raw["provenance"],
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
        cases.extend(_expand_figure4_family(family))
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
