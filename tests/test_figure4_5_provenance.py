import hashlib
import json
from pathlib import Path

import pytest

from one_d.publication_metrics import (
    MetricDefinitionUnavailable,
    publication_convergence_metric,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
EVIDENCE_PATH = (
    REPOSITORY_ROOT
    / "configs"
    / "1d"
    / "publication"
    / "figure4_parameter_evidence.json"
)


def _canonical_evidence_checksum(data):
    payload = dict(data)
    payload.pop("content_checksum")
    canonical = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def test_figure4_evidence_is_complete_deterministic_and_non_executable():
    data = json.loads(EVIDENCE_PATH.read_text(encoding="utf-8"))
    assert data["artifact_type"] == "non_executable_figure4_parameter_evidence"
    assert data["executable"] is False
    assert data["training_count_evidence"]["figure4_historical_selection_count"] is None
    assert _canonical_evidence_checksum(data) == data["content_checksum"]["sha256"]

    cases = data["cases"]
    ranks = (8, 16, 24, 32, 40, 48, 56, 64)
    expected = {
        (model, operators, rank)
        for model in ("elementwise", "tensorial")
        for operators in ("projected", "inferred")
        for rank in ranks
    }
    assert len(cases) == 32
    assert {(c["model"], c["operators"], c["N_r"]) for c in cases} == expected
    assert len({c["case_id"] for c in cases}) == 32

    for case in cases:
        assert case["N_q"] == 564 - case["N_r"]
        assert case["training_snapshot_count"] is None
        assert case["gamma"] == {
            "value": None,
            "classification": "not_found",
            "source": None,
        }
        assert case["lambda_L"]["value"] == 0.0
        assert case["lambda_L"]["classification"] == "documented_but_unverified"
        if case["operators"] == "projected":
            assert case["lambda_L"]["applicable"] is False
            assert case["lambda_Q"]["applicable"] is False
            assert case["lambda_Q"]["value"] is None
            assert case["applied_gram_ridge"] == {
                "gamma": None,
                "lambda_L": None,
                "lambda_Q": None,
            }
        else:
            assert case["lambda_L"]["applicable"] is True
            assert case["lambda_Q"] == {
                "value": None,
                "applicable": True,
                "classification": "not_found",
                "source": None,
            }
            assert case["applied_gram_ridge"] == {
                "gamma": None,
                "lambda_L": 0.0,
                "lambda_Q": None,
            }


def test_unavailable_metric_message_enumerates_every_author_decision():
    with pytest.raises(MetricDefinitionUnavailable) as caught:
        publication_convergence_metric()
    message = str(caught.value)
    for decision in (
        "pointwise numerator",
        "denominator field and power",
        "temporal quadrature and endpoint weights",
        "final square-root convention",
        "integration interval",
    ):
        assert decision in message
