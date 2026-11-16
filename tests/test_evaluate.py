
import json
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from src.evaluate import run_evaluation


def test_run_evaluation(tmp_path):
    """Check metrics, provenance, and JSON output."""

    X_test = pd.Series([
        "Example complaint A",
        "Example complaint B",
        "Example complaint C",
        "Example complaint D",
    ])

    y_test = pd.Series(["A", "A", "B", "B"])

    metadata = {
        "dataset_fingerprint": "a" * 64,
        "manifest_fingerprint": "b" * 64,
        "fingerprint_algorithm": "sha256",
        "total_samples": 14,
        "split_counts": {
            "train": 8,
            "validation": 2,
            "test": 4,
        },
    }

    class FakeModel:
        classes_ = np.array(["A", "B"])

        def predict(self, X):
            return np.array(["A", "B", "B", "B"])

    report_path = tmp_path / "test_metrics.json"

    with patch(
        "src.evaluate.train_final_model_with_test_data",
        return_value=(FakeModel(), X_test, y_test, metadata),
    ) as mocked_training:
        results = run_evaluation(
            "dummy.csv",
            str(report_path),
        )

    mocked_training.assert_called_once_with(
        "dummy.csv",
        "data/splits/group_split_manifest.csv",
        include_metadata=True,
    )

    assert results["test_samples"] == 4
    assert results["accuracy"] == pytest.approx(0.75)
    assert results["macro_f1"] == pytest.approx(
        0.7333333333333334
    )

    assert results["confusion_matrix"] == [
        [1, 1],
        [0, 2],
    ]

    assert results["dataset_fingerprint"] == "a" * 64
    assert results["manifest_fingerprint"] == "b" * 64
    assert results["fingerprint_algorithm"] == "sha256"

    assert results["evaluation_type"] == "frozen_holdout"
    assert results["split_source"] == "frozen_complaint_id_manifest"

    assert results["total_samples"] == 14
    assert results["split_counts"] == {
        "train": 8,
        "validation": 2,
        "test": 4,
    }

    assert results["model_fit_samples"] == 10

    assert results["historical_membership_authentication"] == (
        "not_independently_authenticated"
    )

    assert "split_strategy" not in results
    assert "random_state" not in results
    assert "grouping_method" not in results

    assert report_path.exists()

    raw_report = report_path.read_bytes()
    assert raw_report.endswith(b"\n")
    assert not raw_report.endswith((b"\r\n", b"\n\n"))
    saved_results = json.loads(
        raw_report.decode("utf-8"),
        parse_constant=lambda token: pytest.fail(f"Non-standard JSON constant: {token}"),
    )

    assert saved_results == results
    assert "classification_report" in saved_results


def test_run_evaluation_uses_macro_f1_for_imbalanced_classes(
    tmp_path,
):
    """Ensure macro F1 differs from weighted F1 on imbalanced data."""

    X_test = pd.Series([
        "Example A1",
        "Example A2",
        "Example A3",
        "Example A4",
        "Example B1",
        "Example B2",
    ])

    y_test = pd.Series([
        "A", "A", "A", "A", "B", "B",
    ])

    metadata = {
        "dataset_fingerprint": "c" * 64,
        "manifest_fingerprint": "d" * 64,
        "fingerprint_algorithm": "sha256",
        "total_samples": 21,
        "split_counts": {
            "train": 12,
            "validation": 3,
            "test": 6,
        },
    }

    class FakeModel:
        classes_ = np.array(["A", "B"])

        def predict(self, X):
            return np.array([
                "A", "A", "A", "B", "B", "B",
            ])

    report_path = tmp_path / "imbalanced_metrics.json"

    with patch(
        "src.evaluate.train_final_model_with_test_data",
        return_value=(FakeModel(), X_test, y_test, metadata),
    ):
        results = run_evaluation(
            "dummy.csv",
            str(report_path),
        )

    assert results["test_samples"] == 6

    assert results["accuracy"] == pytest.approx(5 / 6)

    assert results["macro_f1"] == pytest.approx(
        0.8285714285714285
    )

    assert results["classification_report"]["macro avg"]["f1-score"] == pytest.approx(
        0.8285714285714285
    )

    assert results["classification_report"]["weighted avg"]["f1-score"] == pytest.approx(
        0.8380952380952381
    )

    assert results["macro_f1"] != pytest.approx(
        results["classification_report"]["weighted avg"]["f1-score"]
    )

    assert results["confusion_matrix"] == [
        [3, 1],
        [0, 2],
    ]

    assert results["model_fit_samples"] == 15
    assert report_path.exists()


@pytest.mark.parametrize("nonfinite", [float("nan"), float("inf"), float("-inf")])
@pytest.mark.parametrize("location", ["accuracy", "metadata"])
def test_evaluation_report_rejects_nonfinite_values(tmp_path, monkeypatch, nonfinite, location):
    texts = pd.Series(["Synthetic complaint A", "Synthetic complaint B"])
    labels = pd.Series(["A", "B"])
    metadata = {
        "dataset_fingerprint": "a" * 64,
        "manifest_fingerprint": "b" * 64,
        "fingerprint_algorithm": "sha256",
        "total_samples": 6,
        "split_counts": {"train": 2, "validation": 2, "test": 2},
    }

    class FakeModel:
        classes_ = np.array(["A", "B"])

        def predict(self, X):
            return np.array(["A", "B"])

    if location == "accuracy":
        monkeypatch.setattr("src.evaluate.accuracy_score", lambda *_: nonfinite)
    else:
        metadata["total_samples"] = nonfinite

    report_path = tmp_path / "nonfinite_metrics.json"
    with patch(
        "src.evaluate.train_final_model_with_test_data",
        return_value=(FakeModel(), texts, labels, metadata),
    ):
        with pytest.raises(ValueError, match="Out of range float values are not JSON compliant"):
            run_evaluation("synthetic.csv", str(report_path))

    # json.dump has begun writing; the failure is serialization, not metric calculation.
    assert report_path.is_file()
    assert report_path.read_bytes().startswith(b"{")
