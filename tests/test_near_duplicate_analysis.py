
import json

from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from scipy.sparse import csr_matrix

from src.data_integrity import compute_dataset_fingerprint, compute_manifest_fingerprint
from src.near_duplicate_analysis import (
    _evaluate_debt_collection_subset,
    _evaluate_subset,
    analyze_near_duplicates,
    maximum_cosine_similarities,
)

def test_near_duplicate_report_consistency():
    """Verify consistency of saved diagnostic results."""

    path = Path("reports/near_duplicate_analysis.json")

    assert path.is_file(), "Near-duplicate report is missing"

    report = json.loads(path.read_text(encoding="utf-8"))

    test_samples = report["test_samples"]
    counts = report["threshold_counts"]

    assert (
        0 <= counts["0.99"]
        <= counts["0.95"]
        <= counts["0.90"]
        <= counts["0.80"]
        <= test_samples
    )

    groups = report["group_performance"]

    assert (
        groups["high_similarity"]["samples"]
        + groups["lower_similarity"]["samples"]
        == test_samples
    )

    distributions = report["group_class_distribution"]

    assert (
        sum(distributions["high_similarity"].values())
        == groups["high_similarity"]["samples"]
    )

    assert (
        sum(distributions["lower_similarity"].values())
        == groups["lower_similarity"]["samples"]
    )

def test_maximum_cosine_similarities():
    """Check maximum cosine similarity for synthetic vectors."""

    train_vectors = csr_matrix([
        [1.0, 0.0],
        [0.0, 1.0],
    ])

    test_vectors = csr_matrix([
        [1.0, 0.0],
        [0.0, 1.0],
        [1.0, 1.0],
        [0.0, 0.0],
    ])

    scores = maximum_cosine_similarities(
        test_vectors,
        train_vectors,
        batch_size=2,
    )

    expected = np.array([
        1.0,
        1.0,
        1 / np.sqrt(2),
        0.0,
    ])

    np.testing.assert_allclose(
        scores,
        expected,
        rtol=1e-10,
        atol=1e-10,
    )


@pytest.mark.parametrize("similar_test_text", [True, False])
def test_frozen_diagnostic_uses_one_split_and_preserves_history(
    tmp_path, monkeypatch, similar_test_text,
):
    from sklearn.metrics import accuracy_score, f1_score

    dataset = pd.DataFrame({
        "Complaint ID": [90, 40, 10, 70, 30, 80, 20, 60],
        "Consumer complaint narrative": [
            "mortgage escrow payment" if similar_test_text else "unknown zebra",
            "overdue debt collector calls",
            "mortgage escrow payment",
            "unknown giraffe",
            "overdue debt collector calls",
            "overdue debt collector calls" if similar_test_text else "unknown badger",
            "mortgage escrow payment",
            "unknown platypus",
        ],
        "Product": [
            "Debt collection", "Debt collection", "Mortgage", "Mortgage",
            "Debt collection", "Debt collection", "Mortgage", "Debt collection",
        ],
    })
    manifest = pd.DataFrame({
        "Complaint ID": [10, 20, 30, 40, 60, 70, 80, 90],
        "split": ["validation", "validation", "train", "train", "test", "test", "test", "test"],
    })
    csv_path = tmp_path / "complaints.csv"
    manifest_path = tmp_path / "manifest.csv"
    dataset.to_csv(csv_path, index=False)
    manifest.to_csv(manifest_path, index=False)
    manifest_path.with_suffix(".json").write_text(json.dumps({
        "dataset_fingerprint": compute_dataset_fingerprint(dataset),
        "manifest_fingerprint": compute_manifest_fingerprint(manifest),
        "fingerprint_algorithm": "sha256",
        "total_samples": len(dataset),
        "split_counts": manifest["split"].value_counts().to_dict(),
    }), encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    historical_path = tmp_path / "reports" / "near_duplicate_analysis.json"
    historical_path.parent.mkdir()
    historical_path.write_bytes(b'{"historical": "unchanged"}\n')
    original_history = historical_path.read_bytes()

    report = analyze_near_duplicates(str(csv_path), manifest_path=str(manifest_path))

    saved_path = tmp_path / "reports" / "near_duplicate_frozen_split.json"
    assert json.loads(saved_path.read_text(encoding="utf-8")) == report
    assert historical_path.read_bytes() == original_history
    assert report["evaluation_type"] == "frozen_holdout"
    assert report["split_strategy"] == "frozen_complaint_id_manifest"
    assert report["similarity_threshold"] == 0.95
    assert report["similarity_vectorizer"] == "model_fitted_tfidf"
    assert report["similarity_reference"] == "train_plus_validation"
    assert report["similarity_fit_excludes_test"] is True
    assert report["train_samples"] == report["test_samples"] == 4
    similar_count = 2 if similar_test_text else 0
    assert report["threshold_counts"] == {
        "0.80": similar_count, "0.90": similar_count,
        "0.95": similar_count, "0.99": similar_count,
    }
    assert report["mean_max_similarity"] == pytest.approx(similar_count / 4)
    assert report["median_max_similarity"] == pytest.approx(similar_count / 4)

    expected_labels = np.array(["Debt collection", "Mortgage", "Debt collection", "Debt collection"])
    expected_predictions = np.array([
        "Mortgage" if similar_test_text else "Debt collection",
        "Debt collection", "Debt collection", "Debt collection",
    ])
    high_mask = np.array([similar_test_text, False, similar_test_text, False])
    for name, mask in (("high_similarity", high_mask), ("lower_similarity", ~high_mask)):
        metrics = report["group_performance"][name]
        assert metrics["samples"] == int(mask.sum())
        assert sum(report["group_class_distribution"][name].values()) == int(mask.sum())
        if mask.any():
            assert metrics["accuracy"] == accuracy_score(expected_labels[mask], expected_predictions[mask])
            assert metrics["macro_f1"] == f1_score(expected_labels[mask], expected_predictions[mask], average="macro", zero_division=0)
        else:
            assert metrics["accuracy"] is None
            assert metrics["macro_f1"] is None
        debt_mask = mask & (expected_labels == "Debt collection")
        debt_metrics = report["debt_collection_performance"][name]
        assert debt_metrics["samples"] == int(debt_mask.sum())
        if debt_mask.any():
            expected_recall = float((expected_predictions[debt_mask] == "Debt collection").mean())
            assert debt_metrics["recall"] == pytest.approx(expected_recall)
            assert debt_metrics["accuracy"] == debt_metrics["recall"]
        else:
            assert debt_metrics["recall"] is None
            assert debt_metrics["accuracy"] is None
        assert "not comparable to overall 11-class macro-F1" in debt_metrics["macro_f1_note"]


@pytest.mark.parametrize("filename", ["near_duplicate_analysis.json", "NEAR_DUPLICATE_ANALYSIS.JSON"])
def test_diagnostic_refuses_historical_report_destination(tmp_path, filename):
    historical_path = tmp_path / filename
    historical_path.write_bytes(b"historical report")
    with pytest.raises(ValueError, match="Historical near-duplicate report is protected"):
        analyze_near_duplicates("unused.csv", report_path=str(historical_path))
    assert historical_path.read_bytes() == b"historical report"


@pytest.mark.parametrize("empty_partition", ["training", "test"])
def test_diagnostic_rejects_empty_partition(tmp_path, monkeypatch, empty_partition):
    empty = pd.Series([], dtype=str)
    texts = pd.Series(["synthetic complaint"])
    labels = pd.Series(["Debt collection"])
    if empty_partition == "training":
        partitions = (empty, empty, texts, empty, empty, labels)
        message = "Training partition"
    else:
        partitions = (texts, texts, empty, labels, labels, empty)
        message = "Test partition"
    # Simulate empty data at the filesystem-loader boundary, before model fitting.
    monkeypatch.setattr(
        "src.near_duplicate_analysis.load_frozen_split", lambda *_: partitions,
    )
    destination = tmp_path / "diagnostic.json"
    with pytest.raises(ValueError, match=message + ".*must not be empty"):
        analyze_near_duplicates("synthetic.csv", report_path=str(destination))
    assert not destination.exists()


@pytest.mark.parametrize(
    "mask, expected",
    [
        ([False, False, False, False], {"samples": 0, "accuracy": None, "macro_f1": None}),
        ([True, True, False, False], {"samples": 2, "accuracy": 0.5, "macro_f1": 1 / 3}),
        ([True, False, False, True], {"samples": 2, "accuracy": 1.0, "macro_f1": 1.0}),
    ],
)
def test_subset_metrics_without_model_training(mask, expected):
    labels = pd.Series(["A", "B", "A", "B"], index=[10, 20, 30, 40])
    predictions = np.array(["A", "A", "B", "B"])
    assert _evaluate_subset(labels, predictions, np.array(mask)) == expected


@pytest.mark.parametrize(
    "mask, samples, recall, legacy_macro_f1",
    [
        ([True, True, True, False], 2, 0.5, 1 / 3),
        ([False, False, True, True], 1, 0.0, 0.0),
        ([False, False, True, False], 0, None, None),
        ([True, True, True, True], 3, 1 / 3, 1 / 6),
    ],
    ids=["mixed_predictions", "all_incorrect", "no_true_debt_examples", "several_predicted_classes"],
)
def test_debt_collection_subset_reports_recall_and_marks_legacy_macro_f1(
    mask, samples, recall, legacy_macro_f1,
):
    labels = pd.Series(
        ["Debt collection", "Debt collection", "Mortgage", "Debt collection"],
        index=[10, 20, 30, 40],
    )
    predictions = np.array(["Debt collection", "Mortgage", "Debt collection", "Credit card"])

    metrics = _evaluate_debt_collection_subset(labels, predictions, np.array(mask))

    assert metrics["samples"] == samples
    if samples:
        assert metrics["recall"] == pytest.approx(recall)
        assert metrics["accuracy"] == pytest.approx(recall)
        assert metrics["macro_f1"] == pytest.approx(legacy_macro_f1)
    else:
        assert metrics["recall"] is None
        assert metrics["accuracy"] is None
        assert metrics["macro_f1"] is None
    assert metrics["macro_f1_note"] == (
        "Deprecated legacy subset macro-F1 over true and predicted labels; "
        "not comparable to overall 11-class macro-F1."
    )


def test_frozen_diagnostic_fits_tfidf_on_train_and_validation_only(tmp_path, monkeypatch):
    from collections import Counter
    from sklearn.feature_extraction.text import TfidfVectorizer

    train_texts = ["trainsentinel card payment", "trainsentinel mortgage payment"]
    validation_texts = ["validationsentinel card payment", "validationsentinel mortgage payment"]
    test_texts = ["heldoutsentinel card payment", "heldoutsentinel mortgage payment"]
    fitting_texts = train_texts + validation_texts
    dataset = pd.DataFrame({
        "Complaint ID": range(1, 7),
        "Consumer complaint narrative": fitting_texts + test_texts,
        "Product": ["Credit card", "Mortgage"] * 3,
    })
    manifest = pd.DataFrame({
        "Complaint ID": dataset["Complaint ID"],
        "split": ["train", "train", "validation", "validation", "test", "test"],
    })
    csv_path = tmp_path / "complaints.csv"
    manifest_path = tmp_path / "manifest.csv"
    dataset.to_csv(csv_path, index=False)
    manifest.to_csv(manifest_path, index=False)
    manifest_path.with_suffix(".json").write_text(json.dumps({
        "dataset_fingerprint": compute_dataset_fingerprint(dataset),
        "manifest_fingerprint": compute_manifest_fingerprint(manifest),
        "fingerprint_algorithm": "sha256",
        "total_samples": len(dataset),
        "split_counts": manifest["split"].value_counts().to_dict(),
    }), encoding="utf-8")

    calls = []

    def observe_vectorizer_method(method_name):
        original = getattr(TfidfVectorizer, method_name)

        def observed(vectorizer, raw_documents, *args, **kwargs):
            documents = list(raw_documents)
            calls.append((method_name, documents, vectorizer))
            if method_name in ("fit", "fit_transform"):
                assert Counter(documents) == Counter(fitting_texts)
            return original(vectorizer, documents, *args, **kwargs)

        return observed

    # Observe sklearn's fitting boundary while retaining real vectorization and training.
    for method_name in ("fit", "fit_transform", "transform"):
        monkeypatch.setattr(
            TfidfVectorizer, method_name, observe_vectorizer_method(method_name),
        )

    analyze_near_duplicates(
        str(csv_path), str(tmp_path / "diagnostic.json"), str(manifest_path),
    )

    fitted = [vectorizer for method, _, vectorizer in calls if method in ("fit", "fit_transform")]
    assert fitted
    for vectorizer in fitted:
        assert {"trainsentinel", "validationsentinel"} <= set(vectorizer.vocabulary_)
        assert "heldoutsentinel" not in vectorizer.vocabulary_
    assert any(method == "transform" and documents == test_texts for method, documents, _ in calls)


def test_frozen_diagnostic_respects_095_similarity_threshold(tmp_path):
    from sklearn.feature_extraction.text import TfidfVectorizer

    fitting_texts = ["common alpha", "common beta"] * 2
    test_texts = [
        "common alpha common beta",
        "common alpha " * 28 + "common beta " * 10,
        "common alpha " * 29 + "common beta " * 10,
    ]
    reference = TfidfVectorizer(min_df=2, ngram_range=(1, 2))
    train_vectors = reference.fit_transform(fitting_texts)
    scores = maximum_cosine_similarities(reference.transform(test_texts), train_vectors)
    np.testing.assert_allclose(
        scores, [0.7680131654973622, 0.9495882034951384, 0.9525835418188402],
        rtol=1e-12, atol=1e-12,
    )
    assert 0.50 < scores[0] < scores[1] < 0.95 < scores[2] < 1

    dataset = pd.DataFrame({
        "Complaint ID": range(1, 8),
        "Consumer complaint narrative": fitting_texts + test_texts,
        "Product": ["Credit card", "Mortgage"] * 2 + ["Debt collection", "Credit card", "Mortgage"],
    })
    manifest = pd.DataFrame({
        "Complaint ID": dataset["Complaint ID"],
        "split": ["train", "train", "validation", "validation", "test", "test", "test"],
    })
    csv_path = tmp_path / "complaints.csv"
    manifest_path = tmp_path / "manifest.csv"
    dataset.to_csv(csv_path, index=False)
    manifest.to_csv(manifest_path, index=False)
    manifest_path.with_suffix(".json").write_text(json.dumps({
        "dataset_fingerprint": compute_dataset_fingerprint(dataset),
        "manifest_fingerprint": compute_manifest_fingerprint(manifest),
        "fingerprint_algorithm": "sha256",
        "total_samples": len(dataset),
        "split_counts": manifest["split"].value_counts().to_dict(),
    }), encoding="utf-8")

    report = analyze_near_duplicates(
        str(csv_path), str(tmp_path / "diagnostic.json"), str(manifest_path),
    )

    assert report["group_performance"]["high_similarity"]["samples"] == 1
    assert report["group_performance"]["lower_similarity"]["samples"] == 2
    assert report["group_class_distribution"] == {
        "high_similarity": {"Mortgage": 1},
        "lower_similarity": {"Debt collection": 1, "Credit card": 1},
    }
    assert report["threshold_counts"] == {"0.80": 2, "0.90": 2, "0.95": 1, "0.99": 0}
    assert report["mean_max_similarity"] == pytest.approx(scores.mean())
    assert report["median_max_similarity"] == pytest.approx(scores[1])
