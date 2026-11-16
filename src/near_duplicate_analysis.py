
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from sklearn.metrics.pairwise import cosine_similarity

from src.train import (
    load_frozen_split,
    train_model,
)


def maximum_cosine_similarities(
    test_vectors,
    train_vectors,
    batch_size=100,
):
    """Return the highest training similarity for each test sample."""

    max_similarities = []

    for start in range(0, test_vectors.shape[0], batch_size):
        batch = test_vectors[start:start + batch_size]

        similarities = cosine_similarity(
            batch,
            train_vectors,
            dense_output=False,
        )

        max_values = similarities.max(axis=1).toarray().ravel()
        max_similarities.extend(max_values.tolist())

    return np.asarray(max_similarities)


def _evaluate_subset(y_test, predictions, mask):
    if not mask.any():
        return {"samples": 0, "accuracy": None, "macro_f1": None}
    return {
        "samples": int(mask.sum()),
        "accuracy": float(accuracy_score(y_test.iloc[mask], predictions[mask])),
        "macro_f1": float(f1_score(
            y_test.iloc[mask],
            predictions[mask],
            average="macro",
            zero_division=0,
        )),
    }


def _evaluate_debt_collection_subset(y_test, predictions, mask):
    """On true Debt collection examples, subset accuracy is class recall."""
    debt_mask = mask & (y_test.to_numpy() == "Debt collection")
    metrics = _evaluate_subset(y_test, predictions, debt_mask)
    return {
        **metrics,
        "recall": metrics["accuracy"],
        "macro_f1_note": (
            "Deprecated legacy subset macro-F1 over true and predicted labels; "
            "not comparable to overall 11-class macro-F1."
        ),
    }


def analyze_near_duplicates(
    csv_path="data/processed/cleaned_complaints.csv",
    report_path="reports/near_duplicate_frozen_split.json",
    manifest_path="data/splits/group_split_manifest.csv",
):
    """Measure similarity and performance on one verified frozen split.

    Diagnostic similarities use the classifier's TF-IDF representation fitted
    on train plus validation, excluding test. This differs from the full-dataset
    TF-IDF representation used to generate near-duplicate groups.
    """
    path = Path(report_path)
    if path.name.casefold() == "near_duplicate_analysis.json":
        raise ValueError(
            "Historical near-duplicate report is protected; choose a new path."
        )

    X_train, X_val, X_test, y_train, y_val, y_test = load_frozen_split(
        csv_path, manifest_path,
    )
    X_train_val = pd.concat([X_train, X_val])
    y_train_val = pd.concat([y_train, y_val])
    if X_train_val.empty:
        raise ValueError("Training partition (train plus validation) must not be empty.")
    if X_test.empty:
        raise ValueError("Test partition must not be empty.")
    model = train_model(X_train_val, y_train_val)
    vectorizer = model.named_steps["tfidf"]

    train_vectors = vectorizer.transform(X_train_val)
    test_vectors = vectorizer.transform(X_test)

    scores = maximum_cosine_similarities(
        test_vectors,
        train_vectors,
    )

    predictions = model.predict(X_test)

    high_similarity_mask = scores >= 0.95
    lower_similarity_mask = ~high_similarity_mask

    group_performance = {
        "high_similarity": _evaluate_subset(y_test, predictions, high_similarity_mask),
        "lower_similarity": _evaluate_subset(y_test, predictions, lower_similarity_mask),
    }

    group_class_distribution = {
        "high_similarity": {
            str(k): int(v)
            for k, v in y_test.iloc[
                high_similarity_mask
            ].value_counts().items()
        },
        "lower_similarity": {
            str(k): int(v)
            for k, v in y_test.iloc[
                lower_similarity_mask
            ].value_counts().items()
        },
    }

    debt_collection_performance = {
        "high_similarity": _evaluate_debt_collection_subset(
            y_test, predictions, high_similarity_mask
        ),
        "lower_similarity": _evaluate_debt_collection_subset(
            y_test, predictions, lower_similarity_mask
        ),
    }

    report = {
        "evaluation_type": "frozen_holdout",
        "split_strategy": "frozen_complaint_id_manifest",
        "similarity_threshold": 0.95,
        "train_samples": len(X_train_val),
        "test_samples": len(X_test),
        "similarity_method": "TF-IDF cosine similarity",
        "similarity_vectorizer": "model_fitted_tfidf",
        "similarity_reference": "train_plus_validation",
        "similarity_fit_excludes_test": True,
        "mean_max_similarity": float(np.mean(scores)),
        "median_max_similarity": float(np.median(scores)),
        "threshold_counts": {
            "0.80": int(np.sum(scores >= 0.80)),
            "0.90": int(np.sum(scores >= 0.90)),
            "0.95": int(np.sum(scores >= 0.95)),
            "0.99": int(np.sum(scores >= 0.99)),
        },
        "group_performance": group_performance,
        "group_class_distribution": group_class_distribution,
        "debt_collection_performance": debt_collection_performance,
    }

    path.parent.mkdir(parents=True, exist_ok=True)

    path.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    return report


if __name__ == "__main__":
    results = analyze_near_duplicates()
    print(json.dumps(results, indent=2))
