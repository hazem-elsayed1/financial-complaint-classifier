
import json
from pathlib import Path

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

from src.train import train_final_model_with_test_data


def run_evaluation(
    csv_path: str,
    report_path: str = "reports/test_metrics.json",
    manifest_path: str = "data/splits/group_split_manifest.csv",
) -> dict:
    """Train and evaluate using a verified frozen holdout split."""

    model, X_test, y_test, metadata = (
        train_final_model_with_test_data(
            csv_path,
            manifest_path,
            include_metadata=True,
        )
    )

    predictions = model.predict(X_test)
    labels = [str(label) for label in model.classes_]

    split_counts = metadata["split_counts"]

    provenance = {
        "evaluation_type": "frozen_holdout",
        "split_source": "frozen_complaint_id_manifest",
        "dataset_fingerprint": metadata["dataset_fingerprint"],
        "manifest_fingerprint": metadata["manifest_fingerprint"],
        "fingerprint_algorithm": metadata["fingerprint_algorithm"],
        "total_samples": metadata["total_samples"],
        "split_counts": split_counts,
        "model_fit_samples": (
            split_counts["train"] + split_counts["validation"]
        ),
        "historical_membership_authentication": (
            "not_independently_authenticated"
        ),
    }

    results = {
        **provenance,
        "test_samples": int(len(y_test)),
        "test_class_distribution": {
            str(label): int(count)
            for label, count in y_test.value_counts().items()
        },
        "accuracy": float(
            accuracy_score(y_test, predictions)
        ),
        "macro_f1": float(
            f1_score(
                y_test,
                predictions,
                labels=labels,
                average="macro",
                zero_division=0,
            )
        ),
        "labels": labels,
        "classification_report": classification_report(
            y_test,
            predictions,
            labels=labels,
            output_dict=True,
            zero_division=0,
        ),
        "confusion_matrix": confusion_matrix(
            y_test,
            predictions,
            labels=labels,
        ).tolist(),
    }

    path = Path(report_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8", newline="\n") as file:
        json.dump(
            results,
            file,
            indent=2,
            allow_nan=False,
        )
        file.write("\n")
    return results


if __name__ == "__main__":
    results = run_evaluation(
        "data/processed/cleaned_complaints.csv"
    )

    print(f"Test samples: {results['test_samples']}")
    print(f"Accuracy: {results['accuracy']:.4f}")
    print(f"Macro F1: {results['macro_f1']:.4f}")
    print("Report saved to reports/test_metrics.json")
