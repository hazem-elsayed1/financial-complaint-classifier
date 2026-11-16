
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, f1_score

from src.grouping import create_text_groups, group_aware_split
from src.train import load_training_data, train_model


def evaluate_group_aware(
    csv_path="data/processed/cleaned_complaints.csv",
    report_path="reports/group_aware_evaluation.json",
    threshold=0.95,
    batch_size=100,
    *,
    overwrite: bool = False,
):
    """Evaluate a newly generated split; replace reports only with overwrite=True."""
    path = Path(report_path)
    if not overwrite and (path.exists() or path.is_symlink()):
        raise FileExistsError(
            f"Report already exists: {path}. Choose a new path or pass overwrite=True."
        )

    X, y = load_training_data(csv_path)

    groups = create_text_groups(
        X,
        threshold=threshold,
        batch_size=batch_size,
    )

    train_idx, test_idx = group_aware_split(
        X,
        y,
        groups,
        test_size=0.20,
        random_state=42,
    )

    X_train = X.iloc[train_idx]
    X_test = X.iloc[test_idx]
    y_train = y.iloc[train_idx]
    y_test = y.iloc[test_idx]

    model = train_model(X_train, y_train)
    predictions = model.predict(X_test)

    report = {
        "evaluation_type": "near_duplicate_group_aware",
        "split_source": "newly_generated_group_shuffle_split",
        "similarity_threshold": threshold,
        "total_samples": int(len(X)),
        "total_groups": int(len(np.unique(groups))),
        "train_samples": int(len(train_idx)),
        "test_samples": int(len(test_idx)),
        "train_groups": int(len(np.unique(groups[train_idx]))),
        "test_groups": int(len(np.unique(groups[test_idx]))),
        "group_overlap": int(
            len(
                set(groups[train_idx])
                & set(groups[test_idx])
            )
        ),
        "accuracy": float(
            accuracy_score(y_test, predictions)
        ),
        "macro_f1": float(
            f1_score(
                y_test,
                predictions,
                average="macro",
                zero_division=0,
            )
        ),
        "train_class_distribution": {
            str(k): int(v)
            for k, v in y_train.value_counts().items()
        },
        "test_class_distribution": {
            str(k): int(v)
            for k, v in y_test.value_counts().items()
        },
    }

    serialized = json.dumps(report, indent=2, allow_nan=False)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w" if overwrite else "x", encoding="utf-8", newline="\n") as file:
        file.write(serialized + "\n")

    return report


if __name__ == "__main__":
    examples = [
        "My credit card was charged twice",
        "My credit card was charged twice",
        "My mortgage payment is incorrect",
        "My student loan balance is wrong",
    ]

    groups = create_text_groups(examples)

    for text, group in zip(examples, groups):
        print(f"Group {group}: {text}")
