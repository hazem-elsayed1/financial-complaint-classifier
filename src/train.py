
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from pathlib import Path
import joblib
import json
import re
from src.data_processing import has_product_label
from src.data_integrity import (
    compute_manifest_fingerprint,
    verify_dataset_fingerprint,
)
def build_model() -> Pipeline:
    """Create the selected TF-IDF + Logistic Regression pipeline."""

    model = Pipeline([
        ("tfidf", TfidfVectorizer(
            min_df=2,
            ngram_range=(1, 2)
        )),
        ("classifier", LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            C=2.0
        ))
    ])

    return model

def train_model(X_train, y_train) -> Pipeline:
    """Build and train the classifier."""

    if not all(has_product_label(label) for label in y_train):
        raise ValueError(
    "All Product labels must be non-empty strings.")

    model = build_model()

    model.fit(X_train, y_train)

    return model

def load_training_data(csv_path: str):
    """Load cleaned complaint texts and labels."""

    df = pd.read_csv(csv_path)

    X = df["Consumer complaint narrative"]
    y = df["Product"]

    return X, y

def _validate_partitions(train_idx, val_idx, test_idx, groups):
    # Reject duplicate indices within each partition.
    for indices in (train_idx, val_idx, test_idx):
        if len(indices) != len(set(indices)):
            raise RuntimeError(
                "Duplicate indices found within a partition."
            )

    # Check overlap between partitions.
    train_set, val_set, test_set = (
        set(indices) for indices in (train_idx, val_idx, test_idx)
    )

    if train_set & val_set or train_set & test_set or val_set & test_set:
        raise RuntimeError("Train, validation, and test indices overlap.")

    # Check complete dataset coverage.
    if train_set | val_set | test_set != set(range(len(groups))):
        raise RuntimeError("Partitions do not cover the dataset exactly.")

    # Check near-duplicate group isolation.
    train_groups, val_groups, test_groups = (
        set(groups[indices]) for indices in (train_idx, val_idx, test_idx)
    )

    if (
        train_groups & val_groups
        or train_groups & test_groups
        or val_groups & test_groups
    ):
        raise RuntimeError("Near-duplicate groups cross partition boundaries.")

def split_data(
    X,
    y,
    test_size=0.20,
    val_size=0.20,
    random_state=42,
    similarity_threshold=0.95,
):
    """
    Create group-disjoint train, validation, and test splits.

    val_size is measured relative to the train+validation
    portion, preserving the original split convention.

    Group construction is unsupervised and uses all narratives.
    The predictive TF-IDF pipeline is fitted only on training data.
    """

    from src.grouping import (
        create_text_groups,
        group_aware_split,
    )

    if len(X) != len(y):
        raise ValueError("X and y must have equal lengths.")

    if not 0 < test_size < 1:
        raise ValueError("Invalid test_size.")

    if not 0 < val_size < 1:
        raise ValueError("Invalid val_size.")

    # Group similar narratives before assigning partitions.
    groups = create_text_groups(
        X,
        threshold=similarity_threshold,
    )

    # Stage 1: Train+validation versus held-out test.
    train_val_idx, test_idx = group_aware_split(
        X,
        y,
        groups,
        test_size=test_size,
        random_state=random_state,
    )

    # Stage 2: Separate validation from training,
    # while respecting the original group assignments.
    train_relative_idx, val_relative_idx = group_aware_split(
        X.iloc[train_val_idx],
        y.iloc[train_val_idx],
        groups[train_val_idx],
        test_size=val_size,
        random_state=random_state + 1,
    )

    train_idx = train_val_idx[train_relative_idx]
    val_idx = train_val_idx[val_relative_idx]

    _validate_partitions(train_idx, val_idx, test_idx, groups)

    # Model training must have every target class.
    all_classes = set(y)
    train_classes = set(y.iloc[train_idx])

    if train_classes != all_classes:
        missing = all_classes - train_classes
        raise ValueError(
            f"Training split is missing classes: {missing}"
        )

    return (
        X.iloc[train_idx],
        X.iloc[val_idx],
        X.iloc[test_idx],
        y.iloc[train_idx],
        y.iloc[val_idx],
        y.iloc[test_idx],
    )



def load_frozen_split(
    csv_path: str,
    manifest_path: str = "data/splits/group_split_manifest.csv",
    include_metadata: bool = False,
):
    """Load train/validation/test data using saved complaint IDs."""

    df = pd.read_csv(csv_path)
    manifest = pd.read_csv(manifest_path)


    metadata_path = Path(manifest_path).with_suffix(".json")

    verify_dataset_fingerprint(
        df,
        str(metadata_path),
    )

    actual_manifest_fingerprint = compute_manifest_fingerprint(manifest)
    with metadata_path.open("r", encoding="utf-8") as file:
        metadata = json.load(file)
    expected_manifest_fingerprint = metadata.get("manifest_fingerprint")
    if (
        not isinstance(expected_manifest_fingerprint, str)
        or not expected_manifest_fingerprint
    ):
        raise ValueError("Missing manifest fingerprint in metadata.")
    if re.fullmatch(r"[0-9a-f]{64}", expected_manifest_fingerprint) is None:
        raise ValueError("Malformed manifest fingerprint in metadata.")
    if actual_manifest_fingerprint != expected_manifest_fingerprint:
        raise ValueError(
            "Manifest fingerprint mismatch. Split assignments may have changed."
        )

    if df["Complaint ID"].isna().any():
        raise ValueError("Dataset contains missing Complaint IDs.")

    if not df["Complaint ID"].is_unique:
        raise ValueError("Dataset contains duplicate Complaint IDs.")

    if manifest["Complaint ID"].isna().any():
        raise ValueError("Manifest contains missing Complaint IDs.")

    if not manifest["Complaint ID"].is_unique:
        raise ValueError("Manifest contains duplicate Complaint IDs.")

    valid_splits = {"train", "validation", "test"}

    if set(manifest["split"]) != valid_splits:
        raise ValueError("Manifest has missing or invalid splits.")

    if set(df["Complaint ID"]) != set(manifest["Complaint ID"]):
        raise ValueError("Dataset and manifest Complaint IDs differ.")

    if type(metadata.get("total_samples")) is not int or metadata["total_samples"] != len(df):
        raise ValueError("Metadata total_samples must be an integer matching the dataset.")
    split_counts = metadata.get("split_counts")
    if (
        not isinstance(split_counts, dict)
        or any(type(count) is not int for count in split_counts.values())
        or split_counts != manifest["split"].value_counts().to_dict()
    ):
        raise ValueError("Metadata split_counts must match the manifest's integer counts.")

    merged = df.merge(
        manifest,
        on="Complaint ID",
        validate="one_to_one",
    )

    text_column = "Consumer complaint narrative"
    label_column = "Product"

    def select_split(name):
        subset = merged.loc[merged["split"] == name]

        return (
            subset[text_column],
            subset[label_column],
        )

    X_train, y_train = select_split("train")
    X_val, y_val = select_split("validation")
    X_test, y_test = select_split("test")

    splits = (
    X_train,
    X_val,
    X_test,
    y_train,
    y_val,
    y_test,
)

    if include_metadata:
        return (*splits, metadata)

    return splits


def run_training(csv_path: str):
    """Load data, split it, and train the classifier."""

    X, y = load_training_data(csv_path)

    X_train, X_val, X_test, y_train, y_val, y_test = split_data(
        X, y
    )

    model = train_model(X_train, y_train)

    return model, X_val, y_val, X_test, y_test

def evaluate_model(model, X_eval, y_eval) -> dict:
    """Evaluate a trained classifier."""

    predictions = model.predict(X_eval)

    accuracy = accuracy_score(y_eval, predictions)

    macro_f1 = f1_score(
        y_eval,
        predictions,
        average="macro"
    )

    return {
        "accuracy": accuracy,
        "macro_f1": macro_f1
    }

def save_model(model: Pipeline, model_path: str) -> None:
    """Save a trained pipeline to disk."""

    path = Path(model_path)

    path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    joblib.dump(model, path)


def train_final_model(csv_path: str) -> Pipeline:
    """Train the selected model on train + validation data."""

    model, _, _ = train_final_model_with_test_data(csv_path)

    return model



def train_final_model_with_test_data(
    csv_path: str,
    manifest_path: str = "data/splits/group_split_manifest.csv",
    include_metadata: bool = False,
):
    """Train on frozen train+validation and return held-out test data."""

    if include_metadata:
        splits = load_frozen_split(
        csv_path,
        manifest_path,
        include_metadata=True,
    )
    else:
        splits = load_frozen_split(
        csv_path,
        manifest_path,
    )
    if include_metadata:
        (
            X_train,
            X_val,
            X_test,
            y_train,
            y_val,
            y_test,
            metadata,
        ) = splits
    else:
        (
            X_train,
            X_val,
            X_test,
            y_train,
            y_val,
            y_test,
        ) = splits

    X_train_val = pd.concat([X_train, X_val])
    y_train_val = pd.concat([y_train, y_val])

    model = train_model(X_train_val, y_train_val)

    if include_metadata:
        return model, X_test, y_test, metadata

    return model, X_test, y_test
