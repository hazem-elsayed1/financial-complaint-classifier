

from pathlib import Path
import json
import pandas as pd
from src.data_integrity import (
    compute_dataset_fingerprint,
    compute_manifest_fingerprint,
)
from src.train import split_data

DATA_PATH = Path("data/processed/cleaned_complaints.csv")
OUTPUT_PATH = Path("data/splits/group_split_manifest.csv")


def main():

    metadata_path = OUTPUT_PATH.with_suffix(".json")

    if OUTPUT_PATH.exists() or metadata_path.exists():
        raise FileExistsError(
            "Split manifest or metadata already exists."
        )

    df = pd.read_csv(DATA_PATH)

    X = df["Consumer complaint narrative"]
    y = df["Product"]

    X_train, X_val, X_test, _, _, _ = split_data(X, y)

    manifest = pd.DataFrame({
        "Complaint ID": df["Complaint ID"],
        "split": "",
    })

    manifest.loc[X_train.index, "split"] = "train"
    manifest.loc[X_val.index, "split"] = "validation"
    manifest.loc[X_test.index, "split"] = "test"

    if not manifest["Complaint ID"].is_unique:
        raise ValueError("Duplicate Complaint IDs detected.")

    if manifest["Complaint ID"].isna().any():
        raise ValueError("Missing Complaint IDs detected.")

    valid_splits = {"train", "validation", "test"}

    if set(manifest["split"]) != valid_splits:
        raise ValueError("Invalid split assignments.")

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )


    metadata = {
        "dataset_fingerprint": compute_dataset_fingerprint(df),
        "manifest_fingerprint": compute_manifest_fingerprint(manifest),
        "fingerprint_algorithm": "sha256",
        "fingerprint_columns": [
            "Complaint ID",
            "Consumer complaint narrative",
            "Product",
        ],
        "total_samples": len(df),
        "split_counts": {
            str(name): int(count)
            for name, count in manifest["split"].value_counts().items()
        },
    }

    manifest.to_csv(OUTPUT_PATH, index=False)

    with metadata_path.open("x", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2)

    print(manifest["split"].value_counts())
    print(f"Manifest saved to {OUTPUT_PATH}")
    print(f"Metadata saved to {metadata_path}")



if __name__ == "__main__":
    main()
