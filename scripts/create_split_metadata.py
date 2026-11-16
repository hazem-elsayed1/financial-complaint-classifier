
import json
from pathlib import Path

import pandas as pd

from src.data_integrity import (
    compute_dataset_fingerprint,
    compute_manifest_fingerprint,
)


DATA_PATH = Path("data/processed/cleaned_complaints.csv")
MANIFEST_PATH = Path("data/splits/group_split_manifest.csv")
METADATA_PATH = MANIFEST_PATH.with_suffix(".json")


def main():
    if not MANIFEST_PATH.exists():
        raise FileNotFoundError("Split manifest does not exist.")

    if METADATA_PATH.exists():
        raise FileExistsError("Split metadata already exists.")

    df = pd.read_csv(DATA_PATH)
    manifest = pd.read_csv(MANIFEST_PATH)
    manifest_fingerprint = compute_manifest_fingerprint(manifest)

    if df["Complaint ID"].isna().any():
        raise ValueError("Dataset contains missing IDs.")

    if not df["Complaint ID"].is_unique:
        raise ValueError("Dataset contains duplicate IDs.")

    if manifest["Complaint ID"].isna().any():
        raise ValueError("Manifest contains missing IDs.")

    if not manifest["Complaint ID"].is_unique:
        raise ValueError("Manifest contains duplicate IDs.")

    if set(df["Complaint ID"]) != set(manifest["Complaint ID"]):
        raise ValueError("Dataset and manifest IDs do not match.")

    valid_splits = {"train", "validation", "test"}

    if set(manifest["split"]) != valid_splits:
        raise ValueError("Invalid split labels.")

    metadata = {
        "dataset_fingerprint": compute_dataset_fingerprint(df),
        "manifest_fingerprint": manifest_fingerprint,
        "fingerprint_algorithm": "sha256",
        "fingerprint_columns": [
            "Complaint ID",
            "Consumer complaint narrative",
            "Product",
        ],
        "total_samples": int(len(df)),
        "split_counts": {
            str(name): int(count)
            for name, count in manifest["split"].value_counts().items()
        },
    }

    with METADATA_PATH.open("x", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2)

    print(f"Metadata saved to {METADATA_PATH}")


if __name__ == "__main__":
    main()
