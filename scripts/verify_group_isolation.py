"""Read-only isolation check for current frozen assignments under V1 grouping.

V1 uses full-dataset unigram/bigram TF-IDF (default min_df=1), cosine >= 0.95,
and connected components. This transductive check does not authenticate the
manifest's historical origin or establish isolation under classifier TF-IDF.
Results are printed to stdout; no artifacts are written.
"""

import argparse
import json

import pandas as pd

from src.data_integrity import compute_dataset_fingerprint, compute_manifest_fingerprint
from src.grouping import create_text_groups
from src.train import load_frozen_split


def _load_verified_frames(data_path, manifest_path):
    metadata = load_frozen_split(data_path, manifest_path, include_metadata=True)[-1]
    dataset = pd.read_csv(data_path)
    manifest = pd.read_csv(manifest_path)
    # Separate reads must match the snapshots already validated by the frozen loader.
    if compute_dataset_fingerprint(dataset) != metadata["dataset_fingerprint"]:
        raise ValueError("Dataset fingerprint changed after frozen-loader validation.")
    if compute_manifest_fingerprint(manifest) != metadata["manifest_fingerprint"]:
        raise ValueError("Manifest fingerprint changed after frozen-loader validation.")
    return dataset, manifest, metadata


def verify_group_isolation(data_path, manifest_path):
    """Validate frozen artifacts and report V1 group overlap without changing them."""
    dataset, manifest, metadata = _load_verified_frames(data_path, manifest_path)
    groups = create_text_groups(
        dataset["Consumer complaint narrative"], threshold=0.95, batch_size=100,
    )
    assignments = pd.DataFrame({
        "Complaint ID": dataset["Complaint ID"],
        "group": groups,
    }).merge(manifest[["Complaint ID", "split"]], on="Complaint ID", validate="one_to_one")
    group_sets = {
        split: set(assignments.loc[assignments["split"] == split, "group"])
        for split in ("train", "validation", "test")
    }
    overlaps = {
        "train_validation": len(group_sets["train"] & group_sets["validation"]),
        "train_test": len(group_sets["train"] & group_sets["test"]),
        "validation_test": len(group_sets["validation"] & group_sets["test"]),
    }
    return {
        "verification_status": "failed" if any(overlaps.values()) else "passed",
        "verification_scope": "current_frozen_assignments_v1_grouping",
        "historical_membership_authentication": "not_independently_authenticated",
        "similarity_threshold": 0.95,
        "total_records": len(dataset),
        "total_groups": int(assignments["group"].nunique()),
        "split_counts": assignments["split"].value_counts().to_dict(),
        "shared_group_counts": overlaps,
        "dataset_fingerprint": metadata["dataset_fingerprint"],
        "manifest_fingerprint": metadata["manifest_fingerprint"],
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help="Cleaned complaint CSV")
    parser.add_argument("--manifest", required=True, help="Frozen manifest CSV with adjacent .json metadata")
    args = parser.parse_args(argv)
    try:
        report = verify_group_isolation(args.data, args.manifest)
    except (OSError, ValueError) as error:
        print(json.dumps({"verification_status": "error", "error": str(error)}, allow_nan=False))
        return 2
    print(json.dumps(report, indent=2, allow_nan=False))
    return 0 if report["verification_status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
