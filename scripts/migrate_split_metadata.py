"""Propose frozen metadata updates; never replace the original metadata.

Legacy pd.read_csv parsing infers integer Complaint IDs, which the existing
dataset fingerprint serializes as JSON numbers (101, not "101" or 101.0).
This utility retains that parser and hash format. CSV ID tokens must already
be canonical decimal integers: no leading zeros, whitespace or float syntax.

Without an independently trusted historical manifest, validating the dataset
and split counts cannot authenticate historical ID-to-split membership.
"""

import argparse
import json
from pathlib import Path
import re
import warnings

import pandas as pd
from pandas.api.types import is_integer_dtype

from src.data_integrity import (
    compute_dataset_fingerprint,
    compute_manifest_fingerprint,
)


def read_legacy_csv(path):
    """Reject ambiguous ID representations instead of normalizing them."""
    frame = pd.read_csv(path)
    if "Complaint ID" not in frame:
        raise ValueError(f"Missing Complaint ID column: {path}")
    ids = frame["Complaint ID"]
    if not is_integer_dtype(ids.dtype) or ids.isna().any():
        raise ValueError(f"Complaint IDs must use legacy integer representation: {path}")
    raw_ids = pd.read_csv(
        path, usecols=["Complaint ID"], dtype=str, keep_default_na=False,
    )["Complaint ID"]
    if raw_ids.tolist() != ids.map(str).tolist():
        raise ValueError(f"Inconsistent Complaint ID representation: {path}")
    return frame


def validate_split_counts(metadata, manifest):
    counts = metadata.get("split_counts")
    if (
        not isinstance(counts, dict)
        or any(type(count) is not int for count in counts.values())
        or counts != manifest["split"].value_counts().to_dict()
    ):
        raise ValueError("Saved split_counts do not match the manifest.")
    if "total_samples" not in metadata:
        raise ValueError("Missing total_samples in legacy metadata.")
    if (
        type(metadata["total_samples"]) is not int
        or metadata["total_samples"] != len(manifest)
    ):
        raise ValueError("Saved total_samples does not match the manifest.")


def validate_historical_manifest(manifest, manifest_path, trusted_manifest_path):
    if trusted_manifest_path is None:
        warnings.warn(
            "Historical split membership cannot be independently authenticated: "
            "no trusted historical manifest supplied. Dataset fingerprint "
            "verification does not prove that split assignments never changed.",
            UserWarning,
            stacklevel=2,
        )
        return
    if Path(manifest_path).samefile(trusted_manifest_path):
        raise ValueError("Trusted historical manifest must be a separate file.")
    trusted = read_legacy_csv(trusted_manifest_path)
    compute_manifest_fingerprint(trusted)
    current_mapping = manifest.set_index("Complaint ID")["split"].sort_index()
    trusted_mapping = trusted.set_index("Complaint ID")["split"].sort_index()
    if not current_mapping.equals(trusted_mapping):
        raise ValueError("Manifest assignments differ from the trusted historical manifest.")


def propose_metadata(data_path, manifest_path, metadata_path, trusted_manifest_path=None):
    """Validate existing artifacts and return a proposal without writing files."""
    with Path(metadata_path).open("r", encoding="utf-8") as file:
        metadata = json.load(file)
    if not isinstance(metadata, dict):
        raise ValueError("Metadata must be a JSON object.")
    expected = metadata.get("dataset_fingerprint")
    if not isinstance(expected, str) or re.fullmatch(r"[0-9a-f]{64}", expected) is None:
        raise ValueError("Missing or malformed dataset fingerprint.")
    if metadata.get("fingerprint_algorithm") != "sha256":
        raise ValueError("Unsupported fingerprint algorithm.")
    dataset = read_legacy_csv(data_path)
    if compute_dataset_fingerprint(dataset) != expected:
        raise ValueError("Dataset fingerprint mismatch.")
    manifest = read_legacy_csv(manifest_path)
    fingerprint = compute_manifest_fingerprint(manifest)
    if set(dataset["Complaint ID"]) != set(manifest["Complaint ID"]):
        raise ValueError("Dataset and manifest Complaint IDs differ.")
    validate_split_counts(metadata, manifest)
    if "manifest_fingerprint" in metadata and metadata["manifest_fingerprint"] != fingerprint:
        raise ValueError("Conflicting existing manifest fingerprint.")
    validate_historical_manifest(manifest, manifest_path, trusted_manifest_path)
    return {**metadata, "manifest_fingerprint": fingerprint}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--metadata", required=True, type=Path)
    parser.add_argument("--trusted-manifest", type=Path)
    parser.add_argument("--output", type=Path, help="New candidate JSON; never overwritten")
    args = parser.parse_args(argv)
    proposal = propose_metadata(
        args.data, args.manifest, args.metadata, args.trusted_manifest,
    )
    serialized = json.dumps(proposal, indent=2, ensure_ascii=False, allow_nan=False)
    if args.output is not None:
        with args.output.open("x", encoding="utf-8") as file:
            file.write(serialized + "\n")
    print(serialized)


if __name__ == "__main__":
    main()
