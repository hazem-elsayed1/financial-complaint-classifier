import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

from src.data_integrity import compute_dataset_fingerprint, compute_manifest_fingerprint


def write_metadata(data_path, manifest_path):
    dataset = pd.read_csv(data_path)
    manifest = pd.read_csv(manifest_path)
    manifest_path.with_suffix(".json").write_text(json.dumps({
        "dataset_fingerprint": compute_dataset_fingerprint(dataset),
        "manifest_fingerprint": compute_manifest_fingerprint(manifest),
        "fingerprint_algorithm": "sha256",
        "total_samples": len(dataset),
        "split_counts": manifest["split"].value_counts().to_dict(),
    }), encoding="utf-8")


@pytest.fixture
def frozen_files(tmp_path):
    dataset = pd.DataFrame({
        "Complaint ID": [301, 101, 201, 302, 102, 202],
        "Consumer complaint narrative": [
            "Debt collector keeps calling", "Credit card charged twice", "Mortgage escrow balance",
            "Debt collector keeps calling", "Credit card charged twice", "Mortgage escrow balance",
        ],
        "Product": ["Debt collection", "Credit card", "Mortgage"] * 2,
    })
    manifest = pd.DataFrame({
        "Complaint ID": [202, 301, 102, 201, 302, 101],
        "split": ["validation", "test", "train", "validation", "test", "train"],
    })
    data_path = tmp_path / "complaints.csv"
    manifest_path = tmp_path / "manifest.csv"
    dataset.to_csv(data_path, index=False)
    manifest.to_csv(manifest_path, index=False)
    write_metadata(data_path, manifest_path)
    return data_path, manifest_path


def run_verifier(data_path, manifest_path):
    originals = {path: path.read_bytes() for path in data_path.parent.iterdir()}
    process = subprocess.run(
        [sys.executable, "-B", "-m", "scripts.verify_group_isolation",
         "--data", str(data_path), "--manifest", str(manifest_path)],
        capture_output=True, text=True, check=False,
    )
    assert {path: path.read_bytes() for path in data_path.parent.iterdir()} == originals
    return process.returncode, json.loads(process.stdout)


def test_valid_manifest_passes_v1_group_isolation_without_writes(frozen_files):
    status, report = run_verifier(*frozen_files)
    assert status == 0
    assert report["verification_status"] == "passed"
    assert report["total_records"] == 6
    assert report["total_groups"] == 3
    assert report["split_counts"] == {"train": 2, "validation": 2, "test": 2}
    assert report["shared_group_counts"] == {"train_validation": 0, "train_test": 0, "validation_test": 0}
    assert report["similarity_threshold"] == 0.95
    assert report["verification_scope"] == "current_frozen_assignments_v1_grouping"
    assert report["historical_membership_authentication"] == "not_independently_authenticated"


@pytest.mark.parametrize(
    "first_id, second_id, shared_pair",
    [(102, 201, "train_validation"), (102, 301, "train_test"), (202, 301, "validation_test")],
)
def test_cross_partition_groups_fail_with_matching_fingerprints(
    frozen_files, first_id, second_id, shared_pair,
):
    data_path, manifest_path = frozen_files
    manifest = pd.read_csv(manifest_path).set_index("Complaint ID")
    manifest.loc[[first_id, second_id], "split"] = manifest.loc[[second_id, first_id], "split"].to_numpy()
    manifest.reset_index().to_csv(manifest_path, index=False)
    write_metadata(data_path, manifest_path)

    status, report = run_verifier(data_path, manifest_path)

    assert status == 1
    assert report["verification_status"] == "failed"
    assert report["total_groups"] == 3
    assert report["split_counts"] == {"train": 2, "validation": 2, "test": 2}
    expected = {"train_validation": 0, "train_test": 0, "validation_test": 0}
    expected[shared_pair] = 2
    assert report["shared_group_counts"] == expected


def test_row_reordering_preserves_id_aligned_verification(frozen_files):
    data_path, manifest_path = frozen_files
    first = run_verifier(data_path, manifest_path)
    pd.read_csv(data_path).sample(frac=1, random_state=17).to_csv(data_path, index=False)
    pd.read_csv(manifest_path).sample(frac=1, random_state=43).to_csv(manifest_path, index=False)
    assert run_verifier(data_path, manifest_path) == first


@pytest.mark.parametrize("tamper", ["assignments", "saved_digest"])
def test_manifest_fingerprint_violation_fails_before_group_verification(frozen_files, tamper):
    data_path, manifest_path = frozen_files
    metadata_path = manifest_path.with_suffix(".json")
    if tamper == "assignments":
        manifest = pd.read_csv(manifest_path).set_index("Complaint ID")
        manifest.loc[[102, 301], "split"] = ["test", "train"]
        manifest.reset_index().to_csv(manifest_path, index=False)
    else:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        metadata["manifest_fingerprint"] = "0" * 64
        metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    status, report = run_verifier(data_path, manifest_path)

    assert status == 2
    assert report["verification_status"] == "error"
    assert "Manifest fingerprint mismatch" in report["error"]
    assert "total_groups" not in report


@pytest.mark.parametrize("tamper", ["dataset", "metadata_counts"])
def test_verifier_retains_frozen_loader_integrity_checks(frozen_files, tamper):
    data_path, manifest_path = frozen_files
    if tamper == "dataset":
        dataset = pd.read_csv(data_path)
        dataset.loc[0, "Consumer complaint narrative"] = "Changed complaint narrative"
        dataset.to_csv(data_path, index=False)
        expected_error = "Dataset fingerprint mismatch"
    else:
        metadata_path = manifest_path.with_suffix(".json")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        metadata["total_samples"] = 7
        metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
        expected_error = "total_samples"

    status, report = run_verifier(data_path, manifest_path)

    assert status == 2
    assert report["verification_status"] == "error"
    assert expected_error in report["error"]


@pytest.mark.parametrize("changed_artifact", ["dataset", "manifest"])
def test_changed_frames_after_loader_validation_are_rejected(
    frozen_files, monkeypatch, changed_artifact,
):
    from scripts.verify_group_isolation import verify_group_isolation

    data_path, manifest_path = frozen_files
    target = data_path if changed_artifact == "dataset" else manifest_path
    originals = {path: path.read_bytes() for path in data_path.parent.iterdir()}
    original_read_csv = pd.read_csv
    reads = 0

    def read_changed_frame(path, *args, **kwargs):
        nonlocal reads
        frame = original_read_csv(path, *args, **kwargs)
        if Path(path) == target:
            reads += 1
            if reads == 2:
                if changed_artifact == "dataset":
                    frame.loc[0, "Consumer complaint narrative"] = "Changed after validation"
                else:
                    frame.loc[frame["Complaint ID"] == 102, "split"] = "test"
                    frame.loc[frame["Complaint ID"] == 301, "split"] = "train"
        return frame

    monkeypatch.setattr(pd, "read_csv", read_changed_frame)
    with pytest.raises(ValueError, match="fingerprint changed after frozen-loader validation"):
        verify_group_isolation(data_path, manifest_path)
    assert {path: path.read_bytes() for path in data_path.parent.iterdir()} == originals
