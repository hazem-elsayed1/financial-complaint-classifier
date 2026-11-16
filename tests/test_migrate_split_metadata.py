import json

import pandas as pd
import pytest

from scripts.migrate_split_metadata import main, propose_metadata
from src.data_integrity import compute_dataset_fingerprint, compute_manifest_fingerprint


@pytest.fixture
def legacy_files(tmp_path):
    dataset = pd.DataFrame({
        "Complaint ID": [101, 102, 103, 104],
        "Consumer complaint narrative": ["Card issue", "Loan issue", "Bank issue", "Fee issue"],
        "Product": ["A", "B", "A", "B"],
    })
    manifest = pd.DataFrame({
        "Complaint ID": dataset["Complaint ID"],
        "split": ["train", "train", "validation", "test"],
    })
    paths = {
        "data_path": tmp_path / "complaints.csv",
        "manifest_path": tmp_path / "manifest.csv",
        "metadata_path": tmp_path / "manifest.json",
        "trusted_manifest_path": tmp_path / "trusted.csv",
    }
    dataset.to_csv(paths["data_path"], index=False)
    manifest.to_csv(paths["manifest_path"], index=False)
    manifest.iloc[::-1].to_csv(paths["trusted_manifest_path"], index=False)
    paths["metadata_path"].write_text(json.dumps({
        "dataset_fingerprint": compute_dataset_fingerprint(dataset),
        "fingerprint_algorithm": "sha256",
        "split_counts": {"train": 2, "validation": 1, "test": 1},
        "total_samples": 4,
        "historical_notes": {"seed": 42, "reviewed": False, "notes": [None, "حفظ"]},
        "fingerprint_columns": list(dataset.columns),
    }), encoding="utf-8")
    return paths


def cli_args(paths):
    return [
        "--data", str(paths["data_path"]),
        "--manifest", str(paths["manifest_path"]),
        "--metadata", str(paths["metadata_path"]),
        "--trusted-manifest", str(paths["trusted_manifest_path"]),
    ]


@pytest.mark.parametrize("write_candidate", [False, True])
def test_migration_preserves_legacy_fields_and_input_files(
    legacy_files, tmp_path, capsys, write_candidate,
):
    originals = {path: path.read_bytes() for path in legacy_files.values()}
    metadata = json.loads(legacy_files["metadata_path"].read_text(encoding="utf-8"))
    candidate = tmp_path / "candidate.json"
    args = cli_args(legacy_files)
    if write_candidate:
        args += ["--output", str(candidate)]
    main(args)
    proposal = json.loads(capsys.readouterr().out)

    assert proposal == {
        **metadata,
        "manifest_fingerprint": compute_manifest_fingerprint(
            pd.read_csv(legacy_files["manifest_path"])
        ),
    }
    assert {path: path.read_bytes() for path in originals} == originals
    if write_candidate:
        assert json.loads(candidate.read_text(encoding="utf-8")) == proposal
    else:
        assert not candidate.exists()


@pytest.mark.parametrize(
    "field, replacement, message",
    [
        ("dataset_fingerprint", None, "Missing or malformed dataset fingerprint"),
        ("dataset_fingerprint", "", "Missing or malformed dataset fingerprint"),
        ("dataset_fingerprint", "g" * 64, "Missing or malformed dataset fingerprint"),
        ("dataset_fingerprint", 123, "Missing or malformed dataset fingerprint"),
        ("dataset_fingerprint", "0" * 64, "Dataset fingerprint mismatch"),
        ("fingerprint_algorithm", "md5", "Unsupported fingerprint algorithm"),
        ("split_counts", {"train": 1, "validation": 1, "test": 2}, "split_counts"),
        ("split_counts", {"train": 2, "validation": True, "test": 1}, "split_counts"),
        ("split_counts", None, "split_counts"),
        ("total_samples", 5, "total_samples"),
    ],
)
def test_migration_rejects_invalid_legacy_metadata(
    legacy_files, field, replacement, message,
):
    path = legacy_files["metadata_path"]
    metadata = json.loads(path.read_text(encoding="utf-8"))
    metadata[field] = replacement
    path.write_text(json.dumps(metadata), encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        propose_metadata(**legacy_files)


def test_migration_rejects_missing_dataset_fingerprint(legacy_files):
    path = legacy_files["metadata_path"]
    metadata = json.loads(path.read_text(encoding="utf-8"))
    del metadata["dataset_fingerprint"]
    path.write_text(json.dumps(metadata), encoding="utf-8")
    with pytest.raises(ValueError, match="Missing or malformed dataset fingerprint"):
        propose_metadata(**legacy_files)


@pytest.mark.parametrize("total_samples", ["missing", None, True, 4.0, "4", -1, 5])
def test_migration_rejects_missing_or_invalid_total_without_candidate(
    legacy_files, tmp_path, capsys, total_samples,
):
    metadata_path = legacy_files["metadata_path"]
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if total_samples == "missing":
        del metadata["total_samples"]
        message = "Missing total_samples in legacy metadata"
    else:
        metadata["total_samples"] = total_samples
        message = "Saved total_samples does not match the manifest"
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    originals = {path: path.read_bytes() for path in legacy_files.values()}
    candidate = tmp_path / "candidate.json"

    with pytest.raises(ValueError, match=message):
        main(cli_args(legacy_files) + ["--output", str(candidate)])

    assert not candidate.exists()
    assert capsys.readouterr().out == ""
    assert {path: path.read_bytes() for path in originals} == originals


@pytest.mark.parametrize(
    "column, replacement, message",
    [
        ("Complaint ID", 999, "Complaint IDs differ"),
        ("Complaint ID", 102, "duplicate Complaint IDs"),
        ("split", "holdout", "missing or invalid split assignments"),
        ("split", None, "missing or invalid split assignments"),
    ],
)
def test_migration_rejects_invalid_manifest(legacy_files, column, replacement, message):
    path = legacy_files["manifest_path"]
    manifest = pd.read_csv(path)
    manifest.loc[0, column] = replacement
    manifest.to_csv(path, index=False)
    with pytest.raises(ValueError, match=message):
        propose_metadata(**legacy_files)


def test_migration_rejects_count_preserving_tampering_against_trusted_copy(legacy_files):
    path = legacy_files["manifest_path"]
    manifest = pd.read_csv(path)
    manifest.loc[[0, 3], "split"] = ["test", "train"]
    manifest.to_csv(path, index=False)
    with pytest.raises(ValueError, match="assignments differ from the trusted historical"):
        propose_metadata(**legacy_files)


def test_migration_without_trusted_copy_warns_about_historical_authentication(legacy_files):
    del legacy_files["trusted_manifest_path"]
    with pytest.warns(UserWarning, match="cannot be independently authenticated"):
        proposal = propose_metadata(**legacy_files)
    assert "manifest_fingerprint" in proposal


@pytest.mark.parametrize("existing", ["matching", "conflicting", None])
def test_migration_preserves_matching_fingerprint_and_rejects_conflicts(legacy_files, existing):
    path = legacy_files["metadata_path"]
    metadata = json.loads(path.read_text(encoding="utf-8"))
    fingerprint = compute_manifest_fingerprint(pd.read_csv(legacy_files["manifest_path"]))
    metadata["manifest_fingerprint"] = {
        "matching": fingerprint, "conflicting": "0" * 64, None: None,
    }[existing]
    path.write_text(json.dumps(metadata), encoding="utf-8")
    original = path.read_bytes()
    if existing == "matching":
        assert propose_metadata(**legacy_files) == metadata
    else:
        with pytest.raises(ValueError, match="Conflicting existing manifest fingerprint"):
            propose_metadata(**legacy_files)
    assert path.read_bytes() == original


@pytest.mark.parametrize("destination", ["candidate", "metadata_path", "manifest_path"])
def test_migration_never_overwrites_existing_destination(legacy_files, tmp_path, destination):
    path = legacy_files.get(destination, tmp_path / "candidate.json")
    if destination == "candidate":
        path.write_text("existing candidate", encoding="utf-8")
    original = path.read_bytes()
    with pytest.raises(FileExistsError):
        main(cli_args(legacy_files) + ["--output", str(path)])
    assert path.read_bytes() == original


@pytest.mark.parametrize("path_key", ["data_path", "manifest_path", "trusted_manifest_path"])
@pytest.mark.parametrize("spelling", ["00101", "101.0", " 101"])
def test_migration_rejects_noncanonical_id_representations(legacy_files, path_key, spelling):
    path = legacy_files[path_key]
    path.write_text(path.read_text(encoding="utf-8").replace("101,", spelling + ","), encoding="utf-8")
    with pytest.raises(ValueError, match="[Ii].*(representation)"):
        propose_metadata(**legacy_files)


def test_legacy_integer_ids_survive_csv_roundtrip_but_string_ids_are_inferred(tmp_path):
    dataset = pd.DataFrame({
        "Complaint ID": [101, 102],
        "Consumer complaint narrative": ["Card issue", "Loan issue"],
        "Product": ["A", "B"],
    })
    path = tmp_path / "complaints.csv"
    dataset.to_csv(path, index=False)
    restored = pd.read_csv(path)
    assert pd.api.types.is_integer_dtype(restored["Complaint ID"].dtype)
    assert compute_dataset_fingerprint(restored) == compute_dataset_fingerprint(dataset)

    string_ids = dataset.astype({"Complaint ID": str})
    string_ids.to_csv(path, index=False)
    inferred = pd.read_csv(path)
    assert compute_dataset_fingerprint(inferred) == compute_dataset_fingerprint(dataset)
    assert compute_dataset_fingerprint(inferred) != compute_dataset_fingerprint(string_ids)
