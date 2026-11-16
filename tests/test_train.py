
import json
import pandas as pd
import pytest
from src.train import split_data

from unittest.mock import patch
import numpy as np

from src.train import train_final_model_with_test_data
from src.train import train_model, save_model
from src.predict import load_model
from src.data_integrity import (
    compute_dataset_fingerprint,
    compute_manifest_fingerprint,
)


def test_split_data():
    """Check reproducibility and group-disjoint partitions."""
    from src.group_aware_evaluation import create_text_groups

    # Repeated narratives intentionally represent duplicates.
    texts = [
        f"Financial complaint about account number {i}"
        for i in range(100)
    ]

    texts += [
        "My credit card payment was charged twice",
        "My credit card payment was charged twice",
        "My mortgage balance is incorrect",
        "My mortgage balance is incorrect",
    ]

    X = pd.Series(texts)
    y = pd.Series(
        ["A" if i % 2 == 0 else "B" for i in range(104)]
    )

    first = split_data(X, y)
    second = split_data(X, y)

    X_train, X_val, X_test, y_train, y_val, y_test = first

    # Every original observation appears exactly once.
    train_ids = set(X_train.index)
    val_ids = set(X_val.index)
    test_ids = set(X_test.index)

    assert train_ids.isdisjoint(val_ids)
    assert train_ids.isdisjoint(test_ids)
    assert val_ids.isdisjoint(test_ids)

    assert train_ids | val_ids | test_ids == set(X.index)

    # Verify actual near-duplicate group separation.
    groups = create_text_groups(X)

    train_groups = set(groups[X_train.index.to_numpy()])
    val_groups = set(groups[X_val.index.to_numpy()])
    test_groups = set(groups[X_test.index.to_numpy()])

    assert train_groups.isdisjoint(val_groups)
    assert train_groups.isdisjoint(test_groups)
    assert val_groups.isdisjoint(test_groups)

    # All target classes must be available in training.
    assert set(y_train.unique()) == set(y.unique())

    # Repeated splitting with same seed is deterministic.
    for part1, part2 in zip(first, second):
        pd.testing.assert_series_equal(part1, part2)

    # Sanity checks: all partitions are non-empty.
    assert len(X_train) > 0
    assert len(X_val) > 0
    assert len(X_test) > 0

    assert len(y_train) == len(X_train)
    assert len(y_val) == len(X_val)
    assert len(y_test) == len(X_test)



def test_duplicate_texts_never_cross_partitions():
    """Identical narratives must belong to one partition."""
    texts = [
        f"Unique financial complaint narrative {i}"
        for i in range(100)
    ]

    texts.extend([
        "Repeated credit card billing complaint",
        "Repeated credit card billing complaint",
        "Repeated credit card billing complaint",
    ])

    X = pd.Series(texts)
    y = pd.Series(
        ["A" if i % 2 == 0 else "B" for i in range(100)]
        + ["A", "A", "A"]
    )

    X_train, X_val, X_test, *_ = split_data(X, y)

    repeated = "Repeated credit card billing complaint"

    counts = [
        int((part == repeated).sum())
        for part in (X_train, X_val, X_test)
    ]

    assert sorted(counts) == [0, 0, 3]

@pytest.mark.parametrize("invalid_label", [None, float("nan"), pd.NA, "", " \t\n", 123, True, False])
def test_training_rejects_invalid_product_labels(invalid_label):
    texts = ["credit card payment", "credit card charged", "mortgage payment", "mortgage loan"]
    labels = ["Credit card", "Credit card", "Mortgage", invalid_label]
    with pytest.raises(ValueError, match="Product labels must be non-empty strings"):
        train_model(texts, labels)


def test_model_save_load_roundtrip(tmp_path):
    """Verify that saving and loading preserves predictions."""

    texts = [
        "credit card payment failed",
        "credit card payment declined",
        "credit card billing problem",
        "credit card charged twice",
        "mortgage monthly payment issue",
        "mortgage interest rate problem",
        "mortgage loan payment delayed",
        "mortgage lender complaint",
    ]

    labels = [
        "Credit card",
        "Credit card",
        "Credit card",
        "Credit card",
        "Mortgage",
        "Mortgage",
        "Mortgage",
        "Mortgage",
    ]

    model = train_model(texts, labels)

    sample_complaints = [
        "credit card payment issue",
        "mortgage loan problem",
    ]

    predictions_before = model.predict(sample_complaints)

    model_path = tmp_path / "test_model.joblib"

    save_model(model, str(model_path))

    assert model_path.exists()

    loaded_model = load_model(str(model_path))

    predictions_after = loaded_model.predict(sample_complaints)

    np.testing.assert_array_equal(
        predictions_before,
        predictions_after,
    )




def test_final_training_uses_only_train_validation():
    """Ensure final training excludes frozen test samples."""

    X_train = pd.Series(
        ["Complaint A", "Complaint B"],
        index=[101, 102],
    )
    X_val = pd.Series(
        ["Complaint C"],
        index=[103],
    )
    X_test = pd.Series(
        ["Complaint D", "Complaint E"],
        index=[104, 105],
    )

    y_train = pd.Series(
        ["A", "B"],
        index=[101, 102],
    )
    y_val = pd.Series(
        ["A"],
        index=[103],
    )
    y_test = pd.Series(
        ["B", "A"],
        index=[104, 105],
    )

    captured = {}

    def fake_train_model(X, y):
        captured["X"] = X.copy()
        captured["y"] = y.copy()
        return "fake_model"

    with patch(
        "src.train.load_frozen_split",
        return_value=(
            X_train,
            X_val,
            X_test,
            y_train,
            y_val,
            y_test,
        ),
    ) as mocked_loader, patch(
        "src.train.train_model",
        side_effect=fake_train_model,
    ):
        model, returned_X_test, returned_y_test = (
            train_final_model_with_test_data("dummy.csv")
        )

    assert model == "fake_model"

    mocked_loader.assert_called_once_with(
        "dummy.csv",
        "data/splits/group_split_manifest.csv",
    )

    # Final training uses Train + Validation only.
    assert set(captured["X"].index) == {101, 102, 103}
    assert set(captured["y"].index) == {101, 102, 103}

    # Test samples never enter model.fit().
    assert set(captured["X"].index).isdisjoint(
        set(X_test.index)
    )

    # Held-out test remains unchanged.
    pd.testing.assert_series_equal(
        returned_X_test,
        X_test,
    )
    pd.testing.assert_series_equal(
        returned_y_test,
        y_test,
    )


def test_final_training_returns_metadata_when_requested():
    """Ensure verified metadata is returned when explicitly requested."""

    X_train = pd.Series(["Complaint A", "Complaint B"])
    X_val = pd.Series(["Complaint C"])
    X_test = pd.Series(["Complaint D"])

    y_train = pd.Series(["A", "B"])
    y_val = pd.Series(["A"])
    y_test = pd.Series(["B"])

    metadata = {
        "dataset_fingerprint": "a" * 64,
        "manifest_fingerprint": "b" * 64,
        "fingerprint_algorithm": "sha256",
        "total_samples": 4,
        "split_counts": {
            "train": 2,
            "validation": 1,
            "test": 1,
        },
    }

    with patch(
        "src.train.load_frozen_split",
        return_value=(
            X_train,
            X_val,
            X_test,
            y_train,
            y_val,
            y_test,
            metadata,
        ),
    ) as mocked_loader, patch(
        "src.train.train_model",
        return_value="fake_model",
    ):
        model, returned_X_test, returned_y_test, returned_metadata = (
            train_final_model_with_test_data(
                "dummy.csv",
                include_metadata=True,
            )
        )

    mocked_loader.assert_called_once_with(
        "dummy.csv",
        "data/splits/group_split_manifest.csv",
        include_metadata=True,
    )

    assert model == "fake_model"
    pd.testing.assert_series_equal(returned_X_test, X_test)
    pd.testing.assert_series_equal(returned_y_test, y_test)
    assert returned_metadata == metadata



def test_load_frozen_split_is_order_independent(tmp_path):
    """Verify frozen split assignments remain stable after shuffling."""
    from src.train import load_frozen_split

    df = pd.DataFrame({
        "Complaint ID": [101, 102, 103, 104, 105, 106],
        "Consumer complaint narrative": [
            "Complaint A",
            "Complaint B",
            "Complaint C",
            "Complaint D",
            "Complaint E",
            "Complaint F",
        ],
        "Product": ["A", "B", "A", "B", "A", "B"],
    })

    manifest = pd.DataFrame({
        "Complaint ID": [101, 102, 103, 104, 105, 106],
        "split": [
            "train", "train",
            "validation", "validation",
            "test", "test",
        ],
    })

    csv_path = tmp_path / "complaints.csv"
    manifest_path = tmp_path / "manifest.csv"

    manifest.to_csv(manifest_path, index=False)
    df.to_csv(csv_path, index=False)


    metadata_path = manifest_path.with_suffix(".json")

    metadata_path.write_text(
        json.dumps({
            "dataset_fingerprint": compute_dataset_fingerprint(df),
            "manifest_fingerprint": compute_manifest_fingerprint(manifest),
            "fingerprint_algorithm": "sha256",
            "total_samples": len(df),
            "split_counts": manifest["split"].value_counts().to_dict(),
        }),
        encoding="utf-8",
    )


    first = load_frozen_split(
        str(csv_path),
        str(manifest_path),
    )

    # Change only the row order.
    df.sample(frac=1, random_state=123).to_csv(
        csv_path,
        index=False,
    )
    manifest.sample(frac=1, random_state=123).to_csv(
        manifest_path,
        index=False,
    )

    second = load_frozen_split(
        str(csv_path),
        str(manifest_path),
    )

    for first_part, second_part in zip(first, second):
        assert sorted(first_part.tolist()) == sorted(
            second_part.tolist()
        )

    X_train, X_val, X_test, y_train, y_val, y_test = first

    assert set(X_train) == {"Complaint A", "Complaint B"}
    assert set(X_val) == {"Complaint C", "Complaint D"}
    assert set(X_test) == {"Complaint E", "Complaint F"}

    assert len(y_train) == 2
    assert len(y_val) == 2
    assert len(y_test) == 2


def test_load_frozen_split_rejects_missing_ids(tmp_path):
    """Reject datasets whose IDs do not match the manifest."""
    from src.train import load_frozen_split

    df = pd.DataFrame({
        "Complaint ID": [101, 102, 103, 104],
        "Consumer complaint narrative": [
            "Complaint A",
            "Complaint B",
            "Complaint C",
            "Complaint D",
        ],
        "Product": ["A", "B", "A", "B"],
    })

    # Complaint 104 is missing from the manifest.
    manifest = pd.DataFrame({
        "Complaint ID": [101, 102, 103],
        "split": ["train", "validation", "test"],
    })

    csv_path = tmp_path / "complaints.csv"
    manifest_path = tmp_path / "manifest.csv"

    df.to_csv(csv_path, index=False)
    manifest.to_csv(manifest_path, index=False)


    metadata_path = manifest_path.with_suffix(".json")

    metadata_path.write_text(
        json.dumps({
            "dataset_fingerprint": compute_dataset_fingerprint(df),
            "manifest_fingerprint": compute_manifest_fingerprint(manifest),
            "fingerprint_algorithm": "sha256",
            "total_samples": len(df),
            "split_counts": manifest["split"].value_counts().to_dict(),
        }),
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match="Dataset and manifest Complaint IDs differ",
    ):
        load_frozen_split(
            str(csv_path),
            str(manifest_path),
        )

def test_create_split_manifest_rejects_existing_output(
    tmp_path,
    monkeypatch,
):
    """Manifest creation must not overwrite existing files."""
    import pytest

    from scripts import create_split_manifest

    manifest_path = tmp_path / "manifest.csv"
    metadata_path = tmp_path / "manifest.json"

    # Simulate an existing frozen manifest.
    manifest_path.write_text(
        "Complaint ID,split\n101,train\n",
        encoding="utf-8",
    )

    # Point the script to temporary output files.
    monkeypatch.setattr(
        create_split_manifest,
        "OUTPUT_PATH",
        manifest_path,
    )

    with pytest.raises(
        FileExistsError,
        match="Split manifest or metadata already exists",
    ):
        create_split_manifest.main()

    # Verify the original file was not modified.
    assert manifest_path.read_text(
        encoding="utf-8",
    ) == "Complaint ID,split\n101,train\n"

    # Metadata should not have been created.
    assert not metadata_path.exists()


@pytest.fixture
def frozen_split_files(tmp_path):
    df = pd.DataFrame({
        "Complaint ID": [101, 102, 103, 104, 105, 106],
        "Consumer complaint narrative": [f"Complaint {i}" for i in range(6)],
        "Product": ["A", "B", "A", "B", "A", "B"],
    })
    manifest = pd.DataFrame({
        "Complaint ID": df["Complaint ID"],
        "split": ["train", "train", "validation", "validation", "test", "test"],
    })
    csv_path = tmp_path / "complaints.csv"
    manifest_path = tmp_path / "manifest.csv"
    df.to_csv(csv_path, index=False)
    manifest.to_csv(manifest_path, index=False)
    manifest_path.with_suffix(".json").write_text(json.dumps({
        "dataset_fingerprint": compute_dataset_fingerprint(df),
        "manifest_fingerprint": compute_manifest_fingerprint(manifest),
        "fingerprint_algorithm": "sha256",
        "total_samples": len(df),
        "split_counts": manifest["split"].value_counts().to_dict(),
    }), encoding="utf-8")
    return csv_path, manifest_path


def test_load_frozen_split_rejects_count_preserving_swap(frozen_split_files):
    from src.train import load_frozen_split

    csv_path, manifest_path = frozen_split_files
    manifest = pd.read_csv(manifest_path)
    original_counts = manifest["split"].value_counts().sort_index()
    manifest.loc[[0, 4], "split"] = ["test", "train"]
    pd.testing.assert_series_equal(
        manifest["split"].value_counts().sort_index(), original_counts,
    )
    manifest.to_csv(manifest_path, index=False)

    with pytest.raises(ValueError, match="Manifest fingerprint mismatch"):
        load_frozen_split(str(csv_path), str(manifest_path))


@pytest.mark.parametrize("fingerprint", [None, "", 123])
def test_load_frozen_split_rejects_missing_fingerprint(
    frozen_split_files, fingerprint,
):
    from src.train import load_frozen_split

    csv_path, manifest_path = frozen_split_files
    metadata_path = manifest_path.with_suffix(".json")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if fingerprint is None:
        del metadata["manifest_fingerprint"]
    else:
        metadata["manifest_fingerprint"] = fingerprint
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    with pytest.raises(ValueError, match="Missing manifest fingerprint"):
        load_frozen_split(str(csv_path), str(manifest_path))


def test_create_split_metadata_preserves_manifest_and_existing_metadata(
    frozen_split_files, monkeypatch,
):
    from scripts import create_split_metadata

    csv_path, manifest_path = frozen_split_files
    metadata_path = manifest_path.parent / "new_metadata.json"
    original_manifest = manifest_path.read_bytes()
    monkeypatch.setattr(create_split_metadata, "DATA_PATH", csv_path)
    monkeypatch.setattr(create_split_metadata, "MANIFEST_PATH", manifest_path)
    monkeypatch.setattr(create_split_metadata, "METADATA_PATH", metadata_path)
    create_split_metadata.main()

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert metadata["manifest_fingerprint"] == compute_manifest_fingerprint(
        pd.read_csv(manifest_path)
    )
    assert manifest_path.read_bytes() == original_manifest
    original_metadata = metadata_path.read_bytes()
    with pytest.raises(FileExistsError, match="Split metadata already exists"):
        create_split_metadata.main()
    assert metadata_path.read_bytes() == original_metadata


def test_create_split_metadata_rejects_invalid_manifest(
    frozen_split_files, monkeypatch,
):
    from scripts import create_split_metadata

    csv_path, manifest_path = frozen_split_files
    manifest = pd.read_csv(manifest_path)
    manifest.loc[0, "split"] = "holdout"
    manifest.to_csv(manifest_path, index=False)
    original_manifest = manifest_path.read_bytes()
    metadata_path = manifest_path.parent / "new_metadata.json"
    monkeypatch.setattr(create_split_metadata, "DATA_PATH", csv_path)
    monkeypatch.setattr(create_split_metadata, "MANIFEST_PATH", manifest_path)
    monkeypatch.setattr(create_split_metadata, "METADATA_PATH", metadata_path)

    with pytest.raises(ValueError, match="missing or invalid split assignments"):
        create_split_metadata.main()
    assert not metadata_path.exists()
    assert manifest_path.read_bytes() == original_manifest


def test_create_split_manifest_writes_matching_fingerprint(tmp_path, monkeypatch):
    from scripts import create_split_manifest
    from src.train import load_frozen_split

    df = pd.DataFrame({
        "Complaint ID": range(100),
        "Consumer complaint narrative": [
            f"Financial complaint about account number {i}" for i in range(100)
        ],
        "Product": ["A" if i % 2 == 0 else "B" for i in range(100)],
    })
    csv_path = tmp_path / "complaints.csv"
    manifest_path = tmp_path / "manifest.csv"
    df.to_csv(csv_path, index=False)
    monkeypatch.setattr(create_split_manifest, "DATA_PATH", csv_path)
    monkeypatch.setattr(create_split_manifest, "OUTPUT_PATH", manifest_path)
    create_split_manifest.main()

    metadata = json.loads(
        manifest_path.with_suffix(".json").read_text(encoding="utf-8")
    )
    assert metadata["manifest_fingerprint"] == compute_manifest_fingerprint(
        pd.read_csv(manifest_path)
    )
    partitions = load_frozen_split(str(csv_path), str(manifest_path))
    assert sum(len(part) for part in partitions[:3]) == len(df)


@pytest.mark.parametrize(
    "field, replacement, message",
    [
        ("total_samples", None, "total_samples"),
        ("total_samples", True, "total_samples"),
        ("total_samples", 6.0, "total_samples"),
        ("total_samples", "6", "total_samples"),
        ("total_samples", 5, "total_samples"),
        ("split_counts", None, "split_counts"),
        ("split_counts", [], "split_counts"),
        ("split_counts", {}, "split_counts"),
        ("split_counts", {"train": 2, "validation": 2}, "split_counts"),
        ("split_counts", {"train": 1, "validation": 2, "test": 3}, "split_counts"),
        ("split_counts", {"train": 2.0, "validation": 2, "test": 2}, "split_counts"),
        ("dataset_fingerprint", "bad", "Malformed dataset fingerprint"),
        ("dataset_fingerprint", 123, "Missing dataset fingerprint"),
        ("manifest_fingerprint", "g" * 64, "Malformed manifest fingerprint"),
        ("fingerprint_algorithm", "md5", "Unsupported fingerprint algorithm"),
    ],
)
def test_frozen_loader_rejects_malformed_or_inconsistent_metadata(
    frozen_split_files, field, replacement, message,
):
    from src.train import load_frozen_split

    csv_path, manifest_path = frozen_split_files
    metadata_path = manifest_path.with_suffix(".json")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata[field] = replacement
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        load_frozen_split(str(csv_path), str(manifest_path))


@pytest.mark.parametrize("field", ["total_samples", "split_counts", "dataset_fingerprint"])
def test_frozen_loader_requires_metadata_fields(frozen_split_files, field):
    from src.train import load_frozen_split

    csv_path, manifest_path = frozen_split_files
    metadata_path = manifest_path.with_suffix(".json")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    del metadata[field]
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    with pytest.raises(ValueError, match="total_samples|split_counts|Missing dataset fingerprint"):
        load_frozen_split(str(csv_path), str(manifest_path))


@pytest.mark.parametrize("serialized", ["[]", "null", "true", "{invalid json"])
def test_frozen_loader_rejects_invalid_metadata_json(frozen_split_files, serialized):
    from src.train import load_frozen_split

    csv_path, manifest_path = frozen_split_files
    manifest_path.with_suffix(".json").write_text(serialized, encoding="utf-8")
    with pytest.raises(ValueError):
        load_frozen_split(str(csv_path), str(manifest_path))


@pytest.mark.parametrize("batch_size", [0, -1, 1.0, 1.5, "2", None, True, False, np.bool_(True)])
def test_grouping_rejects_invalid_batch_sizes(batch_size):
    from src.grouping import create_text_groups

    with pytest.raises(ValueError, match="batch_size must be a positive integer"):
        create_text_groups(["Card complaint"], batch_size=batch_size)


@pytest.mark.parametrize("invalid", [None, float("nan"), pd.NA, "", " \t\n", 123, True])
def test_grouping_rejects_invalid_narratives(invalid):
    from src.grouping import create_text_groups

    with pytest.raises(ValueError, match="Complaint narratives must be non-empty strings"):
        create_text_groups(["Card complaint", invalid])


@pytest.mark.parametrize("texts", [None, "Card complaint", b"Card complaint", 123])
def test_grouping_rejects_scalar_inputs(texts):
    from src.grouping import create_text_groups

    with pytest.raises(ValueError, match="texts must be an iterable"):
        create_text_groups(texts)


@pytest.mark.parametrize("batch_size", [1, np.int64(2)])
def test_grouping_accepts_integer_batches_without_modifying_text(batch_size):
    from src.grouping import create_text_groups

    texts = ["  Mixed CASE card complaint  ", "Mortgage account billing", "  Mixed CASE card complaint  "]
    original = texts.copy()
    groups = create_text_groups(iter(texts), batch_size=batch_size)
    np.testing.assert_array_equal(groups, [0, 1, 0])
    assert texts == original

@pytest.mark.parametrize(
    "threshold",
    [True, False, None, "0.95", float("nan"), float("inf"), 0, 1.1],
)
def test_create_text_groups_rejects_invalid_threshold(threshold):
    from src.grouping import create_text_groups

    with pytest.raises(ValueError):
        create_text_groups(
            ["complaint one", "complaint two"],
            threshold=threshold,
        )


@pytest.mark.parametrize(
    "train_idx, val_idx, test_idx, groups, message",
    [
        ([0, 1], [1, 2], [3], [0, 1, 2, 3], "indices overlap"),
        ([0, 1], [2], [1, 3], [0, 1, 2, 3], "indices overlap"),
        ([0, 1], [2], [2, 3], [0, 1, 2, 3], "indices overlap"),
        ([0], [1], [2], [0, 1, 2, 3], "do not cover the dataset"),
        ([0, 1], [2], [3], [0, 1, 0, 3], "groups cross partition boundaries"),
        ([0, 1], [2], [3], [0, 1, 2, 0], "groups cross partition boundaries"),
        ([0, 1], [2], [3], [0, 1, 2, 2], "groups cross partition boundaries"),
    ],
)
def test_partition_safety_raises_explicit_errors(train_idx, val_idx, test_idx, groups, message):
    from src.train import _validate_partitions

    with pytest.raises(RuntimeError, match=message):
        _validate_partitions(np.array(train_idx), np.array(val_idx), np.array(test_idx), np.array(groups))


@pytest.mark.parametrize(
    "seed_arguments, expected_seeds",
    [({}, [42, 43]), ({"random_state": 17}, [17, 18])],
    ids=["default_seed", "caller_seed_17"],
)
def test_two_stage_split_seeds_and_positional_mapping(monkeypatch, seed_arguments, expected_seeds):
    from sklearn.model_selection import GroupShuffleSplit

    groups = np.repeat(np.arange(20), 2)
    index = [1000 + 7 * position for position in reversed(range(len(groups)))]
    texts = pd.Series([f"sentinelgroup{group:02d}" for group in groups], index=index)
    labels = pd.Series(["A" if group % 2 == 0 else "B" for group in groups], index=index)
    calls = []
    original_split = GroupShuffleSplit.split

    def observe_split(splitter, X, y=None, groups=None):
        train_positions, test_positions = next(original_split(splitter, X, y, groups))
        calls.append({
            "seed": splitter.random_state,
            "texts": X.copy(),
            "labels": y.copy(),
            "groups": np.asarray(groups).copy(),
            "train_positions": train_positions.copy(),
            "test_positions": test_positions.copy(),
        })
        return iter([(train_positions, test_positions)])

    # Observe sklearn's seed boundary while keeping grouping and both splits real.
    monkeypatch.setattr(GroupShuffleSplit, "split", observe_split)
    partitions = split_data(texts, labels, **seed_arguments)

    assert [call["seed"] for call in calls] == expected_seeds
    first, second = calls
    pd.testing.assert_series_equal(first["texts"], texts)
    pd.testing.assert_series_equal(first["labels"], labels)
    np.testing.assert_array_equal(first["groups"], groups)

    train_val_positions = first["train_positions"]
    pd.testing.assert_series_equal(second["texts"], texts.iloc[train_val_positions])
    pd.testing.assert_series_equal(second["labels"], labels.iloc[train_val_positions])
    np.testing.assert_array_equal(second["groups"], groups[train_val_positions])
    expected_positions = (
        train_val_positions[second["train_positions"]],
        train_val_positions[second["test_positions"]],
        first["test_positions"],
    )
    for positions, returned_texts, returned_labels in zip(
        expected_positions, partitions[:3], partitions[3:],
    ):
        assert len(positions) > 0
        pd.testing.assert_series_equal(returned_texts, texts.iloc[positions])
        pd.testing.assert_series_equal(returned_labels, labels.iloc[positions])

    np.testing.assert_array_equal(
        np.sort(np.concatenate(expected_positions)), np.arange(len(texts)),
    )
    train_groups, val_groups, test_groups = (set(groups[positions]) for positions in expected_positions)
    assert train_groups.isdisjoint(val_groups)
    assert train_groups.isdisjoint(test_groups)
    assert val_groups.isdisjoint(test_groups)
