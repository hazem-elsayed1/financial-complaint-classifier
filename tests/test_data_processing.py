
import pandas as pd
import pytest
import subprocess
import sys
from src.data_processing import remove_missing_narratives
from src.data_processing import (
    has_product_label,
    remove_conflicting_labels,
    remove_exact_duplicates,
    remove_normalized_duplicates,
    clean_dataset
)
from src.data_integrity import compute_manifest_fingerprint

def test_remove_missing_narratives():
    """Remove missing narratives without modifying original data."""

    df = pd.DataFrame({
        "Consumer complaint narrative": [
            "My payment failed",
            None,
            "My account was blocked"
        ],
        "Product": [
            "Credit card",
            "Mortgage",
            "Checking or savings account"
        ]
    })

    result = remove_missing_narratives(df)

    assert len(result) == 2
    assert result["Consumer complaint narrative"].isna().sum() == 0
    assert len(df) == 3


def test_remove_conflicting_labels():
    df = pd.DataFrame({
        "Consumer complaint narrative": [
            "My card was blocked",
            "My card was blocked",
            "My loan was rejected"
        ],
        "Product": [
            "Credit card",
            "Debt collection",
            "Student loan"
        ]
    })

    result = remove_conflicting_labels(df)

    assert len(result) == 1
    assert result["Product"].iloc[0] == "Student loan"


def test_remove_exact_duplicates():
    df = pd.DataFrame({
        "Consumer complaint narrative": [
            "My payment failed",
            "My payment failed",
            "My account was blocked"
        ],
        "Product": [
            "Credit card",
            "Credit card",
            "Checking or savings account"
        ]
    })

    result = remove_exact_duplicates(df)

    assert len(result) == 2
    assert result["Consumer complaint narrative"].is_unique
def test_remove_normalized_duplicates():
    """Remove normalized duplicates and preserve original text."""

    df = pd.DataFrame({
        "Consumer complaint narrative": [
            "My BANK charged me XXXX",
            "  my bank charged me xxxx  ",
            "My loan was rejected"
        ],
        "Product": [
            "Checking or savings account",
            "Checking or savings account",
            "Student loan"
        ]
    })

    result = remove_normalized_duplicates(df)

    assert len(result) == 2
    assert result["normalized_text"].is_unique

    assert result["Consumer complaint narrative"].iloc[0] == (
        "My BANK charged me XXXX"
    )

    assert len(df) == 3


def test_clean_dataset_pipeline():
    """Test all cleaning steps working together."""

    df = pd.DataFrame({
        "Consumer complaint narrative": [
            "My card was stolen",
            "My card was stolen",
            "My BANK charged me XXXX",
            "my bank charged me xxxx",
            "My loan was rejected",
            None,
            "My account was blocked",
            "My account was blocked"
        ],
        "Product": [
            "Credit card",
            "Credit card",
            "Checking account",
            "Checking account",
            "Student loan",
            "Mortgage",
            "Credit card",
            "Debt collection"
        ]
    })

    result = clean_dataset(df)

    assert len(result) == 3
    assert len(df) == 8
    assert result["normalized_text"].is_unique
    assert result["Consumer complaint narrative"].isna().sum() == 0

    assert set(result["Product"]) == {
        "Credit card",
        "Checking account",
        "Student loan"
    }


def test_clean_dataset_removes_empty_normalized_text():
    """Texts that become empty after normalization must be removed."""

    df = pd.DataFrame({
        "Consumer complaint narrative": [
            "XXXX",
            "   ",
            "My credit card was charged twice.",
        ],
        "Product": [
            "Credit card",
            "Credit card",
            "Credit card",
        ],
    })

    result = clean_dataset(df)

    assert len(result) == 1
    assert (
        result.iloc[0]["Consumer complaint narrative"]
        == "My credit card was charged twice."
    )


def test_clean_dataset_removes_normalized_label_conflicts():
    """Remove all rows with conflicting normalized labels."""

    df = pd.DataFrame({
        "Consumer complaint narrative": [
            "My card was blocked XXXX",
            "my card was blocked",
            "My mortgage payment is incorrect.",
        ],
        "Product": [
            "Credit card",
            "Checking or savings account",
            "Mortgage",
        ],
    })

    result = clean_dataset(df)

    assert len(result) == 1

    assert (
        result.iloc[0]["Consumer complaint narrative"]
        == "My mortgage payment is incorrect."
    )


@pytest.mark.parametrize("label", ["Credit card", " \tMortgage\n"])
def test_product_label_accepts_nonblank_strings(label):
    assert has_product_label(label) is True


@pytest.mark.parametrize("invalid_label", [None, float("nan"), pd.NA, "", " \t\n", 123, True, False])
def test_clean_dataset_removes_invalid_labels_before_deduplication(invalid_label):
    df = pd.DataFrame({
        "Consumer complaint narrative": [
            "My BANK charged me XXXX",
            "My BANK charged me XXXX",
            "  my bank charged me xxxx  ",
            "Mortgage payment issue",
        ],
        "Product": [invalid_label, " Credit card ", " Credit card ", "Mortgage"],
    })
    original = df.copy(deep=True)

    cleaned = clean_dataset(df)

    assert cleaned["Consumer complaint narrative"].tolist() == [
        "My BANK charged me XXXX", "Mortgage payment issue",
    ]
    assert cleaned["Product"].tolist() == [" Credit card ", "Mortgage"]
    assert cleaned["normalized_text"].tolist() == [
        "my bank charged me", "mortgage payment issue",
    ]
    pd.testing.assert_frame_equal(df, original)


def test_clean_dataset_with_only_invalid_labels_returns_no_rows():
    df = pd.DataFrame({
        "Consumer complaint narrative": ["Card complaint", "Mortgage complaint"],
        "Product": [None, "   "],
    })
    assert clean_dataset(df).empty


@pytest.mark.parametrize("narratives", [[], [None]])
def test_clean_dataset_preserves_columns_when_no_narratives_remain(narratives):
    df = pd.DataFrame({
        "Consumer complaint narrative": narratives,
        "Product": ["Credit card"] * len(narratives),
    })
    cleaned = clean_dataset(df)
    assert cleaned.empty
    assert set(df.columns).issubset(cleaned.columns)


def test_data_processing_cli(tmp_path):
    """Verify that the CLI cleans and saves a CSV file."""

    input_path = tmp_path / "raw.csv"
    output_path = tmp_path / "processed" / "cleaned.csv"

    df = pd.DataFrame({
        "Consumer complaint narrative": [
            "XXXX",
            "My credit card was charged twice.",
            "My credit card was charged twice.",
            "My mortgage payment was incorrect.",
        ],
        "Product": [
            "Credit card",
            "Credit card",
            "Credit card",
            "Mortgage",
        ],
    })

    df.to_csv(input_path, index=False)

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "src.data_processing",
            "--input",
            str(input_path),
            "--output",
            str(output_path),
        ],
        capture_output=True,
        text=True,
        check=True,
    )

    assert output_path.is_file()

    cleaned = pd.read_csv(output_path)

    assert len(cleaned) == 2
    assert set(cleaned["Product"]) == {
        "Credit card",
        "Mortgage",
    }

    assert "Original rows: 4" in result.stdout
    assert "Cleaned rows: 2" in result.stdout


def test_dataset_fingerprint_detects_changes():
    from src.data_integrity import compute_dataset_fingerprint

    df = pd.DataFrame({
        "Complaint ID": [101, 102, 103],
        "Consumer complaint narrative": [
            "Credit card payment issue",
            "Mortgage payment issue",
            "Student loan issue",
        ],
        "Product": [
            "Credit card",
            "Mortgage",
            "Student loan",
        ],
    })

    original_hash = compute_dataset_fingerprint(df)

    # Changing row order should not affect the fingerprint.
    shuffled = df.sample(frac=1, random_state=42)

    assert compute_dataset_fingerprint(shuffled) == original_hash

    # Changing a complaint text must change the fingerprint.
    modified = df.copy()
    modified.loc[
        modified["Complaint ID"] == 101,
        "Consumer complaint narrative",
    ] = "Completely different complaint"

    assert compute_dataset_fingerprint(modified) != original_hash


def test_verify_dataset_fingerprint_rejects_modified_data(tmp_path):
    import json
    import pytest

    from src.data_integrity import (
        compute_dataset_fingerprint,
        verify_dataset_fingerprint,
    )

    df = pd.DataFrame({
        "Complaint ID": [101, 102],
        "Consumer complaint narrative": [
            "Credit card complaint",
            "Mortgage complaint",
        ],
        "Product": ["Credit card", "Mortgage"],
    })

    metadata_path = tmp_path / "metadata.json"

    metadata = {
        "dataset_fingerprint": compute_dataset_fingerprint(df),
        "fingerprint_algorithm": "sha256",
    }

    metadata_path.write_text(
        json.dumps(metadata),
        encoding="utf-8",
    )

    # Original dataset should pass verification.
    verify_dataset_fingerprint(df, str(metadata_path))

    # Modify a complaint without changing its ID.
    modified = df.copy()
    modified.loc[
        modified["Complaint ID"] == 101,
        "Consumer complaint narrative",
    ] = "Completely different complaint"

    with pytest.raises(
        ValueError,
        match="Dataset fingerprint mismatch",
    ):
        verify_dataset_fingerprint(
            modified,
            str(metadata_path),
        )


def test_manifest_fingerprint_is_stable_and_detects_assignment_changes():
    import hashlib

    manifest = pd.DataFrame({
        "Complaint ID": [101, 102, 103, 104],
        "split": ["train", "train", "validation", "test"],
    })
    canonical = (
        '[{"Complaint ID":101,"split":"train"},'
        '{"Complaint ID":102,"split":"train"},'
        '{"Complaint ID":103,"split":"validation"},'
        '{"Complaint ID":104,"split":"test"}]'
    )
    fingerprint = compute_manifest_fingerprint(manifest)
    assert fingerprint == hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    assert compute_manifest_fingerprint(manifest.iloc[::-1]) == fingerprint

    moved = manifest.copy()
    moved.loc[0, "split"] = "test"
    assert compute_manifest_fingerprint(moved) != fingerprint


@pytest.mark.parametrize(
    "manifest, message",
    [
        (pd.DataFrame({"Complaint ID": [1]}), "Missing required manifest columns"),
        (pd.DataFrame({"split": ["train"]}), "Missing required manifest columns"),
        (pd.DataFrame({"Complaint ID": [1, None, 3], "split": ["train", "validation", "test"]}), "missing Complaint IDs"),
        (pd.DataFrame({"Complaint ID": [1, 1, 3], "split": ["train", "validation", "test"]}), "duplicate Complaint IDs"),
        (pd.DataFrame({"Complaint ID": [1, 2], "split": ["train", "test"]}), "missing or invalid split assignments"),
        (pd.DataFrame({"Complaint ID": [], "split": []}), "missing or invalid split assignments"),
    ],
)
def test_manifest_fingerprint_rejects_invalid_manifest(manifest, message):
    with pytest.raises(ValueError, match=message):
        compute_manifest_fingerprint(manifest)


@pytest.mark.parametrize("assignment", [None, float("nan"), pd.NA, "", "holdout"])
def test_manifest_fingerprint_rejects_invalid_assignment(assignment):
    manifest = pd.DataFrame({
        "Complaint ID": [1, 2, 3, 4],
        "split": ["train", "validation", "test", assignment],
    })
    with pytest.raises(ValueError, match="missing or invalid split assignments"):
        compute_manifest_fingerprint(manifest)


@pytest.mark.parametrize("column", ["Consumer complaint narrative", "Product"])
@pytest.mark.parametrize("invalid", [None, float("nan"), pd.NA, "", " \t\n", 123, True])
def test_dataset_fingerprint_rejects_invalid_text_and_labels(column, invalid):
    from src.data_integrity import compute_dataset_fingerprint

    dataset = pd.DataFrame({
        "Complaint ID": [101, 102],
        "Consumer complaint narrative": ["Card payment issue", "Mortgage issue"],
        "Product": ["Credit card", "Mortgage"],
    })
    dataset[column] = dataset[column].astype(object)
    dataset.loc[0, column] = invalid
    with pytest.raises(ValueError, match="must be non-empty strings"):
        compute_dataset_fingerprint(dataset)


def test_dataset_fingerprint_preserves_legacy_serialization_and_values():
    import hashlib
    from src.data_integrity import compute_dataset_fingerprint

    dataset = pd.DataFrame({
        "Complaint ID": [102, 101],
        "Consumer complaint narrative": ["  Mixed CASE complaint  ", "رسوم البنك"],
        "Product": [" Credit card ", "Mortgage"],
    })
    original = dataset.copy(deep=True)
    canonical = (
        '[{"Complaint ID": 101, "Consumer complaint narrative": "رسوم البنك", "Product": "Mortgage"}, '
        '{"Complaint ID": 102, "Consumer complaint narrative": "  Mixed CASE complaint  ", "Product": " Credit card "}]'
    )
    expected = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    assert compute_dataset_fingerprint(dataset) == expected
    assert compute_dataset_fingerprint(dataset.iloc[::-1]) == expected
    pd.testing.assert_frame_equal(dataset, original)


@pytest.mark.parametrize(
    "id_csv, id_dtype, expected",
    [
        ("10\n2\n3\n", None, "de5b0edcdedc746b1dc4fc362b869851998ae5940e3dd26352daad3b51cc721c"),
        ("10\n2\n3\n", str, "58e5b3bbf2b01628fd2d67449ceacd6d2db2a21e4521a0b61e38dcf33d186e56"),
        ("10.0\n2.0\n3.0\n", None, "8d2e8794e79f6438a24f75b5c86b73fff563dc6950b933ffb83df8432eabd781"),
    ],
    ids=["inferred_integer_ids", "explicit_string_ids", "inferred_float_ids"],
)
def test_dataset_fingerprint_matches_golden_sha256(id_csv, id_dtype, expected):
    """Freeze independently hashed, hand-written legacy JSON byte vectors.

    Select ID/narrative/Product; sort records by typed ID, and object keys
    alphabetically. JSON uses separators (', ', ': '), literal Unicode,
    standard quote/backslash/newline escaping, default=str, UTF-8, no BOM,
    indentation or final newline. Numeric order is 2, 3, 10; string order
    is '10', '2', '3'. Integers, quoted strings and decimal floats differ.
    Expected hex digests were calculated from literal JSON using hashlib,
    without either production fingerprint function or json.dumps.
    """
    from io import StringIO
    from src.data_integrity import compute_dataset_fingerprint

    ids = pd.read_csv(StringIO("Complaint ID\n" + id_csv), dtype=id_dtype)["Complaint ID"]
    dataset = pd.DataFrame({
        "Product": ["Credit card", "قرض", "Mortgage"],
        "Consumer complaint narrative": ['  Café "fee"  ', "رسوم البنك", "Line one\nPath \\bank"],
        "Complaint ID": ids,
    })
    original = dataset.copy(deep=True)

    assert compute_dataset_fingerprint(dataset) == expected
    assert compute_dataset_fingerprint(dataset.iloc[::-1]) == expected
    pd.testing.assert_frame_equal(dataset, original)


@pytest.mark.parametrize(
    "complaint_ids, expected",
    [
        ([10, 2, 3], "dd75d70cc514d927d84d4cb6c7e82ed5a1be21a444be26a3fc99f19d74250e9a"),
        (["ع-3", "A-1", "é-2"], "6ab333215ae4e4564bce152a08901a728d52d9a26217bb183bcdca48488a4efc"),
    ],
    ids=["integer_ids", "unicode_string_ids"],
)
def test_manifest_fingerprint_matches_golden_sha256(complaint_ids, expected):
    """Freeze independently hashed, hand-written compact JSON byte vectors.

    Only ID/split are serialized, keys alphabetically sorted, records sorted
    by typed ID. Separators are (',', ':'), Unicode is literal, NaN forbidden;
    bytes are UTF-8 without BOM, indentation or final newline. Numeric order
    is 2, 3, 10; Unicode code-point order is 'A-1', 'é-2', 'ع-3'.
    Integer vector (the Unicode vector substitutes its corresponding IDs):
    [{"Complaint ID":2,"split":"train"},{"Complaint ID":3,"split":"validation"},{"Complaint ID":10,"split":"test"}]
    Expected digests were calculated with hashlib from literal JSON bytes,
    without the production function or json.dumps.
    """
    manifest = pd.DataFrame({
        "split": ["test", "train", "validation"],
        "Complaint ID": complaint_ids,
    })
    original = manifest.copy(deep=True)

    assert compute_manifest_fingerprint(manifest) == expected
    assert compute_manifest_fingerprint(manifest.iloc[::-1]) == expected
    pd.testing.assert_frame_equal(manifest, original)
