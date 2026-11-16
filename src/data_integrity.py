
import hashlib
import json
import re
from pathlib import Path
import pandas as pd
from src.data_processing import has_product_label


def compute_dataset_fingerprint(df: pd.DataFrame) -> str:
    """Compute a row-order-independent dataset fingerprint."""

    required_columns = [
        "Complaint ID",
        "Consumer complaint narrative",
        "Product",
    ]

    missing = set(required_columns) - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing required columns: {sorted(missing)}"
        )

    if df["Complaint ID"].isna().any():
        raise ValueError("Missing Complaint IDs.")

    if not df["Complaint ID"].is_unique:
        raise ValueError("Duplicate Complaint IDs.")

    if not all(
        isinstance(text, str) and bool(text.strip())
        for text in df["Consumer complaint narrative"]
    ):
        raise ValueError("Complaint narratives must be non-empty strings.")
    if not df["Product"].map(has_product_label).all():
        raise ValueError("Product labels must be non-empty strings.")

    subset = df[required_columns].sort_values(
        by="Complaint ID",
    )

    records = subset.to_dict(orient="records")

    serialized = json.dumps(
        records,
        sort_keys=True,
        ensure_ascii=False,
        default=str,
    )

    return hashlib.sha256(
        serialized.encode("utf-8")
    ).hexdigest()


def compute_manifest_fingerprint(manifest: pd.DataFrame) -> str:
    """Hash validated ID/split assignments independently of row order.

    Format: records sorted by Complaint ID, encoded as UTF-8 JSON with
    sorted keys, separators (',', ':'), unescaped Unicode and no NaN.
    Only Complaint ID and split are serialized; ID types are preserved.
    """
    required_columns = ["Complaint ID", "split"]
    missing = set(required_columns) - set(manifest.columns)
    if missing:
        raise ValueError(f"Missing required manifest columns: {sorted(missing)}")
    if manifest["Complaint ID"].isna().any():
        raise ValueError("Manifest contains missing Complaint IDs.")
    if not manifest["Complaint ID"].is_unique:
        raise ValueError("Manifest contains duplicate Complaint IDs.")
    if set(manifest["split"]) != {"train", "validation", "test"}:
        raise ValueError("Manifest has missing or invalid split assignments.")

    records = manifest[required_columns].sort_values(
        "Complaint ID",
    ).to_dict(orient="records")
    serialized = json.dumps(
        records,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def verify_dataset_fingerprint(
    df: pd.DataFrame,
    metadata_path: str,
) -> None:
    """Verify dataset contents against saved fingerprint metadata."""

    path = Path(metadata_path)

    with path.open("r", encoding="utf-8") as file:
        metadata = json.load(file)

    if not isinstance(metadata, dict):
        raise ValueError("Fingerprint metadata must be a JSON object.")
    expected = metadata.get("dataset_fingerprint")

    if not isinstance(expected, str) or not expected:
        raise ValueError("Missing dataset fingerprint in metadata.")
    if re.fullmatch(r"[0-9a-f]{64}", expected) is None:
        raise ValueError("Malformed dataset fingerprint in metadata.")

    if metadata.get("fingerprint_algorithm") != "sha256":
        raise ValueError("Unsupported fingerprint algorithm.")

    actual = compute_dataset_fingerprint(df)

    if actual != expected:
        raise ValueError(
            "Dataset fingerprint mismatch. "
            "The dataset may have changed since the "
            "split metadata was created."
        )
