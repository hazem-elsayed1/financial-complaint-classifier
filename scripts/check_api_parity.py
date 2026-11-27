"""Read-only parity check: API predictions versus direct inference.

Loads a saved model directly with ``load_model`` and classifies fixed
synthetic complaints with ``predict_complaint``. Separately starts the FastAPI
app in-process (no network) on the same model file and posts the same
complaints to ``/predict``. Reports whether every label matches.

The model file is hashed before and after the check; a change fails the
check. The model is never written. Results print to stdout and are saved
only when ``--output`` names a file that does not exist yet. The file is
created exclusively, so an existing file is never replaced, including one
that appears while the check runs.

    python -m scripts.check_api_parity --model models/ticket_classifier.joblib
"""

import argparse
import hashlib
import json
import platform
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

from fastapi.testclient import TestClient

from src.api import DEFAULT_MODEL_PATH, create_app
from src.predict import load_model, predict_complaint


# Invented narratives written for this check; none come from the CFPB data.
SYNTHETIC_COMPLAINTS = [
    "My credit card was charged twice for the same purchase and the bank has not reversed it.",
    "The mortgage servicer misapplied my monthly payment and now reports me as late.",
    "A debt collector keeps calling my workplace about a debt I already paid off.",
    "My credit report lists an account I never opened and the bureau will not remove it.",
    "The bank closed my checking account without notice and is holding my deposit.",
    "My student loan servicer put my loans into forbearance without asking me.",
    "The dealership financing company added fees to my auto loan that I never agreed to.",
    "I sent money through a transfer app to the wrong person and the company refuses to help.",
    "The payday lender withdrew more from my account than the agreed repayment amount.",
    "My prepaid card was frozen and customer service will not tell me why.",
    "The personal loan company keeps changing my due date and charging late fees.",
    "  credit card   interest rate went up without any warning  ",
    "CREDIT CARD FRAUD CHARGES NOT REFUNDED",
    "Mi tarjeta de crédito tiene cargos que no reconozco — ünïcode test.",
    "late fee " * 200,
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def check_api_parity(model_path: str, texts=SYNTHETIC_COMPLAINTS) -> dict:
    path = Path(model_path)
    if not path.is_file():
        raise FileNotFoundError(f"Model file not found: {path}")

    sha_before = sha256_file(path)

    direct_model = load_model(str(path))
    direct_predictions = [predict_complaint(direct_model, text) for text in texts]

    cases = []
    with TestClient(create_app(str(path))) as client:
        for text, direct in zip(texts, direct_predictions):
            response = client.post("/predict", json={"text": text})
            api = response.json().get("product") if response.status_code == 200 else None
            cases.append({
                "text": text,
                "direct_prediction": direct,
                "api_status_code": response.status_code,
                "api_prediction": api,
                "match": response.status_code == 200 and api == direct,
            })

    sha_after = sha256_file(path)
    mismatches = sum(not case["match"] for case in cases)
    passed = mismatches == 0 and sha_before == sha_after

    return {
        "parity_status": "passed" if passed else "failed",
        "checked_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "model_file": path.name,
        "model_sha256_before": sha_before,
        "model_sha256_after": sha_after,
        "model_unchanged": sha_before == sha_after,
        "input_source": "synthetic_complaints",
        "total_cases": len(cases),
        "matching_cases": len(cases) - mismatches,
        "mismatched_cases": mismatches,
        "environment": {
            "python": platform.python_version(),
            "scikit-learn": version("scikit-learn"),
            "joblib": version("joblib"),
            "fastapi": version("fastapi"),
        },
        "cases": cases,
    }


def _error(message: str) -> int:
    print(json.dumps({"parity_status": "error", "error": message}))
    return 2


def write_new_report(output: Path, serialized: str) -> None:
    """Create ``output`` exclusively; raise FileExistsError if it exists."""

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8", newline="\n") as file:
        file.write(serialized)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default=DEFAULT_MODEL_PATH, help="Trusted saved model (.joblib)")
    parser.add_argument("--output", help="Optional new JSON report path; existing files are never replaced")
    args = parser.parse_args(argv)

    output = Path(args.output) if args.output else None

    # Fail fast before loading the model; the exclusive create below is the real guard.
    if output is not None and output.exists():
        return _error(f"Output already exists and will not be replaced: {output}")

    try:
        report = check_api_parity(args.model)
    except (OSError, ValueError) as error:
        return _error(str(error))

    serialized = json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    if output is not None:
        try:
            write_new_report(output, serialized)
        except FileExistsError:
            return _error(f"Output already exists and will not be replaced: {output}")
        except OSError as error:
            return _error(f"Could not write report: {error}")

    summary = {key: value for key, value in report.items() if key != "cases"}
    print(json.dumps(summary, indent=2, allow_nan=False))
    return 0 if report["parity_status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
