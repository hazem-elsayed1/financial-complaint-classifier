import json
import shutil

import pytest

import scripts.check_api_parity as parity_cli
import src.api
from scripts.check_api_parity import SYNTHETIC_COMPLAINTS, check_api_parity, main, sha256_file


@pytest.fixture
def model_copy(synthetic_model_path, tmp_path):
    """A private copy of the synthetic model so byte checks are isolated."""

    path = tmp_path / "input_model.joblib"
    shutil.copyfile(synthetic_model_path, path)
    return path


def test_parity_passes_for_synthetic_model(synthetic_model_path):
    report = check_api_parity(str(synthetic_model_path))

    assert report["parity_status"] == "passed"
    assert report["total_cases"] == len(SYNTHETIC_COMPLAINTS)
    assert report["mismatched_cases"] == 0
    assert report["model_unchanged"] is True
    assert report["model_sha256_before"] == sha256_file(synthetic_model_path)
    assert all(case["api_status_code"] == 200 for case in report["cases"])
    assert all(case["api_prediction"] == case["direct_prediction"] for case in report["cases"])


def test_parity_detects_api_divergence(synthetic_model_path, monkeypatch):
    monkeypatch.setattr(src.api, "predict_complaint", lambda model, text: "Diverged")

    report = check_api_parity(str(synthetic_model_path), ["credit card", "mortgage loan"])

    assert report["parity_status"] == "failed"
    assert report["mismatched_cases"] == 2


def test_cli_writes_new_report(model_copy, tmp_path, capsys):
    output = tmp_path / "nested" / "parity.json"
    model_bytes = model_copy.read_bytes()

    assert main(["--model", str(model_copy), "--output", str(output)]) == 0

    saved = json.loads(output.read_text(encoding="utf-8"))
    assert saved["parity_status"] == "passed"
    assert len(saved["cases"]) == len(SYNTHETIC_COMPLAINTS)
    assert "cases" not in json.loads(capsys.readouterr().out)
    assert model_copy.read_bytes() == model_bytes


def test_cli_without_output_prints_only(model_copy, tmp_path, capsys, monkeypatch):
    monkeypatch.chdir(tmp_path)

    assert main(["--model", str(model_copy)]) == 0

    assert json.loads(capsys.readouterr().out)["parity_status"] == "passed"
    assert sorted(path.name for path in tmp_path.iterdir()) == ["input_model.joblib"]


def test_cli_refuses_existing_report_and_leaves_files_intact(model_copy, tmp_path, capsys):
    report = tmp_path / "recorded_parity.json"
    report_bytes = b'{"parity_status": "passed", "recorded": true}\r\n\xff not even utf-8'
    report.write_bytes(report_bytes)
    model_bytes = model_copy.read_bytes()

    assert main(["--model", str(model_copy), "--output", str(report)]) == 2

    result = json.loads(capsys.readouterr().out)
    assert result["parity_status"] == "error"
    assert "already exists" in result["error"]
    assert report.read_bytes() == report_bytes
    assert model_copy.read_bytes() == model_bytes


def test_cli_refuses_report_created_during_check(model_copy, tmp_path, monkeypatch, capsys):
    report = tmp_path / "parity.json"
    racing_bytes = b"written by another process while the check ran\n"
    model_bytes = model_copy.read_bytes()
    real_check = parity_cli.check_api_parity

    def check_then_race(model_path):
        result = real_check(model_path)
        report.write_bytes(racing_bytes)
        return result

    monkeypatch.setattr(parity_cli, "check_api_parity", check_then_race)

    assert main(["--model", str(model_copy), "--output", str(report)]) == 2

    assert "already exists" in json.loads(capsys.readouterr().out)["error"]
    assert report.read_bytes() == racing_bytes
    assert model_copy.read_bytes() == model_bytes


def test_cli_refuses_model_path_as_output(model_copy):
    model_bytes = model_copy.read_bytes()

    assert main(["--model", str(model_copy), "--output", str(model_copy)]) == 2

    assert model_copy.read_bytes() == model_bytes


def test_cli_has_no_overwrite_option(model_copy, tmp_path):
    report = tmp_path / "parity.json"
    report.write_bytes(b"keep")

    with pytest.raises(SystemExit) as exit_info:
        main(["--model", str(model_copy), "--output", str(report), "--overwrite"])

    assert exit_info.value.code == 2
    assert report.read_bytes() == b"keep"


def test_cli_reports_missing_model(tmp_path, capsys):
    assert main(["--model", str(tmp_path / "missing.joblib")]) == 2
    assert json.loads(capsys.readouterr().out)["parity_status"] == "error"
