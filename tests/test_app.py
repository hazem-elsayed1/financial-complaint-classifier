from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

import src.predict
from src.api import MODEL_PATH_ENV_VAR
from src.predict import load_model, predict_complaint
from src.train import build_model, save_model


APP_PATH = Path(__file__).resolve().parent.parent / "app.py"

CREDIT_CARD_TEXT = (
    "My credit card payment was charged twice, and the bank has refused "
    "to correct the transaction."
)
EMPTY_INPUT_MESSAGE = (
    "Enter a complaint or choose a synthetic example, then select **Classify complaint**."
)
STALE_RESULT_MESSAGE = (
    "The complaint has changed since the last prediction. "
    "Select **Classify complaint** to classify the new text."
)


def run_app(monkeypatch, model_path) -> AppTest:
    monkeypatch.setenv(MODEL_PATH_ENV_VAR, str(model_path))
    at = AppTest.from_file(str(APP_PATH), default_timeout=30)
    return at.run()


def classify(at: AppTest, text: str) -> AppTest:
    at.text_area(key="complaint_text").input(text).run()
    return at.button(key="classify").click().run()


def predictions(at: AppTest) -> list[str]:
    """Product labels shown in the result panel (rendered as :primary[...] subheaders)."""

    return [
        s.value.removeprefix(":primary[").removesuffix("]")
        for s in at.subheader
        if s.value.startswith(":primary[")
    ]


@pytest.fixture
def app(monkeypatch, synthetic_model_path):
    at = run_app(monkeypatch, synthetic_model_path)
    assert not at.exception
    return at


# --- Initial state -------------------------------------------------------------


def test_initial_state_explains_the_flow(app):
    assert app.title[0].value == "Financial Complaint Classifier"
    assert "Classify complaint" in app.markdown[0].value
    assert "result panel" in app.markdown[0].value

    example = app.selectbox(key="example")
    assert example.label == "Try a synthetic example (optional)"
    assert example.value is None

    complaint = app.text_area(key="complaint_text")
    assert complaint.label == "Enter your financial complaint"
    assert complaint.value == ""

    button = app.button(key="classify")
    assert button.label == "Classify complaint"
    assert not button.disabled

    assert [h.value for h in app.header] == ["Complaint", "Result"]
    assert any("No prediction yet" in m.value for m in app.markdown)
    assert predictions(app) == []
    assert not app.warning and not app.error and not app.info and not app.success


# --- Empty input ---------------------------------------------------------------


@pytest.mark.parametrize("text", ["", "   ", " \n\t "])
def test_empty_input_shows_guidance_and_no_prediction(app, text):
    classify(app, text)

    assert not app.exception
    assert [w.value for w in app.warning] == [EMPTY_INPUT_MESSAGE]
    assert predictions(app) == []
    assert not app.success


# --- Successful prediction -----------------------------------------------------


def test_prediction_matches_direct_inference_in_one_panel(app, synthetic_model_path, monkeypatch):
    calls = []
    real_predict = src.predict.predict_complaint

    def spy(model, text):
        calls.append(text)
        return real_predict(model, text)

    monkeypatch.setattr(src.predict, "predict_complaint", spy)
    padded = f"  {CREDIT_CARD_TEXT}\n"

    classify(app, padded)

    direct = predict_complaint(load_model(str(synthetic_model_path)), CREDIT_CARD_TEXT)
    assert not app.exception
    assert predictions(app) == [direct] == ["Credit card"]
    assert calls == [CREDIT_CARD_TEXT]  # same stripped text the original app classified
    assert not app.success  # the separate success banner is gone
    assert not app.warning and not app.error and not app.info
    assert any(c.value.startswith("For the complaint: “My credit card payment") for c in app.caption)


def test_synthetic_example_fills_the_complaint_and_classifies(app):
    app.selectbox(key="example").select("Misapplied mortgage payment").run()

    assert app.text_area(key="complaint_text").value.startswith("The mortgage servicer")
    app.button(key="classify").click().run()

    assert predictions(app) == ["Mortgage"]


def test_markdown_in_complaint_is_shown_literally(app):
    classify(app, "credit card **charged** twice [see](http://example.com) :red[alert]")

    caption = next(c.value for c in app.caption if c.value.startswith("For the complaint"))
    assert r"\*\*charged\*\*" in caption
    assert r"\[see\]\(http\://example\.com\)" in caption
    assert r"\:red\[alert\]" in caption  # Streamlit color directives are escaped too


# --- Stale results ---------------------------------------------------------------


def test_edited_text_hides_the_previous_prediction(app):
    classify(app, CREDIT_CARD_TEXT)
    assert predictions(app) == ["Credit card"]

    app.text_area(key="complaint_text").input("My mortgage lender lost my payment.").run()

    assert predictions(app) == []
    assert [i.value for i in app.info] == [STALE_RESULT_MESSAGE]

    app.button(key="classify").click().run()
    assert predictions(app) == ["Mortgage"]
    assert not app.info


def test_choosing_a_new_example_hides_the_previous_prediction(app):
    classify(app, CREDIT_CARD_TEXT)

    app.selectbox(key="example").select("Calls about a paid debt").run()

    assert predictions(app) == []
    assert [i.value for i in app.info] == [STALE_RESULT_MESSAGE]


def test_editing_text_clears_the_example_selection(app):
    app.selectbox(key="example").select("Duplicate credit card charge").run()
    app.text_area(key="complaint_text").input("A different complaint about my mortgage.").run()

    assert app.selectbox(key="example").value is None


# --- Model problems --------------------------------------------------------------


def test_missing_model_disables_classification(monkeypatch, tmp_path):
    missing = tmp_path / "absent.joblib"
    at = run_app(monkeypatch, missing)

    assert not at.exception
    assert len(at.error) == 1
    assert "The trained model was not found." in at.error[0].value
    assert str(missing) in at.error[0].value
    assert at.button(key="classify").disabled
    assert any("unavailable until a model is loaded" in m.value for m in at.markdown)
    assert at.text_area(key="complaint_text").label == "Enter your financial complaint"


def test_unfitted_model_disables_classification(monkeypatch, tmp_path):
    path = tmp_path / "unfitted.joblib"
    save_model(build_model(), str(path))

    at = run_app(monkeypatch, path)

    assert not at.exception
    assert "could not be used" in at.error[0].value
    assert at.button(key="classify").disabled


def test_prediction_failure_shows_an_error_not_a_result(app, monkeypatch):
    def broken(model, text):
        raise RuntimeError("inference failed")

    monkeypatch.setattr(src.predict, "predict_complaint", broken)

    classify(app, CREDIT_CARD_TEXT)

    assert not app.exception
    assert predictions(app) == []
    assert ["could not be classified" in e.value for e in app.error] == [True]
    assert "inference failed" not in app.error[0].value
