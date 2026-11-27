from unittest.mock import Mock

import pytest
from fastapi.testclient import TestClient
from sklearn.exceptions import NotFittedError
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline

import src.api
from src.api import (
    DEFAULT_MODEL_PATH,
    MODEL_PATH_ENV_VAR,
    create_app,
    resolve_model_path,
)
from src.predict import load_model, predict_complaint
from src.train import build_model, save_model, train_model
from conftest import SYNTHETIC_TRAINING_LABELS, SYNTHETIC_TRAINING_TEXTS


@pytest.fixture
def client(synthetic_model_path):
    with TestClient(create_app(str(synthetic_model_path))) as test_client:
        yield test_client


# --- Successful predictions -------------------------------------------------


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("credit card payment issue", "Credit card"),
        ("mortgage loan problem", "Mortgage"),
    ],
)
def test_predict_returns_category(client, synthetic_model_path, text, expected):
    response = client.post("/predict", json={"text": text})

    assert response.status_code == 200
    assert response.json() == {"product": expected}

    direct_model = load_model(str(synthetic_model_path))
    assert response.json()["product"] == predict_complaint(direct_model, text)


def test_predict_passes_text_to_model_unchanged(client):
    spy = Mock(wraps=client.app.state.model.predict)
    client.app.state.model.predict = spy
    text = "  Credit card charged twice — ünïcode \n"

    response = client.post("/predict", json={"text": text})

    assert response.status_code == 200
    spy.assert_called_once_with([text])


def test_health_reports_ok(client):
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_model_is_loaded_once_at_startup(synthetic_model_path, monkeypatch):
    loader = Mock(wraps=load_model)
    monkeypatch.setattr(src.api, "load_model", loader)

    app = create_app(str(synthetic_model_path))
    loader.assert_not_called()

    with TestClient(app) as test_client:
        loader.assert_called_once_with(str(synthetic_model_path))
        for _ in range(3):
            assert test_client.post("/predict", json={"text": "credit card"}).status_code == 200

    loader.assert_called_once()


def test_model_path_from_environment(synthetic_model_path, monkeypatch):
    monkeypatch.setenv(MODEL_PATH_ENV_VAR, str(synthetic_model_path))

    with TestClient(create_app()) as test_client:
        response = test_client.post("/predict", json={"text": "mortgage loan problem"})

    assert response.json() == {"product": "Mortgage"}


def test_explicit_model_path_overrides_environment(synthetic_model_path, tmp_path, monkeypatch):
    monkeypatch.setenv(MODEL_PATH_ENV_VAR, str(tmp_path / "missing.joblib"))

    with TestClient(create_app(str(synthetic_model_path))) as test_client:
        assert test_client.get("/health").status_code == 200


def test_default_model_path(monkeypatch):
    monkeypatch.delenv(MODEL_PATH_ENV_VAR, raising=False)
    assert resolve_model_path() == DEFAULT_MODEL_PATH

    monkeypatch.setenv(MODEL_PATH_ENV_VAR, "")
    assert resolve_model_path() == DEFAULT_MODEL_PATH


# --- Invalid input ------------------------------------------------------------


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"text": None},
        {"text": 123},
        {"text": 1.5},
        {"text": True},
        {"text": ["credit card"]},
        {"text": {"value": "credit card"}},
        {"text": "credit card", "unexpected": "field"},
        {"complaint": "credit card"},
        ["credit card"],
        "credit card",
    ],
)
def test_predict_rejects_malformed_payloads(client, payload):
    spy = Mock(wraps=client.app.state.model.predict)
    client.app.state.model.predict = spy

    response = client.post("/predict", json=payload)

    assert response.status_code == 422
    spy.assert_not_called()


@pytest.mark.parametrize("text", ["", " ", "   \n\t  "])
def test_predict_rejects_empty_text(client, text):
    spy = Mock(wraps=client.app.state.model.predict)
    client.app.state.model.predict = spy

    response = client.post("/predict", json={"text": text})

    assert response.status_code == 422
    assert response.json() == {"detail": "Complaint text must be a non-empty string."}
    spy.assert_not_called()


def test_predict_rejects_non_json_body(client):
    response = client.post(
        "/predict",
        content=b"credit card",
        headers={"Content-Type": "text/plain"},
    )

    assert response.status_code == 422


def test_predict_rejects_invalid_json(client):
    response = client.post(
        "/predict",
        content=b'{"text": ',
        headers={"Content-Type": "application/json"},
    )

    assert response.status_code == 422


def test_predict_rejects_get(client):
    assert client.get("/predict").status_code == 405


# --- Startup with a missing model --------------------------------------------


def test_startup_fails_for_missing_explicit_path(tmp_path):
    missing = tmp_path / "missing.joblib"
    app = create_app(str(missing))

    with pytest.raises(FileNotFoundError, match="missing.joblib"):
        with TestClient(app):
            pass


def test_startup_fails_for_missing_environment_path(tmp_path, monkeypatch):
    monkeypatch.setenv(MODEL_PATH_ENV_VAR, str(tmp_path / "env_missing.joblib"))

    with pytest.raises(FileNotFoundError, match="env_missing.joblib"):
        with TestClient(create_app()):
            pass


def test_startup_fails_for_missing_default_path(tmp_path, monkeypatch):
    monkeypatch.delenv(MODEL_PATH_ENV_VAR, raising=False)
    monkeypatch.chdir(tmp_path)

    with pytest.raises(FileNotFoundError, match="ticket_classifier.joblib"):
        with TestClient(create_app()):
            pass


def test_startup_fails_when_path_is_a_directory(tmp_path):
    with pytest.raises(FileNotFoundError):
        with TestClient(create_app(str(tmp_path))):
            pass


# --- Startup with an unusable model -------------------------------------------


def _save(tmp_path, model, name):
    path = tmp_path / name
    save_model(model, str(path))
    return path


def _fitted_classifier_with_unfitted_vectorizer():
    # scikit-learn's Pipeline check only inspects the last step and misses this.
    model = train_model(SYNTHETIC_TRAINING_TEXTS, SYNTHETIC_TRAINING_LABELS)
    model.steps[0] = ("tfidf", TfidfVectorizer())
    return model


def _fitted_vectorizer_with_unfitted_classifier():
    model = build_model()
    model.named_steps["tfidf"].fit(SYNTHETIC_TRAINING_TEXTS)
    return model


@pytest.mark.parametrize(
    ("make_model", "message"),
    [
        (build_model, "step 'tfidf' is not fitted"),
        (_fitted_classifier_with_unfitted_vectorizer, "step 'tfidf' is not fitted"),
        (_fitted_vectorizer_with_unfitted_classifier, "step 'classifier' is not fitted"),
        (lambda: Pipeline([("noop", "passthrough")]), "no estimator steps"),
        (lambda: {"not": "a model"}, "must be a scikit-learn Pipeline, got dict"),
    ],
    ids=["unfitted", "unfitted-vectorizer", "unfitted-classifier", "passthrough-only", "not-a-pipeline"],
)
def test_startup_rejects_unusable_model(tmp_path, make_model, message):
    path = _save(tmp_path, make_model(), "unusable.joblib")
    model_bytes = path.read_bytes()

    with pytest.raises(ValueError, match=message):
        with TestClient(create_app(str(path))):
            pytest.fail("the app must not start serving requests")

    assert path.read_bytes() == model_bytes


def test_unfitted_model_fails_startup_through_environment(tmp_path, monkeypatch):
    monkeypatch.setenv(MODEL_PATH_ENV_VAR, str(_save(tmp_path, build_model(), "unfitted.joblib")))

    with pytest.raises(ValueError, match="not fitted"):
        with TestClient(create_app()):
            pass


# --- Unexpected inference errors ----------------------------------------------


@pytest.mark.parametrize(
    "error",
    [
        ValueError("model rejected input"),
        NotFittedError("model lost its fit"),
        RuntimeError("model crashed"),
    ],
    ids=["ValueError", "NotFittedError", "RuntimeError"],
)
def test_inference_error_returns_server_error(synthetic_model_path, error):
    app = create_app(str(synthetic_model_path))

    with TestClient(app, raise_server_exceptions=False) as test_client:
        test_client.app.state.model.predict = Mock(side_effect=error)
        response = test_client.post("/predict", json={"text": "credit card payment failed"})

    assert response.status_code == 500
    assert str(error) not in response.text


def test_inference_error_is_not_reported_as_bad_input(client):
    client.app.state.model.predict = Mock(side_effect=ValueError("model rejected input"))

    with pytest.raises(ValueError, match="model rejected input"):
        client.post("/predict", json={"text": "credit card payment failed"})
