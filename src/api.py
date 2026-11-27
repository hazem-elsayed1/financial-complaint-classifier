"""Local FastAPI inference service for the complaint classifier.

The trained pipeline is loaded once at application startup through
``src.predict.load_model`` and every request is classified through
``src.predict.predict_complaint``. No inference logic is duplicated here.

Model path resolution (first match wins):

1. The ``model_path`` argument passed to ``create_app()``.
2. The ``COMPLAINT_MODEL_PATH`` environment variable, read at startup.
3. ``models/ticket_classifier.joblib``, relative to the working directory.

Startup fails with ``FileNotFoundError`` when the resolved file is missing,
and with ``ValueError`` when the file does not hold a fitted scikit-learn
pipeline, so the service never accepts requests without a usable model.

Blank or whitespace-only text is rejected with HTTP 422 before the model is
called; other text is passed to the classifier exactly as received. Any
error raised during inference is left unhandled and returned as HTTP 500.

Only point this service at trusted artifacts: joblib deserialization can
execute arbitrary code.

Run locally:

    python -m uvicorn src.api:app --host 127.0.0.1 --port 8000
"""

import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, StrictStr
from sklearn.exceptions import NotFittedError
from sklearn.pipeline import Pipeline
from sklearn.utils.validation import check_is_fitted

from src.predict import load_model, predict_complaint


MODEL_PATH_ENV_VAR = "COMPLAINT_MODEL_PATH"
DEFAULT_MODEL_PATH = "models/ticket_classifier.joblib"
EMPTY_TEXT_MESSAGE = "Complaint text must be a non-empty string."


class ComplaintRequest(BaseModel):
    """A single complaint narrative to classify."""

    model_config = ConfigDict(extra="forbid")

    text: StrictStr


class PredictionResponse(BaseModel):
    """The predicted financial product category."""

    product: str


class HealthResponse(BaseModel):
    status: str


def resolve_model_path(model_path: str | None = None) -> str:
    """Return the explicit path, else the environment variable, else the default."""

    if model_path is not None:
        return str(model_path)

    return os.environ.get(MODEL_PATH_ENV_VAR) or DEFAULT_MODEL_PATH


def validate_model(model) -> None:
    """Raise ValueError unless ``model`` is a pipeline whose every step is fitted.

    scikit-learn's own check for a Pipeline inspects only the last step and
    treats an empty pipeline as fitted, so each step is checked here.
    """

    if not isinstance(model, Pipeline):
        raise ValueError(
            f"Model must be a scikit-learn Pipeline, got {type(model).__name__}."
        )

    steps = [
        (name, step)
        for name, step in model.steps
        if step is not None and step != "passthrough"
    ]
    if not steps:
        raise ValueError("Model pipeline has no estimator steps.")

    for name, step in steps:
        try:
            check_is_fitted(step)
        except (NotFittedError, TypeError) as error:
            raise ValueError(f"Model pipeline step '{name}' is not fitted.") from error


def create_app(model_path: str | None = None) -> FastAPI:
    """Build the inference app; the model is loaded and validated at startup."""

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        model = load_model(resolve_model_path(model_path))
        validate_model(model)
        app.state.model = model
        try:
            yield
        finally:
            app.state.model = None

    app = FastAPI(
        title="Financial Complaint Classifier API",
        version="1.0.0",
        lifespan=lifespan,
    )

    @app.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        return HealthResponse(status="ok")

    @app.post("/predict", response_model=PredictionResponse)
    def predict(request: ComplaintRequest) -> PredictionResponse:
        # Check input here so model errors are never reported as bad input.
        if not request.text.strip():
            raise HTTPException(status_code=422, detail=EMPTY_TEXT_MESSAGE)

        product = predict_complaint(app.state.model, request.text)

        return PredictionResponse(product=product)

    return app


app = create_app()
