import pytest

from src.train import save_model, train_model


SYNTHETIC_TRAINING_TEXTS = [
    "credit card payment failed",
    "credit card payment declined",
    "credit card billing problem",
    "credit card charged twice",
    "mortgage monthly payment issue",
    "mortgage interest rate problem",
    "mortgage loan payment delayed",
    "mortgage lender complaint",
]

SYNTHETIC_TRAINING_LABELS = ["Credit card"] * 4 + ["Mortgage"] * 4


@pytest.fixture(scope="session")
def synthetic_model_path(tmp_path_factory):
    """A tiny pipeline trained on synthetic text, saved with the production saver."""

    model = train_model(SYNTHETIC_TRAINING_TEXTS, SYNTHETIC_TRAINING_LABELS)
    path = tmp_path_factory.mktemp("synthetic_model") / "synthetic_classifier.joblib"
    save_model(model, str(path))

    return path
