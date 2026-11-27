"""Streamlit demo for the financial complaint classifier.

The app reuses the project's inference code: the model is located with the
same rules as the FastAPI service (``COMPLAINT_MODEL_PATH`` or
``models/ticket_classifier.joblib``), loaded with ``load_model``, checked with
``validate_model``, and every prediction comes from ``predict_complaint``.
"""

import re

import streamlit as st

from src.api import resolve_model_path, validate_model
from src.predict import load_model, predict_complaint


# Invented complaints for trying the app; none come from the CFPB data.
SYNTHETIC_EXAMPLES = {
    "Duplicate credit card charge": (
        "My credit card payment was charged twice, and the bank has refused "
        "to correct the transaction."
    ),
    "Misapplied mortgage payment": (
        "The mortgage servicer misapplied my monthly payment and now reports me as late."
    ),
    "Calls about a paid debt": (
        "A debt collector keeps calling my workplace about a debt I already paid off."
    ),
    "Unknown account on credit report": (
        "My credit report lists an account I never opened and the bureau will not remove it."
    ),
}

EMPTY_INPUT_MESSAGE = (
    "Enter a complaint or choose a synthetic example, then select **Classify complaint**."
)
STALE_RESULT_MESSAGE = (
    "The complaint has changed since the last prediction. "
    "Select **Classify complaint** to classify the new text."
)
EXCERPT_LENGTH = 140


def escape_markdown(text: str) -> str:
    """Show user text and labels literally instead of as Markdown."""

    return re.sub(r"([\\`*_{}\[\]()#+\-.!|<>~:$])", r"\\\1", text)


def excerpt(text: str) -> str:
    text = " ".join(text.split())
    if len(text) <= EXCERPT_LENGTH:
        return text
    return text[: EXCERPT_LENGTH - 1].rstrip() + "…"


@st.cache_resource(show_spinner="Loading the model…")
def get_model(model_path: str):
    """Load and validate the trained pipeline once per model path."""

    model = load_model(model_path)
    validate_model(model)
    return model


def load_model_state(model_path: str):
    """Return (model, problem) where problem is None, "missing" or "invalid"."""

    try:
        return get_model(model_path), None
    except FileNotFoundError:
        return None, "missing"
    except Exception:
        return None, "invalid"


def use_example() -> None:
    choice = st.session_state.example
    if choice is not None:
        st.session_state.complaint_text = SYNTHETIC_EXAMPLES[choice]
    st.session_state.notice = None


def text_edited() -> None:
    # The example picker no longer describes hand-edited text.
    st.session_state.example = None
    st.session_state.notice = None


def classify(model) -> None:
    text = st.session_state.complaint_text
    st.session_state.result = None

    if not text.strip():
        st.session_state.notice = "empty"
        return

    try:
        product = predict_complaint(model, text.strip())
    except Exception:
        st.session_state.notice = "error"
        return

    st.session_state.notice = None
    st.session_state.result = {"text": text, "product": product}


st.set_page_config(
    page_title="Financial Complaint Classifier",
    page_icon=":material/fact_check:",
    layout="centered",
)

for key, default in (
    ("complaint_text", ""),
    ("example", None),
    ("result", None),
    ("notice", None),
):
    st.session_state.setdefault(key, default)

model_path = resolve_model_path()
model, model_problem = load_model_state(model_path)

st.title("Financial Complaint Classifier", anchor=False)
st.markdown(
    "Predicts which of 11 financial products a consumer complaint is about. "
    "Choose a synthetic example or write a complaint, select **Classify complaint**, "
    "and read the predicted product in the result panel."
)

if model_problem == "missing":
    st.error(
        f"**The trained model was not found.** The app looks for "
        f"`{model_path}`. Train and save the model as described in the "
        "README, or set `COMPLAINT_MODEL_PATH` to a trusted model file, then reload this page.",
        icon=":material/error:",
    )
elif model_problem == "invalid":
    st.error(
        "**The model file could not be used.** It must be a trusted, fitted scikit-learn "
        "pipeline saved by this project. Replace it, then reload this page.",
        icon=":material/error:",
    )

st.header("Complaint", anchor=False)
st.selectbox(
    "Try a synthetic example (optional)",
    options=list(SYNTHETIC_EXAMPLES),
    index=None,
    key="example",
    placeholder="Choose an example to fill in the complaint",
    on_change=use_example,
)
st.text_area(
    "Enter your financial complaint",
    key="complaint_text",
    height=160,
    placeholder="Describe the problem in your own words, for example what happened and which company was involved.",
    on_change=text_edited,
)
st.button(
    "Classify complaint",
    key="classify",
    type="primary",
    width="stretch",
    disabled=model is None,
    on_click=classify,
    args=(model,),
)

st.header("Result", anchor=False)
with st.container(border=True, key="result_panel"):
    result = st.session_state.result
    current_text = st.session_state.complaint_text
    notice = st.session_state.notice

    if model is None:
        st.caption("Predicted product")
        st.markdown("Classification is unavailable until a model is loaded.")
    elif notice == "empty":
        st.warning(EMPTY_INPUT_MESSAGE, icon=":material/edit_note:")
    elif notice == "error":
        st.error(
            "The complaint could not be classified. Check the model and application setup.",
            icon=":material/error:",
        )
    elif result is not None and result["text"] == current_text:
        st.caption("Predicted product")
        st.subheader(f":primary[{escape_markdown(result['product'])}]", anchor=False)
        st.caption(f"For the complaint: “{escape_markdown(excerpt(result['text']))}”")
    elif result is not None:
        st.info(STALE_RESULT_MESSAGE, icon=":material/update:")
    else:
        st.caption("Predicted product")
        st.markdown("No prediction yet. Your result will appear here.")

st.caption(
    "Model: TF-IDF features with Logistic Regression, trained on CFPB complaint narratives. "
    "The examples are synthetic. Predictions are informational and should not replace "
    "human review."
)
