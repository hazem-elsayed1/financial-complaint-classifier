# FastAPI inference service for the complaint classifier.
#
# The image contains no model. Mount a trusted, fitted model read-only at
# /models/ticket_classifier.joblib (or point COMPLAINT_MODEL_PATH elsewhere).
# Without a usable model the container exits at startup.
#
#   docker build -t complaint-classifier-api:m2 .
#   docker run --rm --read-only -p 127.0.0.1:8000:8000 \
#     --mount type=bind,source="$PWD/models/ticket_classifier.joblib",target=/models/ticket_classifier.joblib,readonly \
#     complaint-classifier-api:m2

# Python 3.14.6, pinned by digest, matches local development. CI selects
# python-version "3.14" (newest 3.14.x patch), so CI may run a later patch.
FROM python:3.14.6-slim@sha256:7bec7ddcddeff7975d6ba9b4be7dd6f6b2f55e7491539145e2978f7f97ce9144

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_ROOT_USER_ACTION=ignore \
    COMPLAINT_MODEL_PATH=/models/ticket_classifier.joblib

WORKDIR /app

COPY requirements-api.txt .
RUN python -m pip install -r requirements-api.txt

# Only the two modules the service imports; training and evaluation code stay out.
COPY src/api.py src/predict.py src/

RUN useradd --uid 10001 --user-group --no-create-home --shell /usr/sbin/nologin api
USER 10001

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4)"]

CMD ["python", "-m", "uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
