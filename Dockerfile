# Stage 1: train the model. The data and training code stay in this stage.
FROM python:3.13-slim AS train

ENV PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
WORKDIR /build

COPY constraints.txt pyproject.toml ./
COPY src ./src
RUN pip install -c constraints.txt .

COPY data/train.csv data/test.csv ./data/
RUN python -m loan_default.train --out-dir /build/models


# Stage 2: the serving image holds the package and the trained model, not the data.
FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    MODEL_DIR=/app/models

WORKDIR /app

COPY constraints.txt pyproject.toml ./
COPY src ./src
RUN pip install -c constraints.txt ".[serve]"

COPY --from=train /build/models ./models

RUN useradd --uid 10001 --no-create-home appuser
USER 10001

EXPOSE 8000
HEALTHCHECK --interval=15s --timeout=3s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=2)"

CMD ["uvicorn", "loan_default.service:app", "--host", "0.0.0.0", "--port", "8000"]
