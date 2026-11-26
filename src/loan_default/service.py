"""Prediction service: score one loan application with the model saved by train.py.

    uvicorn loan_default.service:app --port 8000

Interactive documentation is served at /docs. Set PREDICTION_LOG to a file path to
append every prediction to it as one JSON line; `python -m loan_default.drift
--current <file>` then compares the logged applications with the training data.
"""

import hashlib
import json
import math
import os
import threading
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sklearn.pipeline import Pipeline

from loan_default.schema import LoanApplication


class Prediction(BaseModel):
    """The model's risk score and the policy decision that follows from it."""

    default_probability: float
    decision: Literal["approve", "reject"]
    approval_cutoff: float


class Health(BaseModel):
    """Liveness information for container health checks."""

    status: Literal["ok"]
    model: str


app = FastAPI(
    title="Loan default risk",
    description="Scores a loan application and applies the approve-the-safest-80% cutoff chosen on training data.",
    version="0.1.0",
)


def model_dir() -> Path:
    """Directory holding model.joblib and metadata.json; set MODEL_DIR to change it."""
    return Path(os.environ.get("MODEL_DIR", "models"))


@lru_cache(maxsize=1)
def artifacts() -> tuple[Pipeline, dict[str, Any]]:
    """Load the fitted pipeline and its metadata once per process.

    The metadata gains `model_sha256`, a hash of the model file, so logged predictions
    can be traced to the exact model that made them.
    """
    directory = model_dir()
    model: Pipeline = joblib.load(directory / "model.joblib")
    metadata: dict[str, Any] = json.loads((directory / "metadata.json").read_text())
    metadata["model_sha256"] = hashlib.sha256((directory / "model.joblib").read_bytes()).hexdigest()
    return model, metadata


_log_lock = threading.Lock()


def log_prediction(application: LoanApplication, prediction: Prediction, model_sha256: str) -> None:
    """Append one prediction to the file named by PREDICTION_LOG; do nothing if it is unset."""
    path = os.environ.get("PREDICTION_LOG")
    if not path:
        return
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "model_sha256": model_sha256,
        **application.model_dump(),
        **prediction.model_dump(),
    }
    with _log_lock, open(path, "a", encoding="utf-8") as log:
        log.write(json.dumps(record) + "\n")


@app.get("/health")
def health() -> Health:
    """Report that the service is up and which model it serves."""
    _, metadata = artifacts()
    return Health(status="ok", model=metadata["model"])


@app.get("/metadata")
def model_metadata() -> dict[str, Any]:
    """Return the metadata written at training time: parameters, metrics, cutoff and data hashes."""
    _, metadata = artifacts()
    return metadata


@app.post("/predict")
def predict(application: LoanApplication) -> Prediction:
    """Score one application; approve it if its default probability is at or below the cutoff."""
    model, metadata = artifacts()
    row = pd.DataFrame([application.model_dump(exclude={"application_id"})])
    probability = float(model.predict_proba(row)[0, 1])
    if not math.isfinite(probability) or not 0 <= probability <= 1:
        raise HTTPException(status_code=500, detail=f"model returned an invalid score: {probability}")
    cutoff = float(metadata["approval_cutoff"])
    # full precision: a rounded score could equal the rounded cutoff and contradict the decision
    prediction = Prediction(
        default_probability=probability,
        decision="approve" if probability <= cutoff else "reject",
        approval_cutoff=cutoff,
    )
    log_prediction(application, prediction, metadata["model_sha256"])
    return prediction
