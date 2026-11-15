"""Prediction service: score one loan application with the model saved by train.py.

    uvicorn loan_default.service:app --port 8000

Interactive documentation is served at /docs. Set PREDICTION_LOG to a file path to
append every prediction to it as one JSON line; `python -m loan_default.drift
--current <file>` then compares the logged applications with the training data.
"""

import hashlib
import json
import os
import threading
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field
from sklearn.pipeline import Pipeline

EmploymentLength = Literal[
    "< 1 year",
    "1 year",
    "2 years",
    "3 years",
    "4 years",
    "5 years",
    "6 years",
    "7 years",
    "8 years",
    "9 years",
    "10+ years",
    "NI",
]


class LoanApplication(BaseModel):
    """Application-time information about one loan. Field names match the training data."""

    revenue: float = Field(gt=0, description="Self-reported annual income, USD")
    loan_amnt: float = Field(gt=0, description="Requested loan amount, USD")
    fico_n: float = Field(ge=300, le=850, description="FICO score (midpoint of the reported range)")
    dti_n: float | None = Field(default=None, ge=0, description="Debt-to-income ratio, percent; omit if unknown")
    emp_length: EmploymentLength = Field(default="NI", description="Employment length; NI means no information")
    purpose: str = Field(min_length=1, description="Loan purpose, e.g. debt_consolidation, credit_card, car")
    home_ownership_n: str = Field(min_length=1, description="RENT, MORTGAGE, OWN or OTHER")
    addr_state: str = Field(pattern="^[A-Z]{2}$", description="Two-letter US state code")


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
    row = pd.DataFrame([application.model_dump()])
    probability = float(model.predict_proba(row)[0, 1])
    cutoff = float(metadata["approval_cutoff"])
    prediction = Prediction(
        default_probability=round(probability, 4),
        decision="approve" if probability <= cutoff else "reject",
        approval_cutoff=round(cutoff, 4),
    )
    log_prediction(application, prediction, metadata["model_sha256"])
    return prediction
