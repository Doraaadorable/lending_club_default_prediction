import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from loan_default.registry import export, promote
from loan_default.tracking import log_run

mlflow = pytest.importorskip("mlflow")


def save_run(directory: Path, coefficient_sign: float) -> dict[str, object]:
    """Fit a tiny pipeline and write it with metadata, the way train.py does."""
    X = pd.DataFrame({"x": np.linspace(-1, 1, 40)})
    y = (coefficient_sign * X["x"] > 0).astype(int)
    model = Pipeline([("model", LogisticRegression())]).fit(X, y)
    metadata: dict[str, object] = {
        "model": "LogisticRegression",
        "model_params": {"C": 1.0},
        "seed": 42,
        "approval_rate": 0.8,
        "approval_cutoff": 0.5,
        "n_train": 40,
        "n_test": 0,
        "metrics": {"test": {"roc_auc": 1.0}},
        "data_sha256": {"train": "a" * 64, "test": "b" * 64},
        "scikit_learn": "1.9.1",
    }
    directory.mkdir(parents=True)
    joblib.dump(model, directory / "model.joblib")
    (directory / "metadata.json").write_text(json.dumps(metadata))
    return metadata


def test_promoted_version_is_the_one_exported(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)  # MLflow stores artifacts relative to the working directory
    uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    for version, sign in ((1, 1.0), (2, -1.0)):
        metadata = save_run(tmp_path / f"run{version}", sign)
        log_run(metadata, tmp_path / f"run{version}", tracking_uri=uri, experiment="test", register_as="demo")

    promote(1, name="demo", tracking_uri=uri)
    assert export(tmp_path / "served", name="demo", tracking_uri=uri) == 1

    served = joblib.load(tmp_path / "served" / "model.joblib")
    assert served.predict(pd.DataFrame({"x": [0.9]}))[0] == 1  # version 1 learned "positive x defaults"
    metadata = json.loads((tmp_path / "served" / "metadata.json").read_text())
    assert metadata["approval_cutoff"] == 0.5
    assert metadata["registered_model"] == {"name": "demo", "version": 1, "alias": "champion"}

    promote(2, name="demo", tracking_uri=uri)
    assert export(tmp_path / "served", name="demo", tracking_uri=uri) == 2
    assert joblib.load(tmp_path / "served" / "model.joblib").predict(pd.DataFrame({"x": [0.9]}))[0] == 0
