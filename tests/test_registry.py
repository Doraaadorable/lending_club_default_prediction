import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from loan_default.registry import export, promote
from loan_default.service import release_dir
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

    served = joblib.load(release_dir(tmp_path / "served") / "model.joblib")
    assert served.predict(pd.DataFrame({"x": [0.9]}))[0] == 1  # version 1 learned "positive x defaults"
    metadata = json.loads((release_dir(tmp_path / "served") / "metadata.json").read_text())
    assert metadata["approval_cutoff"] == 0.5
    assert metadata["registered_model"] == {"name": "demo", "version": 1, "alias": "champion"}

    promote(2, name="demo", tracking_uri=uri)
    assert export(tmp_path / "served", name="demo", tracking_uri=uri) == 2
    assert joblib.load(release_dir(tmp_path / "served") / "model.joblib").predict(pd.DataFrame({"x": [0.9]}))[0] == 0


def test_the_project_pipeline_can_be_registered_and_exported(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The real pipeline contains types MLflow's model format must be told to trust."""
    from loan_default.train import TrainConfig, train

    monkeypatch.chdir(tmp_path)
    data = Path(__file__).resolve().parents[1] / "data"
    pd.read_csv(data / "train.csv", nrows=3000, low_memory=False).to_csv(tmp_path / "train.csv", index=False)
    pd.read_csv(data / "test.csv", nrows=500, low_memory=False).to_csv(tmp_path / "test.csv", index=False)
    config = TrainConfig(
        train_path=tmp_path / "train.csv",
        test_path=tmp_path / "test.csv",
        out_dir=tmp_path / "run",
        model_params={"max_iter": 20, "max_leaf_nodes": 7},
        require_beats_baseline=False,  # a tiny sample; this test is about registration, not quality
    )
    metadata = train(config)
    uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    log_run(metadata, config.out_dir, tracking_uri=uri, experiment="test", register_as="loans")
    promote(1, name="loans", tracking_uri=uri)
    export(tmp_path / "served", name="loans", tracking_uri=uri)

    served = joblib.load(release_dir(tmp_path / "served") / "model.joblib")
    original = joblib.load(config.out_dir / "model.joblib")
    rows = pd.read_csv(config.test_path, low_memory=False).drop(columns=["Default"]).head(20)
    np.testing.assert_allclose(served.predict_proba(rows), original.predict_proba(rows))


def test_a_weaker_version_needs_force(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    for version, auc in ((1, 0.70), (2, 0.65)):
        metadata = save_run(tmp_path / f"run{version}", 1.0)
        metadata["metrics"] = {"test": {"roc_auc": auc}}
        (tmp_path / f"run{version}" / "metadata.json").write_text(json.dumps(metadata))
        log_run(metadata, tmp_path / f"run{version}", tracking_uri=uri, experiment="test", register_as="demo")

    promote(1, name="demo", tracking_uri=uri)
    with pytest.raises(ValueError, match="below champion version 1"):
        promote(2, name="demo", tracking_uri=uri)
    promote(2, name="demo", tracking_uri=uri, force=True)
    assert export(tmp_path / "served", name="demo", tracking_uri=uri) == 2


def test_each_export_is_a_new_release_and_current_points_at_the_latest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    for version, sign in ((1, 1.0), (2, -1.0)):
        metadata = save_run(tmp_path / f"run{version}", sign)
        log_run(metadata, tmp_path / f"run{version}", tracking_uri=uri, experiment="test", register_as="demo")
    promote(1, name="demo", tracking_uri=uri)
    export(tmp_path / "served", name="demo", tracking_uri=uri)
    first = release_dir(tmp_path / "served")
    promote(2, name="demo", tracking_uri=uri, force=True)
    export(tmp_path / "served", name="demo", tracking_uri=uri)
    second = release_dir(tmp_path / "served")
    assert first != second and first.exists()  # the old release is left untouched for rollback
    assert json.loads((second / "metadata.json").read_text())["registered_model"]["version"] == 2
    assert (tmp_path / "served" / "current").is_symlink()
    assert not list((tmp_path / "served").glob(".current*"))  # no half-written link is left behind
