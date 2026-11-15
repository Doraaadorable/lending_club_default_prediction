from pathlib import Path

import pytest

from loan_default.tracking import flatten_metrics, log_run

mlflow = pytest.importorskip("mlflow")

METADATA = {
    "model": "HistGradientBoostingClassifier",
    "model_params": {"learning_rate": 0.05, "max_iter": 20},
    "seed": 42,
    "approval_rate": 0.8,
    "approval_cutoff": 0.25,
    "n_train": 100,
    "n_test": 50,
    "metrics": {"test": {"roc_auc": 0.66, "brier": 0.17}, "test_approval_rate_at_cutoff": 0.8},
    "data_sha256": {"train": "a" * 64, "test": "b" * 64},
    "scikit_learn": "1.9.1",
}


def test_nested_metrics_get_dotted_names() -> None:
    assert flatten_metrics({"test": {"roc_auc": 0.66}, "cutoff_rate": 0.8}) == {
        "test.roc_auc": 0.66,
        "cutoff_rate": 0.8,
    }


def test_run_is_recorded_with_params_metrics_and_artifacts(tmp_path: Path) -> None:
    out_dir = tmp_path / "models"
    out_dir.mkdir()
    (out_dir / "model.joblib").write_bytes(b"model")
    (out_dir / "metadata.json").write_text("{}")
    uri = f"sqlite:///{tmp_path / 'mlflow.db'}"

    run_id = log_run(METADATA, out_dir, tracking_uri=uri, experiment="test")

    run = mlflow.get_run(run_id)
    assert run.data.params["learning_rate"] == "0.05"
    assert run.data.params["approval_rate"] == "0.8"
    assert run.data.metrics["test.roc_auc"] == 0.66
    assert run.data.metrics["approval_cutoff"] == 0.25
    assert run.data.tags["train_data_sha256"] == "a" * 64
    artifacts = {a.path for a in mlflow.MlflowClient(uri).list_artifacts(run_id)}
    assert artifacts == {"model.joblib", "metadata.json"}
