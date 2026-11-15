"""Record a training run in MLflow: parameters, metrics and the saved artifacts.

MLflow is an optional dependency (`pip install -e ".[tracking]"`); it is imported
only when a run is actually logged, so training works without it.
"""

from pathlib import Path
from typing import Any

DEFAULT_TRACKING_URI = "sqlite:///mlflow.db"
DEFAULT_EXPERIMENT = "loan-default"


def flatten_metrics(metrics: dict[str, Any], prefix: str = "") -> dict[str, float]:
    """Turn nested metric groups into flat names: {"test": {"roc_auc": 0.66}} -> {"test.roc_auc": 0.66}."""
    flat: dict[str, float] = {}
    for name, value in metrics.items():
        key = f"{prefix}{name}"
        if isinstance(value, dict):
            flat.update(flatten_metrics(value, prefix=f"{key}."))
        else:
            flat[key] = float(value)
    return flat


def log_run(
    metadata: dict[str, Any],
    out_dir: Path,
    tracking_uri: str = DEFAULT_TRACKING_URI,
    experiment: str = DEFAULT_EXPERIMENT,
) -> str:
    """Log one finished training run and return its MLflow run id."""
    import mlflow

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment)
    with mlflow.start_run() as run:
        mlflow.log_params(metadata["model_params"])
        mlflow.log_params(
            {
                "model": metadata["model"],
                "seed": metadata["seed"],
                "approval_rate": metadata["approval_rate"],
                "n_train": metadata["n_train"],
                "n_test": metadata["n_test"],
            }
        )
        mlflow.log_metrics(flatten_metrics(metadata["metrics"]))
        mlflow.log_metric("approval_cutoff", metadata["approval_cutoff"])
        mlflow.set_tags(
            {
                "train_data_sha256": metadata["data_sha256"]["train"],
                "test_data_sha256": metadata["data_sha256"]["test"],
                "scikit_learn": metadata["scikit_learn"],
            }
        )
        mlflow.log_artifact(str(out_dir / "model.joblib"))
        mlflow.log_artifact(str(out_dir / "metadata.json"))
        run_id: str = run.info.run_id
    return run_id
