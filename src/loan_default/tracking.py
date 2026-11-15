"""Record a training run in MLflow: parameters, metrics and the saved artifacts.

MLflow is an optional dependency (`pip install -e ".[tracking]"`); it is imported
only when a run is actually logged, so training works without it.
"""

from pathlib import Path
from typing import Any

DEFAULT_TRACKING_URI = "sqlite:///mlflow.db"
DEFAULT_EXPERIMENT = "loan-default"
DEFAULT_MODEL_NAME = "loan-default"

# MLflow saves scikit-learn models with skops, which refuses types it cannot verify as safe
# to load. These are the ones this project's pipeline contains; each was reviewed:
# our own cleaning function, a numpy dtype, the gradient-boosting tree nodes, and the
# fold splitter inside the target encoder. A new type in the pipeline makes registration
# fail until it is reviewed and added here.
SKOPS_TRUSTED_TYPES = [
    "loan_default.preprocessing.clean",
    "numpy.dtype",
    "sklearn.ensemble._hist_gradient_boosting.predictor.TreePredictor",
    "sklearn.model_selection._split.StratifiedKFold",
]


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
    register_as: str | None = None,
) -> str:
    """Log one finished training run and return its MLflow run id.

    With `register_as`, the fitted pipeline is also logged as an MLflow model and added
    to the model registry under that name as a new version.
    """
    import joblib
    import mlflow
    import mlflow.sklearn

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
        if register_as:
            mlflow.sklearn.log_model(
                joblib.load(out_dir / "model.joblib"),
                name="model",
                registered_model_name=register_as,
                skops_trusted_types=SKOPS_TRUSTED_TYPES,
            )
        run_id: str = run.info.run_id
    return run_id
