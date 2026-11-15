"""Train the final model and save it with its metadata.

    python -m loan_default.train                       # data/ -> models/
    python -m loan_default.train --out-dir somewhere

Fits the tuned gradient boosting pipeline chosen in notebook 05 on the
2007-2015 sample, evaluates it on the 2016 sample next to the FICO-only
baseline, and writes:

    model.joblib    preprocessing and model as one fitted pipeline
    metadata.json   parameters, metrics, the approval cutoff, and data hashes
"""

import argparse
import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import sklearn
from numpy.typing import NDArray
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from loan_default.policy import approve_safest, cutoff_at, policy_results
from loan_default.preprocessing import build_preprocessor, load_split

# Best of the 40 configurations in notebook 05's randomized search, as printed there (4 decimals).
TUNED_PARAMS: dict[str, float | int] = {
    "learning_rate": 0.0249,
    "max_iter": 510,
    "max_leaf_nodes": 15,
    "min_samples_leaf": 337,
    "l2_regularization": 0.0257,
    "max_features": 0.5823,
}


@dataclass(frozen=True)
class TrainConfig:
    """Everything a training run depends on."""

    train_path: Path = Path("data/train.csv")
    test_path: Path = Path("data/test.csv")
    out_dir: Path = Path("models")
    approval_rate: float = 0.8
    cv_folds: int = 5
    seed: int = 42
    model_params: dict[str, float | int] = field(default_factory=lambda: dict(TUNED_PARAMS))


def build_model(params: dict[str, float | int], seed: int) -> Pipeline:
    """Shared preprocessing followed by gradient boosting (unfitted)."""
    return Pipeline(
        [
            ("preprocess", build_preprocessor()),
            ("model", HistGradientBoostingClassifier(**params, random_state=seed)),
        ]
    )


def build_baseline(seed: int) -> Pipeline:
    """Logistic regression on FICO alone (unfitted): the benchmark the model has to beat."""
    fico = Pipeline([("imputer", SimpleImputer(strategy="median", add_indicator=True)), ("scaler", StandardScaler())])
    return Pipeline(
        [
            ("preprocess", ColumnTransformer([("fico", fico, ["fico_n"])])),
            ("model", LogisticRegression(max_iter=2000, random_state=seed)),
        ]
    )


def default_scores(model: Pipeline, X: pd.DataFrame) -> NDArray[np.float64]:
    """Predicted probability of default for each loan."""
    scores: NDArray[np.float64] = model.predict_proba(X)[:, 1]
    return scores


def out_of_fold_scores(model: Pipeline, X: pd.DataFrame, y: pd.Series, folds: int, seed: int) -> NDArray[np.float64]:
    """Score every training loan with a model that did not see it during fitting."""
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    scores: NDArray[np.float64] = cross_val_predict(model, X, y, cv=cv, method="predict_proba", n_jobs=-1)[:, 1]
    return scores


def evaluate(y: pd.Series, scores: NDArray[np.float64], approval_rate: float) -> dict[str, float]:
    """Ranking, probability and policy metrics for one set of scores."""
    policy = policy_results(y, approve_safest(scores, approval_rate))
    return {
        "roc_auc": float(roc_auc_score(y, scores)),
        "average_precision": float(average_precision_score(y, scores)),
        "brier": float(brier_score_loss(y, scores)),
        "mean_predicted_default": float(scores.mean()),
        "actual_default_rate": float(y.mean()),
        "default_rate_approved": policy["default_rate_approved"],
        "default_rate_rejected": policy["default_rate_rejected"],
    }


def sha256_of(path: Path) -> str:
    """Hash a file so the metadata records exactly which data was used."""
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def train(config: TrainConfig) -> dict[str, Any]:
    """Fit, evaluate and save the model; return the metadata that was written."""
    X_train, y_train = load_split(config.train_path)
    X_test, y_test = load_split(config.test_path)
    if not set(X_train["id"]).isdisjoint(X_test["id"]):
        raise ValueError("train and test share loan ids")

    model = build_model(config.model_params, config.seed)
    baseline = build_baseline(config.seed)

    # The cutoff comes from out-of-fold training scores, so the test set plays no part in choosing it.
    oof = out_of_fold_scores(model, X_train, y_train, config.cv_folds, config.seed)
    cutoff = cutoff_at(oof, config.approval_rate)

    model.fit(X_train, y_train)
    baseline.fit(X_train, y_train)
    test_scores = default_scores(model, X_test)

    metadata: dict[str, Any] = {
        "model": "HistGradientBoostingClassifier",
        "model_params": config.model_params,
        "seed": config.seed,
        "approval_rate": config.approval_rate,
        "approval_cutoff": cutoff,
        "n_train": int(len(y_train)),
        "n_test": int(len(y_test)),
        "metrics": {
            "train_out_of_fold": evaluate(y_train, oof, config.approval_rate),
            "test": evaluate(y_test, test_scores, config.approval_rate),
            "test_fico_baseline": evaluate(y_test, default_scores(baseline, X_test), config.approval_rate),
            "test_approval_rate_at_cutoff": float((test_scores <= cutoff).mean()),
        },
        "data_sha256": {"train": sha256_of(config.train_path), "test": sha256_of(config.test_path)},
        "scikit_learn": sklearn.__version__,
    }

    config.out_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, config.out_dir / "model.joblib")
    (config.out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return metadata


def parse_args(argv: list[str] | None = None) -> TrainConfig:
    """Build the configuration from command-line arguments."""
    defaults = asdict(TrainConfig())
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--train-path", type=Path, default=defaults["train_path"])
    parser.add_argument("--test-path", type=Path, default=defaults["test_path"])
    parser.add_argument("--out-dir", type=Path, default=defaults["out_dir"])
    parser.add_argument("--approval-rate", type=float, default=defaults["approval_rate"])
    args = parser.parse_args(argv)
    return TrainConfig(
        train_path=args.train_path, test_path=args.test_path, out_dir=args.out_dir, approval_rate=args.approval_rate
    )


def main(argv: list[str] | None = None) -> None:
    """Train from the command line and print the headline numbers."""
    config = parse_args(argv)
    metadata = train(config)
    test, fico = metadata["metrics"]["test"], metadata["metrics"]["test_fico_baseline"]
    print(f"Saved {config.out_dir / 'model.joblib'}")
    print(f"Test ROC-AUC {test['roc_auc']:.4f} (FICO only {fico['roc_auc']:.4f})")
    print(
        f"Default rate among the safest {config.approval_rate:.0%}: "
        f"{test['default_rate_approved']:.2%} (FICO only {fico['default_rate_approved']:.2%})"
    )
    print(f"Approval cutoff from training data: {metadata['approval_cutoff']:.4f}")


if __name__ == "__main__":
    main()
