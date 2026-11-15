"""Train the final model and save it with its metadata.

    python -m loan_default.train                       # data/ -> models/
    python -m loan_default.train --out-dir somewhere
    python -m loan_default.train --track               # also record the run in MLflow
    python -m loan_default.train --register            # record the run and add it to the model registry

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
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from loan_default.policy import approve_safest, cutoff_at, policy_results
from loan_default.preprocessing import build_preprocessor, load_split
from loan_default.schema import FIELDS, check_frame
from loan_default.tracking import DEFAULT_MODEL_NAME, DEFAULT_TRACKING_URI, log_run

# Best of the 40 configurations in notebook 05's randomized search, at full precision.
# Reproduced by `python -m loan_default.search`; all 40 are in reports/search_results.csv.
TUNED_PARAMS: dict[str, float | int] = {
    "learning_rate": 0.024918939689346737,
    "max_iter": 510,
    "max_leaf_nodes": 15,
    "min_samples_leaf": 337,
    "l2_regularization": 0.025720264197882346,
    "max_features": 0.5823279265714709,
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
    require_beats_baseline: bool = True


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


FICO_BANDS = {"fico_below_680": (0, 680), "fico_680_719": (680, 720), "fico_720_up": (720, 1000)}
MIN_SLICE_SIZE = 200


def slice_metrics(X: pd.DataFrame, y: pd.Series, scores: NDArray[np.float64]) -> dict[str, dict[str, float]]:
    """ROC-AUC, default rate and size within FICO bands and housing groups large enough to score."""
    masks = {name: X["fico_n"].between(low, high, inclusive="left") for name, (low, high) in FICO_BANDS.items()}
    masks |= {f"home_{value}": X["home_ownership_n"] == value for value in sorted(X["home_ownership_n"].unique())}
    result = {}
    for name, mask in masks.items():
        y_slice = y[mask.to_numpy()]
        if len(y_slice) < MIN_SLICE_SIZE or y_slice.nunique() < 2:
            continue
        result[name] = {
            "n": float(len(y_slice)),
            "default_rate": float(y_slice.mean()),
            "roc_auc": float(roc_auc_score(y_slice, scores[mask.to_numpy()])),
        }
    return result


IMPORTANCE_SAMPLE = 5000
IMPORTANCE_REPEATS = 3


def input_importance(model: Pipeline, X: pd.DataFrame, y: pd.Series, seed: int) -> dict[str, float]:
    """Drop in test ROC-AUC when each application field is shuffled, on a sample of the test set."""
    sample = X.sample(min(IMPORTANCE_SAMPLE, len(X)), random_state=seed)
    result = permutation_importance(
        model, sample, y.loc[sample.index], scoring="roc_auc", n_repeats=IMPORTANCE_REPEATS, random_state=seed
    )
    by_column = dict(zip(sample.columns, result.importances_mean, strict=True))
    return {field: round(float(by_column[field]), 4) for field in FIELDS}


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
    check_frame(X_train, "training data")
    check_frame(X_test, "test data")
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
    baseline_scores = default_scores(baseline, X_test)
    model_auc, baseline_auc = roc_auc_score(y_test, test_scores), roc_auc_score(y_test, baseline_scores)
    if config.require_beats_baseline and model_auc <= baseline_auc:
        raise ValueError(
            f"model test ROC-AUC {model_auc:.4f} does not beat the FICO-only baseline ({baseline_auc:.4f})"
        )

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
            "test_fico_baseline": evaluate(y_test, baseline_scores, config.approval_rate),
            "test_slices": slice_metrics(X_test, y_test, test_scores),
            "test_input_importance": input_importance(model, X_test, y_test, config.seed),
            "test_approval_rate_at_cutoff": float((test_scores <= cutoff).mean()),
        },
        "data_sha256": {"train": sha256_of(config.train_path), "test": sha256_of(config.test_path)},
        "scikit_learn": sklearn.__version__,
    }

    config.out_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, config.out_dir / "model.joblib")
    (config.out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return metadata


def parse_args(argv: list[str] | None = None) -> tuple[TrainConfig, argparse.Namespace]:
    """Build the configuration from command-line arguments; also return the raw arguments."""
    defaults = asdict(TrainConfig())
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--train-path", type=Path, default=defaults["train_path"])
    parser.add_argument("--test-path", type=Path, default=defaults["test_path"])
    parser.add_argument("--out-dir", type=Path, default=defaults["out_dir"])
    parser.add_argument("--approval-rate", type=float, default=defaults["approval_rate"])
    parser.add_argument("--learning-rate", type=float, help="override the tuned learning rate")
    parser.add_argument("--max-iter", type=int, help="override the tuned number of boosting iterations")
    parser.add_argument("--track", action="store_true", help="record the run in MLflow")
    parser.add_argument("--tracking-uri", default=DEFAULT_TRACKING_URI, help="MLflow tracking URI")
    parser.add_argument(
        "--register", action="store_true", help="record the run and register the model (implies --track)"
    )
    args = parser.parse_args(argv)

    params = dict(TUNED_PARAMS)
    if args.learning_rate is not None:
        params["learning_rate"] = args.learning_rate
    if args.max_iter is not None:
        params["max_iter"] = args.max_iter
    config = TrainConfig(
        train_path=args.train_path,
        test_path=args.test_path,
        out_dir=args.out_dir,
        approval_rate=args.approval_rate,
        model_params=params,
    )
    return config, args


def main(argv: list[str] | None = None) -> None:
    """Train from the command line and print the headline numbers."""
    config, args = parse_args(argv)
    metadata = train(config)
    test, fico = metadata["metrics"]["test"], metadata["metrics"]["test_fico_baseline"]
    print(f"Saved {config.out_dir / 'model.joblib'}")
    print(f"Test ROC-AUC {test['roc_auc']:.4f} (FICO only {fico['roc_auc']:.4f})")
    print(
        f"Default rate among the safest {config.approval_rate:.0%}: "
        f"{test['default_rate_approved']:.2%} (FICO only {fico['default_rate_approved']:.2%})"
    )
    print(f"Approval cutoff from training data: {metadata['approval_cutoff']:.4f}")
    if args.track or args.register:
        register_as = DEFAULT_MODEL_NAME if args.register else None
        run_id = log_run(metadata, config.out_dir, tracking_uri=args.tracking_uri, register_as=register_as)
        print(f"Logged MLflow run {run_id} to {args.tracking_uri}")
        if register_as:
            print(f"Registered as a new version of '{register_as}'")


if __name__ == "__main__":
    main()
