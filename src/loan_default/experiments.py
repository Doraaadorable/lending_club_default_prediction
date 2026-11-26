"""Shared harness for the comparison experiments (imputation, embeddings, clustering, ...).

Every experiment answers the same question for a set of candidate pipelines: does it rank
2007-2015 loans better in shuffled 5-fold CV, and does that carry over to the 2016 sample?
Each pipeline is scored the same way so the report tables can be read side by side.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.base import BaseEstimator
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from loan_default.policy import approve_safest
from loan_default.preprocessing import load_split

APPROVAL_RATE = 0.8
SEED = 42


@dataclass(frozen=True)
class Splits:
    """The development sample (2007-2015) and the test sample (2016)."""

    X_train: pd.DataFrame
    y_train: pd.Series
    X_test: pd.DataFrame
    y_test: pd.Series


def load_splits(train_path: Path = Path("data/train.csv"), test_path: Path = Path("data/test.csv")) -> Splits:
    """Read both samples."""
    X_train, y_train = load_split(train_path)
    X_test, y_test = load_split(test_path)
    return Splits(X_train, y_train, X_test, y_test)


def default_rate_approved(y: pd.Series | NDArray[np.int64], scores: NDArray[np.float64]) -> float:
    """Default rate among the lowest-risk APPROVAL_RATE share of loans."""
    return float(np.asarray(y)[approve_safest(scores, APPROVAL_RATE)].mean())


def score_metrics(y: pd.Series | NDArray[np.int64], scores: NDArray[np.float64]) -> dict[str, float]:
    """ROC-AUC, average precision and the default rate among approved loans."""
    return {
        "roc_auc": float(roc_auc_score(y, scores)),
        "average_precision": float(average_precision_score(y, scores)),
        "default_rate_at_80": default_rate_approved(y, scores),
    }


def compare(candidates: Mapping[str, BaseEstimator], splits: Splits, folds: int = 5, n_jobs: int = -1) -> pd.DataFrame:
    """CV metrics on the development sample and test metrics on 2016, one row per candidate."""
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=SEED)
    rows = []
    for name, model in candidates.items():
        oof = cross_val_predict(model, splits.X_train, splits.y_train, cv=cv, method="predict_proba", n_jobs=n_jobs)
        fitted = model.fit(splits.X_train, splits.y_train)
        test = fitted.predict_proba(splits.X_test)[:, 1]
        cv_metrics = score_metrics(splits.y_train, oof[:, 1])
        test_metrics = score_metrics(splits.y_test, test)
        rows.append(
            {"candidate": name}
            | {f"cv_{key}": value for key, value in cv_metrics.items()}
            | {f"test_{key}": value for key, value in test_metrics.items()}
        )
    return pd.DataFrame(rows)


def comparison_table(results: pd.DataFrame, baseline: str | None = None) -> list[str]:
    """Markdown table of `compare` results; with `baseline`, adds the test ROC-AUC change against it."""
    header = "| Candidate | CV ROC-AUC | CV AP | Test ROC-AUC | Test AP | Test default rate, safest 80% |"
    rule = "|---|---:|---:|---:|---:|---:|"
    if baseline is not None:
        header += " Test ROC-AUC vs baseline |"
        rule += "---:|"
        reference = float(results.loc[results["candidate"] == baseline, "test_roc_auc"].iloc[0])
    lines = [header, rule]
    for row in results.itertuples():
        line = (
            f"| {row.candidate} | {row.cv_roc_auc:.4f} | {row.cv_average_precision:.4f} | {row.test_roc_auc:.4f} | "
            f"{row.test_average_precision:.4f} | {row.test_default_rate_at_80 * 100:.2f}% |"
        )
        if baseline is not None:
            line += f" {float(row.test_roc_auc) - reference:+.4f} |"
        lines.append(line)
    return lines


def write_report(out_dir: Path, name: str, report: str, table: pd.DataFrame | None = None) -> None:
    """Save a Markdown report, and optionally its table as CSV, under `out_dir`."""
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{name}.md").write_text(report)
    if table is not None:
        table.to_csv(out_dir / f"{name}.csv", index=False)
