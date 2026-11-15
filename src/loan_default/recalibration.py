"""Assess whether recalibrating on recent loans fixes the model's understated probabilities.

    python -m loan_default.recalibration        # writes reports/recalibration_report.md

The model is trained on 2007-2015. Its scores for January-June 2016 loans are used to fit
a correction, and the corrected probabilities are judged on July-December 2016 loans.
Three corrections are compared, from least to most flexible:

    intercept shift   move every probability by one constant on the log-odds scale
    Platt scaling     a constant and a slope on the log-odds scale
    isotonic          any monotonic mapping

All three keep the ordering of loans (isotonic can only create ties), so ranking metrics
and the approve-the-safest-share policy are unaffected; only the probabilities change.

This is an exploratory check on the test period, not a deployment recipe: in practice the
outcomes of loans issued in the first half of a year are not known by its second half.
"""

import argparse
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray
from scipy.optimize import brentq
from scipy.special import expit, logit
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score

from loan_default.preprocessing import load_split
from loan_default.train import TrainConfig, build_model, default_scores

Calibrator = Callable[[NDArray[np.float64]], NDArray[np.float64]]

CLIP = 1e-6  # keeps log-odds finite for probabilities of exactly 0 or 1
CALIBRATION_MONTHS = range(1, 7)


def log_odds(probabilities: ArrayLike) -> NDArray[np.float64]:
    """Probabilities on the log-odds scale."""
    result: NDArray[np.float64] = logit(np.clip(np.asarray(probabilities, dtype=float), CLIP, 1 - CLIP))
    return result


def fit_intercept_shift(scores: ArrayLike, defaults: ArrayLike) -> Calibrator:
    """Find the constant log-odds shift that makes the mean probability equal the observed default rate."""
    z, target = log_odds(scores), float(np.mean(np.asarray(defaults, dtype=float)))
    shift = float(brentq(lambda d: float(expit(z + d).mean()) - target, -10, 10))

    def calibrate(new_scores: NDArray[np.float64]) -> NDArray[np.float64]:
        result: NDArray[np.float64] = expit(log_odds(new_scores) + shift)
        return result

    return calibrate


def fit_platt(scores: ArrayLike, defaults: ArrayLike) -> Calibrator:
    """Fit a logistic regression of the outcome on the score's log-odds (intercept and slope)."""
    model = LogisticRegression(C=1e6, max_iter=1000).fit(log_odds(scores).reshape(-1, 1), np.asarray(defaults))

    def calibrate(new_scores: NDArray[np.float64]) -> NDArray[np.float64]:
        result: NDArray[np.float64] = model.predict_proba(log_odds(new_scores).reshape(-1, 1))[:, 1]
        return result

    return calibrate


def fit_isotonic(scores: ArrayLike, defaults: ArrayLike) -> Calibrator:
    """Fit a monotonic step function from score to observed default rate."""
    model = IsotonicRegression(y_min=0, y_max=1, out_of_bounds="clip").fit(np.asarray(scores), np.asarray(defaults))

    def calibrate(new_scores: NDArray[np.float64]) -> NDArray[np.float64]:
        result: NDArray[np.float64] = model.predict(new_scores)
        return result

    return calibrate


def identity(scores: NDArray[np.float64]) -> NDArray[np.float64]:
    """The uncorrected scores."""
    return scores


def calibration_error(defaults: ArrayLike, probabilities: ArrayLike, bins: int = 10) -> tuple[float, float]:
    """Average and largest gap between predicted and observed default rates over equal-sized risk bins."""
    y, p = np.asarray(defaults, dtype=float), np.asarray(probabilities, dtype=float)
    order = np.argsort(p, kind="stable")
    gaps = [abs(float(y[idx].mean()) - float(p[idx].mean())) for idx in np.array_split(order, bins)]
    return float(np.mean(gaps)), float(np.max(gaps))


def summarize(defaults: ArrayLike, probabilities: ArrayLike) -> dict[str, float]:
    """Calibration and ranking metrics for one set of probabilities."""
    y, p = np.asarray(defaults, dtype=float), np.asarray(probabilities, dtype=float)
    mean_gap, max_gap = calibration_error(y, p)
    return {
        "mean_predicted": float(p.mean()),
        "actual": float(y.mean()),
        "brier": float(brier_score_loss(y, p)),
        "mean_bin_gap": mean_gap,
        "max_bin_gap": max_gap,
        "roc_auc": float(roc_auc_score(y, p)),
    }


def compare(
    fit_scores: NDArray[np.float64], fit_defaults: ArrayLike, eval_scores: NDArray[np.float64], eval_defaults: ArrayLike
) -> pd.DataFrame:
    """Fit each correction on one period and evaluate it on a later one."""
    calibrators: dict[str, Calibrator] = {
        "No recalibration": identity,
        "Intercept shift": fit_intercept_shift(fit_scores, fit_defaults),
        "Platt scaling": fit_platt(fit_scores, fit_defaults),
        "Isotonic": fit_isotonic(fit_scores, fit_defaults),
    }
    rows = [{"method": name, **summarize(eval_defaults, c(eval_scores))} for name, c in calibrators.items()]
    return pd.DataFrame(rows)


def render_report(results: pd.DataFrame, n_fit: int, n_eval: int, fit_rate: float) -> str:
    """Write the comparison as Markdown."""
    lines = [
        "# Recalibration report",
        "",
        f"Model trained on 2007-2015. Corrections fitted on {n_fit:,} loans issued January-June 2016 "
        f"(default rate {fit_rate * 100:.2f}%) and evaluated on {n_eval:,} loans issued July-December 2016.",
        "",
        "Generated by `python -m loan_default.recalibration`.",
        "",
        "| Method | Mean predicted | Actual | Brier | Mean gap over 10 risk bins | Largest gap | ROC-AUC |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in results.itertuples():
        lines.append(
            f"| {row.method} | {row.mean_predicted * 100:.2f}% | {row.actual * 100:.2f}% | {row.brier:.4f} | "
            f"{row.mean_bin_gap * 100:.2f} pp | {row.max_bin_gap * 100:.2f} pp | {row.roc_auc:.4f} |"
        )
    lines += [
        "",
        "A gap is the absolute difference between the mean predicted probability and the observed default rate "
        "in a bin; bins are tenths of the evaluation loans ordered by predicted probability.",
        "",
    ]
    return "\n".join(lines)


def run(config: TrainConfig, out_dir: Path) -> pd.DataFrame:
    """Train on the training sample, compare corrections across the two halves of 2016, write the report."""
    X_train, y_train = load_split(config.train_path)
    X_test, y_test = load_split(config.test_path)
    scores = default_scores(build_model(config.model_params, config.seed).fit(X_train, y_train), X_test)

    first_half = pd.to_datetime(X_test["issue_d"]).dt.month.isin(CALIBRATION_MONTHS).to_numpy()
    y = y_test.to_numpy()
    results = compare(scores[first_half], y[first_half], scores[~first_half], y[~first_half])

    out_dir.mkdir(parents=True, exist_ok=True)
    report = render_report(results, int(first_half.sum()), int((~first_half).sum()), float(y[first_half].mean()))
    (out_dir / "recalibration_report.md").write_text(report)
    print(report)
    return results


def main(argv: list[str] | None = None) -> None:
    """Run the comparison from the command line."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out-dir", type=Path, default=Path("reports"))
    args = parser.parse_args(argv)
    run(TrainConfig(), args.out_dir)


if __name__ == "__main__":
    main()
