"""Replay 2016 as if the model had gone live on 1 January 2016.

    python -m loan_default.replay      # writes reports/replay_report.md and logs/replay_2016.jsonl

1. Train on 2007-2015 exactly as train.py does.
2. Score each 2016 loan in its issue month and write the predictions in the service's log format.
3. Input monitor: compare each month's applications with the training data (PSI).
4. Outcome monitor: join outcomes and compare predicted and actual default rates by month.
5. For outcomes that become known 3, 6 or 12 months after issue: when the outcome monitor first
   fires, and how far quarterly or monthly recalibration closes the gap. A recalibration uses only
   2016 cohorts whose outcomes are known by then.
"""

import argparse
import hashlib
import json
import tempfile
from dataclasses import replace
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.metrics import brier_score_loss

from loan_default.drift import PSI_MODERATE, feature_drift
from loan_default.monitor import DEFAULT_MAX_GAP, calibration_by_month, join_outcomes, months_over
from loan_default.preprocessing import load_split
from loan_default.recalibration import fit_intercept_shift
from loan_default.schema import FIELDS
from loan_default.train import TrainConfig, default_scores, train

LAGS = (3, 6, 12)
POLICIES: dict[str, tuple[int, ...]] = {"none": (), "quarterly": (1, 4, 7, 10), "monthly": tuple(range(1, 13))}


def build_log(X: pd.DataFrame, scores: NDArray[np.float64], cutoff: float, model_sha256: str) -> pd.DataFrame:
    """Predictions in the service's log format, timestamped with each loan's issue date."""
    log = X[FIELDS].copy()
    log.insert(0, "application_id", X["id"].astype(str))
    log.insert(0, "timestamp", pd.to_datetime(X["issue_d"]).dt.strftime("%Y-%m-%dT00:00:00+00:00"))
    log["model_sha256"] = model_sha256
    log["default_probability"] = np.round(scores, 4)
    log["decision"] = np.where(scores <= cutoff, "approve", "reject")
    log["approval_cutoff"] = round(cutoff, 4)
    return log


def monthly_input_drift(X_train: pd.DataFrame, X_current: pd.DataFrame) -> pd.DataFrame:
    """For each month of current applications: the largest feature PSI against training data, and its feature."""
    months = pd.to_datetime(X_current["issue_d"]).dt.to_period("M").astype(str)
    rows = []
    for month in sorted(months.unique()):
        features = feature_drift(X_train, X_current[months == month])
        top = features.iloc[0]
        rows.append({"month": month, "max_psi": float(top["psi"]), "max_psi_feature": str(top["feature"])})
    return pd.DataFrame(rows)


def recalibrated(
    scores: NDArray[np.float64],
    defaults: NDArray[np.int_],
    issue_month: NDArray[np.int_],
    lag: int,
    refits: tuple[int, ...],
) -> NDArray[np.float64]:
    """Scores after recalibrating at each refit month on the cohorts whose outcomes are known by then.

    A loan issued in month i has a known outcome from month i + lag. Loans issued before the first
    refit with known outcomes keep their original score.
    """
    corrected: NDArray[np.float64] = np.array(scores, dtype=float)
    for month in range(1, 13):
        done = [r for r in refits if r <= month]
        if not done:
            continue
        known = issue_month + lag <= max(done)
        if not known.any():
            continue
        current = issue_month == month
        corrected[current] = fit_intercept_shift(scores[known], defaults[known])(scores[current])
    return corrected


def policy_comparison(
    scores: NDArray[np.float64], defaults: NDArray[np.int_], issue_month: NDArray[np.int_], max_gap: float
) -> pd.DataFrame:
    """Mean absolute monthly gap, months over the limit and Brier score for each lag and policy."""
    months = np.array([f"2016-{m:02d}" for m in issue_month])
    rows = []
    for lag in LAGS:
        for policy, refits in POLICIES.items():
            corrected = recalibrated(scores, defaults, issue_month, lag, refits)
            table = calibration_by_month(
                pd.DataFrame({"month": months, "default_probability": corrected, "default": defaults})
            )
            rows.append(
                {
                    "lag": lag,
                    "policy": policy,
                    "mean_abs_gap": float(table["gap"].abs().mean()),
                    "months_over": len(months_over(table, max_gap)),
                    "brier": float(brier_score_loss(defaults, corrected)),
                }
            )
    return pd.DataFrame(rows)


def alert_month(first_flagged: str | None, lag: int) -> str:
    """Month in which the outcome monitor can first fire, given the outcome delay."""
    if first_flagged is None:
        return "never"
    return str(pd.Period(first_flagged, "M") + lag)


def render(monthly: pd.DataFrame, policies: pd.DataFrame, first_flagged: str | None, max_gap: float) -> str:
    """Markdown report of the replay."""
    input_alerts = monthly.loc[monthly["max_psi"] >= PSI_MODERATE, "month"].tolist()
    lines = [
        "# 2016 replay",
        "",
        "The model trained on 2007-2015 scores every 2016 loan in its issue month, as the deployed service would.",
        "",
        "Generated by `python -m loan_default.replay`.",
        "",
        "## Month by month",
        "",
        "| Month | Loans | Largest input PSI | Mean predicted | Actual | Gap |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in monthly.itertuples():
        lines.append(
            f"| {row.month} | {row.loans:,} | {row.max_psi:.3f} (`{row.max_psi_feature}`) | "
            f"{row.mean_predicted * 100:.2f}% | {row.actual * 100:.2f}% | {row.gap * 100:+.2f} pp |"
        )
    lines += [
        "",
        f"Input monitor (PSI ≥ {PSI_MODERATE:.2f}): "
        + (f"fires in {', '.join(input_alerts)}." if input_alerts else "never fires."),
        "",
        f"Outcome monitor (gap > {max_gap * 100:.0f} points): first month over the limit is {first_flagged or 'none'}.",
        "",
        "## Outcome delay and recalibration",
        "",
        "| Outcomes known after | Outcome alert fires | Mean monthly gap: none | quarterly | monthly "
        "| Months over limit: none | quarterly | monthly |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for lag in LAGS:
        by = policies[policies["lag"] == lag].set_index("policy")
        lines.append(
            f"| {lag} months | {alert_month(first_flagged, lag)} | "
            + " | ".join(f"{by.loc[p, 'mean_abs_gap'] * 100:.2f} pp" for p in POLICIES)
            + " | "
            + " | ".join(str(int(by.loc[p, "months_over"])) for p in POLICIES)
            + " |"
        )
    lines += [
        "",
        "Recalibration is an intercept shift on the log-odds scale, fitted on the 2016 cohorts whose outcomes are "
        "known at each refit.",
        "",
    ]
    return "\n".join(lines)


def run(config: TrainConfig, out_dir: Path, log_dir: Path, max_gap: float = DEFAULT_MAX_GAP) -> str:
    """Run the replay, write the report and the replayed log, and return the report."""
    X_train, _ = load_split(config.train_path)
    X_test, y_test = load_split(config.test_path)

    with tempfile.TemporaryDirectory() as tmp:
        model_dir = Path(tmp)
        metadata = train(replace(config, out_dir=model_dir))
        model = joblib.load(model_dir / "model.joblib")
        model_sha = hashlib.sha256((model_dir / "model.joblib").read_bytes()).hexdigest()
    scores = default_scores(model, X_test)
    log = build_log(X_test, scores, float(metadata["approval_cutoff"]), model_sha)
    outcomes = pd.DataFrame({"application_id": X_test["id"].astype(str), "default": y_test.to_numpy()})

    log_dir.mkdir(parents=True, exist_ok=True)
    log.to_json(log_dir / "replay_2016.jsonl", orient="records", lines=True)
    outcomes.to_csv(log_dir / "replay_2016_outcomes.csv", index=False)

    calibration = calibration_by_month(join_outcomes(log, outcomes))
    monthly = calibration.merge(monthly_input_drift(X_train, X_test), on="month")
    flagged = months_over(calibration, max_gap)
    issue_month = pd.to_datetime(X_test["issue_d"]).dt.month.to_numpy()
    policies = policy_comparison(scores, y_test.to_numpy(), issue_month, max_gap)

    report = render(monthly, policies, flagged[0] if flagged else None, max_gap)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "replay_report.md").write_text(report)
    monthly.to_csv(out_dir / "replay_monthly.csv", index=False)
    policies.to_csv(out_dir / "replay_policies.csv", index=False)
    (out_dir / "replay_summary.json").write_text(
        json.dumps({"first_month_over_limit": flagged[0] if flagged else None, "max_gap": max_gap}, indent=2) + "\n"
    )
    return report


def main(argv: list[str] | None = None) -> None:
    """Run the replay from the command line."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out-dir", type=Path, default=Path("reports"))
    parser.add_argument("--log-dir", type=Path, default=Path("logs"))
    parser.add_argument("--max-gap", type=float, default=DEFAULT_MAX_GAP)
    args = parser.parse_args(argv)
    print(run(TrainConfig(), args.out_dir, args.log_dir, args.max_gap))


if __name__ == "__main__":
    main()
