"""Join outcomes to logged predictions and check calibration month by month.

    python -m loan_default.monitor --log logs/predictions.jsonl --outcomes outcomes.csv --max-gap 0.02

`outcomes.csv` has one row per resolved loan: `application_id` and `default` (0 or 1).
An application id may appear only once in the outcomes; if an application was scored more
than once, its first valid prediction (the one in force at the decision) is the one compared,
and log rows without a readable time or a score between 0 and 1 are left out and counted.
Predictions are grouped by the month they were made. Exit status: 1 if any month's actual
default rate differs from its mean predicted probability by more than `--max-gap`; 2 if no
prediction could be checked against an outcome, so there was nothing to check.
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

DEFAULT_MAX_GAP = 0.02


def prediction_months(timestamps: pd.Series) -> pd.Series:
    """Calendar month of each prediction, as 'YYYY-MM'."""
    moments = pd.to_datetime(timestamps, utc=True, format="mixed").dt.tz_convert(None)
    return moments.dt.to_period("M").astype(str)


def valid_predictions(log: pd.DataFrame) -> pd.DataFrame:
    """Log rows that can be checked: an application id, a readable time and a score between 0 and 1.

    Times without a zone are read as UTC (the service writes UTC with an offset).
    """
    moments = pd.to_datetime(log["timestamp"], utc=True, format="mixed", errors="coerce")
    scores = pd.to_numeric(log["default_probability"], errors="coerce")
    usable = log["application_id"].notna() & moments.notna() & scores.between(0, 1)
    return log[usable].assign(
        application_id=lambda d: d["application_id"].astype(str),
        _moment=moments[usable],
        default_probability=scores[usable],
    )


def join_outcomes(log: pd.DataFrame, outcomes: pd.DataFrame) -> pd.DataFrame:
    """Predictions that have a known outcome, with their month, score and result.

    Only valid log rows are used (see `valid_predictions`). When an application was scored more than
    once, the first valid score counts: it is the one in force when the decision was made, and a later
    re-score would otherwise move the loan to another month or credit it to a newer model. Rows with the
    same time keep their order in the log.
    """
    right = outcomes.assign(application_id=outcomes["application_id"].astype(str))[["application_id", "default"]]
    repeated = right["application_id"][right["application_id"].duplicated()].unique()
    if len(repeated):
        raise ValueError(f"outcomes list {len(repeated)} application id(s) more than once, e.g. {repeated[0]!r}")
    if not right["default"].isin([0, 1]).all():
        raise ValueError("every outcome must be 0 (repaid) or 1 (defaulted)")
    left = valid_predictions(log).sort_values("_moment", kind="stable").drop_duplicates("application_id", keep="first")
    joined = left.merge(right, on="application_id", how="inner", validate="one_to_one")
    return pd.DataFrame(
        {
            "month": joined["_moment"].dt.tz_convert(None).dt.to_period("M").astype(str),
            "default_probability": joined["default_probability"].astype(float),
            "default": joined["default"].astype(int),
        }
    )


def calibration_by_month(joined: pd.DataFrame) -> pd.DataFrame:
    """Loans, mean predicted probability, actual default rate and their gap for each month."""
    table = joined.groupby("month").agg(
        loans=("default", "size"), mean_predicted=("default_probability", "mean"), actual=("default", "mean")
    )
    table["gap"] = table["actual"] - table["mean_predicted"]
    return table.reset_index()


def months_over(table: pd.DataFrame, max_gap: float) -> list[str]:
    """Months whose absolute gap exceeds `max_gap`."""
    months: list[str] = table.loc[table["gap"].abs() > max_gap, "month"].tolist()
    return months


def render(table: pd.DataFrame, max_gap: float, source: Path) -> str:
    """Markdown report of the monthly check."""
    lines = [
        "# Outcome monitor",
        "",
        f"Predictions from `{source}` joined to known outcomes. A month is flagged when the actual default rate "
        f"differs from the mean predicted probability by more than {max_gap * 100:.1f} points.",
        "",
        "| Month | Loans | Mean predicted | Actual | Gap | Flagged |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for row in table.itertuples():
        flag = "yes" if abs(row.gap) > max_gap else ""
        lines.append(
            f"| {row.month} | {row.loans:,} | {row.mean_predicted * 100:.2f}% | {row.actual * 100:.2f}% | "
            f"{row.gap * 100:+.2f} pp | {flag} |"
        )
    return "\n".join([*lines, ""])


def main(argv: list[str] | None = None) -> None:
    """Run the check from the command line."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--log", type=Path, required=True, help="prediction log (JSON lines)")
    parser.add_argument("--outcomes", type=Path, required=True, help="CSV with application_id and default")
    parser.add_argument("--max-gap", type=float, default=DEFAULT_MAX_GAP)
    parser.add_argument("--out-dir", type=Path, default=Path("reports"))
    args = parser.parse_args(argv)

    log = pd.read_json(args.log, lines=True, dtype={"application_id": str, "addr_state": str})
    joined = join_outcomes(log, pd.read_csv(args.outcomes, dtype={"application_id": str}))
    logged = int(log["application_id"].dropna().nunique()) if "application_id" in log else 0
    usable = int(valid_predictions(log)["application_id"].nunique()) if len(log) else 0
    table = calibration_by_month(joined)
    checked = int(table["loans"].sum()) if len(table) else 0
    print(
        f"{logged:,} applications logged, {usable:,} with a valid time and score, "
        f"{len(joined):,} matched to an outcome, {checked:,} checked by month.",
        file=sys.stderr,
    )
    if checked == 0:
        print("No prediction could be checked against an outcome: nothing was checked.", file=sys.stderr)
        raise SystemExit(2)
    report = render(table, args.max_gap, args.log)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "outcome_monitor.md").write_text(report)
    print(report)
    flagged = months_over(table, args.max_gap)
    if flagged:
        print(f"Calibration gap above {args.max_gap:.3f} in: {', '.join(flagged)}", file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
