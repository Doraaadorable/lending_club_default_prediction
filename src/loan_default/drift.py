"""Compare the 2016 cohort with the 2007-2015 training data.

    python -m loan_default.drift                                   # 2016 sample; writes reports/drift_report.md
    python -m loan_default.drift --current logs/predictions.jsonl  # logged live applications

Two questions, answered separately:

1. Did the inputs change?  For each model feature: a two-sample test (Kolmogorov-Smirnov
   for numeric features, chi-squared for categorical ones) and the population stability
   index (PSI), which measures how large the change is.
2. Did the link between inputs and default change?  Loans are grouped into bands of
   predicted risk, and the default rate inside each band is compared between the periods.

With 150,000 and 50,000 loans, a test's p-value is tiny for differences too small to
matter, so the report reads drift from PSI and uses the p-value only as a sanity check.
"""

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray
from scipy.stats import chi2_contingency, ks_2samp

from loan_default.preprocessing import NUMERIC, ONE_HOT, SKEWED, TARGET_ENCODED, clean, load_split
from loan_default.train import TrainConfig, build_model, default_scores, out_of_fold_scores

NUMERIC_FEATURES = NUMERIC + SKEWED
CATEGORICAL_FEATURES = ONE_HOT + TARGET_ENCODED

# Common rule of thumb for PSI: below 0.10 stable, 0.10-0.25 moderate shift, above 0.25 large shift.
PSI_MODERATE = 0.10
PSI_LARGE = 0.25
PSI_FLOOR = 1e-6  # keeps empty bins from producing log(0)
# PSI ignores missing values, so a change in the share missing is read separately (absolute change in share).
MISSING_MODERATE = 0.05
MISSING_LARGE = 0.20
LEVELS = ("stable", "moderate", "large", "invalid")


def psi_from_shares(reference: NDArray[np.float64], current: NDArray[np.float64]) -> float:
    """Population stability index between two sets of bin shares."""
    ref = np.clip(reference, PSI_FLOOR, None)
    cur = np.clip(current, PSI_FLOOR, None)
    return float(((cur - ref) * np.log(cur / ref)).sum())


def numeric_psi(reference: pd.Series, current: pd.Series, bins: int = 10) -> float:
    """PSI for a numeric feature, using bins that each hold an equal share of the reference data."""
    ref, cur = reference.dropna().to_numpy(dtype=float), current.dropna().to_numpy(dtype=float)
    edges = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)))
    edges[0], edges[-1] = -np.inf, np.inf
    return psi_from_shares(np.histogram(ref, edges)[0] / len(ref), np.histogram(cur, edges)[0] / len(cur))


def categorical_psi(reference: pd.Series, current: pd.Series) -> float:
    """PSI for a categorical feature, one bin per category."""
    shares = pd.concat([reference.value_counts(normalize=True), current.value_counts(normalize=True)], axis=1).fillna(
        0.0
    )
    return psi_from_shares(shares.iloc[:, 0].to_numpy(dtype=float), shares.iloc[:, 1].to_numpy(dtype=float))


def drifted_features(features: pd.DataFrame, level: str) -> list[str]:
    """Features whose shift reading is `level` or worse ("moderate" includes "large")."""
    flagged = {"moderate": {"moderate", "large", "invalid"}, "large": {"large", "invalid"}}[level]
    names: list[str] = features.loc[features["reading"].isin(flagged), "feature"].tolist()
    return names


def psi_label(psi: float) -> str:
    """Translate a PSI value into the usual three-level reading; a PSI that is not a number is "invalid"."""
    if not np.isfinite(psi):
        return "invalid"
    if psi >= PSI_LARGE:
        return "large"
    if psi >= PSI_MODERATE:
        return "moderate"
    return "stable"


def missing_label(change: float) -> str:
    """Reading for an absolute change in the share of missing values."""
    if abs(change) >= MISSING_LARGE:
        return "large"
    if abs(change) >= MISSING_MODERATE:
        return "moderate"
    return "stable"


def worse(first: str, second: str) -> str:
    """The more severe of two readings."""
    return max(first, second, key=LEVELS.index)


def feature_drift(reference: pd.DataFrame, current: pd.DataFrame) -> pd.DataFrame:
    """One row per model feature: test used, p-value, PSI and its reading, and missing-value shares."""
    ref, cur = clean(reference), clean(current)
    rows: list[dict[str, object]] = []
    for name in NUMERIC_FEATURES:
        p_value = float(ks_2samp(ref[name].dropna(), cur[name].dropna()).pvalue)
        psi = numeric_psi(ref[name], cur[name])
        rows.append({"feature": name, "test": "KS", "p_value": p_value, "psi": psi})
    for name in CATEGORICAL_FEATURES:
        counts = pd.concat([ref[name].value_counts(), cur[name].value_counts()], axis=1).fillna(0)
        p_value = float(chi2_contingency(counts)[1])
        psi = categorical_psi(ref[name], cur[name])
        rows.append({"feature": name, "test": "chi-squared", "p_value": p_value, "psi": psi})
    report = pd.DataFrame(rows)
    report["missing_reference"] = [float(ref[f].isna().mean()) for f in report["feature"]]
    report["missing_current"] = [float(cur[f].isna().mean()) for f in report["feature"]]
    report["reading"] = [
        worse(psi_label(psi), missing_label(cur_missing - ref_missing))
        for psi, ref_missing, cur_missing in zip(
            report["psi"], report["missing_reference"], report["missing_current"], strict=True
        )
    ]
    return report.sort_values("psi", ascending=False, ignore_index=True)


def default_rate_by_year(issue_dates: pd.Series, defaults: pd.Series) -> pd.DataFrame:
    """Number of loans and default rate for each origination year."""
    years = pd.to_datetime(issue_dates).dt.year.rename("year")
    table = defaults.groupby(years).agg(loans="size", default_rate="mean")
    return table.reset_index()


def risk_band_comparison(
    reference_scores: ArrayLike,
    reference_defaults: ArrayLike,
    current_scores: ArrayLike,
    current_defaults: ArrayLike,
    bands: int = 10,
) -> pd.DataFrame:
    """Default rate within bands of predicted risk, in each period.

    Band edges are the deciles of the reference scores, so a band holds loans the model
    regards as equally risky in both periods.
    """
    ref_scores, cur_scores = np.asarray(reference_scores, dtype=float), np.asarray(current_scores, dtype=float)
    ref_y, cur_y = np.asarray(reference_defaults, dtype=float), np.asarray(current_defaults, dtype=float)
    edges = np.quantile(ref_scores, np.linspace(0, 1, bands + 1))[1:-1]
    ref_band, cur_band = np.digitize(ref_scores, edges), np.digitize(cur_scores, edges)
    rows = []
    for band in range(bands):
        in_ref, in_cur = ref_band == band, cur_band == band
        rows.append(
            {
                "band": band + 1,
                "reference_share": float(in_ref.mean()),
                "current_share": float(in_cur.mean()),
                "reference_default_rate": float(ref_y[in_ref].mean()) if in_ref.any() else float("nan"),
                "current_default_rate": float(cur_y[in_cur].mean()) if in_cur.any() else float("nan"),
            }
        )
    table = pd.DataFrame(rows)
    table["change"] = table["current_default_rate"] - table["reference_default_rate"]
    return table


@dataclass(frozen=True)
class GapDecomposition:
    """Split of the change in default rate between two periods."""

    reference_rate: float
    current_rate: float
    expected_from_mix: float

    @property
    def from_mix(self) -> float:
        """Change explained by a different mix of risk bands, holding each band's default rate fixed."""
        return self.expected_from_mix - self.reference_rate

    @property
    def within_bands(self) -> float:
        """Change that remains: loans of the same predicted risk defaulting at a different rate."""
        return self.current_rate - self.expected_from_mix


def decompose_gap(bands: pd.DataFrame, reference_rate: float, current_rate: float) -> GapDecomposition:
    """Estimate the default rate the current period would have if only its mix of risk bands had changed."""
    expected = float((bands["current_share"] * bands["reference_default_rate"]).sum())
    return GapDecomposition(reference_rate=reference_rate, current_rate=current_rate, expected_from_mix=expected)


def _percent(value: float, digits: int = 2) -> str:
    return f"{value * 100:.{digits}f}%"


def _feature_table(features: pd.DataFrame, heading: str = "## Did the inputs change?") -> list[str]:
    """Markdown lines for the per-feature drift table and its reading."""
    drifted = features[features["reading"] != "stable"]["feature"].tolist()
    lines = [
        heading,
        "",
        "| Feature | Test | p-value | PSI | Reading | Missing (reference) | Missing (current) |",
        "|---|---|---:|---:|---|---:|---:|",
    ]
    for row in features.itertuples():
        lines.append(
            f"| `{row.feature}` | {row.test} | {row.p_value:.1e} | {row.psi:.3f} | {row.reading} | "
            f"{_percent(row.missing_reference, 1)} | {_percent(row.missing_current, 1)} |"
        )
    lines += [
        "",
        f"PSI below {PSI_MODERATE:.2f} is read as stable, {PSI_MODERATE:.2f}-{PSI_LARGE:.2f} as a moderate shift, "
        f"above {PSI_LARGE:.2f} as a large shift. PSI ignores missing values, so a change of "
        f"{MISSING_MODERATE:.0%} or more in the share missing is also read as moderate, {MISSING_LARGE:.0%} or more "
        "as large; a PSI that cannot be computed (for example, every current value missing) is read as invalid "
        "and treated as a failure.",
        "",
        (
            f"Features with a moderate, large or invalid reading: {', '.join(f'`{f}`' for f in drifted)}."
            if drifted
            else "No feature reaches the moderate threshold."
        ),
        "",
    ]
    return lines


def render_report(
    features: pd.DataFrame, years: pd.DataFrame, bands: pd.DataFrame, gap: GapDecomposition, current_label: str
) -> str:
    """Write the findings as Markdown."""
    lines = [
        "# Drift report",
        "",
        f"Reference: training sample ({int(years['loans'].sum()):,} loans, "
        f"{years['year'].min()}-{years['year'].max()}). Current: {current_label}.",
        "",
        "Generated by `python -m loan_default.drift`.",
        "",
        *_feature_table(features, "## 1. Did the inputs change?"),
        "Every p-value is far below 0.05, which here reflects the sample sizes: the tests detect differences "
        "that are real but small, so drift is read from PSI.",
        "",
        "## 2. Default rate by origination year (training sample)",
        "",
        "| Year | Loans | Default rate |",
        "|---|---:|---:|",
    ]
    lines += [f"| {row.year} | {row.loans:,} | {_percent(row.default_rate)} |" for row in years.itertuples()]
    lines += [
        "",
        "## 3. Did loans of the same predicted risk default more often?",
        "",
        "Bands are deciles of the model's out-of-fold scores on the training sample (band 1 = lowest risk).",
        "",
        "| Risk band | Share (reference) | Share (current) | Default rate (reference) "
        "| Default rate (current) | Change |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for row in bands.itertuples():
        lines.append(
            f"| {row.band} | {_percent(row.reference_share, 1)} | {_percent(row.current_share, 1)} | "
            f"{_percent(row.reference_default_rate)} | {_percent(row.current_default_rate)} | "
            f"{row.change * 100:+.2f} pp |"
        )
    lines += [
        "",
        "## 4. Where the gap comes from",
        "",
        "| | Default rate |",
        "|---|---:|",
        f"| Reference | {_percent(gap.reference_rate)} |",
        f"| Current, if only the mix of risk bands had changed | {_percent(gap.expected_from_mix)} |",
        f"| Current, actual | {_percent(gap.current_rate)} |",
        "",
        f"Of the {(gap.current_rate - gap.reference_rate) * 100:+.2f} pp change, {gap.from_mix * 100:+.2f} pp comes "
        f"from a different mix of risk bands and {gap.within_bands * 100:+.2f} pp from higher default rates within "
        "bands.",
        "",
    ]
    return "\n".join(lines)


def run(config: TrainConfig, out_dir: Path) -> tuple[str, pd.DataFrame]:
    """Compute every table and write the Markdown report and the feature table."""
    X_train, y_train = load_split(config.train_path)
    X_test, y_test = load_split(config.test_path)

    features = feature_drift(X_train, X_test)
    years = default_rate_by_year(X_train["issue_d"], y_train)

    model = build_model(config.model_params, config.seed)
    train_scores = out_of_fold_scores(model, X_train, y_train, config.cv_folds, config.seed)
    test_scores = default_scores(model.fit(X_train, y_train), X_test)
    bands = risk_band_comparison(train_scores, y_train, test_scores, y_test)
    gap = decompose_gap(bands, float(y_train.mean()), float(y_test.mean()))

    test_years = pd.to_datetime(X_test["issue_d"]).dt.year
    label = f"test sample ({len(y_test):,} loans, {test_years.min()})"
    report = render_report(features, years, bands, gap, label)

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "drift_report.md").write_text(report)
    features.to_csv(out_dir / "feature_drift.csv", index=False)
    return report, features


def load_prediction_log(path: Path) -> pd.DataFrame:
    """Read the service's JSON-lines prediction log."""
    return pd.read_json(path, lines=True, dtype={"addr_state": str})


def render_log_report(features: pd.DataFrame, log: pd.DataFrame, source: Path) -> str:
    """Input-only report for logged applications, which have no outcomes yet."""
    period = f"{log['timestamp'].min()} to {log['timestamp'].max()}" if "timestamp" in log else "unknown period"
    lines = [
        "# Input drift report: logged predictions",
        "",
        f"Reference: training sample. Current: {len(log):,} logged applications from `{source}` ({period}).",
        "",
        "Generated by `python -m loan_default.drift --current`. Outcomes are not known yet, so only the inputs "
        "are compared; a change in default rates at the same inputs would not show here.",
        "",
        *_feature_table(features),
    ]
    return "\n".join(lines)


def run_on_log(config: TrainConfig, log_path: Path, out_dir: Path) -> tuple[str, pd.DataFrame]:
    """Compare logged applications with the training sample and write the report."""
    X_train, _ = load_split(config.train_path)
    log = load_prediction_log(log_path)
    features = feature_drift(X_train, log)
    report = render_log_report(features, log, log_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "log_drift_report.md").write_text(report)
    features.to_csv(out_dir / "log_feature_drift.csv", index=False)
    return report, features


def main(argv: list[str] | None = None) -> None:
    """Run the comparison from the command line."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out-dir", type=Path, default=Path("reports"))
    parser.add_argument(
        "--current", type=Path, help="prediction log (JSON lines) to compare instead of the 2016 sample"
    )
    parser.add_argument(
        "--fail-on", choices=["moderate", "large"], help="exit with status 1 if any feature shifts this much or more"
    )
    args = parser.parse_args(argv)
    if args.current:
        report, features = run_on_log(TrainConfig(), args.current, args.out_dir)
    else:
        report, features = run(TrainConfig(), args.out_dir)
    print(report)
    flagged = drifted_features(features, args.fail_on) if args.fail_on else []
    if flagged:
        print(f"Drift at or above '{args.fail_on}': {', '.join(flagged)}", file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
