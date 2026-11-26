"""Causal inference: does borrowing a large share of income raise the chance of default?

    python -m loan_default.causal        # writes reports/causal_report.md and .csv

A predictive model says that loans which are large for the borrower's income default more often.
A lending-policy team wants to know something different: if the same borrower were offered a
smaller loan, would they be less likely to default? Borrowers who take large loans differ in
other ways (purpose, credit history, housing), so the raw difference mixes the effect of the loan
with the effect of who asks for it.

Treatment   loan amount at least 25% of annual income (37% of 2007-2015 loans)
Outcome     default (charged off)
Adjusted for FICO, DTI, employment length (and whether it is missing), income, loan purpose,
            housing status, state and issue year: everything known at application, except the
            loan amount itself, which defines the treatment.

Estimators of the average treatment effect (ATE), all with 5-fold cross-fitting so no loan is
scored by a model fitted on it:

naive                   difference in default rates
regression adjustment   predict default with and without treatment for every loan (T-learner)
inverse propensity      reweight loans by 1 / P(treatment | covariates) (Hajek estimator)
doubly robust (AIPW)    regression adjustment corrected by the weighted residuals; consistent if
                        either the outcome or the propensity model is right; gives a standard error

Checks: covariate balance before and after weighting, propensity overlap, and three refutation
tests (placebo treatment, an added random confounder, and a random half of the data).

What it cannot fix: confounders the data does not record. Lending Club set the interest rate,
grade and term (36 or 60 months) after seeing the application, and larger loans are more often
60-month loans. Term and rate are partly consequences of the loan size (mediators) and partly
signals of the lender's private information, so the estimate is the effect of a large loan
including the terms that come with it, under the assumption of no unrecorded confounding.
"""

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import OneHotEncoder

from loan_default.experiments import Splits, load_splits, write_report
from loan_default.preprocessing import clean

Array = NDArray[np.float64]
THRESHOLD = 0.25
CLIP = 0.01  # propensities are clipped to [0.01, 0.99] to keep weights finite
FOLDS = 5
SEED = 42
NUMERIC_CONFOUNDERS = ["fico_n", "dti_n", "emp_length", "log_income", "issue_year"]
CATEGORICAL_CONFOUNDERS = ["purpose", "home_ownership_n", "addr_state"]
FICO_BANDS = {"below 680": (0, 680), "680-719": (680, 720), "720 and above": (720, 1000)}


def causal_frame(X: pd.DataFrame, threshold: float = THRESHOLD) -> tuple[pd.DataFrame, NDArray[np.int64]]:
    """Confounders known at application, and the treatment indicator."""
    cleaned = clean(X)
    frame = pd.DataFrame(
        {
            "fico_n": cleaned["fico_n"],
            "dti_n": cleaned["dti_n"],
            "emp_length": cleaned["emp_length"],
            "log_income": np.log(cleaned["revenue"]),
            "issue_year": pd.to_datetime(X["issue_d"]).dt.year.astype(float),
            "purpose": cleaned["purpose"],
            "home_ownership_n": cleaned["home_ownership_n"],
            "addr_state": cleaned["addr_state"],
        }
    )
    treated = (cleaned["loan_to_income"] >= threshold).astype(np.int64).to_numpy()
    return frame.reset_index(drop=True), treated


def nuisance_model(numeric: list[str], seed: int = SEED, max_iter: int = 300) -> Pipeline:
    """Gradient boosting on the confounders; used for both the propensity and the outcome models."""
    columns = ColumnTransformer(
        [
            ("num", SimpleImputer(strategy="median", add_indicator=True), numeric),
            (
                "cat",
                OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=100, sparse_output=False),
                CATEGORICAL_CONFOUNDERS,
            ),
        ]
    )
    model = HistGradientBoostingClassifier(
        learning_rate=0.05, max_iter=max_iter, max_leaf_nodes=15, min_samples_leaf=200, random_state=seed
    )
    return make_pipeline(columns, model)


@dataclass(frozen=True)
class Nuisance:
    """Cross-fitted propensity and outcome predictions for every loan."""

    propensity: Array
    outcome_treated: Array
    outcome_control: Array


def cross_fit(
    W: pd.DataFrame, t: NDArray[np.int64], y: Array, seed: int = SEED, folds: int = FOLDS, max_iter: int = 300
) -> Nuisance:
    """Fit propensity and outcome models on k-1 folds and predict the held-out fold, for each fold."""
    numeric = [c for c in W.columns if c not in CATEGORICAL_CONFOUNDERS]
    propensity, mu1, mu0 = np.zeros(len(y)), np.zeros(len(y)), np.zeros(len(y))
    strata = t * 2 + y.astype(np.int64)  # keep both treatment and outcome balanced across folds
    for fit_idx, held_idx in StratifiedKFold(folds, shuffle=True, random_state=seed).split(W, strata):
        W_fit, W_held = W.iloc[fit_idx], W.iloc[held_idx]
        t_fit, y_fit = t[fit_idx], y[fit_idx]
        propensity[held_idx] = nuisance_model(numeric, seed, max_iter).fit(W_fit, t_fit).predict_proba(W_held)[:, 1]
        treated, control = t_fit == 1, t_fit == 0
        mu1[held_idx] = (
            nuisance_model(numeric, seed, max_iter).fit(W_fit[treated], y_fit[treated]).predict_proba(W_held)[:, 1]
        )
        mu0[held_idx] = (
            nuisance_model(numeric, seed, max_iter).fit(W_fit[control], y_fit[control]).predict_proba(W_held)[:, 1]
        )
    return Nuisance(np.clip(propensity, CLIP, 1 - CLIP), mu1, mu0)


def aipw_scores(t: NDArray[np.int64], y: Array, n: Nuisance) -> Array:
    """Doubly robust score per loan; its mean is the ATE and its spread gives the standard error."""
    out: Array = (
        n.outcome_treated
        - n.outcome_control
        + t * (y - n.outcome_treated) / n.propensity
        - (1 - t) * (y - n.outcome_control) / (1 - n.propensity)
    )
    return out


def estimates(t: NDArray[np.int64], y: Array, n: Nuisance) -> dict[str, dict[str, float]]:
    """ATE from each estimator; AIPW also gets a 95% confidence interval."""
    w1, w0 = t / n.propensity, (1 - t) / (1 - n.propensity)
    scores = aipw_scores(t, y, n)
    se = float(scores.std(ddof=1) / np.sqrt(len(y)))
    aipw = float(scores.mean())
    return {
        "naive difference": {"ate": float(y[t == 1].mean() - y[t == 0].mean())},
        "regression adjustment": {"ate": float((n.outcome_treated - n.outcome_control).mean())},
        "inverse propensity weighting": {"ate": float((w1 * y).sum() / w1.sum() - (w0 * y).sum() / w0.sum())},
        "doubly robust (AIPW)": {"ate": aipw, "se": se, "low": aipw - 1.96 * se, "high": aipw + 1.96 * se},
    }


def standardized_difference(values: Array, t: NDArray[np.int64], weights: Array | None = None) -> float:
    """Difference in (weighted) means between treated and control, in pooled standard deviations."""
    w = np.ones(len(values)) if weights is None else weights
    treated, control = t == 1, t == 0
    mean1 = np.average(values[treated], weights=w[treated])
    mean0 = np.average(values[control], weights=w[control])
    var1 = np.average((values[treated] - mean1) ** 2, weights=w[treated])
    var0 = np.average((values[control] - mean0) ** 2, weights=w[control])
    return float((mean1 - mean0) / np.sqrt((var1 + var0) / 2 + 1e-12))


def balance_table(W: pd.DataFrame, t: NDArray[np.int64], propensity: Array) -> pd.DataFrame:
    """Standardized mean differences before and after inverse propensity weighting."""
    weights = t / propensity + (1 - t) / (1 - propensity)
    columns: dict[str, Array] = {c: W[c].fillna(W[c].median()).to_numpy(dtype=float) for c in NUMERIC_CONFOUNDERS}
    for column, level in [
        ("purpose", "debt_consolidation"),
        ("purpose", "credit_card"),
        ("home_ownership_n", "RENT"),
        ("home_ownership_n", "MORTGAGE"),
    ]:
        columns[f"{column} = {level}"] = (W[column] == level).to_numpy(dtype=float)
    return pd.DataFrame(
        [
            {
                "covariate": name,
                "smd_before": standardized_difference(values, t),
                "smd_after": standardized_difference(values, t, weights),
            }
            for name, values in columns.items()
        ]
    )


def refutations(W: pd.DataFrame, t: NDArray[np.int64], y: Array, seed: int = SEED) -> pd.DataFrame:
    """Re-estimate the AIPW effect under three changes that should (or should not) move it."""
    rng = np.random.default_rng(seed)
    rows = []

    placebo = rng.permutation(t)
    rows.append(("placebo treatment (shuffled)", "close to 0", placebo, W, y))

    noisy = W.copy()
    noisy["random_confounder"] = rng.normal(size=len(W))
    rows.append(("random confounder added", "unchanged", t, noisy, y))

    half = rng.random(len(y)) < 0.5
    rows.append(
        ("random half of the loans", "unchanged, wider interval", t[half], W[half].reset_index(drop=True), y[half])
    )

    out = []
    for name, expected, t_r, W_r, y_r in rows:
        nuisance = cross_fit(W_r, t_r, y_r, seed)
        result = estimates(t_r, y_r, nuisance)["doubly robust (AIPW)"]
        out.append({"test": name, "expected": expected} | result)
    return pd.DataFrame(out)


def effect_by_fico(W: pd.DataFrame, t: NDArray[np.int64], y: Array, n: Nuisance) -> pd.DataFrame:
    """AIPW effect within FICO bands (averaging the doubly robust scores in each band)."""
    scores = aipw_scores(t, y, n)
    rows = []
    for name, (low, high) in FICO_BANDS.items():
        band = W["fico_n"].between(low, high, inclusive="left").to_numpy()
        se = float(scores[band].std(ddof=1) / np.sqrt(band.sum()))
        rows.append(
            {
                "fico_band": name,
                "n": int(band.sum()),
                "share_treated": float(t[band].mean()),
                "ate": float(scores[band].mean()),
                "low": float(scores[band].mean() - 1.96 * se),
                "high": float(scores[band].mean() + 1.96 * se),
            }
        )
    return pd.DataFrame(rows)


def overlap_summary(t: NDArray[np.int64], propensity: Array) -> dict[str, float]:
    """How much of each group has propensities in the thin tails, where weights get large."""
    return {
        "treated_median": float(np.median(propensity[t == 1])),
        "control_median": float(np.median(propensity[t == 0])),
        "share_outside_5_95": float(((propensity < 0.05) | (propensity > 0.95)).mean()),
        "max_weight": float(np.max(t / propensity + (1 - t) / (1 - propensity))),
    }


def _pp(value: float) -> str:
    return f"{value * 100:+.2f} pp"


def render_report(
    results: dict[str, dict[str, dict[str, float]]],
    balance: pd.DataFrame,
    overlap: dict[str, float],
    refuted: pd.DataFrame,
    by_fico: pd.DataFrame,
    treated_share: float,
) -> str:
    """Write the findings as Markdown."""
    lines = [
        "# Causal effect of a large loan relative to income",
        "",
        "Generated by `python -m loan_default.causal`.",
        "",
        f"Treatment: loan amount at least {THRESHOLD:.0%} of annual income ({treated_share:.1%} of 2007-2015 loans). "
        "Outcome: default. Effects are in percentage points (pp) of default probability.",
        "",
        "## Average treatment effect",
        "",
        "| Estimator | 2007-2015 | 2016 (replication) |",
        "|---|---:|---:|",
    ]
    for name in results["2007-2015"]:
        cells = []
        for period in ("2007-2015", "2016"):
            r = results[period][name]
            cell = _pp(r["ate"])
            if "low" in r:
                cell += f" (95% CI {r['low'] * 100:+.2f} to {r['high'] * 100:+.2f})"
            cells.append(cell)
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    lines += [
        "",
        "## Is the comparison fair? Covariate balance",
        "",
        "Standardized mean difference between treated and control loans; below 0.1 is usually read as balanced.",
        "",
        "| Covariate | Before weighting | After weighting |",
        "|---|---:|---:|",
        *[f"| {r.covariate} | {r.smd_before:+.3f} | {r.smd_after:+.3f} |" for r in balance.itertuples()],
        "",
        f"Propensity overlap: median {overlap['treated_median']:.2f} among treated and "
        f"{overlap['control_median']:.2f} among control loans; {overlap['share_outside_5_95']:.1%} of loans "
        f"fall outside 0.05-0.95; largest weight {overlap['max_weight']:.0f}.",
        "",
        "## Refutation tests (AIPW)",
        "",
        "| Test | Expected | Estimate | 95% CI |",
        "|---|---|---:|---|",
        *[
            f"| {r.test} | {r.expected} | {_pp(r.ate)} | {r.low * 100:+.2f} to {r.high * 100:+.2f} pp |"
            for r in refuted.itertuples()
        ],
        "",
        "## Does the effect differ by credit score?",
        "",
        "| FICO band | Loans | Share treated | Effect (AIPW) | 95% CI |",
        "|---|---:|---:|---:|---|",
        *[
            f"| {r.fico_band} | {r.n:,} | {r.share_treated:.1%} | {_pp(r.ate)} | "
            f"{r.low * 100:+.2f} to {r.high * 100:+.2f} pp |"
            for r in by_fico.itertuples()
        ],
        "",
    ]
    return "\n".join(lines)


def run(splits: Splits, out_dir: Path) -> str:
    """Estimate the effect in both periods, check it, and write the report."""
    W, t = causal_frame(splits.X_train)
    y = splits.y_train.to_numpy(dtype=float)
    nuisance = cross_fit(W, t, y)
    W_test, t_test = causal_frame(splits.X_test)
    y_test = splits.y_test.to_numpy(dtype=float)
    results = {
        "2007-2015": estimates(t, y, nuisance),
        "2016": estimates(t_test, y_test, cross_fit(W_test, t_test, y_test)),
    }
    balance = balance_table(W, t, nuisance.propensity)
    refuted = refutations(W, t, y)
    by_fico = effect_by_fico(W, t, y, nuisance)
    report = render_report(results, balance, overlap_summary(t, nuisance.propensity), refuted, by_fico, float(t.mean()))
    table = pd.DataFrame(
        [{"period": period, "estimator": name} | values for period, r in results.items() for name, values in r.items()]
    )
    write_report(out_dir, "causal_report", report, table)
    return report


def main(argv: list[str] | None = None) -> None:
    """Run the analysis from the command line."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out-dir", type=Path, default=Path("reports"))
    args = parser.parse_args(argv)
    print(run(load_splits(), args.out_dir))


if __name__ == "__main__":
    main()
