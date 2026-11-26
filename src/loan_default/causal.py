"""Observational causal inference: is borrowing a large share of income linked to more default, after adjustment?

    python -m loan_default.causal        # writes reports/causal_report.md and .csv

A predictive model says that loans which are large for the borrower's income default more often.
Borrowers who take large loans differ in other ways (purpose, credit history, housing), so the raw
difference mixes the loan with who asks for it. This module estimates the difference that remains
after adjusting for what is recorded at application. It reads as a causal effect only under the
assumptions below, and it describes a binary contrast, not the effect of a lending policy such as
a cap (which would also change who borrows and on what terms).

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

Checks: common support by income (treatment is defined from income, so large loans are absent at
high incomes), raw propensity overlap and effective sample sizes, covariate balance, sensitivity
to clipping, trimming to the overlap region and overlap weighting, and three stability checks
(placebo treatment, an added random confounder, a random half of the data). The stability checks
test the estimator, not the no-unrecorded-confounding assumption.

What it cannot fix: confounders the data does not record. Lending Club set the interest rate,
grade and term (36 or 60 months) after seeing the application, and larger loans are more often
60-month loans. Term and rate may be consequences of the loan size (mediators), signals of the
lender's private information (confounders), or both; the data cannot tell which, so it cannot
say which paths the estimate includes.
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
    """Cross-fitted propensity and outcome predictions for every loan (propensity as predicted, not clipped)."""

    propensity: Array
    outcome_treated: Array
    outcome_control: Array

    def clipped(self, clip: float = CLIP) -> "Nuisance":
        """The same predictions with the propensity clipped to [clip, 1 - clip]."""
        return Nuisance(np.clip(self.propensity, clip, 1 - clip), self.outcome_treated, self.outcome_control)


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
    return Nuisance(propensity, mu1, mu0)


def aipw_scores(t: NDArray[np.int64], y: Array, n: Nuisance) -> Array:
    """Doubly robust score per loan; its mean is the ATE and its spread gives the standard error."""
    out: Array = (
        n.outcome_treated
        - n.outcome_control
        + t * (y - n.outcome_treated) / n.propensity
        - (1 - t) * (y - n.outcome_control) / (1 - n.propensity)
    )
    return out


def estimates(t: NDArray[np.int64], y: Array, n: Nuisance, clip: float = CLIP) -> dict[str, dict[str, float]]:
    """ATE from each estimator (propensities clipped to [clip, 1 - clip]); AIPW also gets a 95% interval."""
    n = n.clipped(clip)
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
    scores = aipw_scores(t, y, n.clipped())
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


def effective_sample_size(weights: Array) -> float:
    """Kish effective sample size: how many equally weighted loans the weights are worth."""
    return float(weights.sum() ** 2 / (weights**2).sum())


def overlap_summary(t: NDArray[np.int64], propensity: Array) -> dict[str, float]:
    """Raw propensity quantiles by group, tail shares, and the effective sample size of the clipped weights."""
    clipped = np.clip(propensity, CLIP, 1 - CLIP)
    weights = t / clipped + (1 - t) / (1 - clipped)
    summary = {
        "share_outside_5_95": float(((propensity < 0.05) | (propensity > 0.95)).mean()),
        "share_clipped": float(((propensity < CLIP) | (propensity > 1 - CLIP)).mean()),
        "max_weight": float(np.max(weights)),
        "ess_treated": effective_sample_size(weights[t == 1]),
        "ess_control": effective_sample_size(weights[t == 0]),
        "n_treated": float((t == 1).sum()),
        "n_control": float((t == 0).sum()),
    }
    for group, mask in (("treated", t == 1), ("control", t == 0)):
        for q in (0.01, 0.5, 0.99):
            summary[f"{group}_q{int(q * 100)}"] = float(np.quantile(propensity[mask], q))
    return summary


INCOME_BANDS = [(0, 40_000), (40_000, 75_000), (75_000, 120_000), (120_000, 140_000), (140_000, float("inf"))]


def support_by_income(W: pd.DataFrame, t: NDArray[np.int64]) -> pd.DataFrame:
    """Loans and share treated per income band: where one group is (nearly) absent there is no comparison."""
    income = np.exp(W["log_income"].to_numpy())
    rows = []
    for low, high in INCOME_BANDS:
        band = (income >= low) & (income < high)
        label = f"${low / 1000:.0f}k and over" if np.isinf(high) else f"${low / 1000:.0f}k-{high / 1000:.0f}k"
        rows.append(
            {"income": label, "n": int(band.sum()), "treated": int(t[band].sum()), "share": float(t[band].mean())}
        )
    return pd.DataFrame(rows)


OVERLAP_REGION = (0.1, 0.9)  # Crump, Hotz, Imbens & Mitnik (2009) rule of thumb


def sensitivity(t: NDArray[np.int64], y: Array, n: Nuisance) -> pd.DataFrame:
    """The adjusted difference under other clipping, trimming and weighting choices, each with its own target."""
    rows = []
    for clip in (0.01, 0.05):
        r = estimates(t, y, n, clip)["doubly robust (AIPW)"]
        rows.append(
            {"estimate": f"AIPW, propensities clipped to [{clip}, {1 - clip}]", "population": "all loans", "n": len(y)}
            | r
        )
    inside = (n.propensity >= OVERLAP_REGION[0]) & (n.propensity <= OVERLAP_REGION[1])
    scores = aipw_scores(
        t[inside], y[inside], Nuisance(n.propensity[inside], n.outcome_treated[inside], n.outcome_control[inside])
    )
    se = float(scores.std(ddof=1) / np.sqrt(inside.sum()))
    rows.append(
        {
            "estimate": f"AIPW, loans with propensity {OVERLAP_REGION[0]}-{OVERLAP_REGION[1]}",
            "population": f"{inside.mean():.0%} of loans, where both groups are common",
            "n": int(inside.sum()),
            "ate": float(scores.mean()),
            "se": se,
            "low": float(scores.mean() - 1.96 * se),
            "high": float(scores.mean() + 1.96 * se),
        }
    )
    w1, w0 = t * (1 - n.propensity), (1 - t) * n.propensity
    overlap = float((w1 * y).sum() / w1.sum() - (w0 * y).sum() / w0.sum())
    rows.append(
        {
            "estimate": "overlap weights (Li, Morgan & Zaslavsky 2018)",
            "population": "loans weighted towards those that could have gone either way",
            "n": len(y),
            "ate": overlap,
        }
    )
    return pd.DataFrame(rows)


def _pp(value: float) -> str:
    return f"{value * 100:+.2f} pp"


def render_report(
    results: dict[str, dict[str, dict[str, float]]],
    balance: pd.DataFrame,
    overlap: dict[str, float],
    refuted: pd.DataFrame,
    by_fico: pd.DataFrame,
    treated_share: float,
    support: dict[str, pd.DataFrame],
    sensitive: pd.DataFrame,
) -> str:
    """Write the findings as Markdown."""
    lines = [
        "# Adjusted effect of a large loan relative to income",
        "",
        "Generated by `python -m loan_default.causal`.",
        "",
        f"Treatment: loan amount at least {THRESHOLD:.0%} of annual income ({treated_share:.1%} of 2007-2015 loans). "
        "Outcome: default. Differences are in percentage points (pp) of default probability.",
        "",
        "These are observational estimates. They equal a causal effect only if the recorded application fields "
        "capture everything that drives both the loan size and default (no unrecorded confounding), and only for "
        "the binary contrast at 25%. They do not describe the effect of a lending policy such as a cap, which would "
        "also change which loans are taken and on what terms. Interest rate, grade and term are not in the data.",
        "",
        "## Adjusted difference (average treatment effect under those assumptions)",
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
        "## Common support by income",
        "",
        "Treatment is defined from income, so at high incomes a large loan is rare or impossible "
        "(the maximum loan was $35,000 up to 2015 and $40,000 in 2016). Where one group is absent, "
        "the all-loans estimate relies on the outcome model extrapolating.",
        "",
        "| Income | Loans 2007-2015 | Treated | Share | Loans 2016 | Treated | Share |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for a, b in zip(support["2007-2015"].itertuples(), support["2016"].itertuples(), strict=True):
        lines.append(
            f"| {a.income} | {a.n:,} | {a.treated:,} | {a.share:.1%} | {b.n:,} | {b.treated:,} | {b.share:.1%} |"
        )
    lines += [
        "",
        "## Propensity overlap (as predicted, before clipping)",
        "",
        "| Group | 1st percentile | Median | 99th percentile |",
        "|---|---:|---:|---:|",
        *[
            f"| {group} | {overlap[f'{group}_q1']:.3f} | {overlap[f'{group}_q50']:.3f} | "
            f"{overlap[f'{group}_q99']:.3f} |"
            for group in ("treated", "control")
        ],
        "",
        f"{overlap['share_outside_5_95']:.1%} of loans have a propensity outside 0.05-0.95 and "
        f"{overlap['share_clipped']:.1%} are clipped at [{CLIP}, {1 - CLIP}]. "
        f"Largest weight {overlap['max_weight']:.0f}. "
        f"Effective sample size of the weights: {overlap['ess_treated']:,.0f} of {overlap['n_treated']:,.0f} treated, "
        f"{overlap['ess_control']:,.0f} of {overlap['n_control']:,.0f} control loans.",
        "",
        "## Covariate balance",
        "",
        "Standardized mean difference between treated and control loans; below 0.1 is usually read as balanced. "
        "Income stays imbalanced after weighting, as the support table explains.",
        "",
        "| Covariate | Before weighting | After weighting |",
        "|---|---:|---:|",
        *[f"| {r.covariate} | {r.smd_before:+.3f} | {r.smd_after:+.3f} |" for r in balance.itertuples()],
        "",
        "## Sensitivity to clipping, trimming and weighting (2007-2015)",
        "",
        "Each row answers a slightly different question: the trimmed and overlap-weighted rows describe the loans "
        "that could plausibly have been either large or small for the borrower, not all loans.",
        "",
        "| Estimate | Population | Loans | Difference | 95% CI |",
        "|---|---|---:|---:|---|",
    ]
    for r in sensitive.itertuples():
        interval = f"{r.low * 100:+.2f} to {r.high * 100:+.2f} pp" if np.isfinite(getattr(r, "low", np.nan)) else "n/a"
        lines.append(f"| {r.estimate} | {r.population} | {r.n:,} | {_pp(r.ate)} | {interval} |")
    lines += [
        "",
        "## Stability checks (AIPW)",
        "",
        "These check that the estimator behaves; they cannot test the no-unrecorded-confounding assumption.",
        "",
        "| Check | Expected | Estimate | 95% CI |",
        "|---|---|---:|---|",
        *[
            f"| {r.test} | {r.expected} | {_pp(r.ate)} | {r.low * 100:+.2f} to {r.high * 100:+.2f} pp |"
            for r in refuted.itertuples()
        ],
        "",
        "## Differences by credit score (a lead for further study, not a policy recommendation)",
        "",
        "| FICO band | Loans | Share treated | Difference (AIPW) | 95% CI |",
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
    balance = balance_table(W, t, nuisance.clipped().propensity)
    refuted = refutations(W, t, y)
    by_fico = effect_by_fico(W, t, y, nuisance)
    support = {"2007-2015": support_by_income(W, t), "2016": support_by_income(W_test, t_test)}
    report = render_report(
        results,
        balance,
        overlap_summary(t, nuisance.propensity),
        refuted,
        by_fico,
        float(t.mean()),
        support,
        sensitivity(t, y, nuisance),
    )
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
