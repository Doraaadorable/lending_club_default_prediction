"""Fairness audit of the approval policy across borrower groups, and two mitigations.

    python -m loan_default.fairness        # writes reports/fairness_report.md and .csv

The data has no race, ethnicity, sex or age, so protected-class disparities cannot be measured
directly. The audit uses the groups we can see, several of which correlate with protected
characteristics in US data: housing status, income band, census region (from state) and whether
employment length was given. The policy is the production one: the cutoff that approves the
safest 80% of 2007-2015 loans (chosen on out-of-fold scores), applied unchanged to 2016.

Per group: approval rate, adverse impact ratio (approval rate over the best-treated group's;
flagged below 0.8, a screening threshold borrowed from the "four-fifths" rule of thumb in US
employment-selection guidelines; it is not a legal test for credit), approval among borrowers who repaid (equal
opportunity), approval among borrowers who defaulted, default rate among approved loans,
calibration (mean predicted minus actual) and ROC-AUC.

Mitigations compared:
1. Remove state, a geographic proxy, from the model (the README's open question).
2. Group-specific cutoffs for income bands, chosen on training scores so every band's repaid
   borrowers are approved at the same rate (equal opportunity post-processing, Hardt et al. 2016).
   Shown to measure the trade-off, compared with a single cutoff at the same approval volume.
   Cutoffs that differ by group raise legal questions under fair-lending law, above all when the
   groups are defined by a protected characteristic, so this is an analysis, not a policy to deploy
   without legal review.
"""

import argparse
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.metrics import roc_auc_score

from loan_default.experiments import Splits, load_splits, score_metrics, write_report
from loan_default.policy import approve_safest, cutoff_at
from loan_default.preprocessing import build_preprocessor, clean
from loan_default.train import TUNED_PARAMS, build_model, out_of_fold_scores

Array = NDArray[np.float64]
SEED = 42
APPROVAL_RATE = 0.8
MIN_GROUP = 300
FOUR_FIFTHS = 0.8

REGIONS = {
    "Northeast": "CT ME MA NH RI VT NJ NY PA",
    "Midwest": "IL IN MI OH WI IA KS MN MO NE ND SD",
    "South": "DE DC FL GA MD NC SC VA WV AL KY MS TN AR LA OK TX",
    "West": "AZ CO ID MT NV NM UT WY AK CA HI OR WA",
}
STATE_TO_REGION = {state: region for region, states in REGIONS.items() for state in states.split()}
INCOME_BANDS: list[tuple[float, float, str]] = [
    (0, 40_000, "under $40k"),
    (40_000, 75_000, "$40k-75k"),
    (75_000, 120_000, "$75k-120k"),
]
INCOME_BANDS.append((120_000, float("inf"), "$120k and over"))


def income_band(revenue: pd.Series) -> pd.Series:
    """Label each income with its band."""
    labels = pd.Series("", index=revenue.index)
    for low, high, label in INCOME_BANDS:
        labels[(revenue >= low) & (revenue < high)] = label
    return labels


GROUPINGS: dict[str, Callable[[pd.DataFrame], pd.Series]] = {
    "Housing status": lambda X: X["home_ownership_n"].astype(str),
    "Income band": lambda X: income_band(X["revenue"]),
    "Census region": lambda X: X["addr_state"].map(STATE_TO_REGION).fillna("Other"),
    "Employment length": lambda X: clean(X)["emp_length"].isna().map({True: "not given", False: "given"}),
}


def group_metrics(groups: pd.Series, y: Array, scores: Array, approved: NDArray[np.bool_]) -> pd.DataFrame:
    """Approval, error and calibration metrics for each group with at least MIN_GROUP loans."""
    rows = []
    for group in sorted(groups.unique()):
        m = (groups == group).to_numpy()
        if m.sum() < MIN_GROUP:
            continue
        repaid, defaulted = m & (y == 0), m & (y == 1)
        rows.append(
            {
                "group": group,
                "n": int(m.sum()),
                "default_rate": float(y[m].mean()),
                "approval_rate": float(approved[m].mean()),
                "approval_if_repaid": float(approved[repaid].mean()),
                "approval_if_defaulted": float(approved[defaulted].mean()),
                "default_rate_approved": float(y[m & approved].mean()) if (m & approved).any() else float("nan"),
                "calibration_gap": float(scores[m].mean() - y[m].mean()),
                "roc_auc": float(roc_auc_score(y[m], scores[m])) if 0 < y[m].mean() < 1 else float("nan"),
            }
        )
    table = pd.DataFrame(rows)
    table["adverse_impact_ratio"] = table["approval_rate"] / table["approval_rate"].max()
    return table


def disparities(table: pd.DataFrame) -> dict[str, float]:
    """Gaps between the best- and worst-treated groups."""
    return {
        "demographic_parity_difference": float(table["approval_rate"].max() - table["approval_rate"].min()),
        "min_adverse_impact_ratio": float(table["adverse_impact_ratio"].min()),
        "equal_opportunity_difference": float(table["approval_if_repaid"].max() - table["approval_if_repaid"].min()),
        "equalized_odds_difference": float(
            max(
                table["approval_if_repaid"].max() - table["approval_if_repaid"].min(),
                table["approval_if_defaulted"].max() - table["approval_if_defaulted"].min(),
            )
        ),
    }


def equal_opportunity_cutoffs(groups: pd.Series, y: Array, scores: Array, approval_rate: float) -> dict[str, float]:
    """Per-group cutoffs that approve the same share of repaid borrowers in every group.

    The common share is the one the single production cutoff achieves on these (training) scores.
    """
    single = cutoff_at(scores, approval_rate)
    target = float((scores[y == 0] <= single).mean())
    cutoffs = {}
    for group in groups.unique():
        repaid_scores = scores[(groups == group).to_numpy() & (y == 0)]
        cutoffs[str(group)] = float(np.quantile(repaid_scores, target)) if len(repaid_scores) else single
    return cutoffs


def apply_cutoffs(
    groups: pd.Series, scores: Array, cutoffs: dict[str, float], fallback: float | None = None
) -> NDArray[np.bool_]:
    """Approve each loan whose score is at or below its group's cutoff.

    A group without a cutoff of its own uses `fallback`; with no fallback it is an error, never a silent rejection.
    """
    limits = groups.astype(str).map(cutoffs)
    if limits.isna().any():
        if fallback is None:
            missing = sorted(set(groups.astype(str)[limits.isna()]))
            raise ValueError(f"no cutoff for group(s) {missing} and no fallback cutoff")
        limits = limits.fillna(fallback)
    limits = limits.to_numpy(dtype=float)
    out: NDArray[np.bool_] = scores <= limits
    return out


def _pct(x: float) -> str:
    return f"{x * 100:.1f}%"


def group_table_lines(table: pd.DataFrame) -> list[str]:
    """Markdown table of group metrics."""
    lines = [
        "| Group | Loans | Default rate | Approved | Impact ratio | Approved if repaid | Approved if defaulted "
        "| Default rate, approved | Predicted - actual | ROC-AUC |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in table.itertuples():
        flag = " ⚠" if r.adverse_impact_ratio < FOUR_FIFTHS else ""
        lines.append(
            f"| {r.group} | {r.n:,} | {_pct(r.default_rate)} | {_pct(r.approval_rate)} | "
            f"{r.adverse_impact_ratio:.2f}{flag} | {_pct(r.approval_if_repaid)} | {_pct(r.approval_if_defaulted)} | "
            f"{_pct(r.default_rate_approved)} | {r.calibration_gap * 100:+.1f} pp | {r.roc_auc:.3f} |"
        )
    return lines


def run(splits: Splits, out_dir: Path) -> str:
    """Audit the production policy on 2016, compare the two mitigations, and write the report."""
    y_train, y_test = splits.y_train.to_numpy(), splits.y_test.to_numpy()
    models = {"production": build_model(TUNED_PARAMS, SEED)}
    no_state = build_model(TUNED_PARAMS, SEED)
    no_state.set_params(preprocess=build_preprocessor(include_state=False))
    models["without state"] = no_state

    oof, test, cutoffs = {}, {}, {}
    for name, model in models.items():
        oof[name] = out_of_fold_scores(model, splits.X_train, splits.y_train, 5, SEED)
        cutoffs[name] = cutoff_at(oof[name], APPROVAL_RATE)
        test[name] = model.fit(splits.X_train, splits.y_train).predict_proba(splits.X_test)[:, 1]

    lines = [
        "# Fairness audit",
        "",
        "Generated by `python -m loan_default.fairness`. 2016 test loans, scored by the model trained on 2007-2015, "
        f"approved when the score is at or below the training cutoff for {APPROVAL_RATE:.0%} approval. "
        f"Groups with fewer than {MIN_GROUP} loans are left out. ⚠ marks an adverse impact ratio below "
        f"{FOUR_FIFTHS}, a screening threshold for this audit (from the four-fifths rule of thumb in US "
        "employment-selection guidelines), not a legal test for credit decisions.",
        "",
    ]
    rows = []
    for grouping, make_groups in GROUPINGS.items():
        groups = pd.Series(make_groups(splits.X_test), index=splits.X_test.index).reset_index(drop=True)
        lines += [f"## {grouping}", ""]
        for name in models:
            approved = test[name] <= cutoffs[name]
            table = group_metrics(groups, y_test, test[name], approved)
            gaps = disparities(table)
            rows.append({"grouping": grouping, "model": name, "policy": "single cutoff"} | gaps)
            if name == "production":
                lines += group_table_lines(table) + [""]
        lines.append(
            "| Model | Approval gap | Lowest impact ratio | Equal opportunity gap | Equalized odds gap |\n"
            "|---|---:|---:|---:|---:|"
        )
        for row in rows:
            if row["grouping"] == grouping:
                lines.append(
                    f"| {row['model']} | {row['demographic_parity_difference'] * 100:.1f} pp | "
                    f"{row['min_adverse_impact_ratio']:.2f} | {row['equal_opportunity_difference'] * 100:.1f} pp | "
                    f"{row['equalized_odds_difference'] * 100:.1f} pp |"
                )
        lines.append("")

    # Mitigation 1: dropping state, overall cost
    overall = {name: score_metrics(splits.y_test, test[name]) for name in models}
    lines += [
        "## Mitigation 1: remove state from the model",
        "",
        "| Model | Test ROC-AUC | Default rate, safest 80% |",
        "|---|---:|---:|",
        *[f"| {name} | {m['roc_auc']:.4f} | {m['default_rate_at_80'] * 100:.2f}% |" for name, m in overall.items()],
        "",
    ]

    # Mitigation 2: equal-opportunity cutoffs for income bands, chosen on training scores
    train_groups = income_band(splits.X_train["revenue"]).reset_index(drop=True)
    test_groups = income_band(splits.X_test["revenue"]).reset_index(drop=True)
    band_cutoffs = equal_opportunity_cutoffs(train_groups, y_train, oof["production"], APPROVAL_RATE)
    single = test["production"] <= cutoffs["production"]
    per_group = apply_cutoffs(test_groups, test["production"], band_cutoffs, fallback=cutoffs["production"])
    same_volume = approve_safest(test["production"], float(per_group.mean()))  # single cutoff, same approvals
    before = group_metrics(test_groups, y_test, test["production"], single)
    after = group_metrics(test_groups, y_test, test["production"], per_group)
    lines += [
        "## Mitigation 2: equal-opportunity cutoffs by income band",
        "",
        "Cutoffs per band (chosen on 2007-2015 out-of-fold scores): "
        + ", ".join(f"{band} {value:.3f}" for band, value in sorted(band_cutoffs.items()))
        + f"; single cutoff {cutoffs['production']:.3f}.",
        "",
        "| Policy | Approved | Default rate, approved | Equal opportunity gap | Lowest impact ratio |",
        "|---|---:|---:|---:|---:|",
    ]
    policies = [
        ("single cutoff (production)", single, before),
        (
            "same approval volume, ranked within 2016 (not a cutoff fixed in advance)",
            same_volume,
            group_metrics(test_groups, y_test, test["production"], same_volume),
        ),
        ("cutoff per income band", per_group, after),
    ]
    for label, approved, table in policies:
        gaps = disparities(table)
        rows.append({"grouping": "Income band", "model": "production", "policy": label} | gaps)
        lines.append(
            f"| {label} | {_pct(float(approved.mean()))} | {_pct(float(y_test[approved].mean()))} | "
            f"{gaps['equal_opportunity_difference'] * 100:.1f} pp | {gaps['min_adverse_impact_ratio']:.2f} |"
        )
    lines += ["", "After per-band cutoffs:", "", *group_table_lines(after), ""]

    report = "\n".join(lines)
    write_report(out_dir, "fairness_report", report, pd.DataFrame(rows))
    return report


def main(argv: list[str] | None = None) -> None:
    """Run the audit from the command line."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out-dir", type=Path, default=Path("reports"))
    args = parser.parse_args(argv)
    print(run(load_splits(), args.out_dir))


if __name__ == "__main__":
    main()
