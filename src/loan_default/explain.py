"""Explain the production model: which inputs drive its scores, and why a given applicant was rejected.

    python -m loan_default.explain        # writes reports/explainability_report.md and figures

Global
    SHAP values (TreeSHAP, exact for tree ensembles) on a 3,000-loan sample of 2016, in log-odds.
    Columns created from one application field (purpose one-hot columns, the employment missing
    flag) are summed back into that field, so the explanation speaks the applicant's language.
    Permutation importance and partial dependence / ICE curves give a model-agnostic second view.
Local
    For individual applicants: the fields that pushed their score up most, relative to the average
    applicant in the SHAP background. These resemble the reasons an adverse action notice lists, but
    they are model contributions, not validated decline reasons: they are measured against an average
    applicant rather than the approval cutoff, related fields (loan amount, income, loan-to-income)
    can describe one mechanism twice, and their wording has not been reviewed. The API does not
    return them.
"""

import argparse
from pathlib import Path
from types import ModuleType

import numpy as np
import pandas as pd
import shap
from numpy.typing import NDArray
from sklearn.inspection import PartialDependenceDisplay
from sklearn.pipeline import Pipeline

from loan_default.experiments import Splits, load_splits, write_report
from loan_default.policy import cutoff_at
from loan_default.preprocessing import clean
from loan_default.train import TUNED_PARAMS, build_model, input_importance, out_of_fold_scores

Array = NDArray[np.float64]
SEED = 42
SAMPLE = 3000
FIELDS = [
    "fico_n",
    "dti_n",
    "emp_length",
    "revenue",
    "loan_amnt",
    "loan_to_income",
    "purpose",
    "home_ownership_n",
    "addr_state",
]
REASONS = {
    "fico_n": "Credit score",
    "dti_n": "Debt payments high relative to income",
    "emp_length": "Length of employment",
    "revenue": "Income",
    "loan_amnt": "Amount requested",
    "loan_to_income": "Amount requested is large relative to income",
    "purpose": "Purpose of the loan",
    "home_ownership_n": "Housing status",
    "addr_state": "State of residence",
}


def field_of(column: str) -> str:
    """Application field a transformed column comes from, e.g. 'cat__purpose_car' -> 'purpose'."""
    name = column.split("__", 1)[-1].removeprefix("missingindicator_")
    for field in sorted(FIELDS, key=len, reverse=True):
        if name == field or name.startswith(f"{field}_"):
            return field
    raise ValueError(f"cannot map column {column!r} to an application field")


def by_field(values: Array, columns: list[str]) -> pd.DataFrame:
    """Sum per-column SHAP values into one column per application field."""
    frame = pd.DataFrame(values, columns=columns)
    return frame.T.groupby([field_of(c) for c in columns], sort=False).sum().T


def shap_values(model: Pipeline, X: pd.DataFrame) -> tuple[pd.DataFrame, float, Array]:
    """Field-level SHAP values (log-odds), the base value, and the per-column values."""
    columns = list(model["preprocess"]["columns"].get_feature_names_out())
    matrix = np.asarray(model["preprocess"].transform(X), dtype=float)
    explainer = shap.TreeExplainer(model["model"])
    values = np.asarray(explainer.shap_values(matrix), dtype=float)
    base = float(np.ravel(explainer.expected_value)[0])
    fields = by_field(values, columns)
    fields.index = X.index
    return fields, base, values


def additivity_error(model: Pipeline, X: pd.DataFrame, fields: pd.DataFrame, base: float) -> float:
    """Largest gap between base + sum of SHAP values and the model's own log-odds (should be ~0)."""
    p = model.predict_proba(X)[:, 1]
    log_odds = np.log(p / (1 - p))
    return float(np.max(np.abs(base + fields.sum(axis=1).to_numpy() - log_odds)))


def global_importance(fields: pd.DataFrame, raw: pd.DataFrame) -> pd.DataFrame:
    """Mean |SHAP| per field and, for numeric fields, whether higher values raise or lower risk."""
    rows = []
    for field in fields.columns:
        direction = ""
        if pd.api.types.is_numeric_dtype(raw[field]):
            known = raw[field].notna().to_numpy()
            corr = np.corrcoef(raw.loc[known, field], fields.loc[known, field])[0, 1]
            direction = (
                "higher value, higher attribution"
                if corr > 0.1
                else "higher value, lower attribution"
                if corr < -0.1
                else "mixed"
            )
        rows.append({"field": field, "mean_abs_shap": float(fields[field].abs().mean()), "direction": direction})
    return pd.DataFrame(rows).sort_values("mean_abs_shap", ascending=False, ignore_index=True)


LOAN_SIZE_GROUP = ("loan_amnt", "revenue", "loan_to_income")


def grouped_importance(fields: pd.DataFrame) -> float:
    """Mean |SHAP| of loan amount, income and loan-to-income together: they share one mechanism."""
    return float(fields[list(LOAN_SIZE_GROUP)].sum(axis=1).abs().mean())


def top_contributors(contributions: pd.Series, k: int = 3) -> list[str]:
    """The k fields that raised this applicant's score most (positive SHAP), described in plain words."""
    raising = contributions[contributions > 0].sort_values(ascending=False)
    return [REASONS[field] for field in raising.index[:k]]


def describe_value(field: str, value: object) -> str:
    """Applicant's value for a field, formatted for the report."""
    if field in {"revenue", "loan_amnt"}:
        return f"${float(value):,.0f}"  # type: ignore[arg-type]
    if field == "loan_to_income":
        return f"{float(value):.2f}"  # type: ignore[arg-type]
    if field == "emp_length":
        return "not given" if pd.isna(value) else f"{float(value):.0f} years"  # type: ignore[arg-type]
    if field in {"fico_n", "dti_n"}:
        return f"{float(value):.0f}"  # type: ignore[arg-type]
    return str(value)


def local_examples(fields: pd.DataFrame, raw: pd.DataFrame, scores: Array, cutoff: float) -> list[str]:
    """Markdown for three applicants: a clear approval, one just past the cutoff, and a clear rejection."""
    order = np.argsort(scores)
    above = np.flatnonzero(scores[order] > cutoff)
    picks = {
        "Clear approval": order[len(order) // 10],
        "Just rejected": order[above[0]] if len(above) else order[-1],
        "Clear rejection": order[-max(1, len(order) // 50)],
    }
    lines = []
    for label, i in picks.items():
        row, contributions = raw.iloc[i], fields.iloc[i]
        decision = "approve" if scores[i] <= cutoff else "reject"
        lines += [
            f"### {label}: predicted default {scores[i]:.1%}, decision {decision} (cutoff {cutoff:.1%})",
            "",
            "| Field | Applicant's value | Contribution (log-odds) |",
            "|---|---|---:|",
        ]
        for field in contributions.abs().sort_values(ascending=False).index:
            lines.append(f"| {field} | {describe_value(field, row[field])} | {contributions[field]:+.3f} |")
        if decision == "reject":
            lines += [
                "",
                "Largest contributions to the score: "
                + "; ".join(f"{n}. {r}" for n, r in enumerate(top_contributors(contributions), 1)),
            ]
        lines.append("")
    return lines


def _pyplot() -> ModuleType:
    """matplotlib.pyplot with a non-interactive backend, imported only when figures are drawn."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def plot_global(importance: pd.DataFrame, fields: pd.DataFrame, raw: pd.DataFrame, figures: Path) -> None:
    """Bar chart of mean |SHAP| per field and a beeswarm of the numeric fields."""
    plt = _pyplot()
    figures.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 4))
    ordered = importance.iloc[::-1]
    ax.barh(ordered["field"], ordered["mean_abs_shap"], color="#3b6ea8")
    ax.set_xlabel("Mean |SHAP value| (log-odds of default)")
    ax.set_title("What drives the model's scores (2016 sample)")
    fig.tight_layout()
    fig.savefig(figures / "shap_importance.png", dpi=150)
    plt.close(fig)

    numeric = [f for f in importance["field"] if pd.api.types.is_numeric_dtype(raw[f])]
    shap.summary_plot(fields[numeric].to_numpy(), raw[numeric], feature_names=numeric, show=False)
    plt.title("SHAP values by field: colour = the applicant's value")
    plt.tight_layout()
    plt.savefig(figures / "shap_beeswarm.png", dpi=150)
    plt.close("all")


def plot_dependence(fields: pd.DataFrame, raw: pd.DataFrame, figures: Path) -> None:
    """SHAP value against the applicant's value for the four strongest numeric fields."""
    plt = _pyplot()
    fig, axes = plt.subplots(2, 2, figsize=(9, 7))
    for ax, field in zip(axes.ravel(), ["fico_n", "loan_to_income", "dti_n", "revenue"], strict=True):
        ax.scatter(raw[field], fields[field], s=4, alpha=0.4, color="#3b6ea8")
        ax.axhline(0, color="grey", lw=0.8)
        ax.set_xlabel(field)
        ax.set_ylabel("SHAP value")
        if field == "revenue":
            ax.set_xscale("log")
    fig.suptitle("How each field moves the score (SHAP dependence)")
    fig.tight_layout()
    fig.savefig(figures / "shap_dependence.png", dpi=150)
    plt.close(fig)


def plot_partial_dependence(model: Pipeline, X: pd.DataFrame, figures: Path) -> None:
    """Partial dependence with individual (ICE) curves on the raw application fields."""
    plt = _pyplot()
    fig, ax = plt.subplots(1, 3, figsize=(12, 4))
    X = X.astype({"fico_n": float, "dti_n": float, "loan_amnt": float})  # integer columns would be rounded
    PartialDependenceDisplay.from_estimator(
        model,
        X,
        ["fico_n", "dti_n", "loan_amnt"],
        kind="both",
        subsample=150,
        random_state=SEED,
        grid_resolution=30,
        ax=ax,
        response_method="predict_proba",
    )
    fig.suptitle("Partial dependence (bold) and individual applicants (ICE), probability of default")
    fig.tight_layout()
    fig.savefig(figures / "partial_dependence.png", dpi=150)
    plt.close(fig)


def render_report(
    importance: pd.DataFrame,
    permutation: dict[str, float],
    error: float,
    base: float,
    examples: list[str],
    loan_size: float,
) -> str:
    """Write the findings as Markdown."""
    lines = [
        "# Explainability",
        "",
        f"Generated by `python -m loan_default.explain`. SHAP values for the production model on a {SAMPLE:,}-loan "
        f"sample of 2016, in log-odds of default. Base value {base:+.3f} "
        f"(probability {1 / (1 + np.exp(-base)):.1%}); base + contributions reproduce the model's log-odds to within "
        f"{error:.1e}.",
        "",
        "## Global importance",
        "",
        "| Field | Mean abs. SHAP | Direction | Permutation importance (drop in test ROC-AUC) |",
        "|---|---:|---|---:|",
    ]
    for r in importance.itertuples():
        perm = permutation.get(r.field)
        perm_text = f"{perm:.4f}" if perm is not None else "n/a (derived field)"
        lines.append(f"| {r.field} | {r.mean_abs_shap:.3f} | {r.direction} | {perm_text} |")
    lines += [
        "",
        "Direction is the sign of the correlation between a field's value and its SHAP value in this model, "
        "not a causal effect.",
        "",
        "`loan_to_income` is computed from `loan_amnt` and `revenue`, so the three share one mechanism and how SHAP "
        "splits credit among them depends on the model's representation. Taken together their mean absolute SHAP "
        f"value is {loan_size:.3f}. Permutation importance shuffles a raw field and recomputes the ratio, so it "
        "measures each raw field's direct and derived paths together.",
        "",
        "![SHAP importance](figures/shap_importance.png)",
        "",
        "![SHAP beeswarm](figures/shap_beeswarm.png)",
        "",
        "![SHAP dependence](figures/shap_dependence.png)",
        "",
        "![Partial dependence](figures/partial_dependence.png)",
        "",
        "## Individual applicants: largest contributions to the score",
        "",
        "Contributions are measured against the average applicant in the SHAP background, not against the approval "
        "cutoff. They are not validated adverse action reasons, and the service does not return them.",
        "",
        *examples,
        "State of residence can appear among the largest contributions; geography can proxy for protected "
        "characteristics (see the fairness report).",
        "",
    ]
    return "\n".join(lines)


def run(splits: Splits, out_dir: Path) -> str:
    """Fit the production model, explain it, and write the report and figures."""
    model = build_model(TUNED_PARAMS, SEED)
    cutoff = cutoff_at(out_of_fold_scores(model, splits.X_train, splits.y_train, 5, SEED), 0.8)
    model.fit(splits.X_train, splits.y_train)

    sample = splits.X_test.sample(SAMPLE, random_state=SEED)
    raw = clean(sample)
    fields, base, _ = shap_values(model, sample)
    error = additivity_error(model, sample, fields, base)
    importance = global_importance(fields, raw)
    permutation = input_importance(model, splits.X_test, splits.y_test, SEED)
    scores = model.predict_proba(sample)[:, 1]

    figures = out_dir / "figures"
    plot_global(importance, fields, raw, figures)
    plot_dependence(fields, raw, figures)
    plot_partial_dependence(model, sample, figures)
    report = render_report(
        importance, permutation, error, base, local_examples(fields, raw, scores, cutoff), grouped_importance(fields)
    )
    write_report(out_dir, "explainability_report", report, importance)
    return report


def main(argv: list[str] | None = None) -> None:
    """Run the explanation from the command line."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out-dir", type=Path, default=Path("reports"))
    args = parser.parse_args(argv)
    print(run(load_splits(), args.out_dir))


if __name__ == "__main__":
    main()
