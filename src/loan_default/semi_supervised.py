"""Semi-supervised learning: what unlabeled loans add when only a few outcomes are known.

    python -m loan_default.semi_supervised        # writes reports/semi_supervised_report.md and .csv

In lending, many applications never get an outcome: rejected applicants (reject inference) and
recent loans that have not yet been repaid or charged off. We simulate this by hiding the outcome
of most 2007-2015 loans, keeping 1%, 5% or 20% of them labeled, and scoring every method on the
fully labeled 2016 sample. Each setting is repeated over three random labeled subsets.

Methods, all with the same gradient boosting learner:

labeled only            ignore the unlabeled loans
self-training           classic threshold self-training (as in scikit-learn's SelfTrainingClassifier):
                        3 rounds, each adding the unlabeled loans predicted with probability 0.9 or
                        more, with their predicted class
balanced self-training  the same, but each of 3 rounds adds 10% of the loans, as confident
                        defaults and confident repayments in the labeled base rate, so the
                        pseudo-labels do not drift to "repaid"
cluster soft labels     cluster all loans (k-means); give each unlabeled loan its cluster's
                        labeled default rate as a soft label (one weighted copy per class)
label spreading         the same soft-label idea over a k-nearest-neighbour graph
cluster features        fit k-means on all loans and add the distances to the centres as features
                        (unlabeled loans only shape the representation)

Every method stops boosting early on the same held-out 15% of the labeled loans, which never get
pseudo-labels and are never duplicated, so early stopping always judges against real outcomes.
The state target encoding is cross-fitted on the labeled training loans, so no loan's encoding
uses its own outcome.

What this measures: whether unlabeled loans from the same population help when few outcomes are
known at random. Rejected applicants are not a random subset, so this does not show what reject
inference would achieve.
"""

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.cluster import KMeans
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.semi_supervised import LabelSpreading

from loan_default.experiments import Splits, load_splits, score_metrics, write_report
from loan_default.preprocessing import build_preprocessor
from loan_default.train import TUNED_PARAMS, build_model

Array = NDArray[np.float64]
LABEL_SHARES = (0.01, 0.05, 0.20)
VALIDATION_SHARE = 0.15
REPEATS = 3
UNLABELED = -1
SEED = 42
METHODS = (
    "labeled only",
    "self-training",
    "balanced self-training",
    "cluster soft labels",
    "label spreading",
    "cluster features",
)


def learner(seed: int = SEED) -> HistGradientBoostingClassifier:
    """Gradient boosting with early stopping, so it suits 1,500 labeled loans as well as 150,000."""
    return HistGradientBoostingClassifier(
        learning_rate=0.1,
        max_iter=300,
        max_leaf_nodes=15,
        min_samples_leaf=40,
        l2_regularization=1.0,
        early_stopping=True,
        n_iter_no_change=10,
        random_state=seed,
    )


def hide_labels(y: pd.Series, share: float, seed: int) -> NDArray[np.int64]:
    """Copy of y with all but a stratified `share` of the outcomes replaced by -1."""
    rng = np.random.default_rng(seed)
    labels = np.full(len(y), UNLABELED, dtype=np.int64)
    values = y.to_numpy()
    for cls in (0, 1):
        members = np.flatnonzero(values == cls)
        keep = rng.choice(members, max(1, int(round(share * len(members)))), replace=False)
        labels[keep] = cls
    return labels


@dataclass
class Prepared:
    """Preprocessed matrices: the training pool with partly hidden labels, the early-stopping set, and the test set."""

    train: Array
    test: Array
    labels: NDArray[np.int64]
    val: Array
    val_labels: NDArray[np.int64]

    @property
    def labeled(self) -> NDArray[np.bool_]:
        """Rows whose outcome is known."""
        out: NDArray[np.bool_] = self.labels != UNLABELED
        return out


def prepare(splits: Splits, labels: NDArray[np.int64], seed: int = SEED) -> Prepared:
    """Set aside a stratified early-stopping set from the labeled loans, then preprocess.

    The production preprocessing is fitted on the remaining labeled loans with fit_transform, so their
    state encoding is cross-fitted; the early-stopping set, the unlabeled pool and 2016 are transformed.
    """
    rng = np.random.default_rng(seed)
    held_out = np.zeros(len(labels), dtype=bool)
    for cls in (0, 1):
        members = np.flatnonzero(labels == cls)
        held_out[rng.choice(members, max(1, int(round(VALIDATION_SHARE * len(members)))), replace=False)] = True
    fit_rows = (labels != UNLABELED) & ~held_out
    preprocessor = build_preprocessor()
    fitted = np.asarray(preprocessor.fit_transform(splits.X_train[fit_rows], labels[fit_rows]), dtype=float)
    train = np.asarray(preprocessor.transform(splits.X_train), dtype=float)
    train[fit_rows] = fitted
    pool = ~held_out
    return Prepared(
        train[pool],
        np.asarray(preprocessor.transform(splits.X_test), dtype=float),
        labels[pool],
        train[held_out],
        labels[held_out],
    )


def fit(
    data: Prepared, X: Array, y: NDArray[np.int64] | Array, seed: int, weight: Array | None = None
) -> HistGradientBoostingClassifier:
    """Fit the shared learner, stopping early on the held-out labeled loans."""
    return learner(seed).fit(X, y, sample_weight=weight, X_val=data.val, y_val=data.val_labels)


def fit_soft(data: Prepared, soft: Array, seed: int) -> HistGradientBoostingClassifier:
    """Fit on labeled rows plus two weighted copies of each unlabeled row: y=1 with weight p, y=0 with 1-p."""
    known = data.labeled
    X = np.vstack([data.train[known], data.train[~known], data.train[~known]])
    y = np.concatenate([data.labels[known], np.ones((~known).sum()), np.zeros((~known).sum())])
    weights = np.concatenate([np.ones(known.sum()), soft[~known], 1 - soft[~known]])
    return fit(data, X, y, seed, weights)


def n_clusters_for(n_labeled: int) -> int:
    """About 30 labeled loans per cluster, so each cluster's default rate rests on some evidence."""
    return int(np.clip(n_labeled // 30, 10, 200))


def cluster_soft_labels(data: Prepared, seed: int) -> Array:
    """Each loan's cluster default rate among labeled members, shrunk towards the overall rate."""
    known = data.labeled
    k = n_clusters_for(int(known.sum()))
    clusters = KMeans(k, n_init=3, random_state=seed).fit_predict(data.train)
    base = float(data.labels[known].mean())
    rates = np.full(k, base)
    for c in range(k):
        members = known & (clusters == c)
        rates[c] = (data.labels[members].sum() + 10 * base) / (members.sum() + 10)  # 10 virtual loans at base
    out: Array = rates[clusters]
    return out


def spread_soft_labels(data: Prepared) -> Array:
    """Default probability for every loan from label spreading over a 10-nearest-neighbour graph."""
    spreading = LabelSpreading(kernel="knn", n_neighbors=10, alpha=0.2, max_iter=50)
    spreading.fit(data.train, data.labels)
    column = list(spreading.classes_).index(1)
    out: Array = spreading.label_distributions_[:, column]
    return out


def balanced_self_training(data: Prepared, seed: int, rounds: int = 3, per_round: float = 0.1) -> Array:
    """Self-training that adds pseudo-labels in the labeled class ratio; returns test scores."""
    labels = data.labels.copy()
    base = float(labels[labels != UNLABELED].mean())
    for _ in range(rounds):
        known = labels != UNLABELED
        model = fit(data, data.train[known], labels[known], seed)
        pool = np.flatnonzero(~known)
        if len(pool) == 0:
            break
        scores = model.predict_proba(data.train[pool])[:, 1]
        n_add = int(per_round * len(data.labels))
        n_default = int(round(base * n_add))
        order = np.argsort(scores)
        labels[pool[order[: n_add - n_default]]] = 0  # most confident repayments
        labels[pool[order[::-1][:n_default]]] = 1  # most confident defaults
    known = labels != UNLABELED
    out: Array = fit(data, data.train[known], labels[known], seed).predict_proba(data.test)[:, 1]
    return out


def threshold_self_training(data: Prepared, seed: int, rounds: int = 3, threshold: float = 0.9) -> Array:
    """Self-training that adds every unlabeled loan predicted with probability >= threshold; returns test scores."""
    labels = data.labels.copy()
    for _ in range(rounds):
        known = labels != UNLABELED
        pool = np.flatnonzero(~known)
        if len(pool) == 0:
            break
        proba = fit(data, data.train[known], labels[known], seed).predict_proba(data.train[pool])[:, 1]
        confident = (proba >= threshold) | (proba <= 1 - threshold)
        if not confident.any():
            break
        labels[pool[confident]] = (proba[confident] >= 0.5).astype(np.int64)
    known = labels != UNLABELED
    out: Array = fit(data, data.train[known], labels[known], seed).predict_proba(data.test)[:, 1]
    return out


def test_scores(method: str, data: Prepared, seed: int) -> Array:
    """2016 scores from one method trained on the partially labeled training set."""
    known = data.labeled
    if method == "labeled only":
        model = fit(data, data.train[known], data.labels[known], seed)
    elif method == "self-training":
        return threshold_self_training(data, seed)
    elif method == "balanced self-training":
        return balanced_self_training(data, seed)
    elif method == "cluster soft labels":
        model = fit_soft(data, cluster_soft_labels(data, seed), seed)
    elif method == "label spreading":
        model = fit_soft(data, spread_soft_labels(data), seed)
    elif method == "cluster features":
        kmeans = KMeans(n_clusters_for(int(known.sum())), n_init=3, random_state=seed).fit(data.train)
        augmented_train = np.hstack([data.train, kmeans.transform(data.train)])
        augmented_test = np.hstack([data.test, kmeans.transform(data.test)])
        augmented = Prepared(
            data.train, data.test, data.labels, np.hstack([data.val, kmeans.transform(data.val)]), data.val_labels
        )
        scores: Array = fit(augmented, augmented_train[known], data.labels[known], seed).predict_proba(augmented_test)
        return scores[:, 1]
    else:
        raise ValueError(f"unknown method {method!r}")
    out: Array = model.predict_proba(data.test)[:, 1]
    return out


def experiment(
    splits: Splits,
    shares: tuple[float, ...] = LABEL_SHARES,
    repeats: int = REPEATS,
    methods: tuple[str, ...] = METHODS,
    verbose: bool = False,
) -> pd.DataFrame:
    """One row per label share, repeat and method, with 2016 test metrics."""
    rows = []
    for share in shares:
        for repeat in range(repeats):
            labels = hide_labels(splits.y_train, share, seed=SEED + repeat)
            data = prepare(splits, labels, seed=SEED + repeat)
            for method in methods:
                metrics = score_metrics(splits.y_test, test_scores(method, data, SEED + repeat))
                if verbose:
                    print(
                        f"{share:.0%} labeled, repeat {repeat}, {method}: ROC-AUC {metrics['roc_auc']:.4f}", flush=True
                    )
                rows.append({"label_share": share, "repeat": repeat, "method": method} | metrics)
    return pd.DataFrame(rows)


def paired_differences(results: pd.DataFrame, baseline: str = "labeled only") -> pd.DataFrame:
    """Each method's test ROC-AUC minus the baseline's on the same labeled subset: mean, min and max over repeats."""
    base = results[results["method"] == baseline].set_index(["label_share", "repeat"])["roc_auc"]
    others = results[results["method"] != baseline].copy()
    others["difference"] = (
        others["roc_auc"].to_numpy()
        - base.loc[list(zip(others["label_share"], others["repeat"], strict=True))].to_numpy()
    )
    return others.groupby(["label_share", "method"], sort=False)["difference"].agg(["mean", "min", "max"])


def render_report(results: pd.DataFrame, full: dict[str, float]) -> str:
    """Average over repeats, one table per metric."""
    summary = results.groupby(["label_share", "method"], sort=False).agg(
        roc_auc=("roc_auc", "mean"), roc_auc_sd=("roc_auc", "std"), default_rate=("default_rate_at_80", "mean")
    )
    shares = list(dict.fromkeys(results["label_share"]))
    methods = list(dict.fromkeys(results["method"]))
    counts = {share: int(round(share * 150_000)) for share in shares}
    lines = [
        "# Semi-supervised learning with few known outcomes",
        "",
        "Generated by `python -m loan_default.semi_supervised`. Every model is scored on the fully labeled "
        f"2016 sample; values are means over {results['repeat'].nunique()} random labeled subsets "
        "(standard deviation in brackets).",
        "",
        "## Test ROC-AUC",
        "",
        "| Method | " + " | ".join(f"{s:.0%} labeled (~{counts[s]:,})" for s in shares) + " |",
        "|---|" + "---:|" * len(shares),
    ]
    for method in methods:
        cells = [
            f"{summary.loc[(s, method), 'roc_auc']:.4f} ({summary.loc[(s, method), 'roc_auc_sd']:.4f})" for s in shares
        ]
        lines.append(f"| {method} | " + " | ".join(cells) + " |")
    lines += [
        "",
        "## Test default rate among the safest 80%",
        "",
        "| Method | " + " | ".join(f"{s:.0%} labeled" for s in shares) + " |",
        "|---|" + "---:|" * len(shares),
    ]
    for method in methods:
        cells = [f"{summary.loc[(s, method), 'default_rate'] * 100:.2f}%" for s in shares]
        lines.append(f"| {method} | " + " | ".join(cells) + " |")
    paired = paired_differences(results)
    lines += [
        "",
        "## Change against labeled only, paired by labeled subset",
        "",
        "Each method minus labeled only on the same labeled subset, test ROC-AUC; mean (smallest to largest) "
        f"over the {results['repeat'].nunique()} subsets.",
        "",
        "| Method | " + " | ".join(f"{s:.0%} labeled" for s in shares) + " |",
        "|---|" + "---:|" * len(shares),
    ]
    for method in methods[1:]:
        cells = []
        for s in shares:
            d = paired.loc[(s, method)]
            cells.append(f"{d['mean']:+.4f} ({d['min']:+.4f} to {d['max']:+.4f})")
        lines.append(f"| {method} | " + " | ".join(cells) + " |")
    lines += [
        "",
        f"Reference, the production model with every outcome known: test ROC-AUC {full['roc_auc']:.4f}, "
        f"default rate among the safest 80% {full['default_rate_at_80'] * 100:.2f}%.",
        "",
    ]
    return "\n".join(lines)


def run(splits: Splits, out_dir: Path) -> str:
    """Run the experiment and write the report."""
    results = experiment(splits, verbose=True)
    production = build_model(TUNED_PARAMS, SEED).fit(splits.X_train, splits.y_train)
    full = score_metrics(splits.y_test, production.predict_proba(splits.X_test)[:, 1])
    report = render_report(results, full)
    write_report(out_dir, "semi_supervised_report", report, results)
    return report


def main(argv: list[str] | None = None) -> None:
    """Run the experiment from the command line."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out-dir", type=Path, default=Path("reports"))
    args = parser.parse_args(argv)
    print(run(load_splits(), args.out_dir))


if __name__ == "__main__":
    main()
