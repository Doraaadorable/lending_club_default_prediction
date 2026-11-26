"""Unsupervised views of the borrowers: principal components and clusters.

    python -m loan_default.unsupervised        # writes reports/unsupervised_report.md and figures

1. Dimensionality reduction. PCA on the preprocessed features: how many directions carry the
   variation, what they mean, and whether a model trained on a few components keeps its accuracy.
2. Clustering. K-means on the borrowers' financial profile (FICO, DTI, employment length, income,
   loan size, loan-to-income), with k chosen by silhouette score. Each cluster is described in
   plain terms and its default rate is compared between 2007-2015 and 2016.
3. Clusters as features. Distances to the cluster centres are added to the tuned model's inputs.

No step here sees the default labels except the final comparison of default rates by cluster.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from loan_default.experiments import Splits, compare, comparison_table, load_splits, write_report
from loan_default.preprocessing import NUMERIC, SKEWED, build_preprocessor, clean
from loan_default.train import TUNED_PARAMS, build_model

SEED = 42
PROFILE = NUMERIC + SKEWED  # fico_n, dti_n, emp_length, revenue, loan_amnt, loan_to_income
K_RANGE = range(2, 11)
SILHOUETTE_SAMPLE = 10_000
VARIANCE_TARGETS = (0.80, 0.90, 0.95)


class ClusterDistances(TransformerMixin, BaseEstimator):  # type: ignore[misc]
    """Append the distance from each row to each k-means centre (learned in fit) to the feature matrix."""

    def __init__(self, n_clusters: int = 5, seed: int = SEED) -> None:
        """Store the settings; the centres are learned in fit()."""
        self.n_clusters = n_clusters
        self.seed = seed

    def fit(self, X: ArrayLike, y: object = None) -> "ClusterDistances":  # noqa: ARG002
        """Learn the cluster centres (the labels are not used)."""
        self.kmeans_ = KMeans(self.n_clusters, n_init=4, random_state=self.seed).fit(np.asarray(X, dtype=float))
        return self

    def transform(self, X: ArrayLike) -> NDArray[np.float64]:
        """The original columns followed by one distance column per centre."""
        values = np.asarray(X, dtype=float)
        out: NDArray[np.float64] = np.hstack([values, self.kmeans_.transform(values)])
        return out


def profile_matrix(X: pd.DataFrame, scaler: StandardScaler | None = None) -> tuple[NDArray[np.float64], StandardScaler]:
    """Financial profile used for clustering: log of skewed columns, median-filled, standardized."""
    cleaned = clean(X)[PROFILE].copy()
    cleaned[SKEWED] = np.log1p(cleaned[SKEWED])
    cleaned = cleaned.fillna(cleaned.median())
    if scaler is None:
        scaler = StandardScaler().fit(cleaned)
    values: NDArray[np.float64] = scaler.transform(cleaned)
    return values, scaler


def feature_matrix(X_train: pd.DataFrame, y_train: pd.Series) -> tuple[NDArray[np.float64], list[str]]:
    """The production preprocessing without the state encoding (the only step that uses the target)."""
    preprocessor = build_preprocessor(include_state=False).fit(X_train, y_train)
    names = [name.split("__", 1)[1] for name in preprocessor["columns"].get_feature_names_out()]
    return np.asarray(preprocessor.transform(X_train), dtype=float), names


def pca_summary(matrix: NDArray[np.float64]) -> tuple[pd.DataFrame, dict[float, int], PCA]:
    """Explained variance per component, components needed per variance target, and the fitted PCA."""
    pca = PCA(random_state=SEED).fit(StandardScaler().fit_transform(matrix))
    cumulative = np.cumsum(pca.explained_variance_ratio_)
    needed = {target: int(np.searchsorted(cumulative, target) + 1) for target in VARIANCE_TARGETS}
    table = pd.DataFrame(
        {
            "component": [f"PC{i + 1}" for i in range(len(cumulative))],
            "explained": pca.explained_variance_ratio_,
            "cumulative": cumulative,
        }
    )
    return table, needed, pca


def top_loadings(pca: PCA, names: list[str], component: int, k: int = 4) -> str:
    """The k features with the largest absolute loading on one component, with signs."""
    weights = pca.components_[component]
    order = np.argsort(-np.abs(weights))[:k]
    return ", ".join(f"{names[i]} ({weights[i]:+.2f})" for i in order)


def choose_k(profile: NDArray[np.float64], k_range: range = K_RANGE, seed: int = SEED) -> pd.DataFrame:
    """Inertia and silhouette score (on a sample) for each number of clusters."""
    rng = np.random.default_rng(seed)
    sample = profile[rng.choice(len(profile), min(SILHOUETTE_SAMPLE, len(profile)), replace=False)]
    rows = []
    for k in k_range:
        kmeans = KMeans(k, n_init=4, random_state=seed).fit(profile)
        rows.append(
            {
                "k": k,
                "inertia": float(kmeans.inertia_),
                "silhouette": float(silhouette_score(sample, kmeans.predict(sample), random_state=seed)),
            }
        )
    return pd.DataFrame(rows)


SILHOUETTE_TOLERANCE = 0.01


def pick_k(k_table: pd.DataFrame, tolerance: float = SILHOUETTE_TOLERANCE) -> int:
    """Largest k whose silhouette is within `tolerance` of the best: more segments for about the same separation."""
    close = k_table[k_table["silhouette"] > k_table["silhouette"].max() - tolerance]
    return int(close["k"].max())


def cluster_profiles(splits: Splits, k: int, seed: int = SEED) -> tuple[pd.DataFrame, KMeans, StandardScaler]:
    """Fit k-means on training profiles; describe each cluster and its default rate in both periods."""
    train_profile, scaler = profile_matrix(splits.X_train)
    test_profile, _ = profile_matrix(splits.X_test, scaler)
    kmeans = KMeans(k, n_init=10, random_state=seed).fit(train_profile)
    train_labels, test_labels = kmeans.predict(train_profile), kmeans.predict(test_profile)
    raw = clean(splits.X_train)
    rows = []
    for cluster in range(k):
        members = train_labels == cluster
        rows.append(
            {
                "cluster": cluster,
                "share_train": float(members.mean()),
                "share_test": float((test_labels == cluster).mean()),
                "fico": float(raw.loc[members, "fico_n"].mean()),
                "income": float(raw.loc[members, "revenue"].median()),
                "loan": float(raw.loc[members, "loan_amnt"].median()),
                "loan_to_income": float(raw.loc[members, "loan_to_income"].median()),
                "dti": float(raw.loc[members, "dti_n"].median()),
                "emp_length": float(raw.loc[members, "emp_length"].mean()),
                "default_rate_train": float(splits.y_train[members].mean()),
                "default_rate_test": float(splits.y_test[test_labels == cluster].mean()),
            }
        )
    table = pd.DataFrame(rows).sort_values("default_rate_train", ignore_index=True)
    table["description"] = [describe(row, table) for _, row in table.iterrows()]
    return table, kmeans, scaler


TRAITS = {
    "fico": ("lower FICO", "higher FICO"),
    "income": ("lower income", "higher income"),
    "loan_to_income": ("small loan for income", "large loan for income"),
    "dti": ("low DTI", "high DTI"),
    "emp_length": ("short employment", "long employment"),
}


def describe(row: pd.Series, table: pd.DataFrame, n_traits: int = 2) -> str:
    """Plain-language label: the traits on which this cluster stands furthest from the average cluster."""
    z = {column: (row[column] - table[column].mean()) / (table[column].std(ddof=0) + 1e-12) for column in TRAITS}
    strongest = sorted(z, key=lambda column: -abs(z[column]))[:n_traits]
    return ", ".join(TRAITS[c][1] if z[c] > 0 else TRAITS[c][0] for c in strongest)


def plot_clusters(profile: NDArray[np.float64], labels: NDArray[np.int64], path: Path) -> None:
    """Training borrowers on the first two principal components of their profile, coloured by cluster."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rng = np.random.default_rng(SEED)
    sample = rng.choice(len(profile), min(8000, len(profile)), replace=False)
    coords = PCA(2, random_state=SEED).fit(profile).transform(profile[sample])
    fig, ax = plt.subplots(figsize=(7, 5.5))
    for cluster in np.unique(labels):
        mask = labels[sample] == cluster
        ax.scatter(coords[mask, 0], coords[mask, 1], s=4, alpha=0.5, label=f"cluster {cluster}")
    ax.set_xlabel("Profile component 1")
    ax.set_ylabel("Profile component 2")
    ax.set_title("K-means clusters of 2007-2015 borrowers (8,000-loan sample)")
    ax.legend(markerscale=4, fontsize=8)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def candidates(n_components: int, k: int) -> dict[str, Pipeline]:
    """Production model, the same model on PCA components only, and with cluster distances added."""
    production = build_model(TUNED_PARAMS, SEED)
    on_pca = build_model(TUNED_PARAMS, SEED)
    on_pca.steps.insert(1, ("scale", StandardScaler()))
    on_pca.steps.insert(2, ("pca", PCA(n_components, random_state=SEED)))
    with_clusters = build_model(TUNED_PARAMS, SEED)
    with_clusters.steps.insert(1, ("clusters", ClusterDistances(k)))
    return {
        "production features": production,
        f"first {n_components} principal components only": on_pca,
        f"production + distances to {k} cluster centres": with_clusters,
    }


def render_report(
    variance: pd.DataFrame,
    needed: dict[float, int],
    loadings: list[str],
    k_table: pd.DataFrame,
    k: int,
    profiles: pd.DataFrame,
    results: pd.DataFrame,
) -> str:
    """Write the findings as Markdown."""
    lines = [
        "# Unsupervised analysis: principal components and clusters",
        "",
        "Generated by `python -m loan_default.unsupervised`.",
        "",
        "## Principal components of the preprocessed features",
        "",
        f"{len(variance)} standardized features. Components needed: "
        + ", ".join(f"{n} for {t:.0%} of the variance" for t, n in needed.items())
        + ".",
        "",
        "| Component | Variance explained | Cumulative | Largest loadings |",
        "|---|---:|---:|---|",
    ]
    for i, row in enumerate(variance.head(3).itertuples()):
        lines.append(f"| {row.component} | {row.explained:.1%} | {row.cumulative:.1%} | {loadings[i]} |")
    lines += [
        "",
        "## Choosing the number of clusters",
        "",
        "| k | Inertia | Silhouette |",
        "|---:|---:|---:|",
        *[f"| {r.k} | {r.inertia:,.0f} | {r.silhouette:.3f} |" for r in k_table.itertuples()],
        "",
        f"Silhouette scores are low and fall slowly with k: the borrowers form a continuum rather than "
        f"well-separated groups. We use k = {k}, the largest k within {SILHOUETTE_TOLERANCE} of the best score, "
        "so the clusters are a readable summary of the population rather than natural segments.",
        "",
        f"## The {k} borrower clusters, safest first",
        "",
        "| Cluster | Share 2007-2015 | Share 2016 | Mean FICO | Median income | Median loan | Loan / income "
        "| Median DTI | Default 2007-2015 | Default 2016 | Description |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for r in profiles.itertuples():
        lines.append(
            f"| {r.cluster} | {r.share_train:.1%} | {r.share_test:.1%} | {r.fico:.0f} | ${r.income:,.0f} | "
            f"${r.loan:,.0f} | {r.loan_to_income:.2f} | {r.dti:.1f} | {r.default_rate_train:.1%} | "
            f"{r.default_rate_test:.1%} | {r.description} |"
        )
    lines += [
        "",
        "![Clusters](figures/clusters.png)",
        "",
        "## Do components or clusters help the model?",
        "",
        *comparison_table(results, baseline=str(results["candidate"].iloc[0])),
        "",
    ]
    return "\n".join(lines)


def run(splits: Splits, out_dir: Path) -> str:
    """Run the analysis and write the report and figures."""
    matrix, names = feature_matrix(splits.X_train, splits.y_train)
    variance, needed, pca = pca_summary(matrix)
    loadings = [top_loadings(pca, names, i) for i in range(3)]

    profile, _ = profile_matrix(splits.X_train)
    k_table = choose_k(profile)
    k = pick_k(k_table)
    profiles, kmeans, _ = cluster_profiles(splits, k)
    plot_clusters(profile, kmeans.predict(profile), out_dir / "figures" / "clusters.png")

    results = compare(candidates(needed[0.95], k), splits)
    report = render_report(variance, needed, loadings, k_table, k, profiles, results)
    write_report(out_dir, "unsupervised_report", report, results)
    profiles.to_csv(out_dir / "cluster_profiles.csv", index=False)
    return report


def main(argv: list[str] | None = None) -> None:
    """Run the analysis from the command line."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out-dir", type=Path, default=Path("reports"))
    args = parser.parse_args(argv)
    print(run(load_splits(), args.out_dir))


if __name__ == "__main__":
    main()
