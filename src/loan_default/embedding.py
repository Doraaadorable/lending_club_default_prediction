"""Entity embeddings for the categorical fields, learned by a small neural network.

    python -m loan_default.embedding        # writes reports/embedding_report.md and a figure

Each category (a state, a loan purpose, a housing status, a ZIP prefix) is mapped to a short
vector of numbers that the network learns while predicting default (Guo & Berkhahn, 2016).
Categories with similar default behaviour end up close together, so the vectors carry more
than one smoothed default rate (target encoding) and stay short where one-hot encoding would
need one column per level.

The network is written in NumPy so the project needs no deep-learning framework:

    [embedding per categorical field] + [scaled numeric fields]
        -> dense layer (ReLU) -> one output (sigmoid) = probability of default

trained with mini-batch Adam on log loss, with weight decay and early stopping on a
held-out 10% of the rows it is given. It is used two ways:

1. on its own, as a classifier;
2. as an encoder: its learned vectors replace one-hot and target encoding as inputs to the
   tuned gradient boosting model.

ZIP code is excluded from the production model as a geographic proxy (README, Fairness). It is
embedded here only to measure what that exclusion costs in accuracy.
"""

import argparse
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

from loan_default.experiments import Splits, compare, comparison_table, load_splits, write_report
from loan_default.preprocessing import NUMERIC, SKEWED, clean
from loan_default.train import TUNED_PARAMS, build_model

Array = NDArray[np.float64]
CATEGORICAL = ("purpose", "home_ownership_n", "addr_state")
WITH_ZIP = (*CATEGORICAL, "zip_code")
SEED = 42


def clean_keep_zip(df: pd.DataFrame) -> pd.DataFrame:
    """The shared cleaning step, with the ZIP prefix put back for the embedding experiment."""
    cleaned = clean(df)
    cleaned["zip_code"] = df["zip_code"].astype(str).to_numpy()
    return cleaned


def embedding_size(n_levels: int, max_dim: int) -> int:
    """Rule of thumb from the entity-embedding literature: half the number of levels, capped."""
    return int(max(1, min(max_dim, (n_levels + 1) // 2)))


def _sigmoid(z: Array) -> Array:
    """Logistic function without clipping, so it stays consistent with the gradient in backward()."""
    out: Array = np.exp(-np.logaddexp(0.0, -z))
    return out


def forward(params: dict[str, Array], numeric: Array, codes: NDArray[np.int64]) -> tuple[Array, dict[str, Array]]:
    """Probability of default for each row, and the intermediate values backward() needs."""
    n_fields = codes.shape[1]
    pieces = [params[f"E{j}"][codes[:, j]] for j in range(n_fields)]
    inputs = np.hstack([*pieces, numeric])
    hidden_in = inputs @ params["W1"] + params["b1"]
    hidden = np.maximum(hidden_in, 0.0)
    logit = hidden @ params["w2"] + params["b2"]
    return _sigmoid(logit), {"inputs": inputs, "hidden_in": hidden_in, "hidden": hidden, "logit": logit}


def loss(params: dict[str, Array], numeric: Array, codes: NDArray[np.int64], y: Array, weight_decay: float) -> float:
    """Mean log loss plus L2 weight decay on the weights and embeddings.

    Log loss is computed from the logits (log(1 + e^z) - y z), which is exact and stable at any score.
    """
    _, cache = forward(params, numeric, codes)
    logit = cache["logit"]
    log_loss = np.mean(np.logaddexp(0.0, logit) - y * logit)
    penalty = sum(float((value**2).sum()) for name, value in params.items() if not name.startswith("b"))
    return float(log_loss + 0.5 * weight_decay * penalty)


def backward(
    params: dict[str, Array], cache: dict[str, Array], p: Array, codes: NDArray[np.int64], y: Array, weight_decay: float
) -> dict[str, Array]:
    """Gradient of loss() with respect to every parameter."""
    n = len(y)
    d_logit = (p - y) / n
    grads = {"w2": cache["hidden"].T @ d_logit, "b2": np.array([d_logit.sum()])}
    d_hidden = np.outer(d_logit, params["w2"]) * (cache["hidden_in"] > 0)
    grads["W1"] = cache["inputs"].T @ d_hidden
    grads["b1"] = d_hidden.sum(axis=0)
    d_inputs = d_hidden @ params["W1"].T
    start = 0
    for j in range(codes.shape[1]):
        table = params[f"E{j}"]
        width = table.shape[1]
        grad = np.zeros_like(table)
        np.add.at(grad, codes[:, j], d_inputs[:, start : start + width])
        grads[f"E{j}"] = grad
        start += width
    for name, value in params.items():
        if not name.startswith("b"):
            grads[name] = grads[name] + weight_decay * value
    return grads


class EntityEmbeddingNet(ClassifierMixin, TransformerMixin, BaseEstimator):  # type: ignore[misc]
    """Small network that learns an embedding per category while predicting default.

    `predict_proba` gives the network's own probability of default; `transform` returns the
    learned embedding vectors of each row's categories, for use as features by another model.
    Levels seen fewer than `min_count` times in training share one "rare or unseen" vector.
    """

    def __init__(
        self,
        categorical: Sequence[str] = CATEGORICAL,
        numeric: Sequence[str] = tuple(NUMERIC + SKEWED),
        skewed: Sequence[str] = tuple(SKEWED),
        min_count: int = 30,
        max_dim: int = 10,
        hidden: int = 32,
        epochs: int = 15,
        batch_size: int = 1024,
        learning_rate: float = 0.003,
        weight_decay: float = 1e-5,
        patience: int = 3,
        seed: int = SEED,
    ) -> None:
        """Store the settings unchanged, as scikit-learn requires; nothing is learned until fit()."""
        self.categorical = categorical
        self.numeric = numeric
        self.skewed = skewed
        self.min_count = min_count
        self.max_dim = max_dim
        self.hidden = hidden
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.patience = patience
        self.seed = seed

    # ---- input preparation -------------------------------------------------------------
    def _numeric_matrix(self, X: pd.DataFrame, fit: bool) -> Array:
        values = X[list(self.numeric)].astype(float).copy()
        skewed = [c for c in self.skewed if c in values.columns]
        values[skewed] = np.log1p(values[skewed])
        if fit:
            self.medians_ = values.median()
            self.flag_columns_ = [c for c in values.columns if values[c].isna().any()]
        flags = values[self.flag_columns_].isna().astype(float).to_numpy()
        filled = values.fillna(self.medians_).to_numpy()
        if fit:
            self.means_, self.stds_ = filled.mean(axis=0), filled.std(axis=0) + 1e-9
        matrix: Array = np.hstack([(filled - self.means_) / self.stds_, flags])
        return matrix

    def _codes(self, X: pd.DataFrame, fit: bool) -> NDArray[np.int64]:
        if fit:
            self.levels_ = {}
            for column in self.categorical:
                counts = X[column].astype(str).value_counts()
                kept = sorted(counts[counts >= self.min_count].index)
                self.levels_[column] = {level: i + 1 for i, level in enumerate(kept)}  # 0 = rare or unseen
        columns = [
            X[c].astype(str).map(self.levels_[c]).fillna(0).astype(np.int64).to_numpy() for c in self.categorical
        ]
        return np.column_stack(columns)

    # ---- training ----------------------------------------------------------------------
    def _init_params(self, n_numeric: int, rng: np.random.Generator) -> dict[str, Array]:
        params: dict[str, Array] = {}
        width = 0
        for j, column in enumerate(self.categorical):
            n_levels = len(self.levels_[column]) + 1
            dim = embedding_size(n_levels, self.max_dim)
            params[f"E{j}"] = rng.normal(0, 0.05, size=(n_levels, dim))
            width += dim
        width += n_numeric
        params["W1"] = rng.normal(0, np.sqrt(2 / width), size=(width, self.hidden))
        params["b1"] = np.zeros(self.hidden)
        params["w2"] = rng.normal(0, np.sqrt(1 / self.hidden), size=self.hidden)
        params["b2"] = np.zeros(1)
        return params

    def fit(self, X: pd.DataFrame, y: ArrayLike) -> "EntityEmbeddingNet":
        """Learn embeddings and weights with Adam, stopping when held-out log loss stops improving."""
        rng = np.random.default_rng(self.seed)
        target = np.asarray(y, dtype=float)
        self.classes_ = np.array([0, 1])
        order = rng.permutation(len(target))
        n_valid = max(1, len(target) // 10)
        valid, train = order[:n_valid], order[n_valid:]
        # vocabulary, medians and scales come from the training part only, so early stopping judges unseen rows
        fit_rows = X.iloc[np.sort(train)]
        self._numeric_matrix(fit_rows, fit=True)
        self._codes(fit_rows, fit=True)
        numeric, codes = self._numeric_matrix(X, fit=False), self._codes(X, fit=False)

        params = self._init_params(numeric.shape[1], rng)
        # start the output at the base rate so early updates go to the features, not the intercept
        base = float(np.clip(target[train].mean(), 1e-3, 1 - 1e-3))
        params["b2"] = np.array([np.log(base / (1 - base))])
        moment = {k: np.zeros_like(v) for k, v in params.items()}
        velocity = {k: np.zeros_like(v) for k, v in params.items()}
        beta1, beta2, step = 0.9, 0.999, 0
        best = (np.inf, {k: v.copy() for k, v in params.items()})
        self.history_: list[float] = []
        waited = 0
        for _ in range(self.epochs):
            train = rng.permutation(train)
            for start in range(0, len(train), self.batch_size):
                batch = train[start : start + self.batch_size]
                p, cache = forward(params, numeric[batch], codes[batch])
                grads = backward(params, cache, p, codes[batch], target[batch], self.weight_decay)
                step += 1
                for name in params:
                    moment[name] = beta1 * moment[name] + (1 - beta1) * grads[name]
                    velocity[name] = beta2 * velocity[name] + (1 - beta2) * grads[name] ** 2
                    m_hat = moment[name] / (1 - beta1**step)
                    v_hat = velocity[name] / (1 - beta2**step)
                    params[name] = params[name] - self.learning_rate * m_hat / (np.sqrt(v_hat) + 1e-8)
            held_out = loss(params, numeric[valid], codes[valid], target[valid], 0.0)
            self.history_.append(held_out)
            if held_out < best[0] - 1e-5:
                best, waited = (held_out, {k: v.copy() for k, v in params.items()}), 0
            else:
                waited += 1
                if waited >= self.patience:
                    break
        self.params_ = best[1]
        return self

    # ---- use ---------------------------------------------------------------------------
    def predict_proba(self, X: pd.DataFrame) -> Array:
        """Probability of each class, columns [repaid, default]."""
        p, _ = forward(self.params_, self._numeric_matrix(X, fit=False), self._codes(X, fit=False))
        out: Array = np.column_stack([1 - p, p])
        return out

    def predict(self, X: pd.DataFrame) -> NDArray[np.int64]:
        """Class at a 0.5 threshold (the project uses scores and an approval share instead)."""
        out: NDArray[np.int64] = (self.predict_proba(X)[:, 1] >= 0.5).astype(np.int64)
        return out

    def transform(self, X: pd.DataFrame) -> Array:
        """Embedding vectors of each row's categories, side by side."""
        codes = self._codes(X, fit=False)
        out: Array = np.hstack([self.params_[f"E{j}"][codes[:, j]] for j in range(len(self.categorical))])
        return out

    def get_feature_names_out(self, input_features: object = None) -> NDArray[np.str_]:  # noqa: ARG002
        """Names like addr_state_emb0, addr_state_emb1, ..."""
        names = [
            f"{column}_emb{k}"
            for j, column in enumerate(self.categorical)
            for k in range(self.params_[f"E{j}"].shape[1])
        ]
        return np.array(names)

    def embedding_table(self, column: str) -> pd.DataFrame:
        """Learned vector of every kept level of one categorical field (plus the rare/unseen vector)."""
        j = list(self.categorical).index(column)
        table = self.params_[f"E{j}"]
        labels = ["(rare or unseen)", *self.levels_[column]]
        return pd.DataFrame(table, index=labels)


def network_pipeline(categorical: Sequence[str] = CATEGORICAL) -> Pipeline:
    """Cleaning followed by the embedding network as the classifier."""
    return Pipeline(
        [
            ("clean", FunctionTransformer(clean_keep_zip)),
            ("net", EntityEmbeddingNet(categorical=categorical)),
        ]
    )


def boosting_on_embeddings(
    categorical: Sequence[str] = CATEGORICAL, params: dict[str, float | int] = TUNED_PARAMS
) -> Pipeline:
    """Tuned gradient boosting whose categorical inputs are learned embeddings instead of one-hot/target encoding."""
    numeric = make_pipeline(SimpleImputer(strategy="median", add_indicator=True), StandardScaler())
    skewed = make_pipeline(FunctionTransformer(np.log1p, feature_names_out="one-to-one"), StandardScaler())
    columns = ColumnTransformer(
        [
            ("num", numeric, NUMERIC),
            ("log", skewed, SKEWED),
            ("emb", EntityEmbeddingNet(categorical=categorical), [*NUMERIC, *SKEWED, *categorical]),
        ]
    )
    return Pipeline(
        [
            ("preprocess", Pipeline([("clean", FunctionTransformer(clean_keep_zip)), ("columns", columns)])),
            ("model", HistGradientBoostingClassifier(**params, random_state=SEED)),
        ]
    )


def nearest_levels(table: pd.DataFrame, level: str, k: int = 3) -> list[str]:
    """The k levels whose embedding is most similar (cosine) to `level`'s."""
    vectors = table.to_numpy()
    norms = np.linalg.norm(vectors, axis=1) + 1e-12
    target = vectors[table.index.get_loc(level)]
    similarity = vectors @ target / (norms * (np.linalg.norm(target) + 1e-12))
    ranked = [table.index[i] for i in np.argsort(-similarity) if table.index[i] != level]
    return [str(name) for name in ranked[:k]]


def state_embedding_summary(net: EntityEmbeddingNet, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    """Per state: loans, default rate, first principal component of its embedding, and nearest states."""
    table = net.embedding_table("addr_state").drop(index="(rare or unseen)")
    centred = table.to_numpy() - table.to_numpy().mean(axis=0)
    first_pc = np.linalg.svd(centred, full_matrices=False)[0][:, 0]
    rates = y.groupby(X["addr_state"].to_numpy()).agg(["size", "mean"])
    summary = pd.DataFrame(
        {
            "state": table.index,
            "loans": rates.loc[table.index, "size"].to_numpy(),
            "default_rate": rates.loc[table.index, "mean"].to_numpy(),
            "embedding_pc1": first_pc,
            "nearest": [", ".join(nearest_levels(table, state)) for state in table.index],
        }
    )
    if np.corrcoef(summary["embedding_pc1"], summary["default_rate"])[0, 1] < 0:
        summary["embedding_pc1"] = -summary["embedding_pc1"]  # sign of a component is arbitrary
    return summary.sort_values("loans", ascending=False, ignore_index=True)


def plot_states(summary: pd.DataFrame, net: EntityEmbeddingNet, path: Path) -> None:
    """Scatter of state embeddings on their first two principal components, coloured by default rate."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    table = net.embedding_table("addr_state").loc[summary["state"]]
    centred = table.to_numpy() - table.to_numpy().mean(axis=0)
    u, s, _ = np.linalg.svd(centred, full_matrices=False)
    coords = u[:, :2] * s[:2]
    fig, ax = plt.subplots(figsize=(7, 5.5))
    points = ax.scatter(coords[:, 0], coords[:, 1], c=summary["default_rate"] * 100, cmap="viridis", s=40)
    for (x, y_), state in zip(coords, summary["state"], strict=True):
        ax.annotate(state, (x, y_), fontsize=7, xytext=(3, 3), textcoords="offset points")
    fig.colorbar(points, ax=ax, label="Training default rate (%)")
    ax.set_xlabel("Embedding component 1")
    ax.set_ylabel("Embedding component 2")
    ax.set_title("State embeddings learned while predicting default")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def render_report(results: pd.DataFrame, states: pd.DataFrame, history: list[float]) -> str:
    """Write the findings as Markdown."""
    correlation = float(np.corrcoef(states["embedding_pc1"], states["default_rate"])[0, 1])
    lines = [
        "# Entity embeddings",
        "",
        "Generated by `python -m loan_default.embedding`.",
        "",
        "## Model comparison",
        "",
        *comparison_table(results, baseline=str(results["candidate"].iloc[0])),
        "",
        "ZIP code stays out of the production model (geographic proxy); "
        "the last two rows measure what leaving it out costs.",
        "",
        "## What the state embeddings learned",
        "",
        f"Network trained on all 2007-2015 loans: {len(history)} epochs, held-out log loss "
        f"{history[0]:.4f} after the first and {min(history):.4f} at best.",
        "",
        f"Correlation between each state's first embedding component and its training default rate: {correlation:.2f}. "
        "The embedding recovers the state's risk level, as target encoding does, and adds further dimensions "
        "that place states with similar borrowers together.",
        "",
        "| State | Loans | Default rate | Embedding component 1 | Most similar states |",
        "|---|---:|---:|---:|---|",
    ]
    for row in states.head(12).itertuples():
        lines.append(
            f"| {row.state} | {row.loans:,} | {row.default_rate * 100:.2f}% | "
            f"{row.embedding_pc1:+.3f} | {row.nearest} |"
        )
    lines += ["", "![State embeddings](figures/state_embeddings.png)", ""]
    return "\n".join(lines)


def run(splits: Splits, out_dir: Path) -> str:
    """Compare the embedding models with the production pipeline and describe the state embeddings."""
    candidates = {
        "one-hot + target encoding (production)": build_model(TUNED_PARAMS, SEED),
        "embedding network on its own": network_pipeline(),
        "boosting on learned embeddings": boosting_on_embeddings(),
        "embedding network, with ZIP (comparison only)": network_pipeline(WITH_ZIP),
        "boosting on learned embeddings, with ZIP (comparison only)": boosting_on_embeddings(WITH_ZIP),
    }
    results = compare(candidates, splits)
    net = network_pipeline().fit(splits.X_train, splits.y_train)["net"]
    states = state_embedding_summary(net, splits.X_train, splits.y_train)
    plot_states(states, net, out_dir / "figures" / "state_embeddings.png")
    report = render_report(results, states, net.history_)
    write_report(out_dir, "embedding_report", report, results)
    states.to_csv(out_dir / "state_embeddings.csv", index=False)
    return report


def main(argv: list[str] | None = None) -> None:
    """Run the comparison from the command line."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out-dir", type=Path, default=Path("reports"))
    args = parser.parse_args(argv)
    print(run(load_splits(), args.out_dir))


if __name__ == "__main__":
    main()
