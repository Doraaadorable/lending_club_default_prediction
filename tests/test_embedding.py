import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone

from loan_default.embedding import (
    EntityEmbeddingNet,
    backward,
    clean_keep_zip,
    embedding_size,
    forward,
    loss,
    nearest_levels,
)


def test_gradients_match_finite_differences() -> None:
    rng = np.random.default_rng(0)
    params = {
        "E0": rng.normal(size=(4, 2)),
        "E1": rng.normal(size=(3, 2)),
        "W1": rng.normal(size=(7, 5)),
        "b1": rng.normal(size=5),
        "w2": rng.normal(size=5),
        "b2": rng.normal(size=1),
    }
    numeric = rng.normal(size=(20, 3))
    codes = np.column_stack([rng.integers(0, 4, 20), rng.integers(0, 3, 20)])
    y = rng.integers(0, 2, 20).astype(float)
    p, cache = forward(params, numeric, codes)
    grads = backward(params, cache, p, codes, y, weight_decay=0.01)
    eps = 1e-6
    for name, value in params.items():
        for index in np.ndindex(value.shape):
            plus = {k: v.copy() for k, v in params.items()}
            minus = {k: v.copy() for k, v in params.items()}
            plus[name][index] += eps
            minus[name][index] -= eps
            numerical = (loss(plus, numeric, codes, y, 0.01) - loss(minus, numeric, codes, y, 0.01)) / (2 * eps)
            assert grads[name][index] == pytest.approx(numerical, rel=1e-4, abs=1e-7), (name, index)


def test_embedding_size_is_half_the_levels_capped() -> None:
    assert embedding_size(3, 10) == 2
    assert embedding_size(51, 10) == 10
    assert embedding_size(1, 10) == 1


@pytest.fixture(scope="module")
def fitted(small_train: tuple[pd.DataFrame, pd.Series]) -> tuple[EntityEmbeddingNet, pd.DataFrame]:
    X, y = small_train
    cleaned = clean_keep_zip(X)
    return EntityEmbeddingNet(epochs=3, min_count=20).fit(cleaned, y), cleaned


def test_network_outputs_probabilities_and_embedding_columns(fitted: tuple[EntityEmbeddingNet, pd.DataFrame]) -> None:
    net, cleaned = fitted
    proba = net.predict_proba(cleaned)
    assert proba.shape == (len(cleaned), 2)
    assert np.allclose(proba.sum(axis=1), 1)
    vectors = net.transform(cleaned)
    assert vectors.shape == (len(cleaned), len(net.get_feature_names_out()))
    assert 1 <= len(net.history_) <= 3


def test_unseen_category_gets_the_shared_rare_vector(fitted: tuple[EntityEmbeddingNet, pd.DataFrame]) -> None:
    net, cleaned = fitted
    row = cleaned.head(1).copy()
    row["addr_state"] = "ZZ"
    table = net.embedding_table("addr_state")
    width = table.shape[1]
    start = sum(net.embedding_table(c).shape[1] for c in ("purpose", "home_ownership_n"))
    assert np.allclose(net.transform(row)[0, start : start + width], table.loc["(rare or unseen)"])


def test_network_clones_like_any_scikit_learn_estimator() -> None:
    net = EntityEmbeddingNet(hidden=8)
    assert clone(net).get_params()["hidden"] == 8


def test_nearest_levels_uses_cosine_similarity() -> None:
    table = pd.DataFrame([[1.0, 0.0], [0.9, 0.1], [0.0, 1.0]], index=["a", "b", "c"])
    assert nearest_levels(table, "a", k=2) == ["b", "c"]
