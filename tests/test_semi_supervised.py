import numpy as np
import pandas as pd
import pytest

from loan_default.experiments import Splits
from loan_default.semi_supervised import METHODS, UNLABELED, experiment, hide_labels, n_clusters_for


def test_hide_labels_keeps_a_stratified_share() -> None:
    y = pd.Series([0] * 900 + [1] * 100)
    labels = hide_labels(y, 0.1, seed=0)
    known = labels != UNLABELED
    assert known.sum() == 100
    assert (labels[known] == y[known]).all()
    assert labels[known].mean() == pytest.approx(0.1)


def test_cluster_count_is_bounded() -> None:
    assert n_clusters_for(30) == 10
    assert n_clusters_for(3000) == 100
    assert n_clusters_for(1_000_000) == 200


def test_every_method_produces_test_metrics(small_splits: Splits) -> None:
    results = experiment(small_splits, shares=(0.2,), repeats=1)
    assert results["method"].tolist() == list(METHODS)
    assert results["roc_auc"].between(0.4, 1).all()
    assert np.isfinite(results["default_rate_at_80"]).all()
