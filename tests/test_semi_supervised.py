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


def test_paired_differences_compare_each_method_on_the_same_subset() -> None:
    from loan_default.semi_supervised import paired_differences

    results = pd.DataFrame(
        {
            "label_share": [0.1] * 4,
            "repeat": [0, 0, 1, 1],
            "method": ["labeled only", "other", "labeled only", "other"],
            "roc_auc": [0.60, 0.62, 0.70, 0.69],
        }
    )
    row = paired_differences(results).loc[(0.1, "other")]
    assert row["mean"] == pytest.approx(0.005)
    assert row["min"] == pytest.approx(-0.01)
    assert row["max"] == pytest.approx(0.02)


def test_early_stopping_set_is_held_out_from_the_training_pool(small_splits: Splits) -> None:
    from loan_default.semi_supervised import prepare

    labels = hide_labels(small_splits.y_train, 0.5, seed=0)
    data = prepare(small_splits, labels, seed=0)
    assert len(data.train) + len(data.val) == len(small_splits.X_train)
    assert (data.val_labels != UNLABELED).all()
    assert data.labeled.sum() + len(data.val) == (labels != UNLABELED).sum()
