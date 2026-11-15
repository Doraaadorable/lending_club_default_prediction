from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from loan_default.search import PARAM_DISTRIBUTIONS, best_params, chronological_folds, search


def test_best_params_come_from_the_top_row_as_plain_numbers() -> None:
    results = pd.DataFrame(
        {
            "learning_rate": [0.024918939689346737, 0.1],
            "max_iter": [510, 200],
            "max_leaf_nodes": [15, 31],
            "min_samples_leaf": [337, 20],
            "l2_regularization": [0.025720264197882346, 1.0],
            "max_features": [0.5823279265714709, 0.9],
            "mean_test_ROC_AUC": [0.6634, 0.66],
        }
    )
    params = best_params(results)
    assert params == {
        "learning_rate": 0.024918939689346737,
        "max_iter": 510,
        "max_leaf_nodes": 15,
        "min_samples_leaf": 337,
        "l2_regularization": 0.025720264197882346,
        "max_features": 0.5823279265714709,
    }
    assert all(type(params[name]) is int for name in ("max_iter", "max_leaf_nodes", "min_samples_leaf"))


def test_search_space_matches_notebook_05() -> None:
    assert sorted(PARAM_DISTRIBUTIONS) == sorted(
        f"model__{name}"
        for name in (
            "learning_rate",
            "max_iter",
            "max_leaf_nodes",
            "min_samples_leaf",
            "l2_regularization",
            "max_features",
        )
    )


def test_chronological_folds_validate_each_year_on_earlier_loans_only() -> None:
    dates = pd.Series(["2010-05-01", "2011-01-01", "2012-03-01", "2012-07-01", "2013-02-01", "2011-09-01"])
    folds = chronological_folds(dates, years=(2012, 2013))
    (train_2012, valid_2012), (train_2013, valid_2013) = folds
    assert sorted(train_2012) == [0, 1, 5] and sorted(valid_2012) == [2, 3]
    assert sorted(train_2013) == [0, 1, 2, 3, 5] and list(valid_2013) == [4]
    assert all(np.intersect1d(t, v).size == 0 for t, v in folds)


def test_unknown_fold_scheme_is_rejected() -> None:
    with pytest.raises(ValueError, match="folds must be"):
        search(Path(__file__).resolve().parents[1] / "data" / "train.csv", folds="random")
