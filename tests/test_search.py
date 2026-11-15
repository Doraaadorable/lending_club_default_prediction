import pandas as pd

from loan_default.search import PARAM_DISTRIBUTIONS, best_params


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
