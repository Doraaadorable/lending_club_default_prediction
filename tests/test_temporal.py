from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from loan_default.preprocessing import load_split
from loan_default.temporal import forward_chaining, issue_years, score_summary
from loan_default.train import TrainConfig

DATA = Path(__file__).resolve().parents[1] / "data"


@pytest.fixture(scope="module")
def sample() -> tuple[pd.DataFrame, pd.Series]:
    """Up to 800 loans per origination year from 2011 to 2014."""
    X, y = load_split(DATA / "train.csv")
    years = issue_years(X)
    keep = pd.concat(
        [
            years[years == year].sample(min(800, int((years == year).sum())), random_state=0)
            for year in range(2011, 2015)
        ]
    ).index
    return X.loc[keep].reset_index(drop=True), y.loc[keep].reset_index(drop=True)


def test_each_year_is_scored_by_a_model_trained_only_on_earlier_years(sample: tuple[pd.DataFrame, pd.Series]) -> None:
    X, y = sample
    years = issue_years(X)
    config = TrainConfig(model_params={"max_iter": 20, "max_leaf_nodes": 7})
    table = forward_chaining(X, y, years, (2013, 2014), config)
    assert table["evaluation_year"].tolist() == [2013, 2014]
    assert table["n_train"].tolist() == [int((years < 2013).sum()), int((years < 2014).sum())]
    assert table["n_eval"].tolist() == [int((years == 2013).sum()), int((years == 2014).sum())]
    assert table["roc_auc"].between(0, 1).all()


def test_gap_is_actual_minus_predicted() -> None:
    y = pd.Series([0, 0, 1, 1, 1])
    summary = score_summary(y, np.full(5, 0.4))
    assert summary["actual"] == pytest.approx(0.6)
    assert summary["mean_predicted"] == pytest.approx(0.4)
    assert summary["gap"] == pytest.approx(0.2)
