from pathlib import Path

import pandas as pd
import pytest

from loan_default.experiments import Splits, load_splits

DATA = Path(__file__).resolve().parents[1] / "data"


@pytest.fixture(scope="session")
def small_splits() -> Splits:
    """2,000 training loans and 800 test loans: enough for every method to run in seconds."""
    full = load_splits(DATA / "train.csv", DATA / "test.csv")
    train = full.X_train.sample(2000, random_state=0).index
    test = full.X_test.sample(800, random_state=0).index
    return Splits(
        full.X_train.loc[train].reset_index(drop=True),
        full.y_train.loc[train].reset_index(drop=True),
        full.X_test.loc[test].reset_index(drop=True),
        full.y_test.loc[test].reset_index(drop=True),
    )


@pytest.fixture(scope="session")
def small_train(small_splits: Splits) -> tuple[pd.DataFrame, pd.Series]:
    return small_splits.X_train, small_splits.y_train
