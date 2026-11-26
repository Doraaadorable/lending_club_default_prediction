import numpy as np
import pandas as pd
import pytest
from sklearn.impute import IterativeImputer, KNNImputer, SimpleImputer

from loan_default.imputation import mask_at_random, missingness_by_outcome, reconstruction_error
from loan_default.preprocessing import NUMERIC, build_preprocessor, make_imputer


def test_make_imputer_returns_each_kind_and_rejects_unknown_names() -> None:
    assert isinstance(make_imputer("median"), SimpleImputer)
    assert isinstance(make_imputer("iterative"), IterativeImputer)
    assert isinstance(make_imputer("knn"), KNNImputer)
    with pytest.raises(ValueError, match="imputer must be one of"):
        make_imputer("magic")


@pytest.mark.parametrize("strategy", ["median", "mean", "iterative", "knn"])
def test_every_imputer_leaves_no_gaps_and_keeps_the_missing_flag(
    small_train: tuple[pd.DataFrame, pd.Series], strategy: str
) -> None:
    X, y = small_train
    preprocessor = build_preprocessor(numeric_imputer=strategy).fit(X, y)
    names = preprocessor["columns"].get_feature_names_out()
    out = preprocessor.transform(X)
    assert not np.isnan(out).any()
    assert any("missingindicator_emp_length" in name for name in names)


def test_dropping_state_and_flags_removes_their_columns(small_train: tuple[pd.DataFrame, pd.Series]) -> None:
    X, y = small_train
    names = build_preprocessor(add_indicator=False, include_state=False).fit(X, y)["columns"].get_feature_names_out()
    assert not any("addr_state" in name or "missingindicator" in name for name in names)


def test_mask_hides_only_known_values_at_about_the_requested_share() -> None:
    frame = pd.DataFrame({"a": np.arange(10_000, dtype=float), "b": np.r_[np.full(5000, np.nan), np.ones(5000)]})
    masked, hidden = mask_at_random(frame, ["a", "b"], 0.1, seed=0)
    assert hidden["a"].mean() == pytest.approx(0.1, abs=0.01)
    assert not hidden.loc[frame["b"].isna(), "b"].any()  # an already-missing value is not "hidden"
    assert masked.loc[hidden["a"], "a"].isna().all()
    assert masked.loc[~hidden["a"], "a"].equals(frame.loc[~hidden["a"], "a"])


def test_missingness_table_reports_the_employment_gap(small_train: tuple[pd.DataFrame, pd.Series]) -> None:
    X, y = small_train
    table = missingness_by_outcome(X, y).set_index("column")
    assert list(table.index) == NUMERIC
    assert table.loc["emp_length", "share_missing"] > 0


def test_reconstruction_error_has_one_row_per_imputer_and_column(small_train: tuple[pd.DataFrame, pd.Series]) -> None:
    errors = reconstruction_error(small_train[0], strategies=("median", "iterative"))
    assert len(errors) == 2 * len(NUMERIC)
    assert (errors["rmse"] >= errors["mae"]).all()
    assert (errors["n_hidden"] > 0).all()
