import numpy as np
import pandas as pd
import pytest

from loan_default.preprocessing import DROP_COLS, DTI_CAP, clean


@pytest.fixture
def loans() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "id": [1, 2, 3],
            "issue_d": ["2014-01-01"] * 3,
            "revenue": [50_000.0, 100_000.0, 40_000.0],
            "dti_n": [12.5, 999.0, 250.0],
            "loan_amnt": [10_000, 5_000, 20_000],
            "fico_n": [700.0, 680.0, 720.0],
            "experience_c": [1, 1, 1],
            "emp_length": ["10+ years", "NI", "< 1 year"],
            "purpose": ["car", "other", "car"],
            "home_ownership_n": ["RENT", "OWN", "MORTGAGE"],
            "addr_state": ["NY", "CA", "TX"],
            "zip_code": ["100xx", "900xx", "750xx"],
            "title": ["a", "b", "c"],
            "desc": [None, None, None],
        }
    )


def test_employment_length_becomes_years_and_ni_becomes_missing(loans: pd.DataFrame) -> None:
    result = clean(loans)["emp_length"]
    assert result.iloc[0] == 10
    assert np.isnan(result.iloc[1])
    assert result.iloc[2] == 0


def test_dti_placeholder_is_missing_and_extremes_are_capped(loans: pd.DataFrame) -> None:
    result = clean(loans)["dti_n"]
    assert result.iloc[0] == 12.5
    assert np.isnan(result.iloc[1])
    assert result.iloc[2] == DTI_CAP


def test_loan_to_income_is_added(loans: pd.DataFrame) -> None:
    assert clean(loans)["loan_to_income"].tolist() == [0.2, 0.05, 0.5]


def test_excluded_columns_are_dropped(loans: pd.DataFrame) -> None:
    assert not set(DROP_COLS) & set(clean(loans).columns)


def test_input_frame_is_not_modified(loans: pd.DataFrame) -> None:
    before = loans.copy()
    clean(loans)
    pd.testing.assert_frame_equal(loans, before)
