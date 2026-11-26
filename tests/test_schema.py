import numpy as np
import pandas as pd
import pytest

from loan_default.schema import FIELDS, check_frame

VALID = {
    "revenue": 60000.0,
    "loan_amnt": 12000.0,
    "fico_n": 690.0,
    "dti_n": 18.0,
    "emp_length": "5 years",
    "purpose": "debt_consolidation",
    "home_ownership_n": "RENT",
    "addr_state": "NY",
}


def frame(**changes: object) -> pd.DataFrame:
    return pd.DataFrame([VALID, {**VALID, **changes}])


def test_valid_rows_pass() -> None:
    check_frame(frame(), "sample")


def test_missing_dti_and_the_999_placeholder_are_valid() -> None:
    check_frame(frame(dti_n=np.nan), "sample")
    check_frame(frame(dti_n=999.0), "sample")


@pytest.mark.parametrize(
    "change",
    [
        {"fico_n": 900.0},
        {"revenue": 0.0},
        {"revenue": float("inf")},
        {"loan_amnt": float("nan")},
        {"revenue": 1e-310},  # would make loan-to-income infinite
        {"emp_length": "12 years"},
        {"addr_state": "New York"},
    ],
)
def test_rule_breaks_are_reported_with_the_row(change: dict[str, object]) -> None:
    field = next(iter(change))
    with pytest.raises(ValueError, match=f"row 1, {field}"):
        check_frame(frame(**change), "sample")


def test_missing_column_is_reported() -> None:
    with pytest.raises(ValueError, match="missing columns: fico_n"):
        check_frame(frame().drop(columns=["fico_n"]), "sample")


def test_schema_fields_match_the_model_inputs() -> None:
    assert set(FIELDS) == set(VALID)
