import numpy as np
import pandas as pd
import pytest

from loan_default.explain import additivity_error, by_field, field_of, global_importance, shap_values, top_contributors
from loan_default.preprocessing import clean
from loan_default.train import build_model


@pytest.mark.parametrize(
    ("column", "field"),
    [
        ("num__fico_n", "fico_n"),
        ("num__missingindicator_emp_length", "emp_length"),
        ("log__loan_to_income", "loan_to_income"),
        ("log__loan_amnt", "loan_amnt"),
        ("cat__purpose_infrequent_sklearn", "purpose"),
        ("cat__home_ownership_n_RENT", "home_ownership_n"),
        ("state__addr_state", "addr_state"),
    ],
)
def test_each_transformed_column_maps_back_to_its_application_field(column: str, field: str) -> None:
    assert field_of(column) == field


def test_unknown_column_is_an_error() -> None:
    with pytest.raises(ValueError):
        field_of("num__mystery")


def test_by_field_sums_the_columns_of_one_field() -> None:
    values = np.array([[0.1, 0.2, -0.3]])
    out = by_field(values, ["cat__purpose_car", "cat__purpose_other", "num__fico_n"])
    assert out.loc[0, "purpose"] == pytest.approx(0.3)
    assert out.loc[0, "fico_n"] == pytest.approx(-0.3)


def test_top_contributors_list_only_fields_that_raised_the_score_largest_first() -> None:
    contributions = pd.Series({"fico_n": 0.4, "loan_to_income": 0.6, "revenue": -0.2, "dti_n": 0.1})
    assert top_contributors(contributions, k=2) == ["Amount requested is large relative to income", "Credit score"]
    assert len(top_contributors(contributions, k=5)) == 3


def test_shap_values_add_up_to_the_model_log_odds(small_train: tuple[pd.DataFrame, pd.Series]) -> None:
    X, y = small_train
    model = build_model({"max_iter": 30}, 0).fit(X, y)
    fields, base, _ = shap_values(model, X.head(200))
    assert additivity_error(model, X.head(200), fields, base) < 1e-6
    importance = global_importance(fields, clean(X.head(200)))
    assert set(importance["field"]) == set(fields.columns)
    assert importance["mean_abs_shap"].is_monotonic_decreasing


def test_loan_size_group_sums_the_related_fields_before_the_absolute_value() -> None:
    from loan_default.explain import grouped_importance

    fields = pd.DataFrame({"loan_amnt": [0.3, -0.1], "revenue": [-0.3, -0.1], "loan_to_income": [0.1, 0.0]})
    assert grouped_importance(fields) == pytest.approx((0.1 + 0.2) / 2)
