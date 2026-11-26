import numpy as np
import pandas as pd
import pytest

from loan_default.causal import (
    Nuisance,
    aipw_scores,
    causal_frame,
    cross_fit,
    estimates,
    standardized_difference,
)


def confounded(n: int = 6000, effect: float = 0.10, seed: int = 0) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """FICO lowers both the chance of treatment and of default, so the naive difference overstates the effect."""
    rng = np.random.default_rng(seed)
    fico = rng.normal(700, 30, n)
    t = (rng.random(n) < 1 / (1 + np.exp((fico - 700) / 25))).astype(np.int64)
    p = np.clip(0.2 - 0.003 * (fico - 700) + effect * t, 0.0, 1.0)  # stays inside (0, 1) for FICO within 3 sd
    y = (rng.random(n) < p).astype(float)
    W = pd.DataFrame(
        {
            "fico_n": fico,
            "dti_n": rng.normal(18, 5, n),
            "emp_length": rng.integers(0, 11, n).astype(float),
            "log_income": rng.normal(11, 0.5, n),
            "issue_year": 2014.0,
            "purpose": "debt_consolidation",
            "home_ownership_n": "RENT",
            "addr_state": "CA",
        }
    )
    return W, t, y


def test_doubly_robust_estimate_recovers_a_known_effect() -> None:
    W, t, y = confounded()
    result = estimates(t, y, cross_fit(W, t, y, folds=3))
    assert result["naive difference"]["ate"] > 0.15  # confounded
    for estimator in ("regression adjustment", "inverse propensity weighting", "doubly robust (AIPW)"):
        assert abs(result[estimator]["ate"] - 0.10) < 0.04, estimator
        assert abs(result[estimator]["ate"] - 0.10) < abs(result["naive difference"]["ate"] - 0.10) / 2, estimator


def test_aipw_equals_the_outcome_model_difference_when_residuals_are_zero() -> None:
    t = np.array([1, 0, 1, 0])
    nuisance = Nuisance(np.full(4, 0.5), np.array([0.3, 0.3, 0.3, 0.3]), np.array([0.1, 0.1, 0.1, 0.1]))
    y = np.array([0.3, 0.1, 0.3, 0.1])
    assert np.allclose(aipw_scores(t, y, nuisance), 0.2)


def test_standardized_difference_is_zero_for_identical_groups_and_signed_otherwise() -> None:
    values = np.array([1.0, 2.0, 1.0, 2.0])
    assert standardized_difference(values, np.array([1, 1, 0, 0])) == pytest.approx(0.0)
    assert standardized_difference(np.array([2.0, 3.0, 0.0, 1.0]), np.array([1, 1, 0, 0])) > 0


def test_causal_frame_treats_loans_of_at_least_a_quarter_of_income(small_train: tuple[pd.DataFrame, pd.Series]) -> None:
    X = small_train[0]
    W, t = causal_frame(X)
    expected = (X["loan_amnt"] / X["revenue"] >= 0.25).to_numpy()
    assert (t.astype(bool) == expected).all()
    assert "loan_amnt" not in W.columns  # the treatment's own variable is not a confounder


def test_support_by_income_counts_treated_loans_per_band() -> None:
    from loan_default.causal import support_by_income

    W = pd.DataFrame({"log_income": np.log([30_000, 30_000, 200_000, 200_000])})
    table = support_by_income(W, np.array([1, 0, 0, 0])).set_index("income")
    assert table.loc["$0k-40k", "share"] == 0.5
    assert table.loc["$140k and over", "treated"] == 0  # no comparison possible there


def test_effective_sample_size_falls_as_weights_concentrate() -> None:
    from loan_default.causal import effective_sample_size

    assert effective_sample_size(np.ones(100)) == pytest.approx(100)
    assert effective_sample_size(np.r_[np.ones(99), 100.0]) < 10


def test_sensitivity_reports_each_estimate_with_its_population() -> None:
    from loan_default.causal import sensitivity

    W, t, y = confounded(n=3000)
    table = sensitivity(t, y, cross_fit(W, t, y, folds=3, max_iter=60))
    assert len(table) == 4
    assert table["n"].iloc[2] <= len(y)  # trimming keeps only the overlap region
    assert table["ate"].between(0, 0.3).all()


def test_overlap_weights_give_each_group_the_other_groups_propensity() -> None:
    from loan_default.causal import overlap_weights

    t = np.array([1, 0])
    e = np.array([0.8, 0.3])
    assert overlap_weights(t, e).tolist() == pytest.approx([0.2, 0.3])
