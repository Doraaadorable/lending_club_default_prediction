import numpy as np
import pandas as pd
import pytest

from loan_default.fairness import (
    STATE_TO_REGION,
    apply_cutoffs,
    disparities,
    equal_opportunity_cutoffs,
    group_metrics,
    income_band,
)


def test_income_bands_cover_every_income() -> None:
    bands = income_band(pd.Series([10_000, 40_000, 74_999, 75_000, 500_000]))
    assert bands.tolist() == ["under $40k", "$40k-75k", "$40k-75k", "$75k-120k", "$120k and over"]


def test_every_state_and_dc_have_a_region() -> None:
    assert len(STATE_TO_REGION) == 51


def make_groups(n: int = 2000, seed: int = 0) -> tuple[pd.Series, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    groups = pd.Series(np.where(rng.random(n) < 0.5, "a", "b"))
    y = (rng.random(n) < 0.2).astype(int)
    scores = np.clip(0.15 + 0.2 * y + 0.1 * (groups == "b") + rng.normal(0, 0.05, n), 0, 1)
    return groups, y, scores


def test_group_metrics_and_disparities_on_a_biased_score() -> None:
    groups, y, scores = make_groups()
    approved = scores <= np.quantile(scores, 0.8)
    table = group_metrics(groups, y, scores, approved).set_index("group")
    assert table.loc["a", "approval_rate"] > table.loc["b", "approval_rate"]
    assert table.loc["a", "adverse_impact_ratio"] == pytest.approx(1.0)
    gaps = disparities(table.reset_index())
    assert gaps["demographic_parity_difference"] > 0
    assert gaps["equalized_odds_difference"] >= gaps["equal_opportunity_difference"]


def test_equal_opportunity_cutoffs_equalize_approval_of_repaid_borrowers() -> None:
    groups, y, scores = make_groups()
    cutoffs = equal_opportunity_cutoffs(groups, y, scores, 0.8)
    approved = apply_cutoffs(groups, scores, cutoffs)
    table = group_metrics(groups, y, scores, approved)
    assert disparities(table)["equal_opportunity_difference"] < 0.01
