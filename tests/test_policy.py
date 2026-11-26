import numpy as np
import pytest

from loan_default.policy import approve_safest, cutoff_at, policy_results


def test_approves_exactly_the_requested_share_of_lowest_scores() -> None:
    scores = np.array([0.9, 0.1, 0.5, 0.3, 0.7])
    approved = approve_safest(scores, 0.6)
    assert approved.tolist() == [False, True, True, True, False]


def test_ties_are_broken_reproducibly_and_still_hit_the_share() -> None:
    scores = np.zeros(1000)
    first, second = approve_safest(scores, 0.8), approve_safest(scores, 0.8)
    assert first.sum() == 800
    assert (first == second).all()
    assert (approve_safest(scores, 0.8, seed=1) != first).any()


@pytest.mark.parametrize("rate", [-0.1, 1.5])
def test_rate_outside_zero_to_one_is_rejected(rate: float) -> None:
    with pytest.raises(ValueError, match="rate must be between 0 and 1"):
        approve_safest([0.1, 0.2], rate)


def test_policy_results_reports_default_rates_on_each_side() -> None:
    y = np.array([0, 0, 1, 1])
    approved = np.array([True, True, True, False])
    result = policy_results(y, approved)
    assert result["approval_rate"] == 0.75
    assert result["approved"] == 3
    assert result["default_rate_approved"] == pytest.approx(1 / 3)
    assert result["default_rate_rejected"] == 1.0


def test_approving_everyone_leaves_no_rejected_rate() -> None:
    result = policy_results([0, 1], np.array([True, True]))
    assert np.isnan(result["default_rate_rejected"])


def test_cutoff_is_the_highest_approved_score() -> None:
    scores = np.array([0.9, 0.1, 0.5, 0.3, 0.7])
    assert cutoff_at(scores, 0.6) == 0.5


def test_a_cutoff_that_approves_nothing_is_an_error() -> None:
    with pytest.raises(ValueError, match="approves no loans"):
        cutoff_at(np.array([0.1, 0.2, 0.3]), 0.0)
