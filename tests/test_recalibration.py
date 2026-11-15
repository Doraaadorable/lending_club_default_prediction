from collections.abc import Callable

import numpy as np
import pytest

from loan_default.recalibration import (
    Calibrator,
    calibration_error,
    compare,
    fit_intercept_shift,
    fit_isotonic,
    fit_platt,
    summarize,
)

Fit = Callable[[np.ndarray, np.ndarray], Calibrator]


@pytest.fixture
def understated() -> tuple[np.ndarray, np.ndarray]:
    """Scores that rank correctly but sit below the true default probability (true log-odds = score log-odds + 0.5)."""
    rng = np.random.default_rng(0)
    scores = rng.uniform(0.05, 0.5, size=40_000)
    true_probability = 1 / (1 + np.exp(-(np.log(scores / (1 - scores)) + 0.5)))
    defaults = (rng.random(40_000) < true_probability).astype(float)
    return scores, defaults


def test_perfectly_calibrated_bins_have_no_gap() -> None:
    defaults = np.array([0, 1] * 500)
    mean_gap, max_gap = calibration_error(defaults, np.full(1000, 0.5), bins=5)
    assert mean_gap == pytest.approx(0.0) and max_gap == pytest.approx(0.0)


def test_intercept_shift_matches_the_observed_rate(understated: tuple[np.ndarray, np.ndarray]) -> None:
    scores, defaults = understated
    corrected = fit_intercept_shift(scores, defaults)(scores)
    assert corrected.mean() == pytest.approx(defaults.mean(), abs=1e-6)
    assert scores.mean() < corrected.mean()


@pytest.mark.parametrize("fit", [fit_intercept_shift, fit_platt, fit_isotonic])
def test_each_correction_reduces_calibration_error_on_new_data(
    understated: tuple[np.ndarray, np.ndarray], fit: Fit
) -> None:
    scores, defaults = understated
    half = len(scores) // 2
    calibrate = fit(scores[:half], defaults[:half])
    before = summarize(defaults[half:], scores[half:])
    after = summarize(defaults[half:], calibrate(scores[half:]))
    assert after["mean_bin_gap"] < before["mean_bin_gap"] / 3
    assert after["brier"] < before["brier"]


@pytest.mark.parametrize("fit", [fit_intercept_shift, fit_platt])
def test_log_odds_corrections_keep_the_ranking(understated: tuple[np.ndarray, np.ndarray], fit: Fit) -> None:
    scores, defaults = understated
    corrected = fit(scores, defaults)(scores)
    assert (np.argsort(scores, kind="stable") == np.argsort(corrected, kind="stable")).all()


def test_compare_lists_every_method(understated: tuple[np.ndarray, np.ndarray]) -> None:
    scores, defaults = understated
    half = len(scores) // 2
    table = compare(scores[:half], defaults[:half], scores[half:], defaults[half:])
    assert table["method"].tolist() == ["No recalibration", "Intercept shift", "Platt scaling", "Isotonic"]
    assert table.loc[0, "mean_predicted"] == pytest.approx(scores[half:].mean())
