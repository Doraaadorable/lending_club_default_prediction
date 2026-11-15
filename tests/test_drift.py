import numpy as np
import pandas as pd
import pytest

from loan_default.drift import (
    categorical_psi,
    decompose_gap,
    default_rate_by_year,
    numeric_psi,
    psi_label,
    risk_band_comparison,
)


def test_identical_distributions_have_zero_psi() -> None:
    values = pd.Series(np.random.default_rng(0).normal(size=5000))
    assert numeric_psi(values, values) == pytest.approx(0.0, abs=1e-9)


def test_same_distribution_resampled_is_stable_and_shifted_is_not() -> None:
    rng = np.random.default_rng(0)
    reference = pd.Series(rng.normal(size=20_000))
    assert psi_label(numeric_psi(reference, pd.Series(rng.normal(size=20_000)))) == "stable"
    assert psi_label(numeric_psi(reference, pd.Series(rng.normal(loc=1.0, size=20_000)))) == "large"


def test_missing_values_are_ignored_in_numeric_psi() -> None:
    values = pd.Series(np.random.default_rng(0).normal(size=5000))
    with_gaps = pd.concat([values, pd.Series([np.nan] * 500)], ignore_index=True)
    assert numeric_psi(values, with_gaps) == pytest.approx(0.0, abs=1e-9)


def test_categorical_psi_grows_with_the_change_in_shares() -> None:
    reference = pd.Series(["a"] * 70 + ["b"] * 30)
    small = pd.Series(["a"] * 68 + ["b"] * 32)
    big = pd.Series(["a"] * 30 + ["b"] * 70)
    assert 0 < categorical_psi(reference, small) < categorical_psi(reference, big)


def test_category_absent_from_one_period_does_not_break_psi() -> None:
    assert np.isfinite(categorical_psi(pd.Series(["a", "b"]), pd.Series(["a", "c"])))


@pytest.mark.parametrize(("psi", "label"), [(0.02, "stable"), (0.10, "moderate"), (0.24, "moderate"), (0.25, "large")])
def test_psi_reading_thresholds(psi: float, label: str) -> None:
    assert psi_label(psi) == label


def test_default_rate_by_year() -> None:
    table = default_rate_by_year(
        pd.Series(["2014-01-01", "2014-06-01", "2015-03-01", "2015-04-01"]), pd.Series([0, 1, 1, 1])
    )
    assert table["year"].tolist() == [2014, 2015]
    assert table["loans"].tolist() == [2, 2]
    assert table["default_rate"].tolist() == [0.5, 1.0]


def test_gap_from_mix_alone_is_attributed_to_mix() -> None:
    """Same default rate inside each band, but the current period holds more high-risk loans."""
    ref_scores = np.array([0.1] * 500 + [0.9] * 500)
    ref_y = np.array([0] * 450 + [1] * 50 + [0] * 250 + [1] * 250)  # 10% and 50%
    cur_scores = np.array([0.1] * 200 + [0.9] * 800)
    cur_y = np.array([0] * 180 + [1] * 20 + [0] * 400 + [1] * 400)  # still 10% and 50%
    bands = risk_band_comparison(ref_scores, ref_y, cur_scores, cur_y, bands=2)
    gap = decompose_gap(bands, float(ref_y.mean()), float(cur_y.mean()))
    assert gap.from_mix == pytest.approx(0.42 - 0.30)
    assert gap.within_bands == pytest.approx(0.0)


def test_gap_within_bands_is_attributed_to_bands() -> None:
    """Same mix of scores, but every band defaults more often in the current period."""
    scores = np.array([0.1] * 500 + [0.9] * 500)
    ref_y = np.array([0] * 450 + [1] * 50 + [0] * 250 + [1] * 250)  # 10% and 50%
    cur_y = np.array([0] * 400 + [1] * 100 + [0] * 200 + [1] * 300)  # 20% and 60%
    bands = risk_band_comparison(scores, ref_y, scores, cur_y, bands=2)
    gap = decompose_gap(bands, float(ref_y.mean()), float(cur_y.mean()))
    assert gap.from_mix == pytest.approx(0.0)
    assert gap.within_bands == pytest.approx(0.10)
    assert bands["change"].tolist() == pytest.approx([0.10, 0.10])
