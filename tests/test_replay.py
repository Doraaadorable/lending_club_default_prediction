import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from loan_default.replay import alert_month, recalibrated, run
from loan_default.train import TrainConfig

DATA = Path(__file__).resolve().parents[1] / "data"


@pytest.fixture
def understated_year() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Scores of 0.2 for loans that default 30% of the time, 1,000 per month."""
    rng = np.random.default_rng(0)
    months = np.repeat(np.arange(1, 13), 1000)
    defaults = (rng.random(len(months)) < 0.3).astype(int)
    return np.full(len(months), 0.2), defaults, months


def test_no_correction_before_outcomes_are_known(understated_year: tuple[np.ndarray, np.ndarray, np.ndarray]) -> None:
    scores, defaults, months = understated_year
    corrected = recalibrated(scores, defaults, months, lag=3, refits=tuple(range(1, 13)))
    assert (corrected[months <= 3] == 0.2).all()
    assert corrected[months == 4].mean() == pytest.approx(defaults[months == 1].mean(), abs=1e-6)


def test_a_year_long_delay_leaves_the_year_uncorrected(
    understated_year: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> None:
    scores, defaults, months = understated_year
    assert (recalibrated(scores, defaults, months, lag=12, refits=tuple(range(1, 13))) == scores).all()


def test_quarterly_refits_only_change_at_quarter_starts(
    understated_year: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> None:
    scores, defaults, months = understated_year
    corrected = recalibrated(scores, defaults, months, lag=3, refits=(1, 4, 7, 10))
    assert corrected[months == 4][0] == corrected[months == 6][0]
    assert corrected[months == 6][0] != corrected[months == 7][0]


def test_alert_month_adds_the_delay() -> None:
    assert alert_month("2016-03", 6) == "2016-09"
    assert alert_month("2016-11", 3) == "2017-02"
    assert alert_month(None, 3) == "never"


def test_replay_writes_a_report_and_a_service_style_log(tmp_path: Path) -> None:
    pd.read_csv(DATA / "train.csv", nrows=4000, low_memory=False).to_csv(tmp_path / "train.csv", index=False)
    pd.read_csv(DATA / "test.csv", low_memory=False).sample(2400, random_state=0).to_csv(
        tmp_path / "test.csv", index=False
    )
    config = TrainConfig(
        train_path=tmp_path / "train.csv",
        test_path=tmp_path / "test.csv",
        model_params={"max_iter": 20, "max_leaf_nodes": 7},
        require_beats_baseline=False,
    )
    report = run(config, tmp_path / "reports", tmp_path / "logs")

    assert "## Month by month" in report and "2016-12" in report
    first = json.loads((tmp_path / "logs" / "replay_2016.jsonl").read_text().splitlines()[0])
    assert {"timestamp", "application_id", "fico_n", "default_probability", "decision"} <= set(first)
    outcomes = pd.read_csv(tmp_path / "logs" / "replay_2016_outcomes.csv")
    assert list(outcomes.columns) == ["application_id", "default"] and len(outcomes) == 2400
