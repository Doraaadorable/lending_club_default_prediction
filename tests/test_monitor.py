from pathlib import Path

import pandas as pd
import pytest

from loan_default.monitor import calibration_by_month, join_outcomes, main, months_over, prediction_months


def test_months_come_from_timestamps_with_or_without_time_zone() -> None:
    stamps = pd.Series(["2026-03-31T23:30:00+00:00", "2026-04-01", "2026-04-15T08:00:00-04:00"])
    assert prediction_months(stamps).tolist() == ["2026-03", "2026-04", "2026-04"]


def test_only_predictions_with_a_known_outcome_are_joined() -> None:
    log = pd.DataFrame(
        {
            "timestamp": ["2026-01-05", "2026-01-06", "2026-02-01", "2026-02-02"],
            "application_id": ["1", "2", None, "4"],
            "default_probability": [0.1, 0.3, 0.2, 0.4],
        }
    )
    outcomes = pd.DataFrame({"application_id": [1, 2, 3], "default": [0, 1, 1]})
    joined = join_outcomes(log, outcomes)
    assert joined["month"].tolist() == ["2026-01", "2026-01"]
    assert joined["default"].tolist() == [0, 1]


def test_gap_is_actual_minus_predicted_and_flags_use_its_size() -> None:
    joined = pd.DataFrame(
        {
            "month": ["2026-01"] * 2 + ["2026-02"] * 2,
            "default_probability": [0.2, 0.2, 0.5, 0.5],
            "default": [0, 1, 0, 0],
        }
    )
    table = calibration_by_month(joined)
    assert table["gap"].round(6).tolist() == [0.3, -0.5]
    assert months_over(table, 0.4) == ["2026-02"]
    assert months_over(table, 0.6) == []


def write_files(tmp_path: Path, defaults: list[int]) -> tuple[Path, Path]:
    log = tmp_path / "predictions.jsonl"
    pd.DataFrame(
        {"timestamp": ["2026-01-05"] * 4, "application_id": ["a", "b", "c", "d"], "default_probability": [0.25] * 4}
    ).to_json(log, orient="records", lines=True)
    outcomes = tmp_path / "outcomes.csv"
    pd.DataFrame({"application_id": ["a", "b", "c", "d"], "default": defaults}).to_csv(outcomes, index=False)
    return log, outcomes


def test_command_line_passes_when_calibrated(tmp_path: Path) -> None:
    log, outcomes = write_files(tmp_path, [1, 0, 0, 0])
    main(["--log", str(log), "--outcomes", str(outcomes), "--out-dir", str(tmp_path / "out")])
    assert "2026-01" in (tmp_path / "out" / "outcome_monitor.md").read_text()


def test_command_line_fails_when_a_month_is_off(tmp_path: Path) -> None:
    log, outcomes = write_files(tmp_path, [1, 1, 1, 0])
    with pytest.raises(SystemExit) as exit_info:
        main(["--log", str(log), "--outcomes", str(outcomes), "--out-dir", str(tmp_path / "out")])
    assert exit_info.value.code == 1


def test_an_outcome_listed_twice_is_an_error() -> None:
    log = pd.DataFrame({"timestamp": ["2026-01-05"], "application_id": ["1"], "default_probability": [0.1]})
    outcomes = pd.DataFrame({"application_id": ["1", "1"], "default": [0, 1]})
    with pytest.raises(ValueError, match="more than once"):
        join_outcomes(log, outcomes)


def test_a_rescored_application_counts_once_at_its_first_score() -> None:
    log = pd.DataFrame(
        {
            "timestamp": ["2026-01-20", "2026-01-05", "2026-01-07"],
            "application_id": ["1", "1", "2"],
            "default_probability": [0.4, 0.1, 0.2],
        }
    )
    outcomes = pd.DataFrame({"application_id": ["1", "2"], "default": [1, 0]})
    joined = join_outcomes(log, outcomes)
    assert len(joined) == 2
    assert sorted(joined["default_probability"]) == [0.1, 0.2]


def test_no_matched_outcome_fails_instead_of_passing(tmp_path: Path) -> None:
    log = tmp_path / "log.jsonl"
    log.write_text('{"timestamp": "2026-01-05", "application_id": "1", "default_probability": 0.1}\n')
    outcomes = tmp_path / "outcomes.csv"
    outcomes.write_text("application_id,default\n99,1\n")
    with pytest.raises(SystemExit) as stop:
        main(["--log", str(log), "--outcomes", str(outcomes), "--out-dir", str(tmp_path)])
    assert stop.value.code == 2


def test_a_prediction_without_a_time_never_replaces_a_valid_one() -> None:
    log = pd.DataFrame(
        {"timestamp": ["2026-01-05T10:00:00", None], "application_id": ["1", "1"], "default_probability": [0.1, 0.2]}
    )
    joined = join_outcomes(log, pd.DataFrame({"application_id": ["1"], "default": [1]}))
    assert joined["default_probability"].tolist() == [0.1]
    assert len(calibration_by_month(joined)) == 1


def test_only_unusable_predictions_fail_instead_of_passing(tmp_path: Path) -> None:
    log = tmp_path / "log.jsonl"
    log.write_text('{"timestamp": null, "application_id": "1", "default_probability": 0.1}\n')
    outcomes = tmp_path / "outcomes.csv"
    outcomes.write_text("application_id,default\n1,1\n")
    with pytest.raises(SystemExit) as stop:
        main(["--log", str(log), "--outcomes", str(outcomes), "--out-dir", str(tmp_path)])
    assert stop.value.code == 2


def test_an_outcome_that_is_not_binary_is_an_error() -> None:
    log = pd.DataFrame({"timestamp": ["2026-01-05"], "application_id": ["1"], "default_probability": [0.1]})
    with pytest.raises(ValueError, match="0 \\(repaid\\) or 1"):
        join_outcomes(log, pd.DataFrame({"application_id": ["1"], "default": [2]}))


@pytest.mark.parametrize("content", ["", "\n", '{"something": 1}\n'])
def test_an_empty_or_malformed_log_exits_2(tmp_path: Path, content: str) -> None:
    log = tmp_path / "log.jsonl"
    log.write_text(content)
    outcomes = tmp_path / "outcomes.csv"
    outcomes.write_text("application_id,default\n1,1\n")
    with pytest.raises(SystemExit) as stop:
        main(["--log", str(log), "--outcomes", str(outcomes), "--out-dir", str(tmp_path)])
    assert stop.value.code == 2
