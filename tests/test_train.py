import json
from pathlib import Path

import joblib
import pandas as pd
import pytest

from loan_default.train import TUNED_PARAMS, TrainConfig, parse_args, train

DATA = Path(__file__).resolve().parents[1] / "data"
FAST_PARAMS: dict[str, float | int] = {"max_iter": 20, "max_leaf_nodes": 7}


@pytest.fixture(scope="module")
def config(tmp_path_factory: pytest.TempPathFactory) -> TrainConfig:
    """A small sample of the real data and a small model, so training takes seconds."""
    tmp = tmp_path_factory.mktemp("run")
    pd.read_csv(DATA / "train.csv", nrows=4000, low_memory=False).to_csv(tmp / "train.csv", index=False)
    pd.read_csv(DATA / "test.csv", nrows=1000, low_memory=False).to_csv(tmp / "test.csv", index=False)
    return TrainConfig(
        train_path=tmp / "train.csv", test_path=tmp / "test.csv", out_dir=tmp / "models", model_params=FAST_PARAMS
    )


@pytest.fixture(scope="module")
def metadata(config: TrainConfig) -> dict[str, object]:
    return train(config)


def test_writes_model_and_matching_metadata(config: TrainConfig, metadata: dict[str, object]) -> None:
    assert (config.out_dir / "model.joblib").is_file()
    assert json.loads((config.out_dir / "metadata.json").read_text()) == metadata
    assert metadata["n_train"] == 4000 and metadata["n_test"] == 1000


@pytest.mark.usefixtures("metadata")
def test_saved_pipeline_scores_raw_applications(config: TrainConfig) -> None:
    model = joblib.load(config.out_dir / "model.joblib")
    raw = pd.read_csv(config.test_path, low_memory=False).drop(columns=["Default"]).head(50)
    scores = model.predict_proba(raw)[:, 1]
    assert len(scores) == 50
    assert ((scores >= 0) & (scores <= 1)).all()


def test_cutoff_is_a_probability_recorded_with_its_approval_rate(metadata: dict[str, object]) -> None:
    assert metadata["approval_rate"] == 0.8
    cutoff = metadata["approval_cutoff"]
    assert isinstance(cutoff, float) and 0 < cutoff < 1


def test_training_is_reproducible(config: TrainConfig, metadata: dict[str, object], tmp_path: Path) -> None:
    again = train(
        TrainConfig(
            train_path=config.train_path, test_path=config.test_path, out_dir=tmp_path, model_params=FAST_PARAMS
        )
    )
    assert again == metadata


def test_overlapping_ids_are_rejected(config: TrainConfig, tmp_path: Path) -> None:
    leaky = TrainConfig(train_path=config.train_path, test_path=config.train_path, out_dir=tmp_path)
    with pytest.raises(ValueError, match="share loan ids"):
        train(leaky)


def test_command_line_overrides_defaults() -> None:
    config, args = parse_args(["--out-dir", "elsewhere", "--approval-rate", "0.5", "--max-iter", "50"])
    assert config.out_dir == Path("elsewhere")
    assert config.approval_rate == 0.5
    assert config.train_path == Path("data/train.csv")
    assert config.model_params["max_iter"] == 50
    assert config.model_params["learning_rate"] == TUNED_PARAMS["learning_rate"]
    assert args.track is False
