from collections.abc import Iterator
from pathlib import Path

import pandas as pd
import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from loan_default import service  # noqa: E402
from loan_default.train import TrainConfig, train  # noqa: E402

DATA = Path(__file__).resolve().parents[1] / "data"
APPLICATION = {
    "revenue": 60000,
    "loan_amnt": 12000,
    "fico_n": 690,
    "dti_n": 18.0,
    "emp_length": "5 years",
    "purpose": "debt_consolidation",
    "home_ownership_n": "RENT",
    "addr_state": "NY",
}


@pytest.fixture(scope="module")
def client(tmp_path_factory: pytest.TempPathFactory) -> Iterator[TestClient]:
    """The service backed by a small model trained on a sample of the real data."""
    tmp = tmp_path_factory.mktemp("service")
    pd.read_csv(DATA / "train.csv", nrows=4000, low_memory=False).to_csv(tmp / "train.csv", index=False)
    pd.read_csv(DATA / "test.csv", nrows=1000, low_memory=False).to_csv(tmp / "test.csv", index=False)
    train(
        TrainConfig(
            train_path=tmp / "train.csv",
            test_path=tmp / "test.csv",
            out_dir=tmp / "models",
            model_params={"max_iter": 20, "max_leaf_nodes": 7},
        )
    )
    patch = pytest.MonkeyPatch()
    patch.setenv("MODEL_DIR", str(tmp / "models"))
    service.artifacts.cache_clear()
    yield TestClient(service.app)
    patch.undo()
    service.artifacts.cache_clear()


def test_health_reports_the_model(client: TestClient) -> None:
    assert client.get("/health").json() == {"status": "ok", "model": "HistGradientBoostingClassifier"}


def test_metadata_exposes_the_cutoff_and_metrics(client: TestClient) -> None:
    body = client.get("/metadata").json()
    assert 0 < body["approval_cutoff"] < 1
    assert "roc_auc" in body["metrics"]["test"]


def test_prediction_is_a_probability_with_a_consistent_decision(client: TestClient) -> None:
    body = client.post("/predict", json=APPLICATION).json()
    assert 0 <= body["default_probability"] <= 1
    expected = "approve" if body["default_probability"] <= body["approval_cutoff"] else "reject"
    assert body["decision"] == expected


def test_riskier_application_scores_higher(client: TestClient) -> None:
    safe = client.post("/predict", json={**APPLICATION, "fico_n": 820, "loan_amnt": 2000}).json()
    risky = client.post("/predict", json={**APPLICATION, "fico_n": 662, "loan_amnt": 35000}).json()
    assert risky["default_probability"] > safe["default_probability"]


def test_optional_fields_can_be_omitted(client: TestClient) -> None:
    minimal = {k: v for k, v in APPLICATION.items() if k not in {"dti_n", "emp_length"}}
    assert client.post("/predict", json=minimal).status_code == 200


def test_unseen_category_is_scored_not_rejected(client: TestClient) -> None:
    assert client.post("/predict", json={**APPLICATION, "purpose": "boat"}).status_code == 200


@pytest.mark.parametrize(
    "change",
    [
        {"revenue": 0},  # income must be positive
        {"loan_amnt": -5},  # amount must be positive
        {"fico_n": 900},  # outside the FICO range
        {"dti_n": -1},  # negative ratio
        {"emp_length": "12 years"},  # not one of the known values
        {"addr_state": "New York"},  # not a two-letter code
        {"revenue": "a lot"},  # not a number
    ],
)
def test_invalid_input_is_rejected_with_422(client: TestClient, change: dict[str, object]) -> None:
    response = client.post("/predict", json={**APPLICATION, **change})
    assert response.status_code == 422


def test_missing_required_field_is_rejected(client: TestClient) -> None:
    incomplete = {k: v for k, v in APPLICATION.items() if k != "fico_n"}
    assert client.post("/predict", json=incomplete).status_code == 422
