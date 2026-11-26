from sklearn.dummy import DummyClassifier
from sklearn.pipeline import Pipeline

from loan_default.experiments import Splits, compare, comparison_table, score_metrics
from loan_default.train import build_model


def test_compare_scores_every_candidate_on_cv_and_test(small_splits: Splits) -> None:
    candidates = {
        "boosting": build_model({"max_iter": 20}, 0),
        "constant": Pipeline([("model", DummyClassifier(strategy="prior"))]),
    }
    results = compare(candidates, small_splits, folds=3, n_jobs=1)
    assert results["candidate"].tolist() == ["boosting", "constant"]
    assert results.loc[1, "test_roc_auc"] == 0.5  # a constant score cannot rank
    assert results.loc[0, "test_roc_auc"] > 0.55
    lines = comparison_table(results, baseline="constant")
    assert lines[0].endswith("Test ROC-AUC vs baseline |")
    assert lines[-1].endswith("+0.0000 |")


def test_score_metrics_default_rate_uses_the_safest_80_percent() -> None:
    import numpy as np

    y = np.array([0] * 8 + [1] * 2)
    scores = np.linspace(0, 1, 10)  # the two defaults have the highest scores
    metrics = score_metrics(y, scores)
    assert metrics["roc_auc"] == 1.0
    assert metrics["default_rate_at_80"] == 0.0
