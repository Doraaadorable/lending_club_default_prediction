import optuna
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

from loan_default.hpo import (
    FAMILIES,
    automl_model,
    boosting_params,
    cv_objective,
    run_study,
    suggest_boosting,
    trials_table,
)
from loan_default.train import build_model


def test_boosting_space_matches_the_randomized_search() -> None:
    from loan_default.search import PARAM_DISTRIBUTIONS

    trial = optuna.trial.FixedTrial(
        {
            "learning_rate": 0.05,
            "max_iter": 200,
            "max_leaf_nodes": 15,
            "min_samples_leaf": 100,
            "l2_regularization": 0.1,
            "max_features": 0.7,
        }
    )
    assert sorted(f"model__{k}" for k in suggest_boosting(trial)) == sorted(PARAM_DISTRIBUTIONS)


@pytest.mark.parametrize("family", FAMILIES)
def test_automl_builds_a_model_for_every_family(family: str) -> None:
    study = optuna.create_study(sampler=optuna.samplers.TPESampler(seed=0))
    study.enqueue_trial({"family": family})
    trial = study.ask()
    assert isinstance(automl_model(trial), Pipeline)


def test_short_study_runs_and_tracks_the_best_score(small_train: tuple[pd.DataFrame, pd.Series]) -> None:
    X, y = small_train

    def tiny(trial: optuna.trial.BaseTrial) -> Pipeline:
        return build_model({"max_iter": trial.suggest_int("max_iter", 10, 30)}, 0)

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study, seconds = run_study(cv_objective(tiny, X, y, folds=2, prune=True), n_trials=3, prune=True)
    table = trials_table(study)
    assert len(table) == 3 and seconds > 0
    assert table["best_so_far"].is_monotonic_increasing
    assert 0.5 < study.best_value < 1


def test_boosting_params_strip_the_family_prefix() -> None:
    params = {
        f"hgb_{k}": 1
        for k in (
            "learning_rate",
            "max_iter",
            "max_leaf_nodes",
            "min_samples_leaf",
            "l2_regularization",
            "max_features",
        )
    }
    params["family"] = "hist_gradient_boosting"  # type: ignore[assignment]
    assert set(boosting_params(params, "hgb_")) == {
        "learning_rate",
        "max_iter",
        "max_leaf_nodes",
        "min_samples_leaf",
        "l2_regularization",
        "max_features",
    }


def test_a_trial_is_never_pruned_after_its_last_fold(small_train: tuple[pd.DataFrame, pd.Series]) -> None:
    X, y = small_train

    def tiny(trial: optuna.trial.BaseTrial) -> Pipeline:
        return build_model({"max_iter": trial.suggest_int("max_iter", 5, 40)}, 0)

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    pruner = optuna.pruners.ThresholdPruner(lower=0.99)  # would prune every trial at every chance
    study = optuna.create_study(direction="maximize", pruner=pruner)
    study.optimize(cv_objective(tiny, X, y, folds=3, prune=True), n_trials=2)
    for trial in study.trials:
        assert len(trial.intermediate_values) < 3  # stopped before the last fold, not after it
