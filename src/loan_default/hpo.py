"""Bayesian hyperparameter optimization and AutoML with Optuna.

    python -m loan_default.hpo                 # writes reports/hpo_report.md, trial tables and best params
    python -m loan_default.hpo --track         # also logs every trial to MLflow as a nested run

1. Tree-structured Parzen estimator (TPE) for the gradient boosting model, over the same ranges,
   budget (40 configurations) and 5-fold CV as the randomized search in `loan_default.search`.
   Each trial reports its running CV score after every fold, and a median pruner stops trials
   that are clearly behind after the first folds, so the budget goes to promising regions.
2. AutoML as a combined algorithm selection and hyperparameter optimization (CASH) problem:
   one Optuna study chooses the model family (gradient boosting, random forest, extra trees,
   logistic regression, or the entity-embedding network) and that family's settings together.
   It runs on a 50,000-loan sample with 3-fold CV to keep the cost down, then the winner is
   refitted on all 2007-2015 loans and scored on 2016.
"""

import argparse
import json
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import optuna
import pandas as pd
from sklearn.base import BaseEstimator, clone
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline

from loan_default.embedding import EntityEmbeddingNet, network_pipeline
from loan_default.experiments import Splits, load_splits, score_metrics, write_report
from loan_default.preprocessing import build_preprocessor
from loan_default.tracking import DEFAULT_TRACKING_URI
from loan_default.train import TUNED_PARAMS, build_model

SEED = 42
N_TRIALS = 40
FOLDS = 5
AUTOML_TRIALS = 40
AUTOML_SAMPLE = 50_000
AUTOML_FOLDS = 3
FAMILIES = ("hist_gradient_boosting", "random_forest", "extra_trees", "logistic_regression", "embedding_network")


def suggest_boosting(trial: optuna.trial.BaseTrial, prefix: str = "") -> dict[str, float | int]:
    """Same ranges as the randomized search (`loan_default.search.PARAM_DISTRIBUTIONS`)."""
    return {
        "learning_rate": trial.suggest_float(f"{prefix}learning_rate", 0.01, 0.2, log=True),
        "max_iter": trial.suggest_int(f"{prefix}max_iter", 100, 800),
        "max_leaf_nodes": trial.suggest_int(f"{prefix}max_leaf_nodes", 7, 63),
        "min_samples_leaf": trial.suggest_int(f"{prefix}min_samples_leaf", 20, 500),
        "l2_regularization": trial.suggest_float(f"{prefix}l2_regularization", 1e-3, 10, log=True),
        "max_features": trial.suggest_float(f"{prefix}max_features", 0.5, 1.0),
    }


def cv_objective(
    make_model: Callable[[optuna.Trial], BaseEstimator],
    X: pd.DataFrame,
    y: pd.Series,
    folds: int,
    prune: bool,
) -> Callable[[optuna.Trial], float]:
    """Objective: mean ROC-AUC over stratified folds, reported after each fold so a pruner can stop early."""
    splits = list(StratifiedKFold(folds, shuffle=True, random_state=SEED).split(X, y))

    def objective(trial: optuna.Trial) -> float:
        model = make_model(trial)
        scores = []
        for step, (fit_idx, held_idx) in enumerate(splits):
            fitted = clone(model).fit(X.iloc[fit_idx], y.iloc[fit_idx])
            scores.append(roc_auc_score(y.iloc[held_idx], fitted.predict_proba(X.iloc[held_idx])[:, 1]))
            trial.report(float(np.mean(scores)), step)
            if prune and step < len(splits) - 1 and trial.should_prune():  # a finished trial is never pruned
                raise optuna.TrialPruned()
        return float(np.mean(scores))

    return objective


def boosting_model(trial: optuna.trial.BaseTrial) -> Pipeline:
    """The production pipeline with TPE-suggested boosting settings."""
    return build_model(suggest_boosting(trial), SEED)


def automl_model(trial: optuna.trial.BaseTrial) -> BaseEstimator:
    """Choose a model family, then that family's settings: the CASH search space."""
    family = trial.suggest_categorical("family", FAMILIES)
    if family == "hist_gradient_boosting":
        return build_model(suggest_boosting(trial, "hgb_"), SEED)
    if family == "embedding_network":
        net = network_pipeline()
        net.set_params(
            net=EntityEmbeddingNet(
                hidden=trial.suggest_int("net_hidden", 8, 64, log=True),
                max_dim=trial.suggest_int("net_max_dim", 2, 12),
                learning_rate=trial.suggest_float("net_learning_rate", 3e-4, 1e-2, log=True),
                weight_decay=trial.suggest_float("net_weight_decay", 1e-7, 1e-3, log=True),
            )
        )
        return net
    estimator: BaseEstimator
    if family == "logistic_regression":
        estimator = LogisticRegression(C=trial.suggest_float("lr_C", 1e-3, 10, log=True), max_iter=2000)
    else:
        forest = RandomForestClassifier if family == "random_forest" else ExtraTreesClassifier
        prefix = "rf_" if family == "random_forest" else "et_"
        estimator = forest(
            n_estimators=trial.suggest_int(f"{prefix}n_estimators", 100, 400),
            max_depth=trial.suggest_int(f"{prefix}max_depth", 4, 20),
            min_samples_leaf=trial.suggest_int(f"{prefix}min_samples_leaf", 10, 300, log=True),
            max_features=trial.suggest_float(f"{prefix}max_features", 0.1, 0.8),
            n_jobs=-1,
            random_state=SEED,
        )
    return Pipeline([("preprocess", build_preprocessor()), ("model", estimator)])


def run_study(
    objective: Callable[[optuna.Trial], float],
    n_trials: int,
    prune: bool,
    sampler: optuna.samplers.BaseSampler | None = None,
) -> tuple[optuna.Study, float]:
    """Maximize the objective (seeded TPE unless another sampler is given); return the study and the seconds."""
    pruner: optuna.pruners.BasePruner = (
        optuna.pruners.MedianPruner(n_startup_trials=8, n_warmup_steps=1) if prune else optuna.pruners.NopPruner()
    )
    sampler = sampler if sampler is not None else optuna.samplers.TPESampler(seed=SEED)
    study = optuna.create_study(direction="maximize", sampler=sampler, pruner=pruner)
    start = time.perf_counter()
    study.optimize(objective, n_trials=n_trials)
    return study, time.perf_counter() - start


def trials_table(study: optuna.Study) -> pd.DataFrame:
    """One row per trial: number, state, score, best score so far, and parameters."""
    frame = study.trials_dataframe(attrs=("number", "value", "state", "params", "duration"))
    frame["duration"] = frame["duration"].dt.total_seconds()
    complete = frame["value"].where(frame["state"] == "COMPLETE")
    frame["best_so_far"] = complete.cummax()
    frame["fold_fits"] = [len(trial.intermediate_values) for trial in study.trials]
    return frame.rename(columns=lambda c: c.removeprefix("params_"))


def boosting_params(params: dict[str, Any], prefix: str = "") -> dict[str, float | int]:
    """Boosting settings from a trial's parameters, without the prefix."""
    names = ("learning_rate", "max_iter", "max_leaf_nodes", "min_samples_leaf", "l2_regularization", "max_features")
    return {name: params[f"{prefix}{name}"] for name in names}


def log_to_mlflow(studies: dict[str, optuna.Study], tracking_uri: str) -> None:
    """Record each study as a parent run with one nested run per trial."""
    import mlflow

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment("loan-default-hpo")
    for name, study in studies.items():
        with mlflow.start_run(run_name=name):
            mlflow.log_params({f"best_{k}": v for k, v in study.best_params.items()})
            mlflow.log_metric("best_cv_roc_auc", study.best_value)
            for trial in study.trials:
                with mlflow.start_run(run_name=f"{name}-trial-{trial.number}", nested=True):
                    mlflow.log_params(trial.params)
                    mlflow.set_tag("state", trial.state.name)
                    if trial.value is not None:
                        mlflow.log_metric("cv_roc_auc", trial.value)


def sampler_rows(studies: dict[str, tuple[pd.DataFrame, float]]) -> list[str]:
    """Markdown rows comparing samplers run with the same objective, pruner and trial budget."""
    lines = [
        "| Sampler | Completed | Pruned | Fold fits | Minutes | Best after 10 | Best after 20 | Best after 40 "
        "| Median completed |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, (table, seconds) in studies.items():
        complete = table[table["state"] == "COMPLETE"]
        best = [table["best_so_far"].iloc[: min(n, len(table))].max() for n in (10, 20, 40)]
        lines.append(
            f"| {name} | {len(complete)} | {int((table['state'] == 'PRUNED').sum())} | {int(table['fold_fits'].sum())} "
            f"| {seconds / 60:.1f} | {best[0]:.4f} | {best[1]:.4f} | {best[2]:.4f} | {complete['value'].median():.4f} |"
        )
    return lines


def render_report(
    samplers: dict[str, tuple[pd.DataFrame, float]],
    automl: pd.DataFrame,
    automl_seconds: float,
    importance: dict[str, float],
    test: pd.DataFrame,
) -> str:
    """Write the findings as Markdown."""
    n_trials = len(next(iter(samplers.values()))[0])
    lines = [
        "# Hyperparameter optimization and AutoML",
        "",
        "Generated by `python -m loan_default.hpo`.",
        "",
        f"## 1. TPE against random sampling for gradient boosting ({n_trials} trials each, {FOLDS}-fold CV, "
        "the same median pruner)",
        "",
        "Both studies use the same search space, objective, folds and pruner in the same run; only the sampler "
        "differs. TPE's first 10 trials are random start-up trials; with the same seed they are the same 10 "
        "configurations as the random sampler's, so the samplers only differ from trial 11. A trial is pruned only "
        "before its last fold. "
        "Fold fits count the CV folds actually trained, since pruned trials stop early. One seed per sampler: "
        "the comparison shows behaviour on this problem, not a general ranking. The median of completed trials "
        "describes each run; because pruning decides which trials complete, it does not isolate sampler quality.",
        "",
        *sampler_rows(samplers),
    ]
    lines += [
        "",
        "Which settings mattered (Optuna fANOVA importance over the completed TPE trials):",
        "",
        "| Hyperparameter | Importance |",
        "|---|---:|",
        *[f"| {name} | {value:.2f} |" for name, value in importance.items()],
        "",
        f"## 2. AutoML: model family and settings together ({len(automl)} trials, "
        f"{AUTOML_SAMPLE:,}-loan sample, {AUTOML_FOLDS}-fold CV, {automl_seconds / 60:.1f} minutes)",
        "",
        "| Family | Trials | Best CV ROC-AUC | Median CV ROC-AUC |",
        "|---|---:|---:|---:|",
    ]
    by_family = automl[automl["state"] == "COMPLETE"].groupby("family")["value"]
    for family, values in sorted(by_family, key=lambda item: -item[1].max()):
        lines.append(
            f"| {family} | {int((automl['family'] == family).sum())} | {values.max():.4f} | {values.median():.4f} |"
        )
    lines += [
        "",
        f"Only the winning family is refitted on all loans below, so a change in ranking between the "
        f"{AUTOML_SAMPLE:,}-loan search and the full refit cannot be attributed to sample size alone.",
        "",
        "## 3. Winners refitted on all 2007-2015 loans, scored on 2016",
        "",
        "| Model | Test ROC-AUC | Test AP | Test default rate, safest 80% |",
        "|---|---:|---:|---:|",
        *[
            f"| {r.model} | {r.roc_auc:.4f} | {r.average_precision:.4f} | {r.default_rate_at_80 * 100:.2f}% |"
            for r in test.itertuples()
        ],
        "",
    ]
    return "\n".join(lines)


def run(splits: Splits, out_dir: Path, n_trials: int, automl_trials: int, track: bool, tracking_uri: str) -> str:
    """Run both studies, refit the winners, and write the report."""
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    boosting_objective = cv_objective(boosting_model, splits.X_train, splits.y_train, FOLDS, prune=True)
    boosting_study, boosting_seconds = run_study(boosting_objective, n_trials, prune=True)
    random_study, random_seconds = run_study(
        boosting_objective, n_trials, prune=True, sampler=optuna.samplers.RandomSampler(seed=SEED)
    )
    sample = splits.X_train.sample(AUTOML_SAMPLE, random_state=SEED).index
    automl_study, automl_seconds = run_study(
        cv_objective(automl_model, splits.X_train.loc[sample], splits.y_train.loc[sample], AUTOML_FOLDS, prune=False),
        automl_trials,
        prune=False,
    )

    winners: dict[str, BaseEstimator] = {
        "production (randomized search winner)": build_model(TUNED_PARAMS, SEED),
        "TPE winner": build_model(boosting_params(boosting_study.best_params), SEED),
        "random-sampler winner": build_model(boosting_params(random_study.best_params), SEED),
        f"AutoML winner ({automl_study.best_params['family']})": automl_model(
            optuna.trial.FixedTrial(automl_study.best_params)
        ),
    }
    rows = []
    for name, model in winners.items():
        fitted = model.fit(splits.X_train, splits.y_train)
        rows.append({"model": name} | score_metrics(splits.y_test, fitted.predict_proba(splits.X_test)[:, 1]))
    test = pd.DataFrame(rows)

    boosting = trials_table(boosting_study)
    automl = trials_table(automl_study)
    random_trials = trials_table(random_study)
    importance = optuna.importance.get_param_importances(
        boosting_study, evaluator=optuna.importance.FanovaImportanceEvaluator(seed=SEED)
    )
    samplers = {"TPE": (boosting, boosting_seconds), "random": (random_trials, random_seconds)}
    report = render_report(samplers, automl, automl_seconds, importance, test)

    write_report(out_dir, "hpo_report", report, test)
    boosting.to_csv(out_dir / "hpo_tpe_trials.csv", index=False)
    random_trials.to_csv(out_dir / "hpo_random_trials.csv", index=False)
    automl.to_csv(out_dir / "hpo_automl_trials.csv", index=False)
    (out_dir / "best_params_tpe.json").write_text(
        json.dumps(boosting_params(boosting_study.best_params), indent=2) + "\n"
    )
    if track:
        log_to_mlflow(
            {"tpe-boosting": boosting_study, "random-boosting": random_study, "automl-cash": automl_study}, tracking_uri
        )
    return report


def main(argv: list[str] | None = None) -> None:
    """Run the studies from the command line."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out-dir", type=Path, default=Path("reports"))
    parser.add_argument("--trials", type=int, default=N_TRIALS)
    parser.add_argument("--automl-trials", type=int, default=AUTOML_TRIALS)
    parser.add_argument("--track", action="store_true", help="log every trial to MLflow")
    parser.add_argument("--tracking-uri", default=DEFAULT_TRACKING_URI)
    args = parser.parse_args(argv)
    print(run(load_splits(), args.out_dir, args.trials, args.automl_trials, args.track, args.tracking_uri))


if __name__ == "__main__":
    main()
