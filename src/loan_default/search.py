"""Rerun notebook 05's hyperparameter search for gradient boosting and save every result.

    python -m loan_default.search        # writes reports/search_results.csv and reports/best_params.json

Same pipeline, parameter ranges, number of configurations, folds and seeds as the
notebook, so it samples the same 40 configurations. Notebook 05 kept only rounded
values of the winner; this keeps all of them at full precision.
"""

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd
from scipy.stats import loguniform, randint, uniform
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold

from loan_default.preprocessing import load_split
from loan_default.train import build_model

SEED = 42
N_CONFIGURATIONS = 40
FOLDS = 5
PARAM_DISTRIBUTIONS: dict[str, Any] = {
    "model__learning_rate": loguniform(0.01, 0.2),
    "model__max_iter": randint(100, 801),
    "model__max_leaf_nodes": randint(7, 64),
    "model__min_samples_leaf": randint(20, 501),
    "model__l2_regularization": loguniform(1e-3, 10),
    "model__max_features": uniform(0.5, 0.5),
}


def search(train_path: Path, n_jobs: int = -1) -> pd.DataFrame:
    """Run the randomized search and return one row per configuration, best first."""
    X, y = load_split(train_path)
    cv = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=SEED)
    searcher = RandomizedSearchCV(
        build_model({}, SEED),
        PARAM_DISTRIBUTIONS,
        n_iter=N_CONFIGURATIONS,
        scoring={"ROC_AUC": "roc_auc", "AP": "average_precision"},
        refit=False,
        cv=cv,
        n_jobs=n_jobs,
        random_state=SEED,
    )
    searcher.fit(X, y)
    results = pd.DataFrame(searcher.cv_results_)
    keep = [c for c in results.columns if c.startswith(("param_", "mean_test", "std_test", "rank_test", "split"))]
    results = results[keep].rename(columns=lambda c: c.replace("param_model__", ""))
    return results.sort_values("rank_test_ROC_AUC", ignore_index=True)


def best_params(results: pd.DataFrame) -> dict[str, float | int]:
    """Hyperparameters of the configuration with the highest mean ROC-AUC, as plain Python numbers."""
    best = results.iloc[0]
    params: dict[str, float | int] = {}
    for name in (key.removeprefix("model__") for key in PARAM_DISTRIBUTIONS):
        value = best[name]
        params[name] = (
            int(value)
            if float(value).is_integer() and name in {"max_iter", "max_leaf_nodes", "min_samples_leaf"}
            else float(value)
        )
    return params


def main(argv: list[str] | None = None) -> None:
    """Run the search from the command line and save the results."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--train-path", type=Path, default=Path("data/train.csv"))
    parser.add_argument("--out-dir", type=Path, default=Path("reports"))
    args = parser.parse_args(argv)

    results = search(args.train_path)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    results.to_csv(args.out_dir / "search_results.csv", index=False)
    params = best_params(results)
    (args.out_dir / "best_params.json").write_text(json.dumps(params, indent=2) + "\n")
    top = results.iloc[0]
    print(f"Best of {len(results)}: ROC-AUC {top['mean_test_ROC_AUC']:.4f}, AP {top['mean_test_AP']:.4f}")
    print(json.dumps(params, indent=2))


if __name__ == "__main__":
    main()
