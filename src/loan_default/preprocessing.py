"""
Shared preprocessing for the Lending Club default-prediction project (v2).

Training, the notebooks and the prediction service all import from this file,
so the data is cleaned in exactly the same way everywhere:

    from loan_default.preprocessing import load_split, build_preprocessor

    X_train, y_train = load_split("../data/train.csv")
    model = make_pipeline(build_preprocessor(), LogisticRegression())
    model.fit(X_train, y_train)

Requires scikit-learn >= 1.3 (for TargetEncoder).

Changes from v1, based on the EDA (notebook 02):
- removed `has_desc`: descriptions stop after 2014, so the flag is all 0 in the 2016 test set
- added `loan_to_income`: strongest numeric signal in the EDA (default 13% -> 27% across quintiles)
- `addr_state`: smoothed target encoding instead of 50 one-hot columns
"""

from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.experimental import enable_iterative_imputer  # noqa: F401  (makes IterativeImputer importable)
from sklearn.impute import IterativeImputer, KNNImputer, SimpleImputer
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler, TargetEncoder

TARGET = "Default"

# Columns kept out of the model, and why (see README "Data preparation")
DROP_COLS = [
    "id",  # identifier only; no relationship with default
    "issue_d",  # used for the time split, not as a feature
    "experience_c",  # almost constant (5 zeros in 150k training loans)
    "title",  # free text duplicating `purpose`, inconsistent spelling
    "zip_code",  # 879 levels in train; also a proxy for protected attributes (fair lending)
    "desc",  # 86% empty and absent after 2014 -> would not exist in the 2016 test set
]

EMP_LENGTH_MAP = {
    "< 1 year": 0,
    "1 year": 1,
    "2 years": 2,
    "3 years": 3,
    "4 years": 4,
    "5 years": 5,
    "6 years": 6,
    "7 years": 7,
    "8 years": 8,
    "9 years": 9,
    "10+ years": 10,
    # "NI" (no information) is not in the map, so it becomes NaN;
    # the imputer then adds a "was missing" flag (NI borrowers default more)
}

DTI_PLACEHOLDER = 999  # undocumented code, treated as missing
DTI_CAP = 100  # debt payments above 100% of income are capped


def load_split(path: str | Path) -> tuple[pd.DataFrame, pd.Series]:
    """Read train.csv or test.csv and return features X and target y."""
    df = pd.read_csv(path, low_memory=False)
    return df.drop(columns=[TARGET]), df[TARGET]


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """
    Step 1 of the pipeline: fix data-entry quirks and build row-level features.

    Nothing here is *learned* from the data (no means, no medians), so this
    step cannot leak information from the test set.
    """
    df = df.copy()

    # emp_length: text -> number of years; "NI" -> NaN (imputed later, with a flag)
    df["emp_length"] = df["emp_length"].map(EMP_LENGTH_MAP)

    # dti_n: 999 is a placeholder -> NaN; extreme values capped
    df["dti_n"] = df["dti_n"].replace(DTI_PLACEHOLDER, np.nan).clip(upper=DTI_CAP)

    # loan size relative to income: how heavy the loan is for this borrower
    df["loan_to_income"] = df["loan_amnt"] / df["revenue"]

    return df.drop(columns=DROP_COLS, errors="ignore")


NUMERIC = ["fico_n", "dti_n", "emp_length"]
SKEWED = ["revenue", "loan_amnt", "loan_to_income"]  # long right tail -> log first
ONE_HOT = ["purpose", "home_ownership_n"]
TARGET_ENCODED = ["addr_state"]


STATE_SMOOTHING = 100  # a state's rate counts like 100 "virtual" loans at the overall rate


def _make_target_encoder() -> TargetEncoder:
    """Smoothed target encoder with reproducible cross-fitting (works on old and new sklearn)."""
    major, minor = (int(x) for x in sklearn.__version__.split(".")[:2])
    if (major, minor) >= (1, 9):  # newer API: shuffling is set through the cv splitter
        return TargetEncoder(
            target_type="binary", smooth=STATE_SMOOTHING, cv=StratifiedKFold(5, shuffle=True, random_state=42)
        )
    return TargetEncoder(target_type="binary", smooth=STATE_SMOOTHING, random_state=42)


IMPUTERS = ("median", "mean", "iterative", "knn")


def make_imputer(
    strategy: str, add_indicator: bool = True, seed: int = 42
) -> SimpleImputer | IterativeImputer | KNNImputer:
    """Imputer for the numeric block.

    median, mean : one constant per column (the project default is median)
    iterative    : MICE-style chained regressions; each column is predicted from all the others
    knn          : average of the 10 most similar loans on the other numeric columns
    """
    if strategy in {"median", "mean"}:
        return SimpleImputer(strategy=strategy, add_indicator=add_indicator)
    if strategy == "iterative":
        return IterativeImputer(max_iter=10, random_state=seed, add_indicator=add_indicator)
    if strategy == "knn":
        return KNNImputer(n_neighbors=10, add_indicator=add_indicator)
    raise ValueError(f"imputer must be one of {IMPUTERS}, got {strategy!r}")


def _log_skewed_columns(X: pd.DataFrame) -> pd.DataFrame:
    """log(1 + x) on the long-tailed columns, other columns unchanged."""
    X = X.copy()
    X[SKEWED] = np.log1p(X[SKEWED])
    return X


def build_preprocessor(
    numeric_imputer: str = "median", add_indicator: bool = True, include_state: bool = True
) -> Pipeline:
    """
    The full preprocessing pipeline (unfitted).

    clean  ->  ColumnTransformer:
        numeric        : fill missing with median (+ missing flag), standardize
        skewed         : log(1 + x), standardize
        one-hot        : purpose, home ownership; levels with < 500 rows grouped as "infrequent"
        target-encoded : state -> smoothed default rate, learned with internal cross-fitting
                         encoded = (n * state_rate + 100 * overall_rate) / (n + 100)

    The defaults are the production pipeline. The arguments exist for the experiments in
    `loan_default.imputation` and `loan_default.fairness`:

    numeric_imputer : "median" (default), "mean", "iterative" or "knn". The model-based imputers
                      see the numeric and log-scaled skewed columns together, so income and loan
                      size can inform a missing employment length or DTI.
    add_indicator   : keep a 0/1 "was missing" column for each column that had gaps in training
    include_state   : False drops the state encoding (fairness comparison)
    """
    one_hot = OneHotEncoder(
        handle_unknown="infrequent_if_exist",
        min_frequency=500,
        sparse_output=False,
    )
    blocks: list[tuple[str, object, list[str]]]
    if numeric_imputer in {"median", "mean"}:
        numeric = make_pipeline(make_imputer(numeric_imputer, add_indicator), StandardScaler())
        skewed = make_pipeline(
            FunctionTransformer(np.log1p, feature_names_out="one-to-one"),
            StandardScaler(),
        )
        blocks = [("num", numeric, NUMERIC), ("log", skewed, SKEWED)]
    else:
        # scale first so that distances (KNN) and regressions (iterative) treat columns alike
        joint = make_pipeline(
            FunctionTransformer(_log_skewed_columns, feature_names_out="one-to-one"),
            StandardScaler(),
            make_imputer(numeric_imputer, add_indicator),
        )
        blocks = [("num", joint, NUMERIC + SKEWED)]
    blocks.append(("cat", one_hot, ONE_HOT))
    if include_state:
        blocks.append(("state", _make_target_encoder(), TARGET_ENCODED))

    return Pipeline(
        [
            ("clean", FunctionTransformer(clean)),
            ("columns", ColumnTransformer(blocks)),
        ]
    )
