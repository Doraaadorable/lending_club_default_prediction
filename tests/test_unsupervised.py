import numpy as np
import pandas as pd

from loan_default.unsupervised import ClusterDistances, choose_k, describe, pick_k, profile_matrix


def test_cluster_distances_appends_one_column_per_centre() -> None:
    X = np.random.default_rng(0).normal(size=(200, 3))
    out = ClusterDistances(n_clusters=4).fit(X).transform(X)
    assert out.shape == (200, 7)
    assert np.allclose(out[:, :3], X)
    assert (out[:, 3:] >= 0).all()


def test_silhouette_finds_three_well_separated_blobs() -> None:
    rng = np.random.default_rng(0)
    blobs = np.vstack([rng.normal(centre, 0.3, size=(300, 2)) for centre in (-5, 0, 5)])
    table = choose_k(blobs, range(2, 6))
    assert int(table.loc[table["silhouette"].idxmax(), "k"]) == 3


def test_pick_k_takes_the_largest_k_close_to_the_best() -> None:
    table = pd.DataFrame({"k": [2, 3, 4, 5], "silhouette": [0.30, 0.295, 0.292, 0.25]})
    assert pick_k(table, tolerance=0.01) == 4


def test_profile_matrix_is_standardized_without_gaps(small_train: tuple[pd.DataFrame, pd.Series]) -> None:
    values, _ = profile_matrix(small_train[0])
    assert not np.isnan(values).any()
    assert np.allclose(values.mean(axis=0), 0, atol=1e-8)


def test_describe_names_the_most_distinctive_traits() -> None:
    table = pd.DataFrame(
        {
            "fico": [700, 700, 760],
            "income": [60_000, 60_000, 60_000],
            "loan_to_income": [0.2, 0.2, 0.2],
            "dti": [15, 15, 15],
            "emp_length": [5, 5, 5],
        }
    )
    assert describe(table.iloc[2], table, n_traits=1) == "higher FICO"
