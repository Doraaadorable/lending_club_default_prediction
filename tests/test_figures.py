import numpy as np
import pandas as pd

from loan_default.figures import approval_curve


def test_approval_curve_rises_when_the_score_ranks_defaults_last() -> None:
    y = pd.Series([0] * 80 + [1] * 20)
    scores = np.linspace(0, 1, 100)  # defaults have the highest scores
    curve = approval_curve(y, scores, np.array([0.5, 0.8, 1.0]))
    assert curve["default_rate"].tolist() == [0.0, 0.0, 0.2]
