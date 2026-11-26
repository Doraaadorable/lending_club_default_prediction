"""The two headline figures for the README.

    python -m loan_default.figures        # writes reports/figures/approval_curve.png and monthly_calibration.png

1. Default rate among approved 2016 loans against the share approved, for the production model
   and for FICO alone: the decision the model is for.
2. Mean predicted against actual default rate per month of 2016, from the replay: the calibration gap.
   It reads `<out-dir>/replay_monthly.csv`, so run `python -m loan_default.replay --out-dir <out-dir>`
   first when drawing into a new directory.
"""

import argparse
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from loan_default.experiments import load_splits
from loan_default.policy import approve_safest
from loan_default.train import TUNED_PARAMS, build_baseline, build_model, default_scores

if TYPE_CHECKING:
    from matplotlib.axes import Axes

SEED = 42
RATES = np.round(np.arange(0.40, 1.0001, 0.02), 2)
MODEL_COLOR, FICO_COLOR = "#257de0", "#f47230"  # categorical slots 1 and 2, validated for colour-blind separation
TEXT, MUTED, GRID = "#191919", "#666666", "#e6e6e6"


def approval_curve(y: pd.Series, scores: NDArray[np.float64], rates: NDArray[np.float64] = RATES) -> pd.DataFrame:
    """Default rate among the approved loans at each approval rate."""
    values = y.to_numpy()
    return pd.DataFrame(
        {"approval_rate": rates, "default_rate": [float(values[approve_safest(scores, r)].mean()) for r in rates]}
    )


def _axes_style(ax: "Axes") -> None:
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def plot_approval(model: pd.DataFrame, fico: pd.DataFrame, path: Path) -> None:
    """Two lines on one axis, labelled at their right ends, with the 80% policy marked."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4.2))
    # labels at the left end, where the two lines are furthest apart (they meet at 100% approval)
    for frame, color, label, offset in [
        (fico, FICO_COLOR, "FICO alone", (-36, 18)),
        (model, MODEL_COLOR, "Gradient boosting", (10, -14)),
    ]:
        ax.plot(frame["approval_rate"] * 100, frame["default_rate"] * 100, color=color, linewidth=2)
        anchor = frame.iloc[3]
        ax.annotate(
            label,
            (anchor["approval_rate"] * 100, anchor["default_rate"] * 100),
            xytext=offset,
            textcoords="offset points",
            ha="left",
            color=TEXT,
            fontsize=9,
        )
    at80 = {
        name: float(f.loc[np.isclose(f["approval_rate"], 0.8), "default_rate"].iloc[0])
        for name, f in (("model", model), ("fico", fico))
    }
    ax.axvline(80, color=MUTED, linewidth=0.8, linestyle=(0, (3, 3)))
    for value, color in ((at80["model"], MODEL_COLOR), (at80["fico"], FICO_COLOR)):
        ax.plot(80, value * 100, "o", color=color, markersize=8, markeredgecolor="white", markeredgewidth=2)
    ax.annotate(
        f"At 80% approval: {at80['model']:.1%} vs {at80['fico']:.1%}",
        (80, at80["fico"] * 100),
        xytext=(-8, 12),
        textcoords="offset points",
        ha="right",
        color=TEXT,
        fontsize=9,
    )
    ax.set_xlabel("Share of 2016 applicants approved (safest first, %)", color=TEXT, fontsize=9)
    ax.set_ylabel("Default rate among approved (%)", color=TEXT, fontsize=9)
    ax.set_title(
        "Gradient boosting lowers default rates at matched approval volumes", loc="left", color=TEXT, fontsize=11
    )
    _axes_style(ax)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_monthly(monthly: pd.DataFrame, path: Path) -> None:
    """Mean predicted and actual default rate per month of 2016, labelled at their right ends."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    months = pd.to_datetime(monthly["month"]).dt.strftime("%b")
    x = np.arange(len(monthly))
    fig, ax = plt.subplots(figsize=(7, 4.2))
    for column, color, label in [("actual", FICO_COLOR, "Actual"), ("mean_predicted", MODEL_COLOR, "Predicted")]:
        ax.plot(
            x,
            monthly[column] * 100,
            color=color,
            linewidth=2,
            marker="o",
            markersize=5,
            markeredgecolor="white",
            markeredgewidth=1.5,
        )
        ax.annotate(
            label,
            (x[-1], monthly[column].iloc[-1] * 100),
            xytext=(8, 0),
            textcoords="offset points",
            va="center",
            color=TEXT,
            fontsize=9,
        )
    ax.set_xticks(x, months)
    ax.set_xlim(-0.5, len(x) + 0.8)
    ax.set_ylim(0, max(30.0, float(monthly["actual"].max()) * 100 + 3))
    ax.set_ylabel("Default rate (%)", color=TEXT, fontsize=9)
    ax.set_title("2016 replay: the model's probabilities ran about 5 points low", loc="left", color=TEXT, fontsize=11)
    _axes_style(ax)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def run(out_dir: Path) -> pd.DataFrame:
    """Train the production model and the FICO baseline, draw both figures, and return the approval curves."""
    splits = load_splits()
    model = build_model(TUNED_PARAMS, SEED).fit(splits.X_train, splits.y_train)
    fico = build_baseline(SEED).fit(splits.X_train, splits.y_train)
    curves = {
        "model": approval_curve(splits.y_test, default_scores(model, splits.X_test)),
        "fico": approval_curve(splits.y_test, default_scores(fico, splits.X_test)),
    }
    figures = out_dir / "figures"
    plot_approval(curves["model"], curves["fico"], figures / "approval_curve.png")
    plot_monthly(pd.read_csv(out_dir / "replay_monthly.csv"), figures / "monthly_calibration.png")
    table = curves["model"].merge(curves["fico"], on="approval_rate", suffixes=("_model", "_fico"))
    table.to_csv(out_dir / "approval_curve.csv", index=False)
    return table


def main(argv: list[str] | None = None) -> None:
    """Draw the figures from the command line."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out-dir", type=Path, default=Path("reports"))
    args = parser.parse_args(argv)
    print(run(args.out_dir).to_string(index=False))


if __name__ == "__main__":
    main()
