"""Evaluation: metrics, the profit-vs-threshold curve and calibration diagrams.

Runs on the validation split by default. ``--split test`` is executed exactly once,
at the very end of the project (the sealed-test rule from the README).
"""

import argparse
import json
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from churn.config import EconomicsConfig, Params
from churn.data import TARGET
from churn.economics import expected_profit, optimal_threshold, policy_profit, profit_curve
from churn.features import ALL_FEATURES
from churn.log import get_logger

logger = get_logger(__name__)

BLUE = "#2563eb"  # primary: the calibrated model
ORANGE = "#ea580c"  # accent: before calibration / the economic threshold
GRAY = "#6b7280"  # context: baselines, reference lines


def setup_style() -> None:
    plt.rcParams.update(
        {
            "figure.figsize": (8, 4.5),
            "figure.dpi": 110,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.3,
            "axes.titlesize": 12,
            "font.size": 10,
        }
    )


def plot_calibration(
    y: np.ndarray,
    proba_calibrated: np.ndarray,
    proba_uncalibrated: np.ndarray,
    out_path: Path,
    split: str,
) -> None:
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1], "--", color=GRAY, linewidth=1.5, label="идеальная калибровка")
    for proba, color, label in (
        (proba_uncalibrated, ORANGE, "до калибровки"),
        (proba_calibrated, BLUE, "после калибровки (isotonic/sigmoid)"),
    ):
        frac_pos, mean_pred = calibration_curve(y, proba, n_bins=10, strategy="quantile")
        ax.plot(mean_pred, frac_pos, "o-", color=color, linewidth=2, markersize=5, label=label)
    ax.set_xlabel("Средняя предсказанная вероятность оттока")
    ax.set_ylabel("Фактическая доля оттока")
    ax.set_title(f"Reliability diagram ({split})")
    ax.legend(frameon=False)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_profit_curve(
    y: np.ndarray,
    proba: np.ndarray,
    eco: EconomicsConfig,
    out_path: Path,
    split: str,
) -> tuple[float, float]:
    """Draw the main figure of the project. Returns (best_threshold, best_profit)."""
    thresholds, profits = profit_curve(y, proba, eco)
    t_theory = optimal_threshold(eco)
    i_best = int(np.argmax(profits))
    t_best, p_best = float(thresholds[i_best]), float(profits[i_best])

    fig, ax = plt.subplots()
    ax.plot(
        thresholds, profits, color=BLUE, linewidth=2, label="прибыль политики «звонить при P ≥ t»"
    )
    ax.axhline(0, color=GRAY, linewidth=1, linestyle=":")
    ax.axhline(
        profits[0],
        color=GRAY,
        linewidth=1.5,
        linestyle="--",
        label=f"звонить всем: {profits[0]:,.0f} ₽",
    )
    ax.axvline(
        t_theory,
        color=ORANGE,
        linewidth=2,
        linestyle="--",
        label=f"порог из экономики C/(p·V) = {t_theory:.3f}",
    )
    ax.plot([t_best], [p_best], "o", color=BLUE, markersize=8)
    ax.annotate(
        f"эмпирический максимум\nt={t_best:.2f}, {p_best:,.0f} ₽",
        xy=(t_best, p_best),
        xytext=(t_best - 0.28, p_best * 0.75),
        fontsize=9,
        color=BLUE,
    )
    ax.set_xlabel("Порог вероятности t")
    ax.set_ylabel(f"Ожидаемая прибыль на {split}, ₽")
    ax.set_title("Прибыль кампании удержания vs порог принятия решения")
    ax.legend(frameon=False, loc="lower center")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return t_best, p_best


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=["valid", "test"], default="valid")
    args = parser.parse_args()
    split = args.split

    setup_style()
    params = Params.load()
    eco = EconomicsConfig.load()
    model_dir = Path(params.train.model_dir)
    model = joblib.load(model_dir / "model.joblib")
    model_uncal = joblib.load(model_dir / "model_uncalibrated.joblib")

    df = pd.read_parquet(Path(params.data.processed_dir) / f"{split}.parquet")
    x, y = df[ALL_FEATURES], df[TARGET].to_numpy()
    proba = model.predict_proba(x)[:, 1]
    proba_uncal = model_uncal.predict_proba(x)[:, 1]

    t_theory = optimal_threshold(eco)
    figures_dir = Path(params.evaluate.reports_dir) / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    plot_calibration(y, proba, proba_uncal, figures_dir / f"calibration_{split}.png", split)
    t_best, p_best = plot_profit_curve(
        y, proba, eco, figures_dir / f"profit_vs_threshold_{split}.png", split
    )

    n = len(df)
    metrics = {
        "split": split,
        "n_customers": n,
        "pr_auc": float(average_precision_score(y, proba)),
        "roc_auc": float(roc_auc_score(y, proba)),
        "brier": float(brier_score_loss(y, proba)),
        "brier_uncalibrated": float(brier_score_loss(y, proba_uncal)),
        "threshold_theoretical": t_theory,
        "threshold_empirical_best": t_best,
        "profit_at_theoretical_threshold": expected_profit(y, proba, t_theory, eco),
        "profit_at_05": expected_profit(y, proba, 0.5, eco),
        "profit_empirical_best": p_best,
        "profit_call_everyone": policy_profit(y, np.ones(n, dtype=bool), eco),
        "profit_call_nobody": 0.0,
        "n_calls_at_threshold": int((proba >= t_theory).sum()),
    }
    out_path = Path(params.evaluate.reports_dir) / f"metrics_{split}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    logger.info("evaluation finished", extra={"extra_fields": metrics})


if __name__ == "__main__":
    main()
