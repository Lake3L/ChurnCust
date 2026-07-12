"""Coverage of the training/evaluation helpers on synthetic data (fast, CI-friendly)."""

import numpy as np
import pytest

from churn.config import DataParams, EconomicsConfig
from churn.data import TARGET
from churn.economics import optimal_threshold
from tests.conftest import make_customers, make_labels

ECO = EconomicsConfig(value_retained=5000, contact_cost=1000, retention_prob=0.3)


@pytest.fixture(scope="module")
def synthetic_splits():
    from churn.data import split_data
    from churn.features import build_features

    df = build_features(make_customers(900, seed=31))
    df[TARGET] = make_labels(df, seed=32)
    params = DataParams(
        raw_path="",
        interim_path="",
        processed_dir="",
        valid_size=0.2,
        test_size=0.2,
        random_state=42,
    )
    return split_data(df, params)


def test_run_baselines_ordering(synthetic_splits):
    from churn.baselines import run_baselines, to_markdown

    train, valid, _ = synthetic_splits
    rows = run_baselines(train, valid, ECO)
    by_name = {r["name"]: r for r in rows}
    assert len(rows) == 4
    assert by_name["call nobody"]["profit"] == 0.0
    assert by_name["call everyone"]["profit"] < 0  # the whole point of the project
    logreg = rows[-1]
    assert logreg["pr_auc"] > 0.3  # sanity: the model learned something
    md = to_markdown(rows, len(valid))
    assert md.count("|") > 20 and "call everyone" in md


def test_train_helpers_fit_and_metrics(synthetic_splits, tmp_path, monkeypatch):
    import mlflow

    from churn.train import fit_and_log, make_lgbm, make_logreg, proba_metrics

    train, valid, _ = synthetic_splits
    x_train, y_train = train, train[TARGET].to_numpy()
    x_valid, y_valid = valid, valid[TARGET].to_numpy()

    mlflow.set_tracking_uri(f"file:///{tmp_path.as_posix()}/mlruns")
    mlflow.set_experiment("test")
    metrics = fit_and_log(
        make_logreg(1.0),
        run_name="t",
        log_params={"model": "logreg"},
        x_train=x_train,
        y_train=y_train,
        x_valid=x_valid,
        y_valid=y_valid,
    )
    assert {"pr_auc", "roc_auc", "brier"} == set(metrics)
    assert metrics["roc_auc"] > 0.6

    lgbm = make_lgbm(n_estimators=30, verbosity=-1, random_state=0)
    lgbm.fit(x_train, y_train)
    m2 = proba_metrics(y_valid, lgbm.predict_proba(x_valid)[:, 1])
    assert 0 < m2["brier"] < 0.3


def test_evaluate_plots_written(synthetic_splits, tmp_path):
    from churn.evaluate import plot_calibration, plot_profit_curve, setup_style

    _, valid, _ = synthetic_splits
    y = valid[TARGET].to_numpy()
    rng = np.random.default_rng(0)
    proba = np.clip(y * 0.5 + rng.random(len(y)) * 0.45, 0, 1)

    setup_style()
    cal_path = tmp_path / "cal.png"
    plot_calibration(y, proba, proba, cal_path, "valid")
    assert cal_path.exists() and cal_path.stat().st_size > 5000

    profit_path = tmp_path / "profit.png"
    t_best, p_best = plot_profit_curve(y, proba, ECO, profit_path, "valid")
    assert profit_path.exists() and profit_path.stat().st_size > 5000
    assert 0.0 <= t_best <= 1.0
    # the empirical optimum of a decent classifier sits near the theoretical one
    assert abs(t_best - optimal_threshold(ECO)) < 0.45
