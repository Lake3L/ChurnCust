"""Model training: LogReg vs LightGBM (Optuna), probability calibration, MLflow registry.

Model selection happens on the validation split by PR-AUC. The champion is then
wrapped into CalibratedClassifierCV (isotonic vs sigmoid, 5-fold on train) and the
calibration method is chosen by Brier score on valid — calibrated probabilities are
what makes the economic threshold meaningful (see churn.economics).

The sealed test split is NOT read anywhere in this module.
"""

import json
import time
from pathlib import Path

import joblib
import mlflow
import numpy as np
import optuna
import pandas as pd
from lightgbm import LGBMClassifier
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.pipeline import Pipeline

from churn.config import EconomicsConfig, Params
from churn.data import TARGET
from churn.economics import expected_profit, optimal_threshold
from churn.features import ALL_FEATURES, make_preprocessor
from churn.log import get_logger

logger = get_logger(__name__)

EXPERIMENT_NAME = "churn-scoring"
REGISTERED_MODEL = "churn-scoring"
LOGREG_C_GRID = [0.01, 0.1, 1.0, 10.0]


def make_logreg(c: float) -> Pipeline:
    return Pipeline(
        [
            ("prep", make_preprocessor(scale_numeric=True)),
            ("clf", LogisticRegression(C=c, max_iter=2000)),
        ]
    )


def make_lgbm(**kwargs) -> Pipeline:
    return Pipeline(
        [
            ("prep", make_preprocessor(scale_numeric=False)),
            ("clf", LGBMClassifier(**kwargs)),
        ]
    )


def proba_metrics(y_true: np.ndarray, proba: np.ndarray) -> dict[str, float]:
    return {
        "pr_auc": float(average_precision_score(y_true, proba)),
        "roc_auc": float(roc_auc_score(y_true, proba)),
        "brier": float(brier_score_loss(y_true, proba)),
    }


def fit_and_log(
    model: Pipeline,
    run_name: str,
    log_params: dict,
    x_train: pd.DataFrame,
    y_train: np.ndarray,
    x_valid: pd.DataFrame,
    y_valid: np.ndarray,
) -> dict[str, float]:
    with mlflow.start_run(run_name=run_name):
        mlflow.log_params(log_params)
        model.fit(x_train, y_train)
        metrics = proba_metrics(y_valid, model.predict_proba(x_valid)[:, 1])
        mlflow.log_metrics(metrics)
    return metrics


def tune_lightgbm(
    params: Params,
    x_train: pd.DataFrame,
    y_train: np.ndarray,
    x_valid: pd.DataFrame,
    y_valid: np.ndarray,
) -> tuple[dict, float]:
    """Optuna search maximizing PR-AUC on valid. Returns (best_params, best_score)."""
    rs = params.train.random_state

    def objective(trial: optuna.Trial) -> float:
        lgbm_params = {
            "objective": "binary",
            "verbosity": -1,
            "random_state": rs,
            "n_estimators": trial.suggest_int("n_estimators", 200, 900),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.12, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 8, 64),
            "min_child_samples": trial.suggest_int("min_child_samples", 10, 100),
            "feature_fraction": trial.suggest_float("feature_fraction", 0.5, 1.0),
            "bagging_fraction": trial.suggest_float("bagging_fraction", 0.5, 1.0),
            "bagging_freq": 1,
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
        }
        metrics = fit_and_log(
            make_lgbm(**lgbm_params),
            run_name=f"lgbm_trial_{trial.number:03d}",
            log_params={"model": "lightgbm", **lgbm_params},
            x_train=x_train,
            y_train=y_train,
            x_valid=x_valid,
            y_valid=y_valid,
        )
        return metrics["pr_auc"]

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=rs))
    study.optimize(objective, n_trials=params.train.optuna_trials)
    best = {
        "objective": "binary",
        "verbosity": -1,
        "random_state": rs,
        "bagging_freq": 1,
        **study.best_params,
    }
    return best, study.best_value


def main() -> None:
    started = time.time()
    params = Params.load()
    eco = EconomicsConfig.load()
    rs = params.train.random_state

    processed = Path(params.data.processed_dir)
    train = pd.read_parquet(processed / "train.parquet")
    valid = pd.read_parquet(processed / "valid.parquet")
    x_train, y_train = train[ALL_FEATURES], train[TARGET].to_numpy()
    x_valid, y_valid = valid[ALL_FEATURES], valid[TARGET].to_numpy()

    mlflow.set_experiment(EXPERIMENT_NAME)

    # --- candidate 1: regularized logistic regression (C grid) ---
    logreg_results: dict[float, dict] = {}
    for c in LOGREG_C_GRID:
        logreg_results[c] = fit_and_log(
            make_logreg(c),
            run_name=f"logreg_C{c}",
            log_params={"model": "logreg", "C": c},
            x_train=x_train,
            y_train=y_train,
            x_valid=x_valid,
            y_valid=y_valid,
        )
    best_c = max(logreg_results, key=lambda c: logreg_results[c]["pr_auc"])
    best_logreg_score = logreg_results[best_c]["pr_auc"]

    # --- candidate 2: LightGBM with defaults, then Optuna ---
    default_lgbm = {**params.train.lightgbm_base, "random_state": rs}
    fit_and_log(
        make_lgbm(**default_lgbm),
        run_name="lgbm_default",
        log_params={"model": "lightgbm", **default_lgbm},
        x_train=x_train,
        y_train=y_train,
        x_valid=x_valid,
        y_valid=y_valid,
    )
    best_lgbm_params, best_lgbm_score = tune_lightgbm(params, x_train, y_train, x_valid, y_valid)

    # --- champion selection by PR-AUC on valid ---
    if best_lgbm_score >= best_logreg_score:
        champion_name = "lightgbm"
        champion: Pipeline = make_lgbm(**best_lgbm_params)
        champion_params: dict = best_lgbm_params
    else:
        champion_name = "logreg"
        champion = make_logreg(best_c)
        champion_params = {"C": best_c}
    logger.info(
        "champion selected",
        extra={
            "extra_fields": {
                "champion": champion_name,
                "logreg_pr_auc": best_logreg_score,
                "lgbm_pr_auc": best_lgbm_score,
            }
        },
    )

    # uncalibrated champion (kept for the before/after calibration story)
    champion.fit(x_train, y_train)
    uncal_metrics = proba_metrics(y_valid, champion.predict_proba(x_valid)[:, 1])

    # --- calibration: isotonic vs sigmoid, 5-fold on train, judged by Brier on valid ---
    calibration_results: dict[str, dict] = {}
    calibrated_models: dict[str, CalibratedClassifierCV] = {}
    for method in ("isotonic", "sigmoid"):
        calibrated = CalibratedClassifierCV(
            estimator=clone(champion), method=method, cv=params.train.cv_folds
        )
        with mlflow.start_run(run_name=f"calibrated_{method}"):
            mlflow.log_params({"model": champion_name, "calibration": method, **champion_params})
            calibrated.fit(x_train, y_train)
            metrics = proba_metrics(y_valid, calibrated.predict_proba(x_valid)[:, 1])
            mlflow.log_metrics(metrics)
        calibration_results[method] = metrics
        calibrated_models[method] = calibrated

    best_method = min(calibration_results, key=lambda m: calibration_results[m]["brier"])
    final_model = calibrated_models[best_method]
    final_metrics = calibration_results[best_method]

    threshold = optimal_threshold(eco)
    final_proba = final_model.predict_proba(x_valid)[:, 1]
    profit_valid = expected_profit(y_valid, final_proba, threshold, eco)

    # --- final run: log the model, register, set alias ---
    with mlflow.start_run(run_name="best_calibrated") as run:
        mlflow.set_tag("best", "true")
        mlflow.log_params(
            {
                "model": champion_name,
                "calibration": best_method,
                "threshold_theoretical": threshold,
                **champion_params,
            }
        )
        mlflow.log_metrics(
            {
                **final_metrics,
                "brier_uncalibrated": uncal_metrics["brier"],
                "profit_valid_at_threshold": profit_valid,
            }
        )
        signature = infer_signature(x_valid.head(), final_model.predict_proba(x_valid.head()))
        mlflow.sklearn.log_model(
            final_model,
            artifact_path="model",
            signature=signature,
            input_example=x_valid.head(3),
            registered_model_name=REGISTERED_MODEL,
        )
        run_id = run.info.run_id

    client = MlflowClient()
    version = client.search_model_versions(f"name='{REGISTERED_MODEL}' and run_id='{run_id}'")[
        0
    ].version
    client.set_registered_model_alias(REGISTERED_MODEL, "staging", version)

    # --- local artifacts: what the API falls back to without a registry ---
    model_dir = Path(params.train.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(final_model, model_dir / "model.joblib")
    joblib.dump(champion, model_dir / "model_uncalibrated.joblib")
    metadata = {
        "model": champion_name,
        "champion_params": champion_params,
        "calibration": best_method,
        "calibration_comparison": calibration_results,
        "metrics_valid_uncalibrated": uncal_metrics,
        "metrics_valid": final_metrics,
        "profit_valid_at_threshold": profit_valid,
        "threshold_theoretical": threshold,
        "registry_version": str(version),
        "features": ALL_FEATURES,
        "train_seconds": round(time.time() - started, 1),
    }
    with open(model_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    logger.info("training finished", extra={"extra_fields": metadata})


if __name__ == "__main__":
    main()
