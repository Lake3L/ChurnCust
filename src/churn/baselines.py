"""Four baseline policies on the validation split.

They anchor every later comparison: a model that cannot beat "call everyone"
in expected profit is worthless regardless of its AUC.

1. call nobody              — profit is exactly 0 (status quo);
2. call everyone            — negative profit expected: most customers stay anyway;
3. simple business rule     — call month-to-month customers with tenure < 6;
4. LogReg on 5 raw features — minimal ML, decisions at the economic threshold.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from churn.config import EconomicsConfig, Params
from churn.data import TARGET
from churn.economics import optimal_threshold, policy_profit
from churn.log import get_logger

logger = get_logger(__name__)

RULE_DESCRIPTION = "month-to-month & tenure < 6"
LOGREG_NUMERIC = ["tenure_months", "monthly_charges", "total_charges"]
LOGREG_CATEGORICAL = ["contract", "internet_service"]


def make_simple_logreg() -> Pipeline:
    prep = ColumnTransformer(
        [
            ("cat", OneHotEncoder(handle_unknown="ignore"), LOGREG_CATEGORICAL),
            ("num", StandardScaler(), LOGREG_NUMERIC),
        ]
    )
    return Pipeline([("prep", prep), ("clf", LogisticRegression(max_iter=2000))])


def run_baselines(train: pd.DataFrame, valid: pd.DataFrame, eco: EconomicsConfig) -> list[dict]:
    y_valid = valid[TARGET].to_numpy()
    n = len(valid)
    threshold = optimal_threshold(eco)

    rows: list[dict] = []

    nobody = np.zeros(n, dtype=bool)
    rows.append(
        {
            "name": "call nobody",
            "pr_auc": None,
            "roc_auc": None,
            "n_calls": 0,
            "profit": policy_profit(y_valid, nobody, eco),
        }
    )

    everyone = np.ones(n, dtype=bool)
    rows.append(
        {
            "name": "call everyone",
            "pr_auc": None,
            "roc_auc": None,
            "n_calls": n,
            "profit": policy_profit(y_valid, everyone, eco),
        }
    )

    rule = ((valid["contract"] == "Month-to-month") & (valid["tenure_months"] < 6)).to_numpy()
    rows.append(
        {
            "name": f"rule: {RULE_DESCRIPTION}",
            "pr_auc": None,
            "roc_auc": None,
            "n_calls": int(rule.sum()),
            "profit": policy_profit(y_valid, rule, eco),
        }
    )

    model = make_simple_logreg()
    model.fit(train, train[TARGET])
    proba = model.predict_proba(valid)[:, 1]
    rows.append(
        {
            "name": "LogReg on 5 features @ economic threshold",
            "pr_auc": float(average_precision_score(y_valid, proba)),
            "roc_auc": float(roc_auc_score(y_valid, proba)),
            "n_calls": int((proba >= threshold).sum()),
            "profit": policy_profit(y_valid, proba >= threshold, eco),
        }
    )
    return rows


def to_markdown(rows: list[dict], n_customers: int) -> str:
    lines = [
        f"Baseline policies on the validation split (n={n_customers}):",
        "",
        "| Policy | PR-AUC | ROC-AUC | Calls | Expected profit, RUB |",
        "|---|---|---|---|---|",
    ]
    for r in rows:
        pr = f"{r['pr_auc']:.3f}" if r["pr_auc"] is not None else "—"
        roc = f"{r['roc_auc']:.3f}" if r["roc_auc"] is not None else "—"
        lines.append(f"| {r['name']} | {pr} | {roc} | {r['n_calls']} | {r['profit']:,.0f} |")
    return "\n".join(lines) + "\n"


def main() -> None:
    params = Params.load()
    eco = EconomicsConfig.load()
    train = pd.read_parquet(Path(params.data.processed_dir) / "train.parquet")
    valid = pd.read_parquet(Path(params.data.processed_dir) / "valid.parquet")

    rows = run_baselines(train, valid, eco)

    reports_dir = Path(params.evaluate.reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)
    with open(reports_dir / "baselines.json", "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)
    (reports_dir / "baselines.md").write_text(to_markdown(rows, len(valid)), encoding="utf-8")

    for r in rows:
        logger.info("baseline evaluated", extra={"extra_fields": r})


if __name__ == "__main__":
    main()
