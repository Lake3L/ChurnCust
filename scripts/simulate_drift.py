"""Drift simulation: three canonical failure modes, three very different detections.

Scenarios (reference = valid split, current = distorted test split):

1. covariate  — MonthlyCharges +30%: input distribution moves, PSI/Evidently see it;
2. prior      — churners oversampled to ~2x base rate: features move only slightly,
                but the PSI of the *predicted probability* moves a lot;
3. concept    — P(y|x) changes (labels flipped for a random 30% subsample):
                inputs are IDENTICAL, no unsupervised detector can see it —
                only delayed labels reveal the quality collapse.

Output: reports/drift/drift_<scenario>.html (Evidently),
        reports/drift/drift_summary.{json,md} — the README table,
        WARNING to stdout for every PSI alert (the "poor man's alerting hook").
"""

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from churn.config import EconomicsConfig, Params  # noqa: E402
from churn.data import TARGET  # noqa: E402
from churn.economics import expected_profit, optimal_threshold  # noqa: E402
from churn.features import ALL_FEATURES  # noqa: E402
from churn.monitoring import (  # noqa: E402
    PSI_ALERT_THRESHOLD,
    evidently_drift_report,
    psi,
    psi_table,
)

RNG = np.random.default_rng(42)


def scenario_covariate(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["monthly_charges"] = out["monthly_charges"] * 1.30
    # derived features must stay consistent -> recompute the ones touched
    out["charge_diff"] = out["monthly_charges"] - out["charges_per_month_hist"]
    out["avg_service_cost"] = out["monthly_charges"] / out["num_services"].clip(lower=1)
    return out


def scenario_prior(df: pd.DataFrame) -> pd.DataFrame:
    churners = df[df[TARGET] == 1]
    extra = churners.sample(n=len(churners), replace=True, random_state=42)
    return pd.concat([df, extra], ignore_index=True)


def scenario_concept(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    flip = RNG.random(len(out)) < 0.30
    out.loc[flip, TARGET] = 1 - out.loc[flip, TARGET]
    return out


def evaluate_scenario(
    name: str,
    reference: pd.DataFrame,
    current: pd.DataFrame,
    model,
    proba_reference: np.ndarray,
    eco: EconomicsConfig,
    out_dir: Path,
) -> dict:
    from sklearn.metrics import average_precision_score, roc_auc_score

    proba = model.predict_proba(current[ALL_FEATURES])[:, 1]
    y = current[TARGET].to_numpy()
    threshold = optimal_threshold(eco)

    evidently = evidently_drift_report(reference, current, out_dir / f"drift_{name}.html")
    table = psi_table(reference, current)
    proba_psi = psi(proba_reference, proba)

    return {
        "scenario": name,
        **evidently,
        "max_psi_feature": table.iloc[0]["feature"],
        "max_psi": round(float(table.iloc[0]["psi"]), 3),
        "n_psi_alerts": int(table["alert"].sum()),
        "proba_psi": round(proba_psi, 3),
        "proba_psi_alert": proba_psi > PSI_ALERT_THRESHOLD,
        "pr_auc": round(float(average_precision_score(y, proba)), 3),
        "roc_auc": round(float(roc_auc_score(y, proba)), 3),
        "profit": expected_profit(y, proba, threshold, eco),
    }


def to_markdown(rows: list[dict], baseline: dict) -> str:
    lines = [
        "Drift simulation: reference = valid, current = distorted test "
        f"(PSI alert threshold: {PSI_ALERT_THRESHOLD}).",
        "",
        "| Scenario | Evidently: dataset drift | PSI alerts (features) | PSI(proba) "
        "| PR-AUC | ROC-AUC | Profit, RUB |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in [baseline, *rows]:
        drift = "YES" if r["dataset_drift"] else "no"
        proba_flag = " (alert)" if r["proba_psi_alert"] else ""
        lines.append(
            f"| {r['scenario']} | {drift} ({r['n_drifted']}/{r['n_features']} cols) "
            f"| {r['n_psi_alerts']} (max {r['max_psi']}: {r['max_psi_feature']}) "
            f"| {r['proba_psi']}{proba_flag} | {r['pr_auc']} | {r['roc_auc']} "
            f"| {r['profit']:,.0f} |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    params = Params.load()
    eco = EconomicsConfig.load()
    processed = Path(params.data.processed_dir)
    reference = pd.read_parquet(processed / "valid.parquet")
    test = pd.read_parquet(processed / "test.parquet")
    model = joblib.load(Path(params.train.model_dir) / "model.joblib")
    proba_reference = model.predict_proba(reference[ALL_FEATURES])[:, 1]

    out_dir = Path(params.evaluate.reports_dir) / "drift"
    out_dir.mkdir(parents=True, exist_ok=True)

    scenarios = {
        "no_drift (control)": test,
        "covariate": scenario_covariate(test),
        "prior": scenario_prior(test),
        "concept": scenario_concept(test),
    }
    results = []
    for name, current in scenarios.items():
        key = name.split(" ")[0]
        row = evaluate_scenario(key, reference, current, model, proba_reference, eco, out_dir)
        results.append(row)
        status = "ALERT" if (row["n_psi_alerts"] > 0 or row["proba_psi_alert"]) else "ok"
        if status == "ALERT":
            print(  # the alerting hook: grep WARNING in cron/CI
                f"WARNING: drift detected in scenario '{key}' "
                f"(feature alerts: {row['n_psi_alerts']}, proba PSI: {row['proba_psi']})"
            )
        else:
            print(f"scenario '{key}': no PSI alerts")

    baseline, *drifted = results
    with open(out_dir / "drift_summary.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    (out_dir / "drift_summary.md").write_text(to_markdown(drifted, baseline), encoding="utf-8")
    print(f"reports written to {out_dir}")


if __name__ == "__main__":
    main()
