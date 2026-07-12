"""Drift monitoring: Evidently reports for humans, PSI for alerting.

Two complementary tools:
* Evidently DataDriftPreset — rich HTML report for eyeballing what moved;
* hand-rolled PSI (population stability index) — a single number per feature that
  a cron job can threshold without opening any HTML (PSI > 0.2 -> alert).

PSI is also computed for the *predicted probability* distribution: that catches
prior shift even when no single feature moved much.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from churn.features import CATEGORICAL_FEATURES, NUMERIC_FEATURES
from churn.log import get_logger

logger = get_logger(__name__)

PSI_ALERT_THRESHOLD = 0.2


def psi(reference: np.ndarray, current: np.ndarray, bins: int = 10) -> float:
    """PSI between two numeric samples; bins are reference deciles."""
    reference = np.asarray(reference, dtype=float)
    current = np.asarray(current, dtype=float)
    edges = np.unique(np.quantile(reference, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:  # degenerate / constant feature
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf
    ref_frac = np.histogram(reference, edges)[0] / len(reference)
    cur_frac = np.histogram(current, edges)[0] / len(current)
    ref_frac = np.clip(ref_frac, 1e-6, None)
    cur_frac = np.clip(cur_frac, 1e-6, None)
    return float(np.sum((cur_frac - ref_frac) * np.log(cur_frac / ref_frac)))


def psi_categorical(reference: pd.Series, current: pd.Series) -> float:
    categories = sorted(set(reference.unique()) | set(current.unique()))
    ref_frac = np.clip(
        reference.value_counts(normalize=True).reindex(categories).fillna(0).to_numpy(),
        1e-6,
        None,
    )
    cur_frac = np.clip(
        current.value_counts(normalize=True).reindex(categories).fillna(0).to_numpy(),
        1e-6,
        None,
    )
    return float(np.sum((cur_frac - ref_frac) * np.log(cur_frac / ref_frac)))


def psi_table(reference: pd.DataFrame, current: pd.DataFrame) -> pd.DataFrame:
    """PSI per model feature, sorted by severity."""
    rows = []
    for col in NUMERIC_FEATURES:
        rows.append({"feature": col, "psi": psi(reference[col], current[col])})
    for col in CATEGORICAL_FEATURES:
        rows.append({"feature": col, "psi": psi_categorical(reference[col], current[col])})
    table = pd.DataFrame(rows).sort_values("psi", ascending=False).reset_index(drop=True)
    table["alert"] = table["psi"] > PSI_ALERT_THRESHOLD
    return table


def evidently_drift_report(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    out_html: Path,
) -> dict:
    """Evidently DataDriftPreset over model features. Returns a compact summary."""
    from evidently.metric_preset import DataDriftPreset
    from evidently.report import Report

    columns = NUMERIC_FEATURES + CATEGORICAL_FEATURES
    report = Report(metrics=[DataDriftPreset()])
    report.run(
        reference_data=reference[columns],
        current_data=current[columns],
    )
    out_html.parent.mkdir(parents=True, exist_ok=True)
    report.save_html(str(out_html))

    summary = report.as_dict()["metrics"][0]["result"]
    return {
        "n_features": summary["number_of_columns"],
        "n_drifted": summary["number_of_drifted_columns"],
        "share_drifted": round(summary["share_of_drifted_columns"], 3),
        "dataset_drift": bool(summary["dataset_drift"]),
    }
