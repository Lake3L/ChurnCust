"""Shared fixtures.

CI has neither the dataset nor a trained model (data lives under DVC), so the
API tests run against a tiny model trained on synthetic-but-coherent customers.
Tests that need the real data/model skip gracefully when artifacts are absent.
"""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from churn.features import ALL_FEATURES, build_features, make_preprocessor

REPO_ROOT = Path(__file__).resolve().parents[1]
REAL_MODEL_PATH = REPO_ROOT / "models" / "model.joblib"
REAL_VALID_PATH = REPO_ROOT / "data" / "processed" / "valid.parquet"


def make_customers(n: int, seed: int = 42) -> pd.DataFrame:
    """Coherent synthetic customers (dependent services match internet/phone)."""
    rng = np.random.default_rng(seed)
    rows = []
    for _ in range(n):
        internet = rng.choice(["DSL", "Fiber optic", "No"], p=[0.35, 0.4, 0.25])
        phone = rng.choice(["Yes", "No"], p=[0.9, 0.1])

        def internet_dep(p_yes: float, internet: str = internet) -> str:
            if internet == "No":
                return "No internet service"
            return "Yes" if rng.random() < p_yes else "No"

        tenure = int(rng.integers(0, 73))
        monthly = float(np.round(rng.uniform(20, 118), 2))
        total = float(np.round(monthly * tenure * rng.uniform(0.9, 1.1), 2))
        rows.append(
            {
                "gender": str(rng.choice(["Male", "Female"])),
                "senior_citizen": str(rng.choice(["Yes", "No"], p=[0.16, 0.84])),
                "partner": str(rng.choice(["Yes", "No"])),
                "dependents": str(rng.choice(["Yes", "No"], p=[0.3, 0.7])),
                "tenure_months": tenure,
                "phone_service": phone,
                "multiple_lines": (
                    "No phone service" if phone == "No" else str(rng.choice(["Yes", "No"]))
                ),
                "internet_service": internet,
                "online_security": internet_dep(0.35),
                "online_backup": internet_dep(0.4),
                "device_protection": internet_dep(0.4),
                "tech_support": internet_dep(0.35),
                "streaming_tv": internet_dep(0.45),
                "streaming_movies": internet_dep(0.45),
                "contract": str(
                    rng.choice(["Month-to-month", "One year", "Two year"], p=[0.55, 0.2, 0.25])
                ),
                "paperless_billing": str(rng.choice(["Yes", "No"])),
                "payment_method": str(
                    rng.choice(
                        [
                            "Electronic check",
                            "Mailed check",
                            "Bank transfer (automatic)",
                            "Credit card (automatic)",
                        ]
                    )
                ),
                "monthly_charges": monthly,
                "total_charges": total,
            }
        )
    return pd.DataFrame(rows)


def make_labels(df: pd.DataFrame, seed: int = 0) -> np.ndarray:
    """Synthetic churn labels with the same qualitative structure as Telco."""
    rng = np.random.default_rng(seed)
    logit = (
        -1.0
        + 1.4 * (df["contract"] == "Month-to-month").to_numpy()
        + 0.025 * (df["monthly_charges"].to_numpy() - 70)
        - 0.045 * df["tenure_months"].to_numpy()
    )
    p = 1 / (1 + np.exp(-logit))
    return (rng.random(len(df)) < p).astype(int)


@pytest.fixture(scope="session")
def fixture_model_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A tiny but real sklearn pipeline with the same serving contract."""
    model_dir = tmp_path_factory.mktemp("models")
    raw = make_customers(400, seed=1)
    y = make_labels(raw, seed=2)
    featured = build_features(raw)
    model = Pipeline(
        [
            ("prep", make_preprocessor(scale_numeric=True)),
            ("clf", LogisticRegression(max_iter=2000)),
        ]
    )
    model.fit(featured[ALL_FEATURES], y)
    joblib.dump(model, model_dir / "model.joblib")
    (model_dir / "metadata.json").write_text(
        json.dumps({"model": "logreg-fixture", "registry_version": "test"}), encoding="utf-8"
    )
    return model_dir


@pytest.fixture()
def client(fixture_model_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """TestClient over the app with the fixture model and JSONL prediction log."""
    from fastapi.testclient import TestClient

    from app.main import app

    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.setenv("MODEL_DIR", str(fixture_model_dir))
    monkeypatch.setenv("PREDICTIONS_LOG_PATH", str(tmp_path / "predictions.jsonl"))
    with TestClient(app) as test_client:
        test_client.predictions_log = tmp_path / "predictions.jsonl"
        yield test_client


@pytest.fixture()
def valid_customer() -> dict:
    return {
        "gender": "Female",
        "senior_citizen": "No",
        "partner": "Yes",
        "dependents": "No",
        "tenure_months": 2,
        "phone_service": "Yes",
        "multiple_lines": "No",
        "internet_service": "Fiber optic",
        "online_security": "No",
        "online_backup": "No",
        "device_protection": "No",
        "tech_support": "No",
        "streaming_tv": "Yes",
        "streaming_movies": "Yes",
        "contract": "Month-to-month",
        "paperless_billing": "Yes",
        "payment_method": "Electronic check",
        "monthly_charges": 95.7,
        "total_charges": 191.4,
    }
