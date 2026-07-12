"""Model loading and application settings.

The model is loaded ONCE at startup (lifespan), never per request.

Loading order ("auto"):
1. MLflow Model Registry, alias `staging` — if MLFLOW_TRACKING_URI is set;
2. local artifacts from `models/` (produced by `dvc repro`) — the fallback that
   keeps `docker compose up` working on a machine without a populated registry.
"""

import json
from pathlib import Path

import joblib
import pandas as pd
from pydantic_settings import BaseSettings

from churn.config import EconomicsConfig
from churn.economics import optimal_threshold
from churn.features import ALL_FEATURES, build_features
from churn.log import get_logger

logger = get_logger(__name__)


class Settings(BaseSettings):
    mlflow_tracking_uri: str | None = None
    model_uri: str = "models:/churn-scoring@staging"
    model_dir: str = "models"
    database_url: str | None = None
    predictions_log_path: str = "predictions.jsonl"


class ModelService:
    """Holds the loaded model + decision threshold derived from economics config."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.model = None
        self.model_version: str | None = None
        self.model_source: str | None = None
        eco = EconomicsConfig.load()
        self.economics = eco
        self.threshold = optimal_threshold(eco)

    def load(self) -> None:
        if self.settings.mlflow_tracking_uri:
            try:
                self._load_from_mlflow()
                return
            except Exception:
                logger.exception("mlflow registry load failed, falling back to local artifacts")
        self._load_from_local()

    def _load_from_mlflow(self) -> None:
        import mlflow

        mlflow.set_tracking_uri(self.settings.mlflow_tracking_uri)
        self.model = mlflow.sklearn.load_model(self.settings.model_uri)
        self.model_version = self.settings.model_uri
        self.model_source = "mlflow-registry"
        logger.info(
            "model loaded from mlflow",
            extra={"extra_fields": {"uri": self.settings.model_uri}},
        )

    def _load_from_local(self) -> None:
        model_dir = Path(self.settings.model_dir)
        model_path = model_dir / "model.joblib"
        if not model_path.exists():
            logger.error(
                "no model artifact found — run `dvc repro` first",
                extra={"extra_fields": {"path": str(model_path)}},
            )
            return
        self.model = joblib.load(model_path)
        meta_path = model_dir / "metadata.json"
        version = "local"
        if meta_path.exists():
            with open(meta_path, encoding="utf-8") as f:
                meta = json.load(f)
            version = f"{meta.get('model', 'model')}-v{meta.get('registry_version', '?')}"
        self.model_version = version
        self.model_source = "local-artifact"
        logger.info("model loaded from local artifacts", extra={"extra_fields": {"v": version}})

    @property
    def ready(self) -> bool:
        return self.model is not None

    def predict_proba(self, raw: pd.DataFrame) -> pd.Series:
        """Raw request fields -> the SAME build_features() as in training -> proba."""
        featured = build_features(raw)
        proba = self.model.predict_proba(featured[ALL_FEATURES])[:, 1]
        return pd.Series(proba, index=raw.index)
