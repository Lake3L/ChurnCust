"""Prediction logging: Postgres when DATABASE_URL is set, JSONL file otherwise.

Every scored request is persisted (timestamp, input features, probability,
decision, model version) — this is the raw material for drift monitoring.
A logging failure must never break scoring, so all writes are wrapped.
"""

import json
from datetime import UTC, datetime
from pathlib import Path

from sqlalchemy import (
    JSON,
    Column,
    DateTime,
    Float,
    Integer,
    MetaData,
    String,
    Table,
    create_engine,
)

from churn.log import get_logger

logger = get_logger(__name__)

_metadata = MetaData()

predictions_table = Table(
    "predictions",
    _metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("ts", DateTime(timezone=True), nullable=False),
    Column("features", JSON, nullable=False),
    Column("probability", Float, nullable=False),
    Column("decision", String(16), nullable=False),
    Column("threshold", Float, nullable=False),
    Column("model_version", String(128), nullable=False),
)


class PredictionLogger:
    def __init__(self, database_url: str | None, jsonl_path: str) -> None:
        self.engine = None
        self.jsonl_path = Path(jsonl_path)
        if database_url:
            try:
                self.engine = create_engine(database_url, pool_pre_ping=True)
                _metadata.create_all(self.engine)
                logger.info("prediction logging -> postgres")
            except Exception:
                logger.exception("db init failed, falling back to jsonl")
                self.engine = None
        if self.engine is None:
            logger.info(
                "prediction logging -> jsonl",
                extra={"extra_fields": {"path": str(self.jsonl_path)}},
            )

    def log(
        self,
        features: dict,
        probability: float,
        decision: str,
        threshold: float,
        model_version: str,
    ) -> None:
        record = {
            "ts": datetime.now(UTC).isoformat(),
            "features": features,
            "probability": probability,
            "decision": decision,
            "threshold": threshold,
            "model_version": model_version,
        }
        try:
            if self.engine is not None:
                with self.engine.begin() as conn:
                    conn.execute(
                        predictions_table.insert().values(
                            ts=datetime.now(UTC),
                            features=features,
                            probability=probability,
                            decision=decision,
                            threshold=threshold,
                            model_version=model_version,
                        )
                    )
            else:
                with open(self.jsonl_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
        except Exception:
            logger.exception("failed to persist prediction")
