"""FastAPI service: turns a calibrated churn probability into a business decision."""

import time
from contextlib import asynccontextmanager

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import PlainTextResponse

from app.db import PredictionLogger
from app.deps import ModelService, Settings
from app.schemas import (
    BatchRequest,
    BatchResponse,
    CustomerFeatures,
    HealthResponse,
    ScoringResponse,
)
from churn import __version__
from churn.economics import expected_value_of_call
from churn.log import get_logger

logger = get_logger(__name__)

_state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = Settings()
    service = ModelService(settings)
    service.load()  # once at startup, not per request
    _state["service"] = service
    _state["pred_logger"] = PredictionLogger(settings.database_url, settings.predictions_log_path)
    _state["n_predictions"] = 0
    _state["n_calls"] = 0
    _state["latencies_ms"] = []
    yield
    _state.clear()


app = FastAPI(
    title="Churn Scoring Platform",
    version=__version__,
    description="Не «кто уйдёт», а «кому звонить, чтобы заработать»: "
    "калиброванная вероятность оттока + порог из unit-экономики.",
    lifespan=lifespan,
)


def _score_one(service: ModelService, customer: CustomerFeatures) -> ScoringResponse:
    raw = pd.DataFrame([customer.model_dump()])
    probability = float(service.predict_proba(raw).iloc[0])
    decision = "call" if probability >= service.threshold else "no_call"
    return ScoringResponse(
        probability=round(probability, 4),
        decision=decision,
        threshold=round(service.threshold, 4),
        expected_value_of_call=round(expected_value_of_call(probability, service.economics), 2),
        model_version=service.model_version or "unknown",
    )


def _get_ready_service() -> ModelService:
    service: ModelService | None = _state.get("service")
    if service is None or not service.ready:
        raise HTTPException(status_code=503, detail="Model is not loaded. Run `dvc repro` first.")
    return service


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    service: ModelService | None = _state.get("service")
    if service is None or not service.ready:
        return HealthResponse(status="no_model", model_version=None, model_source=None)
    return HealthResponse(
        status="ok",
        model_version=service.model_version,
        model_source=service.model_source,
    )


@app.post("/predict", response_model=ScoringResponse)
def predict(customer: CustomerFeatures) -> ScoringResponse:
    service = _get_ready_service()
    started = time.perf_counter()
    response = _score_one(service, customer)
    latency_ms = (time.perf_counter() - started) * 1000
    _state["n_predictions"] += 1
    _state["n_calls"] += int(response.decision == "call")
    _state["latencies_ms"] = _state["latencies_ms"][-9999:] + [latency_ms]
    _state["pred_logger"].log(
        features=customer.model_dump(),
        probability=response.probability,
        decision=response.decision,
        threshold=response.threshold,
        model_version=response.model_version,
    )
    return response


@app.post("/predict/batch", response_model=BatchResponse)
def predict_batch(request: BatchRequest) -> BatchResponse:
    service = _get_ready_service()
    raw = pd.DataFrame([c.model_dump() for c in request.customers])
    probas = service.predict_proba(raw)
    results = []
    for customer, probability in zip(request.customers, probas, strict=True):
        probability = float(probability)
        decision = "call" if probability >= service.threshold else "no_call"
        response = ScoringResponse(
            probability=round(probability, 4),
            decision=decision,
            threshold=round(service.threshold, 4),
            expected_value_of_call=round(expected_value_of_call(probability, service.economics), 2),
            model_version=service.model_version or "unknown",
        )
        _state["pred_logger"].log(
            features=customer.model_dump(),
            probability=response.probability,
            decision=response.decision,
            threshold=response.threshold,
            model_version=response.model_version,
        )
        results.append(response)
    _state["n_predictions"] += len(results)
    n_calls = sum(r.decision == "call" for r in results)
    _state["n_calls"] += n_calls
    return BatchResponse(results=results, n_calls=n_calls)


@app.get("/metrics", response_class=PlainTextResponse)
def metrics() -> str:
    """Prometheus text format, hand-rolled (no extra dependency)."""
    latencies = sorted(_state.get("latencies_ms", []))
    p95 = latencies[int(0.95 * (len(latencies) - 1))] if latencies else 0.0
    lines = [
        "# TYPE churn_predictions_total counter",
        f"churn_predictions_total {_state.get('n_predictions', 0)}",
        "# TYPE churn_call_decisions_total counter",
        f"churn_call_decisions_total {_state.get('n_calls', 0)}",
        "# TYPE churn_predict_latency_ms_p95 gauge",
        f"churn_predict_latency_ms_p95 {p95:.2f}",
    ]
    return "\n".join(lines) + "\n"
