"""API contract: status codes, response schema, decision logic, prediction logging."""

import json


def test_health_ok(client):
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["model_source"] == "local-artifact"
    assert body["model_version"]


def test_predict_valid_customer(client, valid_customer):
    response = client.post("/predict", json=valid_customer)
    assert response.status_code == 200
    body = response.json()
    assert set(body) == {
        "probability",
        "decision",
        "threshold",
        "expected_value_of_call",
        "model_version",
    }
    assert 0.0 <= body["probability"] <= 1.0
    assert body["threshold"] == 0.6667
    expected_decision = "call" if body["probability"] >= body["threshold"] else "no_call"
    assert body["decision"] == expected_decision
    # EV formula consistency: P * p * V - C
    assert abs(body["expected_value_of_call"] - (body["probability"] * 0.3 * 5000 - 1000)) < 0.51


def test_predict_invalid_category_422(client, valid_customer):
    bad = {**valid_customer, "contract": "Three year"}
    assert client.post("/predict", json=bad).status_code == 422


def test_predict_out_of_range_422(client, valid_customer):
    bad = {**valid_customer, "tenure_months": -5}
    assert client.post("/predict", json=bad).status_code == 422


def test_predict_extra_field_422(client, valid_customer):
    bad = {**valid_customer, "churn": 1}  # nice try
    assert client.post("/predict", json=bad).status_code == 422


def test_predict_batch(client, valid_customer):
    other = {**valid_customer, "tenure_months": 60, "contract": "Two year"}
    response = client.post("/predict/batch", json={"customers": [valid_customer, other]})
    assert response.status_code == 200
    body = response.json()
    assert len(body["results"]) == 2
    assert body["n_calls"] == sum(r["decision"] == "call" for r in body["results"])


def test_predictions_are_logged(client, valid_customer):
    client.post("/predict", json=valid_customer)
    client.post("/predict", json=valid_customer)
    lines = client.predictions_log.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2
    record = json.loads(lines[0])
    assert record["features"]["contract"] == valid_customer["contract"]
    assert record["decision"] in {"call", "no_call"}
    assert record["model_version"]


def test_metrics_endpoint(client, valid_customer):
    client.post("/predict", json=valid_customer)
    response = client.get("/metrics")
    assert response.status_code == 200
    assert "churn_predictions_total 1" in response.text
    assert "churn_predict_latency_ms_p95" in response.text


def test_no_model_returns_503(tmp_path, monkeypatch, valid_customer):
    from fastapi.testclient import TestClient

    from app.main import app

    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.setenv("MODEL_DIR", str(tmp_path / "empty"))
    monkeypatch.setenv("PREDICTIONS_LOG_PATH", str(tmp_path / "p.jsonl"))
    with TestClient(app) as client:
        assert client.get("/health").json()["status"] == "no_model"
        assert client.post("/predict", json=valid_customer).status_code == 503
