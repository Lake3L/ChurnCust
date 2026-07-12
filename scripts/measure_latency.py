"""Latency measurement of the running API: p50/p95/p99 for /predict and /predict/batch.

Usage: start the service (uvicorn or docker compose), then
    uv run python scripts/measure_latency.py [--url http://127.0.0.1:8000] [--n 300]
"""

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests"))

from conftest import make_customers  # noqa: E402  (synthetic but valid customers)


def percentiles(samples_ms: list[float]) -> dict:
    ordered = sorted(samples_ms)

    def pct(p: float) -> float:
        return ordered[min(int(p * (len(ordered) - 1)), len(ordered) - 1)]

    return {
        "p50_ms": round(statistics.median(ordered), 2),
        "p95_ms": round(pct(0.95), 2),
        "p99_ms": round(pct(0.99), 2),
        "mean_ms": round(statistics.fmean(ordered), 2),
        "n": len(ordered),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--n", type=int, default=300)
    args = parser.parse_args()

    health = requests.get(f"{args.url}/health", timeout=5).json()
    if health["status"] != "ok":
        raise SystemExit(f"API is not ready: {health}")

    customers = make_customers(args.n, seed=99).to_dict(orient="records")
    session = requests.Session()

    # warm-up
    for customer in customers[:20]:
        session.post(f"{args.url}/predict", json=customer, timeout=10)

    single_ms = []
    for customer in customers:
        started = time.perf_counter()
        response = session.post(f"{args.url}/predict", json=customer, timeout=10)
        single_ms.append((time.perf_counter() - started) * 1000)
        response.raise_for_status()

    batch_ms = []
    batch = {"customers": customers[:32]}
    for _ in range(50):
        started = time.perf_counter()
        response = session.post(f"{args.url}/predict/batch", json=batch, timeout=30)
        batch_ms.append((time.perf_counter() - started) * 1000)
        response.raise_for_status()

    result = {
        "endpoint_single": percentiles(single_ms),
        "endpoint_batch32": percentiles(batch_ms),
        "model_version": health["model_version"],
    }
    out = Path("reports/latency.json")
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
