.PHONY: install lint format test repro train evaluate up down drift latency mlflow dashboard

install:
	uv sync --all-groups
	uv run pre-commit install

lint:
	uv run ruff format --check .
	uv run ruff check .

format:
	uv run ruff format .
	uv run ruff check --fix .

test:
	uv run pytest

repro:
	uv run dvc repro

train:
	uv run python -m churn.train

evaluate:
	uv run python -m churn.evaluate

up:
	docker compose -f docker/docker-compose.yml up --build -d

down:
	docker compose -f docker/docker-compose.yml down

drift:
	uv run python scripts/simulate_drift.py

latency:
	uv run python scripts/measure_latency.py

mlflow:
	uv run mlflow ui --backend-store-uri ./mlruns

dashboard:
	uv run streamlit run app/dashboard.py
