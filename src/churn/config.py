"""Configuration loading: YAML files + environment overrides."""

import os
from pathlib import Path

import yaml
from pydantic import BaseModel


def repo_root() -> Path:
    """Resolve project root: CHURN_ROOT env var or current working directory.

    Every entry point (dvc stage, make target, uvicorn in the container) is
    launched from the repository root / container workdir, so cwd is a safe default.
    """
    return Path(os.getenv("CHURN_ROOT", ".")).resolve()


class EconomicsConfig(BaseModel):
    """Unit economics of the retention campaign (see config/economics.yaml)."""

    value_retained: float
    contact_cost: float
    retention_prob: float

    @classmethod
    def load(cls, path: Path | str | None = None) -> "EconomicsConfig":
        path = Path(path) if path else repo_root() / "config" / "economics.yaml"
        with open(path, encoding="utf-8") as f:
            return cls(**yaml.safe_load(f))


class DataParams(BaseModel):
    raw_path: str
    interim_path: str
    processed_dir: str
    valid_size: float
    test_size: float
    random_state: int


class TrainParams(BaseModel):
    model_dir: str
    optuna_trials: int
    cv_folds: int
    random_state: int
    lightgbm_base: dict


class EvaluateParams(BaseModel):
    reports_dir: str


class Params(BaseModel):
    """Pipeline parameters (params.yaml), shared by dvc stages."""

    data: DataParams
    train: TrainParams
    evaluate: EvaluateParams

    @classmethod
    def load(cls, path: Path | str | None = None) -> "Params":
        path = Path(path) if path else repo_root() / "params.yaml"
        with open(path, encoding="utf-8") as f:
            return cls(**yaml.safe_load(f))
