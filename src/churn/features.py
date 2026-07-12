"""Feature engineering shared by training and serving.

``build_features`` is THE single place where features are derived. The training
pipeline and the FastAPI service call exactly the same function — this is the
protection against train/serve skew.

Rules of this module:
* only deterministic row-wise transformations — no fitting, no dataset-level
  statistics (anything trainable lives in the sklearn Pipeline created by
  ``make_preprocessor`` and is fit on train only);
* the function must work identically for a 7k-row dataframe and a single-row
  request coming from the API.
"""

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

RAW_CATEGORICAL = [
    "gender",
    "senior_citizen",
    "partner",
    "dependents",
    "phone_service",
    "multiple_lines",
    "internet_service",
    "online_security",
    "online_backup",
    "device_protection",
    "tech_support",
    "streaming_tv",
    "streaming_movies",
    "contract",
    "paperless_billing",
    "payment_method",
]
RAW_NUMERIC = ["tenure_months", "monthly_charges", "total_charges"]

# services counted as "subscribed" when the value is exactly "Yes"
_YES_SERVICES = [
    "phone_service",
    "multiple_lines",
    "online_security",
    "online_backup",
    "device_protection",
    "tech_support",
    "streaming_tv",
    "streaming_movies",
]
_PROTECTION_SERVICES = ["online_security", "online_backup", "device_protection", "tech_support"]

DERIVED_NUMERIC = [
    "charges_per_month_hist",
    "charge_diff",
    "num_services",
    "num_protection_services",
    "internet_no_protection",
    "is_new_customer",
    "month_to_month_new",
    "avg_service_cost",
]

CATEGORICAL_FEATURES = RAW_CATEGORICAL
NUMERIC_FEATURES = RAW_NUMERIC + DERIVED_NUMERIC
ALL_FEATURES = CATEGORICAL_FEATURES + NUMERIC_FEATURES


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add derived features. Deterministic, row-wise, no fitting."""
    out = df.copy()
    tenure = out["tenure_months"].clip(lower=1)

    # historical average bill vs the current one: a recent price hike is a churn signal
    out["charges_per_month_hist"] = out["total_charges"] / tenure
    out["charge_diff"] = out["monthly_charges"] - out["charges_per_month_hist"]

    out["num_services"] = (out["internet_service"] != "No").astype(int) + sum(
        (out[c] == "Yes").astype(int) for c in _YES_SERVICES
    )
    out["num_protection_services"] = sum(
        (out[c] == "Yes").astype(int) for c in _PROTECTION_SERVICES
    )
    # internet without any protection/support services — fragile, high-churn segment
    out["internet_no_protection"] = (
        (out["internet_service"] != "No") & (out["num_protection_services"] == 0)
    ).astype(int)

    out["is_new_customer"] = (out["tenure_months"] < 6).astype(int)
    out["month_to_month_new"] = (
        (out["contract"] == "Month-to-month") & (out["is_new_customer"] == 1)
    ).astype(int)

    out["avg_service_cost"] = out["monthly_charges"] / out["num_services"].clip(lower=1)
    return out


def make_preprocessor(scale_numeric: bool) -> ColumnTransformer:
    """Trainable preprocessing (fit on train only, inside the model Pipeline).

    ``scale_numeric=True`` for logistic regression, ``False`` for tree models.
    ``handle_unknown="ignore"`` so an unseen category degrades gracefully in serving.
    """
    numeric: StandardScaler | str = StandardScaler() if scale_numeric else "passthrough"
    return ColumnTransformer(
        transformers=[
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                CATEGORICAL_FEATURES,
            ),
            ("num", numeric, NUMERIC_FEATURES),
        ],
        verbose_feature_names_out=False,
    )


def main() -> None:
    """dvc stage `featurize`: interim parquet -> processed train/valid/test parquet."""
    from pathlib import Path

    from churn.config import Params
    from churn.data import split_data
    from churn.log import get_logger

    logger = get_logger(__name__)
    params = Params.load()
    interim = pd.read_parquet(params.data.interim_path)
    featured = build_features(interim)
    train, valid, test = split_data(featured, params.data)

    out_dir = Path(params.data.processed_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, part in {"train": train, "valid": valid, "test": test}.items():
        part.to_parquet(out_dir / f"{name}.parquet", index=False)
    logger.info(
        "featurized and split",
        extra={
            "extra_fields": {
                "train": len(train),
                "valid": len(valid),
                "test": len(test),
                "n_features": len(ALL_FEATURES),
            }
        },
    )


if __name__ == "__main__":
    main()
