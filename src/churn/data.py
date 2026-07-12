"""Data loading, cleaning and the stratified train/valid/test split.

The raw file is the full IBM Telco sample (33 columns). Several columns are
dropped as leakage or noise — the reasoning is documented in notebooks/01_eda.ipynb
and in the README:

* ``Churn Reason``  — filled only for churned customers (missing count == number
  of non-churners), pure leakage;
* ``Churn Score``   — output of IBM's own churn model trained on this target;
* ``CLTV``          — computed lifetime value of unknown provenance, dropped
  conservatively;
* ``Churn Label``   — string duplicate of the target ``Churn Value``;
* ``Count/Country/State`` — constants, ``CustomerID`` — identifier;
* geo columns       — single-state dataset, high-cardinality noise for 7k rows.
"""

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from churn.config import DataParams, Params
from churn.log import get_logger

logger = get_logger(__name__)

TARGET = "churn"

LEAKY_COLUMNS = ["Churn Label", "Churn Score", "CLTV", "Churn Reason"]
ID_COLUMNS = ["CustomerID"]
CONSTANT_COLUMNS = ["Count", "Country", "State"]
GEO_COLUMNS = ["City", "Zip Code", "Lat Long", "Latitude", "Longitude"]

RENAME = {
    "Gender": "gender",
    "Senior Citizen": "senior_citizen",
    "Partner": "partner",
    "Dependents": "dependents",
    "Tenure Months": "tenure_months",
    "Phone Service": "phone_service",
    "Multiple Lines": "multiple_lines",
    "Internet Service": "internet_service",
    "Online Security": "online_security",
    "Online Backup": "online_backup",
    "Device Protection": "device_protection",
    "Tech Support": "tech_support",
    "Streaming TV": "streaming_tv",
    "Streaming Movies": "streaming_movies",
    "Contract": "contract",
    "Paperless Billing": "paperless_billing",
    "Payment Method": "payment_method",
    "Monthly Charges": "monthly_charges",
    "Total Charges": "total_charges",
    "Churn Value": TARGET,
}


def load_raw(path: Path | str) -> pd.DataFrame:
    return pd.read_excel(path)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Drop leaky/noise columns, normalize names and dtypes."""
    drop = LEAKY_COLUMNS + ID_COLUMNS + CONSTANT_COLUMNS + GEO_COLUMNS
    out = df.drop(columns=[c for c in drop if c in df.columns]).rename(columns=RENAME)
    # Total Charges is stored as text; blank for brand-new customers (tenure 0)
    out["total_charges"] = pd.to_numeric(out["total_charges"], errors="coerce").fillna(0.0)
    out[TARGET] = out[TARGET].astype(int)
    return out


def split_data(
    df: pd.DataFrame, params: DataParams
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Stratified train/valid/test split (60/20/20 with default params).

    The test part is sealed: it is written once and not touched until the final
    evaluation (see README, project rules).
    """
    train_valid, test = train_test_split(
        df,
        test_size=params.test_size,
        stratify=df[TARGET],
        random_state=params.random_state,
    )
    valid_fraction = params.valid_size / (1.0 - params.test_size)
    train, valid = train_test_split(
        train_valid,
        test_size=valid_fraction,
        stratify=train_valid[TARGET],
        random_state=params.random_state,
    )
    return (
        train.reset_index(drop=True),
        valid.reset_index(drop=True),
        test.reset_index(drop=True),
    )


def main() -> None:
    """dvc stage `prepare`: raw xlsx -> cleaned interim parquet."""
    params = Params.load()
    raw = load_raw(params.data.raw_path)
    interim = clean(raw)
    out_path = Path(params.data.interim_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    interim.to_parquet(out_path, index=False)
    logger.info(
        "prepared interim dataset",
        extra={"extra_fields": {"rows": len(interim), "cols": interim.shape[1]}},
    )


if __name__ == "__main__":
    main()
