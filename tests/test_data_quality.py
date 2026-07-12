"""Data contract: cleaning logic on synthetic raw data + checks on the real dataset.

Real-data checks are skipped when the DVC-managed files are absent (e.g. in CI).
"""

import pandas as pd
import pytest

from churn.config import DataParams
from churn.data import LEAKY_COLUMNS, TARGET, clean, split_data
from tests.conftest import REPO_ROOT, make_customers, make_labels

INTERIM_PATH = REPO_ROOT / "data" / "interim" / "telco_clean.parquet"

ALLOWED = {
    "gender": {"Male", "Female"},
    "contract": {"Month-to-month", "One year", "Two year"},
    "internet_service": {"DSL", "Fiber optic", "No"},
    "payment_method": {
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)",
    },
}


def _raw_like() -> pd.DataFrame:
    """A miniature of the raw IBM file: original column names, text Total Charges."""
    return pd.DataFrame(
        {
            "CustomerID": ["a", "b"],
            "Count": [1, 1],
            "Country": ["United States"] * 2,
            "State": ["California"] * 2,
            "City": ["LA", "SF"],
            "Zip Code": [90001, 94101],
            "Lat Long": ["1, 2", "3, 4"],
            "Latitude": [1.0, 3.0],
            "Longitude": [2.0, 4.0],
            "Gender": ["Male", "Female"],
            "Senior Citizen": ["No", "Yes"],
            "Partner": ["No", "Yes"],
            "Dependents": ["No", "No"],
            "Tenure Months": [0, 24],
            "Phone Service": ["Yes", "Yes"],
            "Multiple Lines": ["No", "Yes"],
            "Internet Service": ["DSL", "Fiber optic"],
            "Online Security": ["Yes", "No"],
            "Online Backup": ["No", "No"],
            "Device Protection": ["No", "Yes"],
            "Tech Support": ["No", "No"],
            "Streaming TV": ["No", "Yes"],
            "Streaming Movies": ["No", "Yes"],
            "Contract": ["Month-to-month", "Two year"],
            "Paperless Billing": ["Yes", "No"],
            "Payment Method": ["Electronic check", "Credit card (automatic)"],
            "Monthly Charges": [45.0, 99.5],
            "Total Charges": [" ", "2388.0"],  # blank for the brand-new customer
            "Churn Label": ["Yes", "No"],
            "Churn Value": [1, 0],
            "Churn Score": [91, 20],
            "CLTV": [3000, 5500],
            "Churn Reason": ["Price", None],
        }
    )


def test_clean_drops_all_leaky_columns():
    out = clean(_raw_like())
    for col in LEAKY_COLUMNS + ["CustomerID", "City", "Latitude"]:
        assert col not in out.columns
    assert TARGET in out.columns


def test_clean_fixes_total_charges():
    out = clean(_raw_like())
    assert out["total_charges"].dtype == "float64"
    assert out.loc[0, "total_charges"] == 0.0  # blank -> 0 for tenure=0
    assert out.loc[1, "total_charges"] == 2388.0


def test_split_is_deterministic_and_stratified():
    df = make_customers(1000, seed=11)
    df[TARGET] = make_labels(df, seed=12)
    params = DataParams(
        raw_path="",
        interim_path="",
        processed_dir="",
        valid_size=0.2,
        test_size=0.2,
        random_state=42,
    )
    train1, valid1, test1 = split_data(df, params)
    train2, valid2, test2 = split_data(df, params)
    pd.testing.assert_frame_equal(train1, train2)
    pd.testing.assert_frame_equal(valid1, valid2)
    pd.testing.assert_frame_equal(test1, test2)

    assert len(train1) + len(valid1) + len(test1) == len(df)
    base = df[TARGET].mean()
    for part in (train1, valid1, test1):
        assert abs(part[TARGET].mean() - base) < 0.03


@pytest.mark.skipif(not INTERIM_PATH.exists(), reason="DVC data not present (CI)")
def test_real_interim_contract():
    df = pd.read_parquet(INTERIM_PATH)
    assert len(df) == 7043
    assert not df.isna().any().any()
    assert set(df[TARGET].unique()) == {0, 1}
    assert 0.20 <= df[TARGET].mean() <= 0.35
    assert df["tenure_months"].between(0, 120).all()
    assert df["monthly_charges"].between(0, 1000).all()
    assert df["total_charges"].between(0, 200000).all()
    for col, allowed in ALLOWED.items():
        assert set(df[col].unique()).issubset(allowed), col
