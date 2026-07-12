"""build_features: shape, hand-checked values, determinism, single-row behavior."""

import numpy as np
import pandas as pd

from churn.features import ALL_FEATURES, DERIVED_NUMERIC, build_features
from tests.conftest import make_customers


def _one_row() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "gender": "Male",
                "senior_citizen": "No",
                "partner": "No",
                "dependents": "No",
                "tenure_months": 10,
                "phone_service": "Yes",
                "multiple_lines": "Yes",
                "internet_service": "DSL",
                "online_security": "Yes",
                "online_backup": "No",
                "device_protection": "No",
                "tech_support": "No",
                "streaming_tv": "No",
                "streaming_movies": "No",
                "contract": "Month-to-month",
                "paperless_billing": "Yes",
                "payment_method": "Electronic check",
                "monthly_charges": 60.0,
                "total_charges": 500.0,
            }
        ]
    )


def test_derived_values_hand_checked():
    out = build_features(_one_row()).iloc[0]
    assert out["charges_per_month_hist"] == 50.0  # 500 / 10
    assert out["charge_diff"] == 10.0  # 60 - 50: current bill higher than history
    # phone + multiple_lines + internet + online_security = 4 services
    assert out["num_services"] == 4
    assert out["num_protection_services"] == 1
    assert out["internet_no_protection"] == 0
    assert out["is_new_customer"] == 0
    assert out["month_to_month_new"] == 0
    assert out["avg_service_cost"] == 15.0  # 60 / 4


def test_all_features_present_no_nan():
    df = build_features(make_customers(200, seed=7))
    assert set(ALL_FEATURES).issubset(df.columns)
    assert not df[ALL_FEATURES].isna().any().any()


def test_deterministic():
    raw = make_customers(50, seed=3)
    a = build_features(raw)
    b = build_features(raw)
    pd.testing.assert_frame_equal(a, b)


def test_zero_tenure_no_division_errors():
    row = _one_row()
    row.loc[0, ["tenure_months", "total_charges"]] = [0, 0.0]
    out = build_features(row).iloc[0]
    assert np.isfinite(out[DERIVED_NUMERIC].astype(float)).all()
    assert out["is_new_customer"] == 1
    assert out["month_to_month_new"] == 1


def test_does_not_mutate_input():
    raw = make_customers(20, seed=5)
    snapshot = raw.copy()
    build_features(raw)
    pd.testing.assert_frame_equal(raw, snapshot)
