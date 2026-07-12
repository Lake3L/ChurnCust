"""Metamorphic test against the real trained model (skipped in CI where DVC data is absent).

Hypothesis from EDA: raising MonthlyCharges with everything else fixed should not
lower churn probability. For a GBDT point-wise monotonicity is NOT guaranteed
(no monotone constraints were imposed), so the test asserts the honest version:

* population-level: the mean probability shift after +25% MonthlyCharges is positive;
* point-wise violations are rare and small (< 15% of customers, drop < 0.05).

If this ever fails, that is a finding to investigate, not a flake: it means the
model has learned a non-monotone price response.
"""

import joblib
import numpy as np
import pytest

from churn.features import ALL_FEATURES, build_features
from tests.conftest import REAL_MODEL_PATH, make_customers


@pytest.mark.skipif(not REAL_MODEL_PATH.exists(), reason="trained model absent (CI)")
def test_higher_monthly_charges_do_not_lower_churn_probability():
    model = joblib.load(REAL_MODEL_PATH)
    base = make_customers(200, seed=21)
    bumped = base.copy()
    bumped["monthly_charges"] = bumped["monthly_charges"] * 1.25

    proba_base = model.predict_proba(build_features(base)[ALL_FEATURES])[:, 1]
    proba_bumped = model.predict_proba(build_features(bumped)[ALL_FEATURES])[:, 1]
    delta = proba_bumped - proba_base

    assert delta.mean() > 0, "population-level monotonicity violated"
    violation_rate = float(np.mean(delta < -0.05))
    assert violation_rate < 0.15, f"too many point-wise violations: {violation_rate:.2%}"
