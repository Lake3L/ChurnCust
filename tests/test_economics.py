"""Sanity checks of the economics layer on synthetic data.

If these formulas are wrong, every business number in the project is wrong,
so they are tested against hand-derived closed forms.
"""

import numpy as np
import pytest

from churn.config import EconomicsConfig
from churn.economics import (
    expected_profit,
    expected_value_of_call,
    optimal_threshold,
    profit_curve,
)

CFG = EconomicsConfig(value_retained=5000, contact_cost=1000, retention_prob=0.3)


def test_optimal_threshold_matches_hand_derivation():
    # P* = C / (p * V) = 1000 / (0.3 * 5000)
    assert optimal_threshold(CFG) == pytest.approx(0.6667, abs=1e-3)


def test_ev_of_call_is_zero_at_threshold():
    t = optimal_threshold(CFG)
    assert expected_value_of_call(t, CFG) == pytest.approx(0.0, abs=1e-9)


def test_call_everyone_profit_closed_form():
    # profit(call all) = N * (base_rate * p * V - C)
    rng = np.random.default_rng(0)
    n, base_rate = 1000, 0.27
    y = (rng.random(n) < base_rate).astype(int)
    proba = rng.random(n)
    profit = expected_profit(y, proba, threshold=0.0, cfg=CFG)
    expected = int(y.sum()) * CFG.retention_prob * CFG.value_retained - n * CFG.contact_cost
    assert profit == pytest.approx(expected)


def test_call_nobody_profit_is_zero():
    y = np.array([0, 1, 1, 0])
    proba = np.array([0.1, 0.9, 0.8, 0.2])
    assert expected_profit(y, proba, threshold=1.01, cfg=CFG) == 0.0


def test_perfect_model_beats_call_everyone():
    rng = np.random.default_rng(1)
    y = (rng.random(2000) < 0.27).astype(int)
    perfect_proba = y.astype(float)
    profit_perfect = expected_profit(y, perfect_proba, optimal_threshold(CFG), CFG)
    profit_all = expected_profit(y, perfect_proba, 0.0, CFG)
    assert profit_perfect > profit_all
    # perfect policy calls exactly the churners: N_churn * (p*V - C)
    expected = int(y.sum()) * (CFG.retention_prob * CFG.value_retained - CFG.contact_cost)
    assert profit_perfect == pytest.approx(expected)


def test_profit_curve_shape_and_ends():
    rng = np.random.default_rng(2)
    y = (rng.random(500) < 0.3).astype(int)
    proba = y * 0.5 + rng.random(500) * 0.4  # max < 0.9, so threshold 1.0 calls nobody
    thresholds, profits = profit_curve(y, proba, CFG)
    assert len(thresholds) == len(profits) == 101
    # at threshold above max(proba) nobody is called -> profit is exactly 0
    assert profits[-1] == 0.0
