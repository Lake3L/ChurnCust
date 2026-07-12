"""Cost-sensitive decision layer: expected profit and the economically optimal threshold.

The retention campaign economics:
    EV(call | P) = P * (p * V - C) + (1 - P) * (-C) = P * p * V - C

where P is the (calibrated!) churn probability, V = value_retained,
C = contact_cost, p = retention_prob. Calling is profitable when EV > 0:

    P > C / (p * V)

With the default config: 1000 / (0.3 * 5000) = 0.667 — not 0.5.
"""

import numpy as np

from churn.config import EconomicsConfig


def optimal_threshold(cfg: EconomicsConfig) -> float:
    """Probability above which calling a customer has positive expected value."""
    return cfg.contact_cost / (cfg.retention_prob * cfg.value_retained)


def expected_value_of_call(proba: float, cfg: EconomicsConfig) -> float:
    """Expected profit (RUB) of calling one customer with churn probability `proba`."""
    return proba * cfg.retention_prob * cfg.value_retained - cfg.contact_cost


def policy_profit(y_true: np.ndarray, calls: np.ndarray, cfg: EconomicsConfig) -> float:
    """Expected profit (RUB) of an arbitrary calling policy given as a boolean mask.

    Evaluated against true outcomes: a called customer who was really churning
    yields p * V - C in expectation (we retain them with probability p),
    a called customer who was staying anyway costs C for nothing.
    Customers we do not call contribute 0 (their churn loss is the status quo baseline).
    """
    y_true = np.asarray(y_true)
    calls = np.asarray(calls, dtype=bool)
    n_called = int(calls.sum())
    n_churners_called = int((y_true[calls] == 1).sum())
    return n_churners_called * cfg.retention_prob * cfg.value_retained - n_called * cfg.contact_cost


def expected_profit(
    y_true: np.ndarray,
    proba: np.ndarray,
    threshold: float,
    cfg: EconomicsConfig,
) -> float:
    """Expected profit (RUB) of the policy "call everyone with proba >= threshold"."""
    return policy_profit(y_true, np.asarray(proba) >= threshold, cfg)


def profit_curve(
    y_true: np.ndarray,
    proba: np.ndarray,
    cfg: EconomicsConfig,
    thresholds: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Profit of the calling policy for a grid of thresholds (for the profit-vs-threshold plot)."""
    if thresholds is None:
        thresholds = np.linspace(0.0, 1.0, 101)
    profits = np.array([expected_profit(y_true, proba, t, cfg) for t in thresholds])
    return thresholds, profits
