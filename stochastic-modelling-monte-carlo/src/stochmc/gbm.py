"""Exact simulation of Geometric Brownian Motion.

dS_t = mu S_t dt + sigma S_t dW_t has the closed-form solution

    S_t = S_0 exp[(mu - sigma^2 / 2) t + sigma W_t],

so paths can be simulated without discretisation error.
"""

import numpy as np


def simulate_gbm_paths(
    initial_price: float,
    drift: float,
    volatility: float,
    time_horizon: float,
    n_steps: int,
    n_paths: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Simulate full GBM price paths on an equally spaced grid.

    Returns (time, prices) with prices of shape (n_paths, n_steps + 1).
    """
    dt = time_horizon / n_steps
    normal_shocks = rng.standard_normal((n_paths, n_steps))

    log_returns = (
        (drift - 0.5 * volatility**2) * dt
        + volatility * np.sqrt(dt) * normal_shocks
    )

    cumulative_log_returns = np.concatenate(
        [np.zeros((n_paths, 1)), np.cumsum(log_returns, axis=1)],
        axis=1,
    )

    time = np.linspace(0, time_horizon, n_steps + 1)

    return time, initial_price * np.exp(cumulative_log_returns)


def gbm_terminal_prices(
    initial_price: float,
    drift: float,
    volatility: float,
    time_horizon: float,
    normal_shocks: np.ndarray,
) -> np.ndarray:
    """Map standard normal shocks Z to terminal prices S_T in one exact step."""
    return initial_price * np.exp(
        (drift - 0.5 * volatility**2) * time_horizon
        + volatility * np.sqrt(time_horizon) * normal_shocks
    )


def simulate_gbm_terminal(
    initial_price: float,
    drift: float,
    volatility: float,
    time_horizon: float,
    n_paths: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Simulate n_paths terminal prices S_T directly (no intermediate steps)."""
    return gbm_terminal_prices(
        initial_price,
        drift,
        volatility,
        time_horizon,
        rng.standard_normal(n_paths),
    )


def gbm_expected_price(
    initial_price: float,
    drift: float,
    time_horizon: float | np.ndarray,
) -> float | np.ndarray:
    """E[S_T] = S_0 exp(mu T)."""
    return initial_price * np.exp(drift * time_horizon)


def gbm_median_price(
    initial_price: float,
    drift: float,
    volatility: float,
    time_horizon: float | np.ndarray,
) -> float | np.ndarray:
    """Median of S_T = S_0 exp((mu - sigma^2 / 2) T)."""
    return initial_price * np.exp(
        (drift - 0.5 * volatility**2) * time_horizon
    )
