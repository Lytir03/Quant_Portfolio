"""Analytical Black-Scholes prices for European options."""

import numpy as np
from scipy.stats import norm


def black_scholes_price(
    initial_price: float,
    strike_price: float | np.ndarray,
    risk_free_rate: float,
    volatility: float,
    time_to_maturity: float,
    option_type: str = "call",
) -> float | np.ndarray:
    """Black-Scholes price of a European call or put (vectorised over strike)."""
    if option_type not in ("call", "put"):
        raise ValueError("option_type must be 'call' or 'put'.")

    sqrt_t = np.sqrt(time_to_maturity)

    d1 = (
        np.log(initial_price / strike_price)
        + (risk_free_rate + 0.5 * volatility**2) * time_to_maturity
    ) / (volatility * sqrt_t)

    d2 = d1 - volatility * sqrt_t

    discount_factor = np.exp(-risk_free_rate * time_to_maturity)

    if option_type == "call":
        return (
            initial_price * norm.cdf(d1)
            - strike_price * discount_factor * norm.cdf(d2)
        )

    return (
        strike_price * discount_factor * norm.cdf(-d2)
        - initial_price * norm.cdf(-d1)
    )
