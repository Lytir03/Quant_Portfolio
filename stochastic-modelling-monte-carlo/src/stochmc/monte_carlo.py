"""Monte Carlo estimators with standard errors and confidence intervals."""

from dataclasses import dataclass

import numpy as np
from scipy.stats import norm

from stochmc.gbm import gbm_terminal_prices


@dataclass(frozen=True)
class MonteCarloEstimate:
    """A Monte Carlo point estimate together with its sampling uncertainty."""

    estimate: float
    standard_error: float
    n_samples: int
    confidence_level: float = 0.95

    @property
    def half_width(self) -> float:
        z = norm.ppf(0.5 + self.confidence_level / 2)
        return z * self.standard_error

    @property
    def confidence_interval(self) -> tuple[float, float]:
        return (
            self.estimate - self.half_width,
            self.estimate + self.half_width,
        )

    def contains(self, value: float) -> bool:
        low, high = self.confidence_interval
        return low <= value <= high


def summarise_samples(
    samples: np.ndarray,
    confidence_level: float = 0.95,
) -> MonteCarloEstimate:
    """Sample mean, standard error s / sqrt(N) and a CLT confidence interval."""
    samples = np.asarray(samples, dtype=float)
    n_samples = samples.size

    return MonteCarloEstimate(
        estimate=float(samples.mean()),
        standard_error=float(samples.std(ddof=1) / np.sqrt(n_samples)),
        n_samples=n_samples,
        confidence_level=confidence_level,
    )


def european_payoff(
    terminal_prices: np.ndarray,
    strike_price: float,
    option_type: str = "call",
) -> np.ndarray:
    """Payoff max(S_T - K, 0) for a call or max(K - S_T, 0) for a put."""
    if option_type == "call":
        return np.maximum(terminal_prices - strike_price, 0)
    if option_type == "put":
        return np.maximum(strike_price - terminal_prices, 0)
    raise ValueError("option_type must be 'call' or 'put'.")


def price_european_option_mc(
    initial_price: float,
    strike_price: float,
    risk_free_rate: float,
    volatility: float,
    time_to_maturity: float,
    n_samples: int,
    rng: np.random.Generator,
    option_type: str = "call",
    antithetic: bool = False,
    confidence_level: float = 0.95,
) -> MonteCarloEstimate:
    """Price a European option as the discounted risk-neutral expected payoff.

    n_samples is the number of payoff evaluations in both modes, so plain
    and antithetic estimators are compared at equal simulation cost. With
    antithetic=True, n_samples / 2 shocks Z are drawn and each is paired
    with -Z; the standard error is computed from the independent pair means.
    """
    discount_factor = np.exp(-risk_free_rate * time_to_maturity)

    def discounted_payoffs(normal_shocks: np.ndarray) -> np.ndarray:
        terminal_prices = gbm_terminal_prices(
            initial_price,
            risk_free_rate,
            volatility,
            time_to_maturity,
            normal_shocks,
        )
        return discount_factor * european_payoff(
            terminal_prices,
            strike_price,
            option_type,
        )

    if not antithetic:
        samples = discounted_payoffs(rng.standard_normal(n_samples))
        return summarise_samples(samples, confidence_level)

    if n_samples % 2 != 0:
        raise ValueError("n_samples must be even for antithetic sampling.")

    normal_shocks = rng.standard_normal(n_samples // 2)
    pair_means = 0.5 * (
        discounted_payoffs(normal_shocks)
        + discounted_payoffs(-normal_shocks)
    )

    pair_summary = summarise_samples(pair_means, confidence_level)

    return MonteCarloEstimate(
        estimate=pair_summary.estimate,
        standard_error=pair_summary.standard_error,
        n_samples=n_samples,
        confidence_level=confidence_level,
    )
