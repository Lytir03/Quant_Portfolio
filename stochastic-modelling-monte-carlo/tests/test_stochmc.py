import numpy as np
import pytest

from stochmc import (
    black_scholes_price,
    european_payoff,
    gbm_expected_price,
    price_european_option_mc,
    scaled_random_walk,
    simulate_gbm_paths,
    simulate_gbm_terminal,
    simulate_random_walks,
    summarise_samples,
)

S0, K, R, SIGMA, T = 100.0, 100.0, 0.05, 0.20, 1.0


@pytest.fixture
def rng():
    return np.random.default_rng(0)


# Random walks

def test_random_walk_shape_and_steps(rng):
    paths = simulate_random_walks(50, 20, rng, initial_position=3)

    assert paths.shape == (50, 21)
    assert np.all(paths[:, 0] == 3)
    assert np.all(np.isin(np.diff(paths, axis=1), [-1, 1]))


def test_random_walk_moments_with_drift(rng):
    p, n_steps = 0.6, 100
    final = simulate_random_walks(100_000, n_steps, rng, probability_up=p)[:, -1]

    assert final.mean() == pytest.approx(n_steps * (2 * p - 1), abs=0.1)
    assert final.var() == pytest.approx(n_steps * 4 * p * (1 - p), rel=0.02)


def test_scaled_random_walk_variance_is_time(rng):
    time, paths = scaled_random_walk(50_000, 400, 2.0, rng)

    assert time[-1] == pytest.approx(2.0)
    assert paths[:, -1].var() == pytest.approx(2.0, rel=0.03)


# GBM

def test_gbm_paths_positive_and_start_at_s0(rng):
    time, prices = simulate_gbm_paths(S0, 0.08, SIGMA, T, 252, 100, rng)

    assert time.shape == (253,)
    assert prices.shape == (100, 253)
    assert np.all(prices[:, 0] == S0)
    assert np.all(prices > 0)


def test_gbm_terminal_mean(rng):
    terminal = simulate_gbm_terminal(S0, 0.08, SIGMA, T, 400_000, rng)
    standard_error = terminal.std(ddof=1) / np.sqrt(terminal.size)

    assert abs(terminal.mean() - gbm_expected_price(S0, 0.08, T)) < 4 * standard_error


def test_risk_neutral_discounted_price_is_martingale(rng):
    terminal = simulate_gbm_terminal(S0, R, SIGMA, T, 400_000, rng)
    discounted = summarise_samples(np.exp(-R * T) * terminal)

    assert abs(discounted.estimate - S0) < 4 * discounted.standard_error


# Black-Scholes

def test_black_scholes_reference_values():
    assert black_scholes_price(S0, K, R, SIGMA, T, "call") == pytest.approx(10.4506, abs=1e-4)
    assert black_scholes_price(S0, K, R, SIGMA, T, "put") == pytest.approx(5.5735, abs=1e-4)


def test_black_scholes_put_call_parity():
    strikes = np.linspace(60, 140, 9)
    call = black_scholes_price(S0, strikes, R, SIGMA, T, "call")
    put = black_scholes_price(S0, strikes, R, SIGMA, T, "put")

    np.testing.assert_allclose(call - put, S0 - strikes * np.exp(-R * T))


def test_black_scholes_rejects_unknown_type():
    with pytest.raises(ValueError):
        black_scholes_price(S0, K, R, SIGMA, T, "straddle")


# Monte Carlo

def test_summarise_samples_matches_formula():
    samples = np.array([1.0, 2.0, 3.0, 4.0])
    result = summarise_samples(samples)

    assert result.estimate == pytest.approx(2.5)
    assert result.standard_error == pytest.approx(samples.std(ddof=1) / 2)
    low, high = result.confidence_interval
    assert low < 2.5 < high


def test_payoffs_are_non_negative():
    prices = np.array([50.0, 100.0, 150.0])

    np.testing.assert_array_equal(european_payoff(prices, 100, "call"), [0, 0, 50])
    np.testing.assert_array_equal(european_payoff(prices, 100, "put"), [50, 0, 0])


@pytest.mark.parametrize("option_type", ["call", "put"])
@pytest.mark.parametrize("antithetic", [False, True])
def test_mc_price_agrees_with_black_scholes(rng, option_type, antithetic):
    result = price_european_option_mc(
        S0, K, R, SIGMA, T, 200_000, rng,
        option_type=option_type,
        antithetic=antithetic,
    )
    exact = black_scholes_price(S0, K, R, SIGMA, T, option_type)

    assert abs(result.estimate - exact) < 4 * result.standard_error


def test_antithetic_reduces_standard_error(rng):
    plain = price_european_option_mc(S0, K, R, SIGMA, T, 100_000, rng)
    anti = price_european_option_mc(S0, K, R, SIGMA, T, 100_000, rng, antithetic=True)

    assert anti.n_samples == plain.n_samples
    assert anti.standard_error < plain.standard_error


def test_antithetic_requires_even_sample_size(rng):
    with pytest.raises(ValueError):
        price_european_option_mc(S0, K, R, SIGMA, T, 1_001, rng, antithetic=True)
