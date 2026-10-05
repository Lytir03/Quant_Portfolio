"""Discrete random walks and their Brownian scaling limit."""

import numpy as np


def simulate_random_walks(
    n_paths: int,
    n_steps: int,
    rng: np.random.Generator,
    probability_up: float = 0.5,
    initial_position: float = 0.0,
) -> np.ndarray:
    """Simulate simple random walks with +1 / -1 steps.

    Returns an array of shape (n_paths, n_steps + 1) whose first column
    is the initial position.
    """
    if not 0 <= probability_up <= 1:
        raise ValueError("probability_up must lie in [0, 1].")

    steps = rng.choice(
        [-1, 1],
        size=(n_paths, n_steps),
        p=[1 - probability_up, probability_up],
    )

    return np.concatenate(
        [
            np.full((n_paths, 1), initial_position, dtype=float),
            initial_position + np.cumsum(steps, axis=1),
        ],
        axis=1,
    )


def scaled_random_walk(
    n_paths: int,
    steps_per_unit_time: int,
    time_horizon: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Simulate the rescaled symmetric walk W_t = S_{nt} / sqrt(n).

    By Donsker's theorem this converges in distribution to standard
    Brownian motion as n = steps_per_unit_time grows.

    Returns (time, paths) with paths of shape (n_paths, n_steps + 1).
    """
    n_steps = int(round(steps_per_unit_time * time_horizon))
    walks = simulate_random_walks(n_paths, n_steps, rng)
    time = np.linspace(0, time_horizon, n_steps + 1)

    return time, walks / np.sqrt(steps_per_unit_time)
