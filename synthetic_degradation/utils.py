"""Helpers de muestreo desde rangos definidos en DamageConfig."""

import numpy as np


def rand_int(rng: np.random.Generator, rango: tuple[int, int]) -> int:
    """Entero aleatorio en el rango cerrado [lo, hi]."""
    lo, hi = rango
    return int(rng.integers(lo, hi + 1))


def rand_float(rng: np.random.Generator, rango: tuple[float, float]) -> float:
    """Float aleatorio uniforme en [lo, hi]."""
    lo, hi = rango
    return float(rng.uniform(lo, hi))
