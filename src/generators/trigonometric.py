from __future__ import annotations

import numpy as np

from . import FunctionDefinition
from .utils import unpack_params


def sine(x: np.ndarray | float, *params: float) -> np.ndarray:
    a, b, c, d = unpack_params(params, 4, "sine")
    x_values = np.asarray(x, dtype=float)
    return a * np.sin(b * x_values + c) + d


def cosine(x: np.ndarray | float, *params: float) -> np.ndarray:
    a, b, c, d = unpack_params(params, 4, "cosine")
    x_values = np.asarray(x, dtype=float)
    return a * np.cos(b * x_values + c) + d


TRIGONOMETRIC_PARAMS = {
    "a": (-5.0, 5.0),
    "b": (0.1, 5.0),
    "c": (-np.pi, np.pi),
    "d": (-5.0, 5.0),
}

FUNCTIONS = [
    FunctionDefinition(
        name="sine",
        category="trigonometric",
        generator=sine,
        params=TRIGONOMETRIC_PARAMS,
    ),
    FunctionDefinition(
        name="cosine",
        category="trigonometric",
        generator=cosine,
        params=TRIGONOMETRIC_PARAMS,
    ),
]
