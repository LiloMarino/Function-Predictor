from __future__ import annotations

import numpy as np

from . import FunctionDefinition
from .utils import unpack_params


def logarithmic(x: np.ndarray | float, *params: float) -> np.ndarray:
    a, b, c, d = unpack_params(params, 4, "logarithmic")
    x_values = np.asarray(x, dtype=float)
    inner = b * x_values + c
    if np.any(inner <= 0):
        raise ValueError("logarithmic domain error: b * x + c must be > 0")
    return a * np.log(inner) + d


LOGARITHMIC_PARAMS = {
    "a": (-5.0, 5.0),
    "b": (0.1, 3.0),
    "c": (0.1, 5.0),
    "d": (-5.0, 5.0),
}

FUNCTIONS = [
    FunctionDefinition(
        name="logarithmic",
        category="logarithmic",
        generator=logarithmic,
        params=LOGARITHMIC_PARAMS,
    )
]
