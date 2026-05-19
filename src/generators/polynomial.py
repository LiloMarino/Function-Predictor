from __future__ import annotations

import numpy as np

from . import FunctionDefinition
from .utils import unpack_params


def linear(x: np.ndarray | float, *params: float) -> np.ndarray:
    a, b = unpack_params(params, 2, "linear")
    x_values = np.asarray(x, dtype=float)
    return a * x_values + b


def quadratic(x: np.ndarray | float, *params: float) -> np.ndarray:
    a, b, c = unpack_params(params, 3, "quadratic")
    x_values = np.asarray(x, dtype=float)
    return a * x_values**2 + b * x_values + c


def cubic(x: np.ndarray | float, *params: float) -> np.ndarray:
    a, b, c, d = unpack_params(params, 4, "cubic")
    x_values = np.asarray(x, dtype=float)
    return a * x_values**3 + b * x_values**2 + c * x_values + d


def quartic(x: np.ndarray | float, *params: float) -> np.ndarray:
    a, b, c, d, e = unpack_params(params, 5, "quartic")
    x_values = np.asarray(x, dtype=float)
    return a * x_values**4 + b * x_values**3 + c * x_values**2 + d * x_values + e


LINEAR_PARAMS = {"a": (-5.0, 5.0), "b": (-5.0, 5.0)}
QUADRATIC_PARAMS = {"a": (-2.0, 2.0), "b": (-5.0, 5.0), "c": (-5.0, 5.0)}
CUBIC_PARAMS = {"a": (-1.0, 1.0), "b": (-3.0, 3.0), "c": (-5.0, 5.0), "d": (-5.0, 5.0)}
QUARTIC_PARAMS = {
    "a": (-0.5, 0.5),
    "b": (-2.0, 2.0),
    "c": (-3.0, 3.0),
    "d": (-5.0, 5.0),
    "e": (-5.0, 5.0),
}

FUNCTIONS = [
    FunctionDefinition(
        name="linear",
        category="polynomial",
        generator=linear,
        params=LINEAR_PARAMS,
    ),
    FunctionDefinition(
        name="quadratic",
        category="polynomial",
        generator=quadratic,
        params=QUADRATIC_PARAMS,
    ),
    FunctionDefinition(
        name="cubic",
        category="polynomial",
        generator=cubic,
        params=CUBIC_PARAMS,
    ),
    FunctionDefinition(
        name="quartic",
        category="polynomial",
        generator=quartic,
        params=QUARTIC_PARAMS,
    ),
]
