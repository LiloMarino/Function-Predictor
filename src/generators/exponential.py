from __future__ import annotations

import numpy as np

from . import FunctionDefinition
from .utils import unpack_params


def exponential(x: np.ndarray | float, *params: float) -> np.ndarray:
    a, b, c = unpack_params(params, 3, "exponential")
    x_values = np.asarray(x, dtype=float)
    return a * np.exp(b * x_values) + c


EXPONENTIAL_PARAMS = {"a": (-5.0, 5.0), "b": (-2.0, 2.0), "c": (-5.0, 5.0)}

FUNCTIONS = [
    FunctionDefinition(
        name="exponential",
        category="exponential",
        generator=exponential,
        params=EXPONENTIAL_PARAMS,
    )
]
