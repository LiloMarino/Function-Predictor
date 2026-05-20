from __future__ import annotations

from typing import overload

import numpy as np

from . import FunctionDefinition, Generator


@overload
def logarithmic(*, a: float, b: float, c: float, d: float) -> Generator: ...


@overload
def logarithmic(**params: float) -> Generator: ...


def logarithmic(**params: float) -> Generator:
    a = float(params["a"])
    b = float(params["b"])
    c = float(params["c"])
    d = float(params["d"])

    def f(x: np.ndarray | float) -> np.ndarray:
        x_values = np.asarray(x, dtype=float)
        inner = b * x_values + c
        if np.any(inner <= 0):
            raise ValueError("logarithmic domain error: b * x + c must be > 0")
        return a * np.log(inner) + d

    return f


LOGARITHMIC_FUNCTIONS = [
    FunctionDefinition(
        name="logarithmic",
        category="logarithmic",
        factory=logarithmic,
        parameter_names=("a", "b", "c", "d"),
    )
]
