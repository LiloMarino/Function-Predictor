from __future__ import annotations

from typing import overload

import numpy as np

from . import FunctionDefinition, Generator


@overload
def exponential(*, a: float, b: float, c: float) -> Generator: ...


@overload
def exponential(**params: float) -> Generator: ...


def exponential(**params: float) -> Generator:
    a = float(params["a"])
    b = float(params["b"])
    c = float(params["c"])

    def f(x: np.ndarray | float) -> np.ndarray:
        x_values = np.asarray(x, dtype=float)
        return a * np.exp(b * x_values) + c

    return f


EXPONENTIAL_FUNCTIONS = [
    FunctionDefinition(
        name="exponential",
        category="exponential",
        factory=exponential,
        parameter_names=("a", "b", "c"),
    )
]
