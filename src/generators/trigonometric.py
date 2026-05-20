from __future__ import annotations

from typing import overload

import numpy as np

from . import FunctionDefinition, Generator


@overload
def sine(*, a: float, b: float, c: float, d: float) -> Generator: ...


@overload
def sine(**params: float) -> Generator: ...


def sine(**params: float) -> Generator:
    a = float(params["a"])
    b = float(params["b"])
    c = float(params["c"])
    d = float(params["d"])

    def f(x: np.ndarray | float) -> np.ndarray:
        x_values = np.asarray(x, dtype=float)
        return a * np.sin(b * x_values + c) + d

    return f


@overload
def cosine(*, a: float, b: float, c: float, d: float) -> Generator: ...


@overload
def cosine(**params: float) -> Generator: ...


def cosine(**params: float) -> Generator:
    a = float(params["a"])
    b = float(params["b"])
    c = float(params["c"])
    d = float(params["d"])

    def f(x: np.ndarray | float) -> np.ndarray:
        x_values = np.asarray(x, dtype=float)
        return a * np.cos(b * x_values + c) + d

    return f


TRIGONOMETRIC_FUNCTIONS = [
    FunctionDefinition(
        name="sine",
        category="trigonometric",
        factory=sine,
        parameter_names=("a", "b", "c", "d"),
    ),
    FunctionDefinition(
        name="cosine",
        category="trigonometric",
        factory=cosine,
        parameter_names=("a", "b", "c", "d"),
    ),
]
