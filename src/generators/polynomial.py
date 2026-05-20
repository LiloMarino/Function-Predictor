from __future__ import annotations

from typing import overload

import numpy as np

from . import FunctionDefinition, Generator


@overload
def linear(*, a: float, b: float) -> Generator: ...


@overload
def linear(**params: float) -> Generator: ...


def linear(**params: float) -> Generator:
    a = float(params["a"])
    b = float(params["b"])

    def f(x: np.ndarray | float) -> np.ndarray:
        x_values = np.asarray(x, dtype=float)
        return a * x_values + b

    return f


@overload
def quadratic(*, a: float, b: float, c: float) -> Generator: ...


@overload
def quadratic(**params: float) -> Generator: ...


def quadratic(**params: float) -> Generator:
    a = float(params["a"])
    b = float(params["b"])
    c = float(params["c"])

    def f(x: np.ndarray | float) -> np.ndarray:
        x_values = np.asarray(x, dtype=float)
        return a * x_values**2 + b * x_values + c

    return f


@overload
def cubic(*, a: float, b: float, c: float, d: float) -> Generator: ...


@overload
def cubic(**params: float) -> Generator: ...


def cubic(**params: float) -> Generator:
    a = float(params["a"])
    b = float(params["b"])
    c = float(params["c"])
    d = float(params["d"])

    def f(x: np.ndarray | float) -> np.ndarray:
        x_values = np.asarray(x, dtype=float)
        return a * x_values**3 + b * x_values**2 + c * x_values + d

    return f


@overload
def quartic(*, a: float, b: float, c: float, d: float, e: float) -> Generator: ...


@overload
def quartic(**params: float) -> Generator: ...


def quartic(**params: float) -> Generator:
    a = float(params["a"])
    b = float(params["b"])
    c = float(params["c"])
    d = float(params["d"])
    e = float(params["e"])

    def f(x: np.ndarray | float) -> np.ndarray:
        x_values = np.asarray(x, dtype=float)
        return a * x_values**4 + b * x_values**3 + c * x_values**2 + d * x_values + e

    return f


POLYNOMIAL_FUNCTIONS = [
    FunctionDefinition(
        name="linear",
        category="polynomial",
        factory=linear,
        parameter_names=("a", "b"),
    ),
    FunctionDefinition(
        name="quadratic",
        category="polynomial",
        factory=quadratic,
        parameter_names=("a", "b", "c"),
    ),
    FunctionDefinition(
        name="cubic",
        category="polynomial",
        factory=cubic,
        parameter_names=("a", "b", "c", "d"),
    ),
    FunctionDefinition(
        name="quartic",
        category="polynomial",
        factory=quartic,
        parameter_names=("a", "b", "c", "d", "e"),
    ),
]
