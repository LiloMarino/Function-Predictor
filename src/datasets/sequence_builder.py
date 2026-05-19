from __future__ import annotations

from typing import Callable, Sequence

import numpy as np


def generate_x_values(
    n_points: int, x_start: float = 0.0, x_step: float = 1.0
) -> np.ndarray:
    if n_points <= 0:
        raise ValueError("n_points must be > 0")
    return x_start + x_step * np.arange(n_points, dtype=float)


def generate_sequence(
    generator: Callable,
    params: Sequence[float],
    n_points: int,
    x_start: float = 0.0,
    x_step: float = 1.0,
    x_values: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    if x_values is None:
        x_values = generate_x_values(n_points, x_start, x_step)
    else:
        x_values = np.asarray(x_values, dtype=float)
        if x_values.ndim != 1:
            raise ValueError("x_values must be 1D")
        if n_points is not None and len(x_values) != n_points:
            raise ValueError("n_points must match x_values length")

    y_values = generator(x_values, *params)
    y_values = np.asarray(y_values, dtype=float)
    if y_values.shape != x_values.shape:
        raise ValueError("generator output must match x_values shape")
    return x_values, y_values


def build_windows(
    values: Sequence[float],
    window_size: int,
    horizon: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    if window_size <= 0:
        raise ValueError("window_size must be > 0")
    if horizon <= 0:
        raise ValueError("horizon must be > 0")

    values_array = np.asarray(values, dtype=float)
    if values_array.ndim != 1:
        raise ValueError("values must be 1D")

    max_start = len(values_array) - window_size - horizon + 1
    if max_start <= 0:
        raise ValueError("not enough values to build windows")

    inputs = np.empty((max_start, window_size), dtype=float)
    targets = np.empty((max_start,), dtype=float)
    for idx in range(max_start):
        inputs[idx] = values_array[idx : idx + window_size]
        targets[idx] = values_array[idx + window_size + horizon - 1]

    return inputs, targets


def build_sequence_dataset(
    generator: Callable,
    params: Sequence[float],
    n_points: int,
    window_size: int,
    horizon: int = 1,
    x_start: float = 0.0,
    x_step: float = 1.0,
    x_values: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    _, y_values = generate_sequence(
        generator=generator,
        params=params,
        n_points=n_points,
        x_start=x_start,
        x_step=x_step,
        x_values=x_values,
    )
    return build_windows(y_values, window_size, horizon)
