from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np

from generators import FunctionDefinition, FunctionInstance, Generator


@dataclass(frozen=True, slots=True)
class SequenceSample:
    function_name: str
    category: str
    generator: Generator
    x_values: np.ndarray
    y_values: np.ndarray


def generate_x_values(
    n_points: int, x_start: float = 0.0, x_step: float = 1.0
) -> np.ndarray:
    if n_points <= 0:
        raise ValueError("n_points must be > 0")
    return x_start + x_step * np.arange(n_points, dtype=float)


def generate_sequence(
    generator: Generator,
    n_points: int | None,
    x_start: float = 0.0,
    x_step: float = 1.0,
    x_values: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    if x_values is None:
        if n_points is None:
            raise ValueError("n_points must be provided when x_values is None")
        x_values = generate_x_values(n_points, x_start, x_step)
    else:
        x_values = np.asarray(x_values, dtype=float)
        if x_values.ndim != 1:
            raise ValueError("x_values must be 1D")
        if n_points is not None and len(x_values) != n_points:
            raise ValueError("n_points must match x_values length")

    y_values = generator(x_values)
    y_values = np.asarray(y_values, dtype=float)
    if y_values.shape != x_values.shape:
        raise ValueError("generator output must match x_values shape")
    return x_values, y_values


def generate_random_functions(
    definitions: Sequence[FunctionDefinition],
    parameter_ranges_by_function: Mapping[str, Mapping[str, tuple[float, float]]],
    n: int,
    rng: np.random.Generator | None = None,
) -> list[FunctionInstance]:
    if n <= 0:
        raise ValueError("n must be > 0")
    if not definitions:
        raise ValueError("definitions must not be empty")

    rng = rng or np.random.default_rng()
    ranges_by_name = _validate_parameter_ranges(
        definitions, parameter_ranges_by_function
    )

    instances: list[FunctionInstance] = []
    for _ in range(n):
        definition = definitions[int(rng.integers(0, len(definitions)))]
        base_ranges = ranges_by_name[definition.name]
        params: dict[str, float] = {}
        for name in definition.parameter_names:
            low, high = base_ranges[name]
            if high <= low:
                raise ValueError(
                    f"invalid range for {definition.name}.{name}: ({low}, {high})"
                )
            params[name] = float(rng.uniform(low, high))

        generator = definition.factory(**params)
        instances.append(
            FunctionInstance(
                name=definition.name,
                category=definition.category,
                generator=generator,
                params=params,
            )
        )

    return instances


def _validate_parameter_ranges(
    definitions: Sequence[FunctionDefinition],
    parameter_ranges_by_function: Mapping[str, Mapping[str, tuple[float, float]]],
) -> dict[str, dict[str, tuple[float, float]]]:
    ranges_by_name: dict[str, dict[str, tuple[float, float]]] = {}
    for definition in definitions:
        if definition.name not in parameter_ranges_by_function:
            raise ValueError(f"missing parameter ranges for {definition.name}")
        ranges = dict(parameter_ranges_by_function[definition.name])
        missing = [name for name in definition.parameter_names if name not in ranges]
        extra = set(ranges) - set(definition.parameter_names)
        if missing or extra:
            details: list[str] = []
            if missing:
                details.append(f"missing {', '.join(missing)}")
            if extra:
                details.append(f"unknown {', '.join(sorted(extra))}")
            detail_text = "; ".join(details)
            raise ValueError(
                f"parameter ranges for {definition.name} are invalid: {detail_text}"
            )
        ranges_by_name[definition.name] = ranges
    return ranges_by_name


def build_windows(
    values: Sequence[float] | np.ndarray,
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


def build_sequences(
    functions: Sequence[FunctionInstance],
    *,
    n_points: int | None = None,
    x_start: float = 0.0,
    x_step: float = 1.0,
    x_range: tuple[float, float] | None = None,
    step: float | None = None,
) -> list[SequenceSample]:
    if not functions:
        return []

    if x_range is not None:
        if step is None:
            raise ValueError("step must be provided when x_range is set")
        if step <= 0:
            raise ValueError("step must be > 0")
        x_start, x_end = x_range
        if x_end <= x_start:
            raise ValueError("x_range must be (start, end) with end > start")
        n_points = int(np.floor((x_end - x_start) / step)) + 1
        x_values = generate_x_values(n_points, x_start, step)
    else:
        if n_points is None:
            raise ValueError("n_points must be provided when x_range is not set")
        x_values = generate_x_values(n_points, x_start, x_step)

    samples: list[SequenceSample] = []
    for instance in functions:
        _, y_values = generate_sequence(
            generator=instance.generator,
            n_points=len(x_values),
            x_values=x_values,
        )
        samples.append(
            SequenceSample(
                function_name=instance.name,
                category=instance.category,
                generator=instance.generator,
                x_values=x_values,
                y_values=y_values,
            )
        )

    return samples


def build_sequence_dataset(
    generator: Generator,
    n_points: int,
    window_size: int,
    horizon: int = 1,
    x_start: float = 0.0,
    x_step: float = 1.0,
    x_values: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    _, y_values = generate_sequence(
        generator=generator,
        n_points=n_points,
        x_start=x_start,
        x_step=x_step,
        x_values=x_values,
    )
    return build_windows(y_values, window_size, horizon)
