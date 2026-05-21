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
    n_points: int,
    x_start: float = 0.0,
    x_step: float = 1.0,
) -> np.ndarray:
    if n_points <= 0:
        raise ValueError("n_points must be > 0")

    return x_start + x_step * np.arange(n_points, dtype=float)


def generate_sequence(
    generator: Generator,
    x_values: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    x_values = np.asarray(x_values, dtype=float)

    if x_values.ndim != 1:
        raise ValueError("x_values must be 1D")

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
        definitions,
        parameter_ranges_by_function,
    )

    instances: list[FunctionInstance] = []

    for _ in range(n):
        definition = definitions[int(rng.integers(0, len(definitions)))]
        ranges = ranges_by_name[definition.name]

        params: dict[str, float] = {}

        for parameter_name in definition.parameter_names:
            low, high = ranges[parameter_name]

            if high <= low:
                raise ValueError(
                    f"invalid range for "
                    f"{definition.name}.{parameter_name}: ({low}, {high})"
                )

            params[parameter_name] = float(rng.uniform(low, high))

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

        missing = [
            parameter_name
            for parameter_name in definition.parameter_names
            if parameter_name not in ranges
        ]

        extra = set(ranges) - set(definition.parameter_names)

        if missing or extra:
            details: list[str] = []

            if missing:
                details.append(f"missing {', '.join(missing)}")

            if extra:
                details.append(f"unknown {', '.join(sorted(extra))}")

            raise ValueError(
                f"parameter ranges for {definition.name} are invalid: "
                f"{'; '.join(details)}"
            )

        ranges_by_name[definition.name] = ranges

    return ranges_by_name


def build_sequences(
    functions: Sequence[FunctionInstance],
    x_values: np.ndarray,
) -> list[SequenceSample]:
    if not functions:
        return []

    x_values = np.asarray(x_values, dtype=float)

    if x_values.ndim != 1:
        raise ValueError("x_values must be 1D")

    samples: list[SequenceSample] = []

    for instance in functions:
        _, y_values = generate_sequence(
            generator=instance.generator,
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
