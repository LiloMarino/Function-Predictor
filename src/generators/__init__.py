from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np


class Generator(Protocol):
    def __call__(self, x: np.ndarray | float) -> np.ndarray: ...


class GeneratorFactory(Protocol):
    def __call__(self, **params: float) -> Generator: ...


@dataclass(frozen=True, slots=True)
class FunctionDefinition:
    name: str
    category: str
    factory: GeneratorFactory
    parameter_names: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class FunctionInstance:
    name: str
    category: str
    generator: Generator
    params: dict[str, float]
