from dataclasses import dataclass
from typing import Callable

import numpy as np

type GeneratorFn = Callable[[np.ndarray | float, float], np.ndarray]


@dataclass(frozen=True, slots=True)
class FunctionDefinition:
    name: str
    category: str
    generator: GeneratorFn
    params: dict[str, tuple[float, float]]
