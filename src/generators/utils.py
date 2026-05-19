from __future__ import annotations


def unpack_params(
    params: tuple[float, ...], expected: int, name: str
) -> tuple[float, ...]:
    if len(params) != expected:
        raise ValueError(f"{name} expects {expected} params, got {len(params)}")
    return params
