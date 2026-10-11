"""System registry: `@register_system("mock")` and lookup by the `impl` key of a system YAML."""
from __future__ import annotations

import importlib
from collections.abc import Callable

from ag.systems.base import System

_REGISTRY: dict[str, type[System]] = {}

# impl name -> module that registers it (imported lazily so heavy deps load only when used)
_BUILTIN = {
    "mock": "ag.systems.mock_system",
}


def register_system(name: str) -> Callable[[type[System]], type[System]]:
    def deco(cls: type[System]) -> type[System]:
        if name in _REGISTRY and _REGISTRY[name] is not cls:
            raise ValueError(f"system impl {name!r} registered twice")
        _REGISTRY[name] = cls
        return cls
    return deco


def get_system_class(impl: str) -> type[System]:
    if impl not in _REGISTRY and impl in _BUILTIN:
        importlib.import_module(_BUILTIN[impl])
    if impl not in _REGISTRY:
        raise KeyError(f"unknown system impl {impl!r}; known: {sorted(set(_REGISTRY) | set(_BUILTIN))}")
    return _REGISTRY[impl]


def create_system(impl: str) -> System:
    return get_system_class(impl)()
