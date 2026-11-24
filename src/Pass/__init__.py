from __future__ import annotations

from importlib import import_module
from typing import Any

__all__ = [
    "predict_pass_epv",
    "predict_pass_surfaces",
]

_EXPORTS = {
    "predict_pass_epv": (".inference", "predict_pass_epv"),
    "predict_pass_surfaces": (".inference", "predict_pass_surfaces"),
}


def __getattr__(name: str) -> Any:
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attr_name = _EXPORTS[name]
    value = getattr(import_module(module_name, __name__), attr_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
