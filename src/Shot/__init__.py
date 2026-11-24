from __future__ import annotations

from importlib import import_module
from typing import Any

__all__ = [
    "BaselineXGArtifacts",
    "BaselineXGConfig",
    "build_baseline_xg_dataset",
    "load_baseline_xg",
    "train_baseline_xg",
    "ShotFeatureBuilder",
    "ShotFeatureConfig",
    "predict_shot_epv",
    "ShotEPVNet",
]

_EXPORTS = {
    "BaselineXGArtifacts": (".baseline_xg", "BaselineXGArtifacts"),
    "BaselineXGConfig": (".baseline_xg", "BaselineXGConfig"),
    "build_baseline_xg_dataset": (".baseline_xg", "build_baseline_xg_dataset"),
    "load_baseline_xg": (".baseline_xg", "load_baseline_xg"),
    "train_baseline_xg": (".baseline_xg", "train_baseline_xg"),
    "ShotFeatureBuilder": (".features", "ShotFeatureBuilder"),
    "ShotFeatureConfig": (".features", "ShotFeatureConfig"),
    "predict_shot_epv": (".inference", "predict_shot_epv"),
    "ShotEPVNet": (".models", "ShotEPVNet"),
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
