from __future__ import annotations

from importlib import import_module
from typing import Any

__all__ = [
    "BallDriveDataConfig",
    "build_ball_drive_canonical_dataset",
    "segment_ball_drives",
    "build_ball_drive_split_manifest",
    "BallDriveFeatureBuilder",
    "BallDriveDPModel",
    "BallDriveDEModel",
    "BallDriveTrainer",
    "predict_ball_drive_epv",
]

_EXPORTS = {
    "BallDriveDataConfig": (".data", "BallDriveDataConfig"),
    "build_ball_drive_canonical_dataset": (".data", "build_ball_drive_canonical_dataset"),
    "segment_ball_drives": (".data", "segment_ball_drives"),
    "build_ball_drive_split_manifest": (".data", "build_ball_drive_split_manifest"),
    "BallDriveFeatureBuilder": (".features", "BallDriveFeatureBuilder"),
    "BallDriveDPModel": (".models", "BallDriveDPModel"),
    "BallDriveDEModel": (".models", "BallDriveDEModel"),
    "BallDriveTrainer": (".trainer", "BallDriveTrainer"),
    "predict_ball_drive_epv": (".inference", "predict_ball_drive_epv"),
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
