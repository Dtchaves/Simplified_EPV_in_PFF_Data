from __future__ import annotations

from contextlib import contextmanager, nullcontext
from pathlib import Path
import sys
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import torch

from ActionSelection.inference import predict_action_selection_probs

from .Pass_epv_missed.soccermap import SoccerMapPassEPVMissed
from .Pass_epv_missed.utils import ToSoccerMapTensor as PassEPVMissedTensorizer
from .Pass_epv_success.soccermap import SoccerMapPassEPVSuccess
from .Pass_epv_success.utils import ToSoccerMapTensor as PassEPVSuccessTensorizer
from .Pass_selection_probability.soccermap import SoccerMapPassSelect
from .Pass_selection_probability.utils import ToSoccerMapTensor as PassSelectionTensorizer
from .Pass_sucess_probability.soccermap import SoccerMapPassSucess
from .data_utils import REPO_ROOT


PASS_MODULE_ROOT = Path(__file__).resolve().parent


def _default_paths() -> Dict[str, Path]:
    return {
        "pass_success": REPO_ROOT / "results" / "models" / "Pass_success_probability.pt",
        "pass_selection": REPO_ROOT / "results" / "models" / "Pass_selection_probability.pt",
        "pe_success": REPO_ROOT / "results" / "models" / "pass_epv_success" / "Pass_epv_success.pt",
        "pe_missed": REPO_ROOT / "results" / "models" / "pass_epv_missed" / "Pass_epv_missed.pt",
    }


@contextmanager
def _temporary_sys_path(path: Optional[Path]):
    if path is None:
        yield
        return

    path_str = str(path)
    sys.path.insert(0, path_str)
    try:
        yield
    finally:
        try:
            sys.path.remove(path_str)
        except ValueError:
            pass


@contextmanager
def _temporary_checkpoint_import_context(path: Optional[Path]):
    module_names = ("soccermap", "utils")
    saved_modules = {name: sys.modules.pop(name) for name in module_names if name in sys.modules}
    try:
        with _temporary_sys_path(path):
            yield
    finally:
        for name in module_names:
            sys.modules.pop(name, None)
        sys.modules.update(saved_modules)


def _load_model(model_path: Path, model_factory, module_dir: Optional[Path] = None):
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    load_context = _temporary_checkpoint_import_context(module_dir) if module_dir is not None else nullcontext()
    with load_context:
        loaded = torch.load(model_path, map_location="cpu", weights_only=False)
    if isinstance(loaded, torch.nn.Module):
        loaded.eval()
        return loaded

    model = model_factory()
    if isinstance(loaded, dict):
        model.load_state_dict(loaded)
    else:
        raise TypeError(f"Unsupported pass model payload type: {type(loaded)!r}")
    model.eval()
    return model


def _normalize_inference_sample(sample: Dict[str, Any]) -> Dict[str, Any]:
    required = ("ball_x_start", "ball_y_start", "team_id", "frame")
    missing = [key for key in required if key not in sample]
    if missing:
        raise KeyError(f"Pass inference sample is missing required keys: {missing}")

    normalized = dict(sample)
    normalized.setdefault("ball_x_end", normalized["ball_x_start"])
    normalized.setdefault("ball_y_end", normalized["ball_y_start"])
    normalized.setdefault("pass_outcome_type", "C")
    return normalized


def _build_ps_input(tensorizer: PassSelectionTensorizer, sample: Dict[str, Any]) -> torch.Tensor:
    matrix, _, _ = tensorizer(_normalize_inference_sample(sample))
    return matrix.unsqueeze(0)


def _build_pe_input(tensorizer: PassEPVSuccessTensorizer | PassEPVMissedTensorizer, sample: Dict[str, Any]) -> torch.Tensor:
    matrix, _, _ = tensorizer(_normalize_inference_sample(sample))
    return matrix.unsqueeze(0)


def _build_destination_mask(
    tensorizer: PassSelectionTensorizer,
    sample: Dict[str, Any],
) -> Optional[torch.Tensor]:
    if "ball_x_end" not in sample or "ball_y_end" not in sample:
        return None
    _, mask, _ = tensorizer(_normalize_inference_sample(sample))
    return mask


def predict_pass_surfaces(
    sample: Dict[str, Any],
    paths: Optional[Dict[str, Path]] = None,
    include_observed_destination_metrics: bool = True,
) -> Dict[str, Any]:
    resolved = _default_paths()
    if paths:
        resolved.update(paths)

    pass_success_model = _load_model(
        resolved["pass_success"],
        lambda: SoccerMapPassSucess(in_channels=13),
        module_dir=PASS_MODULE_ROOT / "Pass_sucess_probability",
    )
    pass_selection_model = _load_model(
        resolved["pass_selection"],
        lambda: SoccerMapPassSelect(in_channels=13),
        module_dir=PASS_MODULE_ROOT / "Pass_selection_probability",
    )
    pe_success_model = _load_model(
        resolved["pe_success"],
        lambda: SoccerMapPassEPVSuccess(in_channels=16),
        module_dir=PASS_MODULE_ROOT / "Pass_epv_success",
    )
    pe_missed_model = _load_model(
        resolved["pe_missed"],
        lambda: SoccerMapPassEPVMissed(in_channels=16),
        module_dir=PASS_MODULE_ROOT / "Pass_epv_missed",
    )

    ps_tensorizer = PassSelectionTensorizer()
    pe_success_tensorizer = PassEPVSuccessTensorizer(pp_model_path=resolved["pass_success"])
    pe_missed_tensorizer = PassEPVMissedTensorizer(pp_model_path=resolved["pass_success"])
    pe_success_tensorizer.pp_model = pass_success_model
    pe_missed_tensorizer.pp_model = pass_success_model

    with torch.no_grad():
        ps_input = _build_ps_input(ps_tensorizer, sample)
        pe_success_input = _build_pe_input(pe_success_tensorizer, sample)
        pe_missed_input = _build_pe_input(pe_missed_tensorizer, sample)

        pass_success_surface = pass_success_model(ps_input)[0, 0].cpu().numpy().astype(float)
        pass_selection_surface = pass_selection_model(ps_input)[0, 0].cpu().numpy().astype(float)
        epv_success_surface = pe_success_model(pe_success_input)[0, 0].cpu().numpy().astype(float)
        epv_missed_surface = pe_missed_model(pe_missed_input)[0, 0].cpu().numpy().astype(float)

    joint_expected_value_surface = (
        (pass_success_surface * epv_success_surface)
        + ((1.0 - pass_success_surface) * epv_missed_surface)
    )
    expected_value_given_pass = float(np.sum(joint_expected_value_surface * pass_selection_surface))

    result: Dict[str, Any] = {
        "expected_value_given_pass": expected_value_given_pass,
        "selection_surface_sum": float(np.sum(pass_selection_surface)),
        "joint_surface_min": float(np.min(joint_expected_value_surface)),
        "joint_surface_max": float(np.max(joint_expected_value_surface)),
    }

    if include_observed_destination_metrics:
        destination_mask = _build_destination_mask(ps_tensorizer, sample)
        if destination_mask is not None:
            destination_mask_np = destination_mask.numpy()[0]
            result.update(
                {
                    "pass_selection_at_observed_destination": float(np.sum(pass_selection_surface * destination_mask_np)),
                    "pass_success_at_observed_destination": float(np.sum(pass_success_surface * destination_mask_np)),
                    "joint_epv_at_observed_destination": float(np.sum(joint_expected_value_surface * destination_mask_np)),
                }
            )

    return {
        **result,
        "pass_success_surface": pass_success_surface,
        "pass_selection_surface": pass_selection_surface,
        "epv_success_surface": epv_success_surface,
        "epv_missed_surface": epv_missed_surface,
        "joint_expected_value_surface": joint_expected_value_surface,
    }


def predict_pass_epv(
    sample: Dict[str, Any],
    action_selection_state: Optional[Dict[str, Any]] = None,
    action_selection_tracking_window: Optional[pd.DataFrame] = None,
    paths: Optional[Dict[str, Path]] = None,
    include_surfaces: bool = False,
) -> Dict[str, Any]:
    surface_output = predict_pass_surfaces(sample=sample, paths=paths)

    result: Dict[str, Any] = {
        "expected_value_given_pass": float(surface_output["expected_value_given_pass"]),
        "selection_surface_sum": float(surface_output["selection_surface_sum"]),
        "joint_surface_min": float(surface_output["joint_surface_min"]),
        "joint_surface_max": float(surface_output["joint_surface_max"]),
    }

    for key in (
        "pass_selection_at_observed_destination",
        "pass_success_at_observed_destination",
        "joint_epv_at_observed_destination",
    ):
        if key in surface_output:
            result[key] = float(surface_output[key])

    if action_selection_state is not None:
        action_probs = predict_action_selection_probs(
            state=action_selection_state,
            tracking_window=action_selection_tracking_window,
        )
        p_pass = float(action_probs["p_pass"])
        result.update(action_probs)
        result["expected_value_weighted_by_action_selection"] = float(
            p_pass * result["expected_value_given_pass"]
        )

    if include_surfaces:
        result.update(
            {
                "pass_success_surface": surface_output["pass_success_surface"],
                "pass_selection_surface": surface_output["pass_selection_surface"],
                "epv_success_surface": surface_output["epv_success_surface"],
                "epv_missed_surface": surface_output["epv_missed_surface"],
                "joint_expected_value_surface": surface_output["joint_expected_value_surface"],
            }
        )

    return result