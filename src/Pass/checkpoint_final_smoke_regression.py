from __future__ import annotations

import importlib
import importlib.util
import json
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import matplotlib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from reward_labels import PassRewardLabeler

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[2]

PP_DIR = ROOT / "src/Pass/Pass_sucess_probability"
PS_DIR = ROOT / "src/Pass/Pass_selection_probability"
PE_SUCCESS_DIR = ROOT / "src/Pass/Pass_epv_success"
PE_MISSED_DIR = ROOT / "src/Pass/Pass_epv_missed"

MISSED_PASS_OUTCOMES = {"D", "B", "O", "S", "G", "I"}


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _load_soccermap_class(package_dir: Path, class_name: str):
    package_str = str(package_dir)
    sys.path.insert(0, package_str)
    try:
        for module_name in ("soccermap", "utils", "dataloader", "trainer", "test"):
            if module_name in sys.modules:
                del sys.modules[module_name]
        module = importlib.import_module("soccermap")
        pixel_fn = getattr(module, "pixel")
        return getattr(module, class_name), pixel_fn
    finally:
        if sys.path and sys.path[0] == package_str:
            sys.path.pop(0)


def _resolve_carrier_velocity(row: pd.Series, frame: pd.DataFrame) -> Tuple[float, float]:
    player_id = int(row["player_id"])

    for col in frame.columns:
        if not col.startswith("original_pId_player_"):
            continue
        raw_value = frame.iloc[0][col]
        if pd.isna(raw_value):
            continue
        if int(raw_value) != player_id:
            continue

        suffix = col.replace("original_pId_player_", "")
        vx_col = f"vx_player_{suffix}"
        vy_col = f"vy_player_{suffix}"

        vx_val = frame.iloc[0][vx_col] if vx_col in frame.columns else 0.0
        vy_val = frame.iloc[0][vy_col] if vy_col in frame.columns else 0.0

        vx = float(vx_val) if pd.notna(vx_val) else 0.0
        vy = float(vy_val) if pd.notna(vy_val) else 0.0
        return vx, vy

    return 0.0, 0.0


def _is_valid_coords(row: pd.Series) -> bool:
    return (
        pd.notna(row.get("ball_x_start"))
        and pd.notna(row.get("ball_y_start"))
        and pd.notna(row.get("ball_x_end"))
        and pd.notna(row.get("ball_y_end"))
    )


def _sample_payload(row: pd.Series, frame: pd.DataFrame) -> Dict:
    vx_carrier, vy_carrier = _resolve_carrier_velocity(row, frame)
    return {
        "ball_x_start": float(row["ball_x_start"]),
        "ball_y_start": float(row["ball_y_start"]),
        "ball_x_end": float(row["ball_x_end"]),
        "ball_y_end": float(row["ball_y_end"]),
        "pass_outcome_type": str(row["pass_outcome_type"]),
        "team_id": int(row["team_id"]),
        "vx_carrier": vx_carrier,
        "vy_carrier": vy_carrier,
        "frame": frame,
    }


def _collect_base_samples(max_samples: int) -> List[Dict]:
    pass_files = sorted((ROOT / "passes").glob("final_pass_track_*.csv"))
    if not pass_files:
        raise FileNotFoundError("No pass CSV files found under 'passes/'.")

    samples: List[Dict] = []
    for pass_file in pass_files:
        if len(samples) >= max_samples:
            break

        df = pd.read_csv(pass_file)
        df = df[df["pass_outcome_type"].notna()].copy()
        if df.empty:
            continue

        for idx, row in df.iterrows():
            if len(samples) >= max_samples:
                break
            if not _is_valid_coords(row):
                continue

            frame = df.loc[[idx]].copy()
            samples.append(
                {
                    "source_file": pass_file.name,
                    "row_index": int(idx),
                    "sample": _sample_payload(row, frame),
                }
            )

    if not samples:
        raise RuntimeError("Could not collect PP/PS smoke samples.")

    return samples


def _collect_pe_samples(
    labeler: PassRewardLabeler,
    max_success_samples: int,
    max_missed_samples: int,
) -> Tuple[List[Dict], List[Dict]]:
    pass_files = sorted((ROOT / "passes").glob("final_pass_track_*.csv"))
    if not pass_files:
        raise FileNotFoundError("No pass CSV files found under 'passes/'.")

    success_samples: List[Dict] = []
    missed_samples: List[Dict] = []

    for pass_file in pass_files:
        if len(success_samples) >= max_success_samples and len(missed_samples) >= max_missed_samples:
            break

        df = pd.read_csv(pass_file)
        df = df[df["pass_outcome_type"].notna()].copy()
        if df.empty:
            continue

        labeled_df, _ = labeler.label_pass_dataframe(
            df,
            source_filename=pass_file.name,
            drop_unlabeled=True,
        )
        if labeled_df.empty:
            continue

        for idx, row in labeled_df.iterrows():
            outcome = str(row["pass_outcome_type"])
            if not _is_valid_coords(row):
                continue

            target_bucket = None
            if outcome == "C" and len(success_samples) < max_success_samples:
                target_bucket = success_samples
            elif outcome in MISSED_PASS_OUTCOMES and len(missed_samples) < max_missed_samples:
                target_bucket = missed_samples

            if target_bucket is None:
                continue

            frame = labeled_df.loc[[idx]].copy()
            target_bucket.append(
                {
                    "source_file": pass_file.name,
                    "row_index": int(idx),
                    "reward_label": int(row["reward_label"]),
                    "reward_status": str(row.get("reward_status", "")),
                    "sample": _sample_payload(row, frame),
                }
            )

            if len(success_samples) >= max_success_samples and len(missed_samples) >= max_missed_samples:
                break

    if not success_samples:
        raise RuntimeError("Could not collect PE-success smoke samples.")
    if not missed_samples:
        raise RuntimeError("Could not collect PE-missed smoke samples.")

    return success_samples, missed_samples


def _split_train_eval(records: Sequence[Dict], train_size: int, eval_size: int) -> Tuple[List[Dict], List[Dict]]:
    if len(records) < (train_size + eval_size):
        if len(records) < 4:
            raise RuntimeError("Not enough smoke samples to create train/eval split.")
        dynamic_train = max(2, int(round(0.7 * len(records))))
        return list(records[:dynamic_train]), list(records[dynamic_train:])

    return list(records[:train_size]), list(records[train_size : train_size + eval_size])


def _stack_pp_or_ps_batch(converter, records: Sequence[Dict]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    matrices: List[torch.Tensor] = []
    masks: List[torch.Tensor] = []
    labels: List[float] = []

    for record in records:
        matrix, mask, target = converter(record["sample"])
        matrices.append(matrix)
        masks.append(mask)
        labels.append(float(target.item()))

    return (
        torch.stack(matrices),
        torch.stack(masks),
        torch.tensor(labels, dtype=torch.float32),
    )


def _stack_pe_batch(
    converter,
    records: Sequence[Dict],
    pp_converter,
    pp_model,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    matrices: List[torch.Tensor] = []
    masks: List[torch.Tensor] = []
    labels: List[float] = []
    max_channel_diff = 0.0

    with torch.no_grad():
        for record in records:
            sample = record["sample"]

            pp_matrix, _, _ = pp_converter(sample)
            pp_surface = pp_model(pp_matrix.unsqueeze(0))[0, 0].cpu().numpy().astype(float)

            matrix, mask, _ = converter(sample)
            channel_16 = matrix.numpy()[15]
            channel_diff = float(np.max(np.abs(channel_16 - pp_surface)))
            max_channel_diff = max(max_channel_diff, channel_diff)

            matrices.append(matrix)
            masks.append(mask)
            labels.append(float(record["reward_label"]))

    return (
        torch.stack(matrices),
        torch.stack(masks),
        torch.tensor(labels, dtype=torch.float32),
        max_channel_diff,
    )


def _save_surface_plot(surface: np.ndarray, title: str, output_path: Path, cmap: str, vmin=None, vmax=None) -> None:
    plt.figure(figsize=(8, 5))
    plt.imshow(surface, cmap=cmap, aspect="auto", vmin=vmin, vmax=vmax)
    plt.colorbar()
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def run_final_smoke_regression_checkpoint() -> None:
    torch.manual_seed(42)
    np.random.seed(42)

    pp_utils = _load_module("pp_utils_smoke", PP_DIR / "utils.py")
    ps_utils = _load_module("ps_utils_smoke", PS_DIR / "utils.py")
    pe_success_utils = _load_module("pe_success_utils_smoke", PE_SUCCESS_DIR / "utils.py")
    pe_missed_utils = _load_module("pe_missed_utils_smoke", PE_MISSED_DIR / "utils.py")

    pp_class, pp_pixel = _load_soccermap_class(PP_DIR, "SoccerMapPassSucess")
    ps_class, ps_pixel = _load_soccermap_class(PS_DIR, "SoccerMapPassSelect")
    pe_success_class, pe_success_pixel = _load_soccermap_class(PE_SUCCESS_DIR, "SoccerMapPassEPVSuccess")
    pe_missed_class, pe_missed_pixel = _load_soccermap_class(PE_MISSED_DIR, "SoccerMapPassEPVMissed")

    metrics_dir = ROOT / "results/metrics/final_smoke_checkpoint"
    heatmaps_dir = ROOT / "results/heatmaps/final_smoke_checkpoint"
    models_dir = ROOT / "results/models/final_smoke_checkpoint"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    heatmaps_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    base_samples = _collect_base_samples(max_samples=20)
    labeler = PassRewardLabeler(
        event_root=ROOT / "data/raw/event",
        horizon_seconds=15.0,
        include_open_play_null=True,
    )
    pe_success_samples, pe_missed_samples = _collect_pe_samples(
        labeler=labeler,
        max_success_samples=20,
        max_missed_samples=20,
    )

    base_train, base_eval = _split_train_eval(base_samples, train_size=12, eval_size=6)
    pe_success_train, pe_success_eval = _split_train_eval(pe_success_samples, train_size=12, eval_size=6)
    pe_missed_train, pe_missed_eval = _split_train_eval(pe_missed_samples, train_size=12, eval_size=6)

    pp_converter = pp_utils.ToSoccerMapTensor()
    ps_converter = ps_utils.ToSoccerMapTensor()

    pp_model = pp_class(in_channels=13)
    ps_model = ps_class(in_channels=13)

    bce_loss = nn.BCELoss()
    mse_loss = nn.MSELoss()

    # PP smoke pass
    pp_train_x, pp_train_mask, pp_train_y = _stack_pp_or_ps_batch(pp_converter, base_train)
    pp_eval_x, pp_eval_mask, pp_eval_y = _stack_pp_or_ps_batch(pp_converter, base_eval)

    pp_optimizer = torch.optim.AdamW(pp_model.parameters(), lr=1e-3, weight_decay=1e-4)
    pp_model.train()
    pp_optimizer.zero_grad()
    pp_train_surface = pp_model(pp_train_x)
    pp_train_pred = pp_pixel(pp_train_surface, pp_train_mask).view(-1)
    pp_train_loss = bce_loss(pp_train_pred, pp_train_y)
    pp_train_loss.backward()
    pp_optimizer.step()

    pp_model.eval()
    with torch.no_grad():
        pp_eval_surface = pp_model(pp_eval_x)
        pp_eval_pred = pp_pixel(pp_eval_surface, pp_eval_mask).view(-1)
        pp_eval_loss = bce_loss(pp_eval_pred, pp_eval_y)

    assert float(pp_eval_surface.min().item()) >= -1e-6
    assert float(pp_eval_surface.max().item()) <= 1.0 + 1e-6

    # PS smoke pass
    ps_train_x, ps_train_mask, _ = _stack_pp_or_ps_batch(ps_converter, base_train)
    ps_eval_x, ps_eval_mask, _ = _stack_pp_or_ps_batch(ps_converter, base_eval)
    ps_train_y = torch.ones(len(ps_train_x), dtype=torch.float32)
    ps_eval_y = torch.ones(len(ps_eval_x), dtype=torch.float32)

    ps_optimizer = torch.optim.AdamW(ps_model.parameters(), lr=1e-3, weight_decay=1e-4)
    ps_model.train()
    ps_optimizer.zero_grad()
    ps_train_surface = ps_model(ps_train_x)
    ps_train_pred = ps_pixel(ps_train_surface, ps_train_mask).view(-1)
    ps_train_loss = bce_loss(ps_train_pred, ps_train_y)
    ps_train_loss.backward()
    ps_optimizer.step()

    ps_model.eval()
    with torch.no_grad():
        ps_eval_surface = ps_model(ps_eval_x)
        ps_eval_pred = ps_pixel(ps_eval_surface, ps_eval_mask).view(-1)
        ps_eval_loss = bce_loss(ps_eval_pred, ps_eval_y)

    ps_eval_sums = ps_eval_surface.sum(dim=(2, 3)).view(-1)
    ps_max_sum_dev = float(torch.max(torch.abs(ps_eval_sums - 1.0)).item())
    assert ps_max_sum_dev < 1e-5, f"PS spatial normalization drifted: {ps_max_sum_dev}"

    # PE smoke passes with explicit PP->PE wiring.
    pe_success_converter = pe_success_utils.ToSoccerMapTensor()
    pe_missed_converter = pe_missed_utils.ToSoccerMapTensor()
    pe_success_converter.pp_model = pp_model
    pe_missed_converter.pp_model = pp_model

    pe_success_model = pe_success_class(in_channels=16)
    pe_missed_model = pe_missed_class(in_channels=16)

    (
        pe_success_train_x,
        pe_success_train_mask,
        pe_success_train_y,
        pe_success_train_pp_diff,
    ) = _stack_pe_batch(pe_success_converter, pe_success_train, pp_converter, pp_model)
    (
        pe_success_eval_x,
        pe_success_eval_mask,
        pe_success_eval_y,
        pe_success_eval_pp_diff,
    ) = _stack_pe_batch(pe_success_converter, pe_success_eval, pp_converter, pp_model)

    pe_success_optimizer = torch.optim.AdamW(pe_success_model.parameters(), lr=1e-3, weight_decay=1e-4)
    pe_success_model.train()
    pe_success_optimizer.zero_grad()
    pe_success_train_surface = pe_success_model(pe_success_train_x)
    pe_success_train_pred = pe_success_pixel(pe_success_train_surface, pe_success_train_mask).view(-1)
    pe_success_train_loss = mse_loss(pe_success_train_pred, pe_success_train_y)
    pe_success_train_loss.backward()
    pe_success_optimizer.step()

    pe_success_model.eval()
    with torch.no_grad():
        pe_success_eval_surface = pe_success_model(pe_success_eval_x)
        pe_success_eval_pred = pe_success_pixel(pe_success_eval_surface, pe_success_eval_mask).view(-1)
        pe_success_eval_loss = mse_loss(pe_success_eval_pred, pe_success_eval_y)

    assert float(pe_success_eval_surface.min().item()) >= -1.0 - 1e-6
    assert float(pe_success_eval_surface.max().item()) <= 1.0 + 1e-6

    (
        pe_missed_train_x,
        pe_missed_train_mask,
        pe_missed_train_y,
        pe_missed_train_pp_diff,
    ) = _stack_pe_batch(pe_missed_converter, pe_missed_train, pp_converter, pp_model)
    (
        pe_missed_eval_x,
        pe_missed_eval_mask,
        pe_missed_eval_y,
        pe_missed_eval_pp_diff,
    ) = _stack_pe_batch(pe_missed_converter, pe_missed_eval, pp_converter, pp_model)

    pe_missed_optimizer = torch.optim.AdamW(pe_missed_model.parameters(), lr=1e-3, weight_decay=1e-4)
    pe_missed_model.train()
    pe_missed_optimizer.zero_grad()
    pe_missed_train_surface = pe_missed_model(pe_missed_train_x)
    pe_missed_train_pred = pe_missed_pixel(pe_missed_train_surface, pe_missed_train_mask).view(-1)
    pe_missed_train_loss = mse_loss(pe_missed_train_pred, pe_missed_train_y)
    pe_missed_train_loss.backward()
    pe_missed_optimizer.step()

    pe_missed_model.eval()
    with torch.no_grad():
        pe_missed_eval_surface = pe_missed_model(pe_missed_eval_x)
        pe_missed_eval_pred = pe_missed_pixel(pe_missed_eval_surface, pe_missed_eval_mask).view(-1)
        pe_missed_eval_loss = mse_loss(pe_missed_eval_pred, pe_missed_eval_y)

    assert float(pe_missed_eval_surface.min().item()) >= -1.0 - 1e-6
    assert float(pe_missed_eval_surface.max().item()) <= 1.0 + 1e-6

    # Save model snapshots.
    torch.save(pp_model.state_dict(), models_dir / "pp_smoke_state_dict.pt")
    torch.save(ps_model.state_dict(), models_dir / "ps_smoke_state_dict.pt")
    torch.save(pe_success_model.state_dict(), models_dir / "pe_success_smoke_state_dict.pt")
    torch.save(pe_missed_model.state_dict(), models_dir / "pe_missed_smoke_state_dict.pt")

    # Save one representative heatmap per component.
    _save_surface_plot(
        pp_eval_surface[0, 0].cpu().numpy(),
        "PP smoke eval surface",
        heatmaps_dir / "pp_smoke_eval_surface.png",
        cmap="viridis",
        vmin=0,
        vmax=1,
    )
    _save_surface_plot(
        ps_eval_surface[0, 0].cpu().numpy(),
        "PS smoke eval surface",
        heatmaps_dir / "ps_smoke_eval_surface.png",
        cmap="viridis",
    )
    _save_surface_plot(
        pe_success_eval_surface[0, 0].cpu().numpy(),
        "PE-success smoke eval surface",
        heatmaps_dir / "pe_success_smoke_eval_surface.png",
        cmap="coolwarm",
        vmin=-1,
        vmax=1,
    )
    _save_surface_plot(
        pe_missed_eval_surface[0, 0].cpu().numpy(),
        "PE-missed smoke eval surface",
        heatmaps_dir / "pe_missed_smoke_eval_surface.png",
        cmap="coolwarm",
        vmin=-1,
        vmax=1,
    )

    example_rows: List[Dict] = []
    for idx, value in enumerate(pp_eval_pred.detach().cpu().numpy().tolist()):
        example_rows.append({"component": "PP", "split": "eval", "index": idx, "target": float(pp_eval_y[idx]), "pred": float(value)})
    for idx, value in enumerate(ps_eval_pred.detach().cpu().numpy().tolist()):
        example_rows.append({"component": "PS", "split": "eval", "index": idx, "target": float(ps_eval_y[idx]), "pred": float(value)})
    for idx, value in enumerate(pe_success_eval_pred.detach().cpu().numpy().tolist()):
        example_rows.append({
            "component": "PE_SUCCESS",
            "split": "eval",
            "index": idx,
            "target": float(pe_success_eval_y[idx]),
            "pred": float(value),
        })
    for idx, value in enumerate(pe_missed_eval_pred.detach().cpu().numpy().tolist()):
        example_rows.append({
            "component": "PE_MISSED",
            "split": "eval",
            "index": idx,
            "target": float(pe_missed_eval_y[idx]),
            "pred": float(value),
        })

    pd.DataFrame(example_rows).to_csv(metrics_dir / "smoke_examples.csv", index=False)

    summary = {
        "status": "FINAL_SMOKE_CHECKPOINT_OK",
        "component_metrics": {
            "PP": {
                "train_samples": int(len(pp_train_x)),
                "eval_samples": int(len(pp_eval_x)),
                "train_loss": float(pp_train_loss.item()),
                "eval_loss": float(pp_eval_loss.item()),
                "eval_surface_min": float(pp_eval_surface.min().item()),
                "eval_surface_max": float(pp_eval_surface.max().item()),
            },
            "PS": {
                "train_samples": int(len(ps_train_x)),
                "eval_samples": int(len(ps_eval_x)),
                "train_loss": float(ps_train_loss.item()),
                "eval_loss": float(ps_eval_loss.item()),
                "max_spatial_sum_deviation": ps_max_sum_dev,
            },
            "PE_SUCCESS": {
                "train_samples": int(len(pe_success_train_x)),
                "eval_samples": int(len(pe_success_eval_x)),
                "train_loss": float(pe_success_train_loss.item()),
                "eval_loss": float(pe_success_eval_loss.item()),
                "eval_surface_min": float(pe_success_eval_surface.min().item()),
                "eval_surface_max": float(pe_success_eval_surface.max().item()),
                "max_pp_channel16_diff_train": float(pe_success_train_pp_diff),
                "max_pp_channel16_diff_eval": float(pe_success_eval_pp_diff),
            },
            "PE_MISSED": {
                "train_samples": int(len(pe_missed_train_x)),
                "eval_samples": int(len(pe_missed_eval_x)),
                "train_loss": float(pe_missed_train_loss.item()),
                "eval_loss": float(pe_missed_eval_loss.item()),
                "eval_surface_min": float(pe_missed_eval_surface.min().item()),
                "eval_surface_max": float(pe_missed_eval_surface.max().item()),
                "max_pp_channel16_diff_train": float(pe_missed_train_pp_diff),
                "max_pp_channel16_diff_eval": float(pe_missed_eval_pp_diff),
            },
        },
        "artifacts": {
            "summary_json": str(metrics_dir / "smoke_summary.json"),
            "examples_csv": str(metrics_dir / "smoke_examples.csv"),
            "heatmaps_dir": str(heatmaps_dir),
            "models_dir": str(models_dir),
        },
    }

    with (metrics_dir / "smoke_summary.json").open("w", encoding="utf-8") as output_file:
        json.dump(summary, output_file, indent=2)

    print("FINAL_SMOKE_CHECKPOINT_OK")
    print(f"PP_eval_loss={summary['component_metrics']['PP']['eval_loss']:.6f}")
    print(f"PS_eval_loss={summary['component_metrics']['PS']['eval_loss']:.6f}")
    print(f"PE_SUCCESS_eval_loss={summary['component_metrics']['PE_SUCCESS']['eval_loss']:.6f}")
    print(f"PE_MISSED_eval_loss={summary['component_metrics']['PE_MISSED']['eval_loss']:.6f}")
    print(f"artifacts_metrics={metrics_dir}")
    print(f"artifacts_heatmaps={heatmaps_dir}")
    print(f"artifacts_models={models_dir}")


if __name__ == "__main__":
    run_final_smoke_regression_checkpoint()