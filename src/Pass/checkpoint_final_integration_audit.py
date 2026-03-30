from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
import numpy as np
import pandas as pd
import torch

from reward_labels import PassRewardLabeler

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[2]
PP_DIR = ROOT / "src/Pass/Pass_sucess_probability"
PE_SUCCESS_DIR = ROOT / "src/Pass/Pass_epv_success"
PE_MISSED_DIR = ROOT / "src/Pass/Pass_epv_missed"

MISSED_PASS_OUTCOMES = {"D", "B", "O", "S", "G", "I"}


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


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


def _is_valid_coordinate_pair(row: pd.Series) -> bool:
    return pd.notna(row.get("ball_x_end")) and pd.notna(row.get("ball_y_end"))


def _collect_labeled_samples(
    labeler: PassRewardLabeler,
    max_success_samples: int,
    max_missed_samples: int,
) -> List[Dict]:
    pass_files = sorted((ROOT / "passes").glob("final_pass_track_*.csv"))
    if not pass_files:
        raise FileNotFoundError("No pass CSV files found under 'passes/'.")

    samples: List[Dict] = []
    success_count = 0
    missed_count = 0

    for pass_file in pass_files:
        if success_count >= max_success_samples and missed_count >= max_missed_samples:
            break

        source_df = pd.read_csv(pass_file)
        source_df = source_df[source_df["pass_outcome_type"].notna()].copy()
        if source_df.empty:
            continue

        labeled_df, _ = labeler.label_pass_dataframe(
            source_df,
            source_filename=pass_file.name,
            drop_unlabeled=True,
        )
        if labeled_df.empty:
            continue

        for idx, row in labeled_df.iterrows():
            outcome = str(row["pass_outcome_type"])
            if outcome == "C":
                if success_count >= max_success_samples:
                    continue
                target_group = "success"
            elif outcome in MISSED_PASS_OUTCOMES:
                if missed_count >= max_missed_samples:
                    continue
                target_group = "missed"
            else:
                continue

            if not _is_valid_coordinate_pair(row):
                continue

            frame = labeled_df.loc[[idx]].copy()
            vx_carrier, vy_carrier = _resolve_carrier_velocity(row, frame)

            samples.append(
                {
                    "sample_group": target_group,
                    "source_file": pass_file.name,
                    "row_index": int(idx),
                    "reward_label": int(row["reward_label"]),
                    "reward_status": str(row.get("reward_status", "")),
                    "sample": {
                        "ball_x_start": float(row["ball_x_start"]),
                        "ball_y_start": float(row["ball_y_start"]),
                        "ball_x_end": float(row["ball_x_end"]),
                        "ball_y_end": float(row["ball_y_end"]),
                        "pass_outcome_type": outcome,
                        "team_id": int(row["team_id"]),
                        "vx_carrier": vx_carrier,
                        "vy_carrier": vy_carrier,
                        "frame": frame,
                    },
                }
            )

            if target_group == "success":
                success_count += 1
            else:
                missed_count += 1

            if success_count >= max_success_samples and missed_count >= max_missed_samples:
                break

    if success_count == 0:
        raise RuntimeError("No reward-labeled successful pass samples were found.")
    if missed_count == 0:
        raise RuntimeError("No reward-labeled missed pass samples were found.")

    return samples


def run_final_integration_checkpoint(
    max_success_samples: int = 5,
    max_missed_samples: int = 5,
) -> None:
    torch.manual_seed(42)
    np.random.seed(42)

    pp_utils = _load_module("pp_utils_integration", PP_DIR / "utils.py")
    pp_soccermap = _load_module("pp_soccermap_integration", PP_DIR / "soccermap.py")
    pe_success_utils = _load_module("pe_success_utils_integration", PE_SUCCESS_DIR / "utils.py")
    pe_success_soccermap = _load_module("pe_success_soccermap_integration", PE_SUCCESS_DIR / "soccermap.py")
    pe_missed_utils = _load_module("pe_missed_utils_integration", PE_MISSED_DIR / "utils.py")
    pe_missed_soccermap = _load_module("pe_missed_soccermap_integration", PE_MISSED_DIR / "soccermap.py")

    labeler = PassRewardLabeler(
        event_root=ROOT / "data/raw/event",
        horizon_seconds=15.0,
        include_open_play_null=True,
    )
    samples = _collect_labeled_samples(
        labeler=labeler,
        max_success_samples=max_success_samples,
        max_missed_samples=max_missed_samples,
    )

    pp_converter = pp_utils.ToSoccerMapTensor()
    pe_success_converter = pe_success_utils.ToSoccerMapTensor()
    pe_missed_converter = pe_missed_utils.ToSoccerMapTensor()

    pp_model = pp_soccermap.SoccerMapPassSucess(in_channels=13).eval()
    pe_success_model = pe_success_soccermap.SoccerMapPassEPVSuccess(in_channels=16).eval()
    pe_missed_model = pe_missed_soccermap.SoccerMapPassEPVMissed(in_channels=16).eval()

    # Explicitly wire PP -> PE so channel 16 in both PE converters comes from the same PP model.
    pe_success_converter.pp_model = pp_model
    pe_missed_converter.pp_model = pp_model

    heatmap_dir = ROOT / "results/heatmaps/final_integration_checkpoint"
    metrics_dir = ROOT / "results/metrics/final_integration_checkpoint"
    models_dir = ROOT / "results/models/final_integration_checkpoint"
    heatmap_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    rows: List[Dict] = []
    reward_targets: List[float] = []
    integrated_predictions: List[float] = []
    pp_channel_diff_success: List[float] = []
    pp_channel_diff_missed: List[float] = []

    with torch.no_grad():
        for sample_idx, sample_meta in enumerate(samples):
            sample = sample_meta["sample"]
            reward_label = int(sample_meta["reward_label"])
            assert reward_label in {-1, 0, 1}, f"Invalid reward label: {reward_label}"

            pp_matrix, pp_mask, _ = pp_converter(sample)
            pp_surface = pp_model(pp_matrix.unsqueeze(0))[0, 0].cpu().numpy().astype(float)
            pp_min = float(pp_surface.min())
            pp_max = float(pp_surface.max())
            assert pp_min >= -1e-6 and pp_max <= 1.0 + 1e-6, (
                f"PP surface out of range [{pp_min}, {pp_max}] for sample {sample_idx}"
            )

            pe_success_matrix, pe_success_mask, _ = pe_success_converter(sample)
            pe_missed_matrix, pe_missed_mask, _ = pe_missed_converter(sample)

            pe_success_np = pe_success_matrix.numpy()
            pe_missed_np = pe_missed_matrix.numpy()
            assert pe_success_np.shape == (16, 68, 104)
            assert pe_missed_np.shape == (16, 68, 104)

            pp_from_success = pe_success_np[15]
            pp_from_missed = pe_missed_np[15]
            max_diff_success = float(np.max(np.abs(pp_from_success - pp_surface)))
            max_diff_missed = float(np.max(np.abs(pp_from_missed - pp_surface)))
            pp_channel_diff_success.append(max_diff_success)
            pp_channel_diff_missed.append(max_diff_missed)
            assert max_diff_success < 1e-6, (
                f"PE-success channel16 is not wired to PP surface (max diff={max_diff_success:.8f})."
            )
            assert max_diff_missed < 1e-6, (
                f"PE-missed channel16 is not wired to PP surface (max diff={max_diff_missed:.8f})."
            )

            assert float(pe_success_mask.sum().item()) == 1.0
            assert float(pe_missed_mask.sum().item()) == 1.0

            pe_success_surface = pe_success_model(pe_success_matrix.unsqueeze(0))[0, 0].cpu().numpy().astype(float)
            pe_missed_surface = pe_missed_model(pe_missed_matrix.unsqueeze(0))[0, 0].cpu().numpy().astype(float)

            success_min = float(pe_success_surface.min())
            success_max = float(pe_success_surface.max())
            missed_min = float(pe_missed_surface.min())
            missed_max = float(pe_missed_surface.max())
            assert success_min >= -1.0 - 1e-6 and success_max <= 1.0 + 1e-6
            assert missed_min >= -1.0 - 1e-6 and missed_max <= 1.0 + 1e-6

            integrated_surface = pp_surface * pe_success_surface + (1.0 - pp_surface) * pe_missed_surface
            integrated_min = float(integrated_surface.min())
            integrated_max = float(integrated_surface.max())
            assert integrated_min >= -1.0 - 1e-6 and integrated_max <= 1.0 + 1e-6

            target_mask = pe_success_mask.numpy()[0]
            pp_at_target = float(np.sum(pp_surface * target_mask))
            pe_success_at_target = float(np.sum(pe_success_surface * target_mask))
            pe_missed_at_target = float(np.sum(pe_missed_surface * target_mask))
            integrated_at_target = float(np.sum(integrated_surface * target_mask))

            reward_targets.append(float(reward_label))
            integrated_predictions.append(integrated_at_target)

            rows.append(
                {
                    "sample_index": sample_idx,
                    "sample_group": sample_meta["sample_group"],
                    "source_file": sample_meta["source_file"],
                    "row_index": sample_meta["row_index"],
                    "pass_outcome_type": sample["pass_outcome_type"],
                    "reward_label": reward_label,
                    "reward_status": sample_meta["reward_status"],
                    "pp_surface_min": pp_min,
                    "pp_surface_max": pp_max,
                    "pp_channel_diff_success": max_diff_success,
                    "pp_channel_diff_missed": max_diff_missed,
                    "pp_at_target": pp_at_target,
                    "pe_success_at_target": pe_success_at_target,
                    "pe_missed_at_target": pe_missed_at_target,
                    "integrated_at_target": integrated_at_target,
                }
            )

            fig, axes = plt.subplots(1, 4, figsize=(18, 4))
            axes[0].imshow(pp_surface, cmap="viridis", aspect="auto", vmin=0, vmax=1)
            axes[0].set_title("PP surface")
            axes[1].imshow(pe_success_surface, cmap="coolwarm", aspect="auto", vmin=-1, vmax=1)
            axes[1].set_title("PE-success surface")
            axes[2].imshow(pe_missed_surface, cmap="coolwarm", aspect="auto", vmin=-1, vmax=1)
            axes[2].set_title("PE-missed surface")
            axes[3].imshow(integrated_surface, cmap="coolwarm", aspect="auto", vmin=-1, vmax=1)
            axes[3].set_title("Integrated PP->PE")

            for ax in axes:
                ax.set_xticks([])
                ax.set_yticks([])

            fig.suptitle(
                (
                    f"sample={sample_idx} outcome={sample['pass_outcome_type']} "
                    f"label={reward_label:+d} integrated={integrated_at_target:.4f}"
                )
            )
            fig.tight_layout()
            fig.savefig(heatmap_dir / f"integration_example_{sample_idx}.png", dpi=200)
            plt.close(fig)

    rows_df = pd.DataFrame(rows)
    rows_df.to_csv(metrics_dir / "integration_examples.csv", index=False)

    reward_arr = np.asarray(reward_targets, dtype=float)
    pred_arr = np.asarray(integrated_predictions, dtype=float)
    mse = float(np.mean((pred_arr - reward_arr) ** 2))
    mae = float(np.mean(np.abs(pred_arr - reward_arr)))

    # Persist integration-checkpoint model snapshots.
    torch.save(pp_model.state_dict(), models_dir / "pp_model_state_dict.pt")
    torch.save(pe_success_model.state_dict(), models_dir / "pe_success_model_state_dict.pt")
    torch.save(pe_missed_model.state_dict(), models_dir / "pe_missed_model_state_dict.pt")

    summary = {
        "status": "FINAL_INTEGRATION_CHECKPOINT_AUDIT_OK",
        "samples_evaluated": int(len(rows_df)),
        "samples_by_group": {
            "success": int((rows_df["sample_group"] == "success").sum()),
            "missed": int((rows_df["sample_group"] == "missed").sum()),
        },
        "wiring_checks": {
            "max_pp_channel_diff_success": float(np.max(pp_channel_diff_success)) if pp_channel_diff_success else None,
            "max_pp_channel_diff_missed": float(np.max(pp_channel_diff_missed)) if pp_channel_diff_missed else None,
        },
        "metric_snapshot": {
            "mse": mse,
            "mae": mae,
            "reward_mean": float(reward_arr.mean()) if reward_arr.size else None,
            "integrated_mean": float(pred_arr.mean()) if pred_arr.size else None,
        },
        "artifacts": {
            "examples_csv": str(metrics_dir / "integration_examples.csv"),
            "summary_json": str(metrics_dir / "integration_summary.json"),
            "heatmap_dir": str(heatmap_dir),
            "models_dir": str(models_dir),
        },
    }

    with (metrics_dir / "integration_summary.json").open("w", encoding="utf-8") as output_file:
        json.dump(summary, output_file, indent=2)

    print("FINAL_INTEGRATION_CHECKPOINT_AUDIT_OK")
    print(f"samples_evaluated={summary['samples_evaluated']}")
    print(f"samples_by_group={summary['samples_by_group']}")
    print(f"wiring_checks={summary['wiring_checks']}")
    print(f"metric_snapshot={{'mse': {mse:.6f}, 'mae': {mae:.6f}}}")
    print(f"artifacts_metrics={metrics_dir}")
    print(f"artifacts_heatmaps={heatmap_dir}")
    print(f"artifacts_models={models_dir}")


if __name__ == "__main__":
    run_final_integration_checkpoint()