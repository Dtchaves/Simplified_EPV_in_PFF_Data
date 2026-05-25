from __future__ import annotations

import importlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List

import matplotlib
import numpy as np
import pandas as pd
import torch

from data_utils import discover_pass_sources, load_or_build_canonical_pass_cache
from reward_labels import PassRewardLabeler

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[2]
PE_DIR = ROOT / "src/Pass/Pass_epv_missed"
MISSED_PASS_OUTCOMES = {"D", "B", "O", "S", "G", "I"}
REQUIRED_COLUMNS = [
    "game_id",
    "game_event_id",
    "possession_event_id",
    "player_id",
    "ball_x_start",
    "ball_y_start",
    "ball_x_end",
    "ball_y_end",
    "team_id",
]


def _load_pe_modules():
    pe_path = str(PE_DIR)
    sys.path.insert(0, pe_path)
    try:
        for module_name in ("utils", "soccermap", "dataloader"):
            if module_name in sys.modules:
                del sys.modules[module_name]
        pe_utils = importlib.import_module("utils")
        pe_soccermap = importlib.import_module("soccermap")
        pe_dataloader = importlib.import_module("dataloader")
        return pe_utils, pe_soccermap, pe_dataloader
    finally:
        if sys.path and sys.path[0] == pe_path:
            sys.path.pop(0)


def run_filter_contract_checks() -> Dict[str, int]:
    sources = discover_pass_sources("data/passes", source_format="auto")
    if not sources:
        raise FileNotFoundError("No canonical pass sources found under data/passes.")

    labeler = PassRewardLabeler(
        event_root=ROOT / "data/raw/event",
        horizon_seconds=15.0,
        include_open_play_null=True,
    )

    label_counter: Counter = Counter()
    status_counter: Counter = Counter()
    rows_after_filters = 0

    for source in sources:
        source_df, _ = load_or_build_canonical_pass_cache(
            source=source,
            required_columns=REQUIRED_COLUMNS,
            source_filename=source.get("source_name"),
            event_root="data/raw/event",
        )
        source_df = source_df[source_df["pass_outcome_type"].notna()].copy()
        source_df = source_df[source_df["pass_outcome_type"].isin(MISSED_PASS_OUTCOMES)].copy()
        if source_df.empty:
            continue

        labeled_df, _ = labeler.label_pass_dataframe(
            source_df,
            source_filename=source.get("source_name"),
            drop_unlabeled=True,
            source_kind=str(source.get("source_kind", "legacy_wide")),
            processed_events_path=source.get("events_path"),
            processed_tracking_path=source.get("tracking_path"),
        )
        if labeled_df.empty:
            continue

        assert (~labeled_df["pass_outcome_type"].eq("C")).all(), (
            f"Found successful outcomes after PE-missed filter in {source.get('source_name')}."
        )

        labels = labeled_df["reward_label"].astype(int)
        unique_values = set(labels.unique().tolist())
        assert unique_values.issubset({-1, 0, 1}), (
            f"Unexpected reward labels in {source.get('source_name')}: {sorted(unique_values)}"
        )

        rows_after_filters += int(len(labeled_df))
        label_counter.update(labels.tolist())
        status_counter.update(labeled_df["reward_status"].tolist())

    assert rows_after_filters > 0, "No rows left after PE-missed + reward filtering."

    return {
        "rows_after_filters": rows_after_filters,
        "label_-1": int(label_counter.get(-1, 0)),
        "label_0": int(label_counter.get(0, 0)),
        "label_1": int(label_counter.get(1, 0)),
        "status_goal_scored": int(status_counter.get("goal_scored", 0)),
        "status_goal_conceded": int(status_counter.get("goal_conceded", 0)),
        "status_no_goal_window": int(status_counter.get("no_goal_window", 0)),
    }


def run_pe_missed_checkpoint() -> None:
    pe_utils, pe_soccermap, pe_dataloader = _load_pe_modules()

    filter_summary = run_filter_contract_checks()

    dataset = pe_dataloader.PFFDataset(
        train_directory="data/passes",
        split_ratio=0.8,
        pass_outcome_filter="MISSED",
        reward_event_directory="data/raw/event",
        reward_horizon_seconds=15.0,
        include_open_play_null=True,
    )

    assert len(dataset) > 0, "PE-missed dataset is empty."

    matrix, mask, label = dataset[0]
    matrix_np = matrix.numpy()
    mask_np = mask.numpy()

    assert matrix_np.shape == (16, 68, 104), matrix_np.shape
    assert mask_np.shape == (1, 68, 104), mask_np.shape

    # Contract checks for PE channels.
    assert np.all(np.isfinite(matrix_np)), "Found NaN/Inf in PE matrix."
    assert float(matrix_np[15].min()) >= -1e-6, "PP-surface channel has values < 0."
    assert float(matrix_np[15].max()) <= 1.0 + 1e-6, "PP-surface channel has values > 1."
    for idx in (11, 12, 13, 14):
        assert float(matrix_np[idx].min()) >= -1e-6, f"Outplayed channel {idx} has negative values."

    model = pe_soccermap.SoccerMapPassEPVMissed(in_channels=16).eval()

    sample_count = min(5, len(dataset))
    samples: List[Dict] = []
    pred_values: List[float] = []
    true_values: List[float] = []

    heatmap_dir = ROOT / "results/heatmaps/pe_missed_checkpoint"
    metrics_dir = ROOT / "results/metrics/pe_missed_checkpoint"
    heatmap_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        for i in range(sample_count):
            matrix_i, mask_i, label_i = dataset[i]
            surface_i = model(matrix_i.unsqueeze(0))
            pred_i = pe_soccermap.pixel(surface_i, mask_i.unsqueeze(0)).view(-1)

            pred_value = float(pred_i.item())
            true_value = float(label_i)

            pred_values.append(pred_value)
            true_values.append(true_value)

            surface_np = surface_i[0, 0].cpu().numpy()
            assert float(surface_np.min()) >= -1.0 - 1e-6, "PE output below -1."
            assert float(surface_np.max()) <= 1.0 + 1e-6, "PE output above 1."

            fig, axes = plt.subplots(1, 3, figsize=(14, 4))
            axes[0].imshow(matrix_i.numpy()[9], cmap="viridis", aspect="auto")
            axes[0].set_title("C10 Att pressure-lines")
            axes[1].imshow(matrix_i.numpy()[15], cmap="viridis", aspect="auto")
            axes[1].set_title("C16 PP surface")
            axes[2].imshow(surface_np, cmap="coolwarm", aspect="auto", vmin=-1, vmax=1)
            axes[2].set_title("PE-missed output")
            for ax in axes:
                ax.set_xticks([])
                ax.set_yticks([])
            fig.suptitle(f"sample={i} label={true_value:.0f} pred={pred_value:.4f}")
            fig.tight_layout()
            fig.savefig(heatmap_dir / f"pe_missed_example_{i}.png", dpi=200)
            plt.close(fig)

            samples.append(
                {
                    "sample_index": i,
                    "true_reward_label": true_value,
                    "predicted_value_at_target": pred_value,
                }
            )

    pred_arr = np.asarray(pred_values, dtype=float)
    true_arr = np.asarray(true_values, dtype=float)
    mse = float(np.mean((pred_arr - true_arr) ** 2))
    mae = float(np.mean(np.abs(pred_arr - true_arr)))

    pd.DataFrame(samples).to_csv(metrics_dir / "pe_missed_examples.csv", index=False)

    summary = {
        "status": "PE_MISSED_CHECKPOINT_AUDIT_OK",
        "dataset_train_rows": int(len(dataset)),
        "feature_shape": [16, 68, 104],
        "filter_summary": filter_summary,
        "range_checks": {
            "channel_16_pp_surface_min": float(matrix_np[15].min()),
            "channel_16_pp_surface_max": float(matrix_np[15].max()),
            "model_output_min": float(pred_arr.min()) if pred_arr.size else None,
            "model_output_max": float(pred_arr.max()) if pred_arr.size else None,
        },
        "metric_snapshot": {
            "samples_evaluated": int(sample_count),
            "mse": mse,
            "mae": mae,
        },
        "artifacts": {
            "examples_csv": str(metrics_dir / "pe_missed_examples.csv"),
            "heatmap_dir": str(heatmap_dir),
        },
    }

    with (metrics_dir / "pe_missed_summary.json").open("w", encoding="utf-8") as output_file:
        json.dump(summary, output_file, indent=2)

    print("PE_MISSED_CHECKPOINT_AUDIT_OK")
    print(f"dataset_train_rows={len(dataset)}")
    print(f"feature_shape={summary['feature_shape']}")
    print(f"filter_summary={filter_summary}")
    print(f"metric_snapshot={{'mse': {mse:.6f}, 'mae': {mae:.6f}}}")
    print(f"artifacts_metrics={metrics_dir}")
    print(f"artifacts_heatmaps={heatmap_dir}")


if __name__ == "__main__":
    run_pe_missed_checkpoint()
