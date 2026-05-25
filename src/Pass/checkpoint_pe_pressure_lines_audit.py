from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from data_utils import discover_pass_sources, load_or_build_canonical_pass_cache


ROOT = Path(__file__).resolve().parents[2]


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


def _collect_samples(max_samples: int = 5) -> List[Tuple[pd.Series, pd.DataFrame, str]]:
    required_cols = [
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

    sources = discover_pass_sources(ROOT / "data" / "passes", source_format="auto", prefer_parquet=True)
    samples: List[Tuple[pd.Series, pd.DataFrame, str]] = []
    for source in sources:
        source_name = str(source.get("source_name") or source.get("source_path") or "source")
        df, _ = load_or_build_canonical_pass_cache(
            source=source,
            required_columns=required_cols,
            source_filename=source_name,
            event_root="data/raw/event",
            source_format="auto",
        )
        valid = df[df["pass_outcome_type"].notna()].copy()
        if valid.empty:
            continue

        pick_idx = int(len(valid) // 2)
        row = valid.iloc[pick_idx]
        frame = row.to_frame().T.copy()
        samples.append((row, frame, source_name))
        if len(samples) >= max_samples:
            break

    if not samples:
        raise RuntimeError("Could not collect samples for pressure-line audit.")

    return samples


def run_pressure_line_fidelity_checkpoint() -> None:
    success_utils = _load_module(
        "pe_success_utils_pressure",
        ROOT / "src/Pass/Pass_epv_success/utils.py",
    )
    missed_utils = _load_module(
        "pe_missed_utils_pressure",
        ROOT / "src/Pass/Pass_epv_missed/utils.py",
    )

    success_converter = success_utils.ToSoccerMapTensor()
    missed_converter = missed_utils.ToSoccerMapTensor()

    # Fallback behavior check for sparse player snapshots.
    fallback_map, fallback_vertical, fallback_horizontal = success_converter._build_pressure_map(
        np.array([], dtype=float),
        np.array([], dtype=float),
        None,
        None,
    )
    assert fallback_map.shape == (68, 104)
    assert fallback_vertical.shape == (3,)
    assert fallback_horizontal.shape == (3,)

    sample_rows = _collect_samples(max_samples=5)

    heatmap_dir = ROOT / "results/heatmaps/pe_pressure_lines_checkpoint"
    metrics_dir = ROOT / "results/metrics/pe_pressure_lines_checkpoint"
    heatmap_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    metrics_rows: List[Dict] = []

    for idx, (row, frame, source_file) in enumerate(sample_rows):
        vx_carrier, vy_carrier = _resolve_carrier_velocity(row, frame)
        sample = {
            "ball_x_start": float(row["ball_x_start"]),
            "ball_y_start": float(row["ball_y_start"]),
            "ball_x_end": float(row["ball_x_end"]),
            "ball_y_end": float(row["ball_y_end"]),
            "pass_outcome_type": row["pass_outcome_type"],
            "team_id": int(row["team_id"]),
            "vx_carrier": vx_carrier,
            "vy_carrier": vy_carrier,
            "frame": frame,
        }

        success_matrix, _, _ = success_converter(sample)
        missed_matrix, _, _ = missed_converter(sample)

        success_att = success_matrix.numpy()[9]
        success_def = success_matrix.numpy()[10]
        missed_att = missed_matrix.numpy()[9]
        missed_def = missed_matrix.numpy()[10]

        for name, pressure_map in (
            ("success_att", success_att),
            ("success_def", success_def),
            ("missed_att", missed_att),
            ("missed_def", missed_def),
        ):
            assert np.all(np.isfinite(pressure_map)), f"Found NaN/Inf in {name}."
            assert float(pressure_map.min()) >= -1.000001, f"{name} below -1."
            assert float(pressure_map.max()) <= 1.000001, f"{name} above 1."
            assert float(np.std(pressure_map)) > 0.005, f"{name} variance too low."
            unique_rounded = int(np.unique(np.round(pressure_map, 3)).size)
            assert unique_rounded > 10, f"{name} too coarse ({unique_rounded} unique values)."

        fig, axes = plt.subplots(2, 2, figsize=(12, 7))
        axes[0, 0].imshow(success_att, cmap="coolwarm", aspect="auto", vmin=-1, vmax=1)
        axes[0, 0].set_title("PE-success C10 Att pressure")
        axes[0, 1].imshow(success_def, cmap="coolwarm", aspect="auto", vmin=-1, vmax=1)
        axes[0, 1].set_title("PE-success C11 Def pressure")
        axes[1, 0].imshow(missed_att, cmap="coolwarm", aspect="auto", vmin=-1, vmax=1)
        axes[1, 0].set_title("PE-missed C10 Att pressure")
        axes[1, 1].imshow(missed_def, cmap="coolwarm", aspect="auto", vmin=-1, vmax=1)
        axes[1, 1].set_title("PE-missed C11 Def pressure")

        for ax in axes.flatten():
            ax.set_xticks([])
            ax.set_yticks([])

        fig.suptitle(f"sample={idx} source={source_file}")
        fig.tight_layout()
        fig.savefig(heatmap_dir / f"pressure_lines_example_{idx}.png", dpi=200)
        plt.close(fig)

        metrics_rows.append(
            {
                "sample_index": idx,
                "source_file": source_file,
                "success_att_std": float(np.std(success_att)),
                "success_def_std": float(np.std(success_def)),
                "missed_att_std": float(np.std(missed_att)),
                "missed_def_std": float(np.std(missed_def)),
                "success_att_unique_rounded": int(np.unique(np.round(success_att, 3)).size),
                "success_def_unique_rounded": int(np.unique(np.round(success_def, 3)).size),
                "missed_att_unique_rounded": int(np.unique(np.round(missed_att, 3)).size),
                "missed_def_unique_rounded": int(np.unique(np.round(missed_def, 3)).size),
            }
        )

    metrics_df = pd.DataFrame(metrics_rows)
    metrics_df.to_csv(metrics_dir / "pressure_lines_examples.csv", index=False)

    summary = {
        "status": "PE_PRESSURE_LINES_CHECKPOINT_AUDIT_OK",
        "samples_evaluated": int(len(metrics_rows)),
        "fallback_defaults": {
            "vertical_lines": [float(v) for v in fallback_vertical.tolist()],
            "horizontal_lines": [float(v) for v in fallback_horizontal.tolist()],
        },
        "aggregate": {
            "success_att_std_mean": float(metrics_df["success_att_std"].mean()),
            "success_def_std_mean": float(metrics_df["success_def_std"].mean()),
            "missed_att_std_mean": float(metrics_df["missed_att_std"].mean()),
            "missed_def_std_mean": float(metrics_df["missed_def_std"].mean()),
            "success_att_unique_min": int(metrics_df["success_att_unique_rounded"].min()),
            "success_def_unique_min": int(metrics_df["success_def_unique_rounded"].min()),
            "missed_att_unique_min": int(metrics_df["missed_att_unique_rounded"].min()),
            "missed_def_unique_min": int(metrics_df["missed_def_unique_rounded"].min()),
        },
        "artifacts": {
            "examples_csv": str(metrics_dir / "pressure_lines_examples.csv"),
            "heatmap_dir": str(heatmap_dir),
        },
    }

    with (metrics_dir / "pressure_lines_summary.json").open("w", encoding="utf-8") as output_file:
        json.dump(summary, output_file, indent=2)

    print("PE_PRESSURE_LINES_CHECKPOINT_AUDIT_OK")
    print(f"samples_evaluated={len(metrics_rows)}")
    print(f"success_att_std_mean={summary['aggregate']['success_att_std_mean']:.6f}")
    print(f"success_def_std_mean={summary['aggregate']['success_def_std_mean']:.6f}")
    print(f"missed_att_std_mean={summary['aggregate']['missed_att_std_mean']:.6f}")
    print(f"missed_def_std_mean={summary['aggregate']['missed_def_std_mean']:.6f}")
    print(f"artifacts_metrics={metrics_dir}")
    print(f"artifacts_heatmaps={heatmap_dir}")


if __name__ == "__main__":
    run_pressure_line_fidelity_checkpoint()
