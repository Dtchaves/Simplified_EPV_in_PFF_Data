import importlib.util
import importlib
from pathlib import Path
import sys

import matplotlib
import numpy as np
import pandas as pd
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[2]


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def load_soccermap_class(package_dir: Path, class_name: str):
    package_str = str(package_dir)
    sys.path.insert(0, package_str)
    try:
        # Reset ambiguous module names so each package resolves its own local imports.
        for module_name in ("soccermap", "utils"):
            if module_name in sys.modules:
                del sys.modules[module_name]
        module = importlib.import_module("soccermap")
        return getattr(module, class_name)
    finally:
        if sys.path and sys.path[0] == package_str:
            sys.path.pop(0)


def build_single_sample(root: Path):
    pass_files = sorted((root / "passes").glob("final_pass_track_*.csv"))
    if not pass_files:
        raise FileNotFoundError("No pass CSV files found under passes/.")

    source_file = None
    row = None
    for csv_path in pass_files:
        df = pd.read_csv(csv_path)
        valid = df[df["pass_outcome_type"].notna()]
        if not valid.empty:
            source_file = csv_path
            row = valid.iloc[0]
            break

    if row is None:
        raise RuntimeError("Could not find a row with non-null pass_outcome_type.")

    frame = row.to_frame().T.copy()
    player_id = int(row["player_id"])

    passer_suffix = None
    for col in frame.columns:
        if col.startswith("original_pId_player_"):
            value = frame.iloc[0][col]
            if pd.notna(value) and int(value) == player_id:
                passer_suffix = col.replace("original_pId_player_", "")
                break

    vx_carrier = 0.0
    vy_carrier = 0.0
    carrier_velocity = 0.0
    if passer_suffix is not None:
        vx_col = f"vx_player_{passer_suffix}"
        vy_col = f"vy_player_{passer_suffix}"
        if vx_col in frame.columns and vy_col in frame.columns:
            vx_value = frame.iloc[0][vx_col]
            vy_value = frame.iloc[0][vy_col]
            vx_carrier = float(vx_value) if pd.notna(vx_value) else 0.0
            vy_carrier = float(vy_value) if pd.notna(vy_value) else 0.0
            carrier_velocity = float(np.hypot(vx_carrier, vy_carrier))

    sample = {
        "ball_x_start": float(row["ball_x_start"]),
        "ball_y_start": float(row["ball_y_start"]),
        "ball_x_end": float(row["ball_x_end"]),
        "ball_y_end": float(row["ball_y_end"]),
        "pass_outcome_type": row["pass_outcome_type"],
        "team_id": int(row["team_id"]),
        "vx_carrier": vx_carrier,
        "vy_carrier": vy_carrier,
        "carrier_velocity": carrier_velocity,
        "frame": frame,
    }
    return sample, source_file


def run_audit():
    pp_utils = load_module(
        "pp_utils", ROOT / "src/Pass/Pass_sucess_probability/utils.py"
    )
    ps_utils = load_module(
        "ps_utils", ROOT / "src/Pass/Pass_selection_probability/utils.py"
    )
    pp_model_class = load_soccermap_class(
        ROOT / "src/Pass/Pass_sucess_probability", "SoccerMapPassSucess"
    )
    ps_model_class = load_soccermap_class(
        ROOT / "src/Pass/Pass_selection_probability", "SoccerMapPassSelect"
    )

    sample, source_file = build_single_sample(ROOT)

    pp_converter = pp_utils.ToSoccerMapTensor()
    ps_converter = ps_utils.ToSoccerMapTensor()

    pp_matrix, _, _ = pp_converter(sample)
    ps_matrix, _, _ = ps_converter(sample)

    pp_matrix_np = pp_matrix.numpy()
    ps_matrix_np = ps_matrix.numpy()

    assert pp_matrix_np.shape == (13, 68, 104), pp_matrix_np.shape
    assert ps_matrix_np.shape == (13, 68, 104), ps_matrix_np.shape

    for name, mat in (("PP", pp_matrix_np), ("PS", ps_matrix_np)):
        assert np.all(mat[11] >= 0.0), f"{name} distance-to-goal has negatives"
        assert np.all(mat[12] >= 0.0), f"{name} distance-to-ball has negatives"
        assert np.max(np.abs(mat[9])) <= 1.000001, (
            f"{name} sin(angle to carrier velocity) outside [-1,1]"
        )
        assert np.max(np.abs(mat[10])) <= 1.000001, (
            f"{name} cos(angle to carrier velocity) outside [-1,1]"
        )

        sin2_cos2 = mat[7] ** 2 + mat[8] ** 2
        max_dev = float(np.max(np.abs(sin2_cos2 - 1.0)))
        assert max_dev < 1e-5, (
            f"{name} sin/cos(angle-to-ball) identity deviation too high: {max_dev}"
        )

    pp_model = pp_model_class(in_channels=13).eval()
    ps_model = ps_model_class(in_channels=13).eval()

    with torch.no_grad():
        pp_surface = pp_model(pp_matrix.unsqueeze(0))
        ps_surface = ps_model(ps_matrix.unsqueeze(0))

    pp_min = float(pp_surface.min().item())
    pp_max = float(pp_surface.max().item())
    assert pp_min >= -1e-6 and pp_max <= 1.0 + 1e-6, (
        f"PP sigmoid range failed: [{pp_min}, {pp_max}]"
    )

    ps_sum = float(ps_surface.sum(dim=(2, 3)).item())
    assert abs(ps_sum - 1.0) < 1e-5, f"PS spatial softmax sum failed: {ps_sum}"

    pp_dir = ROOT / "results/heatmaps/pass_success_checkpoint"
    ps_dir = ROOT / "results/heatmaps/pass_selection_checkpoint"
    pp_dir.mkdir(parents=True, exist_ok=True)
    ps_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 5))
    plt.imshow(pp_matrix_np[0], cmap="viridis", aspect="auto")
    plt.colorbar()
    plt.title("PP Example Input Channel 1 (Attacking locations)")
    plt.tight_layout()
    plt.savefig(pp_dir / "pp_example_input_ch1.png", dpi=200)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.imshow(pp_surface[0, 0].cpu().numpy(), cmap="viridis", aspect="auto")
    plt.colorbar()
    plt.title("PP Example Surface (sigmoid per-cell)")
    plt.tight_layout()
    plt.savefig(pp_dir / "pp_example_surface.png", dpi=200)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.imshow(ps_matrix_np[0], cmap="viridis", aspect="auto")
    plt.colorbar()
    plt.title("PS Example Input Channel 1 (Attacking locations)")
    plt.tight_layout()
    plt.savefig(ps_dir / "ps_example_input_ch1.png", dpi=200)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.imshow(ps_surface[0, 0].cpu().numpy(), cmap="viridis", aspect="auto")
    plt.colorbar()
    plt.title("PS Example Surface (spatial softmax)")
    plt.tight_layout()
    plt.savefig(ps_dir / "ps_example_surface.png", dpi=200)
    plt.close()

    print("CHECKPOINT_AUDIT_OK")
    print(f"source_file={source_file.name}")
    print(f"pp_shape={pp_matrix_np.shape}")
    print(f"ps_shape={ps_matrix_np.shape}")
    print(f"pp_range=[{pp_min:.6f}, {pp_max:.6f}]")
    print(f"ps_spatial_sum={ps_sum:.6f}")
    print(f"pp_examples={pp_dir}")
    print(f"ps_examples={ps_dir}")


if __name__ == "__main__":
    run_audit()
