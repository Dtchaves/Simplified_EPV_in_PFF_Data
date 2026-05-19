"""Generate example heatmaps for trained Pass EPV models.

This script loads the trained models for `pass_epv_success` and `pass_epv_missed`
and saves example input/output heatmaps into `results/heatmaps`.

Usage:
  python generate_heatmaps.py --models success,missed --n 3
"""
from pathlib import Path
import sys
import os
import argparse
import random
import torch
import matplotlib.pyplot as plt


def ensure_src_in_path():
    # Ensure CausalEPV/src is on sys.path so package imports work
    here = Path(__file__).resolve()
    src_dir = here.parents[1]  # .../CausalEPV/src
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))


ensure_src_in_path()

from Pass.Pass_epv_success.test import TestPassEPVSuccess, TestConfig as SuccessTestConfig
from Pass.Pass_epv_missed.test import TestPassEPVMissed, TestConfig as MissedTestConfig


def _save_heatmap(array, path, title=None, cmap="jet"):
    plt.figure(figsize=(6, 5), dpi=100)
    plt.imshow(array, cmap=cmap, interpolation="gaussian", aspect="auto")
    plt.colorbar()
    if title:
        plt.title(title)
    plt.savefig(path, bbox_inches="tight")
    plt.close()


def generate_for_success(n_samples: int, out_dir: str, seed: int = 42):
    cfg = SuccessTestConfig()
    tester = TestPassEPVSuccess(**cfg.__dict__)
    device = cfg.device
    os.makedirs(out_dir, exist_ok=True)

    torch_model = tester.model.to(device)
    dataset = tester.test_loader.dataset

    rng = random.Random(seed)
    for i in range(n_samples):
        idx = rng.randint(0, len(dataset) - 1)
        matriz, mask, target = dataset[idx]

        # Save input channels
        _save_heatmap(matriz[0].numpy(), os.path.join(out_dir, f"pass_epv_success_input_attack_{i}.png"), title=f"Attack (idx={idx})")
        _save_heatmap(matriz[1].numpy(), os.path.join(out_dir, f"pass_epv_success_input_defense_{i}.png"), title=f"Defense (idx={idx})")
        _save_heatmap(matriz[2].numpy(), os.path.join(out_dir, f"pass_epv_success_input_ball_{i}.png"), title=f"Ball (idx={idx})")

        # Model output
        with torch.no_grad():
            batch = matriz.unsqueeze(0).to(device)
            surface = torch_model(batch)
            output_np = surface[0].cpu().detach().numpy().squeeze()
        _save_heatmap(output_np, os.path.join(out_dir, f"pass_epv_success_output_surface_{i}.png"), title=f"Output Surface (idx={idx}, target={target:.3f})")


def generate_for_missed(n_samples: int, out_dir: str, seed: int = 42):
    cfg = MissedTestConfig()
    tester = TestPassEPVMissed(**cfg.__dict__)
    device = cfg.device
    os.makedirs(out_dir, exist_ok=True)

    torch_model = tester.model.to(device)
    dataset = tester.test_loader.dataset

    rng = random.Random(seed + 1)
    for i in range(n_samples):
        idx = rng.randint(0, len(dataset) - 1)
        matriz, mask, target = dataset[idx]

        _save_heatmap(matriz[0].numpy(), os.path.join(out_dir, f"pass_epv_missed_input_attack_{i}.png"), title=f"Attack (idx={idx})")
        _save_heatmap(matriz[1].numpy(), os.path.join(out_dir, f"pass_epv_missed_input_defense_{i}.png"), title=f"Defense (idx={idx})")
        _save_heatmap(matriz[2].numpy(), os.path.join(out_dir, f"pass_epv_missed_input_ball_{i}.png"), title=f"Ball (idx={idx})")

        with torch.no_grad():
            batch = matriz.unsqueeze(0).to(device)
            surface = torch_model(batch)
            output_np = surface[0].cpu().detach().numpy().squeeze()
        _save_heatmap(output_np, os.path.join(out_dir, f"pass_epv_missed_output_surface_{i}.png"), title=f"Output Surface (idx={idx}, target={target:.3f})")


def main():
    parser = argparse.ArgumentParser(description="Generate heatmaps for Pass EPV models")
    parser.add_argument("--models", default="success,missed", help="Comma-separated: success,missed")
    parser.add_argument("--n", type=int, default=1, help="Number of samples per model to generate")
    parser.add_argument("--out", default="results/heatmaps", help="Output directory")
    args = parser.parse_args()

    models = [m.strip().lower() for m in args.models.split(",") if m.strip()]
    out_dir = args.out
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    if "success" in models:
        print("Generating success heatmaps...")
        generate_for_success(args.n, out_dir)
    if "missed" in models:
        print("Generating missed heatmaps...")
        generate_for_missed(args.n, out_dir)

    print(f"Saved heatmaps to {out_dir}")


if __name__ == "__main__":
    main()
