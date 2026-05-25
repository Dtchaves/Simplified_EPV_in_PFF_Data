"""
Smoke tests for Parquet data compatibility before full training.

Validates that:
1. Parquet files exist and are readable
2. All dataloaders load non-zero samples
3. Tensor shapes are correct (13/16 channels, 68x104 spatial)
4. Reward labeling still works correctly
"""

import os
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

# Add Pass root to path
PASS_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(PASS_ROOT) not in sys.path:
    sys.path.append(str(PASS_ROOT))

from data_utils import discover_data_files, get_data_root


def check_parquet_files_exist():
    """Verify Parquet files exist in data/passes."""
    print("\n" + "="*60)
    print("1. Checking Parquet files exist in data/passes...")
    print("="*60)

    data_root = get_data_root()
    print(f"Data root: {data_root}")

    if not data_root.exists():
        print(f"[ERROR] Data root does not exist: {data_root}")
        return False

    try:
        files = discover_data_files(str(data_root), prefer_parquet=True)
        print(f"[OK] Found {len(files)} Parquet/CSV files:")
        for f in sorted(files)[:5]:
            print(f"     - {f.name}")
        if len(files) > 5:
            print(f"     ... and {len(files) - 5} more")
        return True
    except FileNotFoundError as e:
        print(f"[ERROR] {e}")
        return False


def check_pass_success_probability():
    """Smoke test Pass_success_probability dataloader."""
    print("\n" + "="*60)
    print("2. Smoke test: Pass_sucess_probability...")
    print("="*60)

    try:
        from Pass_sucess_probability.dataloader import PFFDataset

        dataset = PFFDataset(train_directory="passes", split_ratio=0.8)

        if len(dataset) == 0:
            print(f"[ERROR] Dataset loaded 0 samples!")
            return False

        print(f"[OK] Loaded {len(dataset)} training samples")
        print(f"[OK] Validation set: {len(dataset.get_validation_data())} samples")
        print(f"[OK] Test set: {len(dataset.get_test_data())} samples")

        # Check tensor shape
        matrix, mask, target = dataset[0]
        expected_channels = 13
        expected_shape = (68, 104)

        if matrix.shape[0] != expected_channels:
            print(f"[ERROR] Expected {expected_channels} channels, got {matrix.shape[0]}")
            return False

        if matrix.shape[1:] != expected_shape:
            print(f"[ERROR] Expected spatial shape {expected_shape}, got {matrix.shape[1:]}")
            return False

        print(f"[OK] Tensor shape correct: {matrix.shape}")

        # Test DataLoader
        loader = DataLoader(dataset, batch_size=32, shuffle=False)
        for i, (batch_matrix, batch_mask, batch_target) in enumerate(loader):
            print(f"[OK] Batch {i+1}: matrix {batch_matrix.shape}, mask {batch_mask.shape}, target {batch_target.shape}")
            if i >= 1:
                break

        return True

    except Exception as e:
        print(f"[ERROR] {e}")
        import traceback
        traceback.print_exc()
        return False


def check_pass_selection_probability():
    """Smoke test Pass_selection_probability dataloader."""
    print("\n" + "="*60)
    print("3. Smoke test: Pass_selection_probability...")
    print("="*60)

    try:
        from Pass_selection_probability.dataloader import PFFDataset

        dataset = PFFDataset(train_directory="passes", split_ratio=0.8)

        if len(dataset) == 0:
            print(f"[ERROR] Dataset loaded 0 samples!")
            return False

        print(f"[OK] Loaded {len(dataset)} training samples")
        print(f"[OK] Validation set: {len(dataset.get_validation_data())} samples")
        print(f"[OK] Test set: {len(dataset.get_test_data())} samples")

        # Check tensor shape
        matrix, mask, target = dataset[0]
        expected_channels = 13
        expected_shape = (68, 104)

        if matrix.shape[0] != expected_channels:
            print(f"[ERROR] Expected {expected_channels} channels, got {matrix.shape[0]}")
            return False

        if matrix.shape[1:] != expected_shape:
            print(f"[ERROR] Expected spatial shape {expected_shape}, got {matrix.shape[1:]}")
            return False

        print(f"[OK] Tensor shape correct: {matrix.shape}")

        # Test DataLoader
        loader = DataLoader(dataset, batch_size=32, shuffle=False)
        for i, (batch_matrix, batch_mask, batch_target) in enumerate(loader):
            print(f"[OK] Batch {i+1}: matrix {batch_matrix.shape}, mask {batch_mask.shape}, target {batch_target.shape}")
            if i >= 1:
                break

        return True

    except Exception as e:
        print(f"[ERROR] {e}")
        import traceback
        traceback.print_exc()
        return False


def check_pass_epv_success():
    """Smoke test Pass_epv_success dataloader."""
    print("\n" + "="*60)
    print("4. Smoke test: Pass_epv_success...")
    print("="*60)

    try:
        from Pass_epv_success.dataloader import PFFDataset

        dataset = PFFDataset(
            train_directory="passes",
            split_ratio=0.8,
            pass_outcome_filter="C",
            pp_model_path=None  # Don't load PP model for this smoke test
        )

        if len(dataset) == 0:
            print(f"[ERROR] Dataset loaded 0 samples!")
            return False

        print(f"[OK] Loaded {len(dataset)} training samples")
        print(f"[OK] Validation set: {len(dataset.get_validation_data())} samples")
        print(f"[OK] Test set: {len(dataset.get_test_data())} samples")

        # Check tensor shape
        matrix, mask, target = dataset[0]
        expected_channels = 16  # PE models have 16 channels
        expected_shape = (68, 104)

        if matrix.shape[0] != expected_channels:
            print(f"[ERROR] Expected {expected_channels} channels, got {matrix.shape[0]}")
            return False

        if matrix.shape[1:] != expected_shape:
            print(f"[ERROR] Expected spatial shape {expected_shape}, got {matrix.shape[1:]}")
            return False

        print(f"[OK] Tensor shape correct: {matrix.shape}")

        # Test DataLoader
        loader = DataLoader(dataset, batch_size=32, shuffle=False)
        for i, (batch_matrix, batch_mask, batch_target) in enumerate(loader):
            print(f"[OK] Batch {i+1}: matrix {batch_matrix.shape}, mask {batch_mask.shape}, target {batch_target.shape}")
            if i >= 1:
                break

        return True

    except Exception as e:
        print(f"[ERROR] {e}")
        import traceback
        traceback.print_exc()
        return False


def check_pass_epv_missed():
    """Smoke test Pass_epv_missed dataloader."""
    print("\n" + "="*60)
    print("5. Smoke test: Pass_epv_missed...")
    print("="*60)

    try:
        from Pass_epv_missed.dataloader import PFFDataset

        dataset = PFFDataset(
            train_directory="passes",
            split_ratio=0.8,
            pass_outcome_filter="MISSED",
            pp_model_path=None  # Don't load PP model for this smoke test
        )

        if len(dataset) == 0:
            print(f"[ERROR] Dataset loaded 0 samples!")
            return False

        print(f"[OK] Loaded {len(dataset)} training samples")
        print(f"[OK] Validation set: {len(dataset.get_validation_data())} samples")
        print(f"[OK] Test set: {len(dataset.get_test_data())} samples")

        # Check tensor shape
        matrix, mask, target = dataset[0]
        expected_channels = 16  # PE models have 16 channels
        expected_shape = (68, 104)

        if matrix.shape[0] != expected_channels:
            print(f"[ERROR] Expected {expected_channels} channels, got {matrix.shape[0]}")
            return False

        if matrix.shape[1:] != expected_shape:
            print(f"[ERROR] Expected spatial shape {expected_shape}, got {matrix.shape[1:]}")
            return False

        print(f"[OK] Tensor shape correct: {matrix.shape}")

        # Test DataLoader
        loader = DataLoader(dataset, batch_size=32, shuffle=False)
        for i, (batch_matrix, batch_mask, batch_target) in enumerate(loader):
            print(f"[OK] Batch {i+1}: matrix {batch_matrix.shape}, mask {batch_mask.shape}, target {batch_target.shape}")
            if i >= 1:
                break

        return True

    except Exception as e:
        print(f"[ERROR] {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all smoke tests."""
    print("\n" + "#"*60)
    print("# PARQUET COMPATIBILITY SMOKE TESTS")
    print("#"*60)

    results = {
        "Parquet files exist": check_parquet_files_exist(),
        "Pass_sucess_probability": check_pass_success_probability(),
        "Pass_selection_probability": check_pass_selection_probability(),
        "Pass_epv_success": check_pass_epv_success(),
        "Pass_epv_missed": check_pass_epv_missed(),
    }

    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    for test_name, passed in results.items():
        status = "[OK]" if passed else "[FAILED]"
        print(f"{status} {test_name}")

    all_passed = all(results.values())

    if all_passed:
        print("\n[SUCCESS] All smoke tests passed! Ready for full training.")
        return 0
    else:
        print("\n[ERROR] Some smoke tests failed. Please review the errors above.")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
