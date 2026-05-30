from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset


class BallDriveTabularDataset(Dataset):
    def __init__(self, x: np.ndarray, y: np.ndarray):
        self.x = torch.tensor(np.asarray(x, dtype=np.float32))
        self.y = torch.tensor(np.asarray(y, dtype=np.float32)).view(-1, 1)

    def __len__(self) -> int:
        return int(self.x.shape[0])

    def __getitem__(self, idx: int):
        return self.x[idx], self.y[idx]


@dataclass
class BallDriveLoaderConfig:
    batch_size: int = 32


def split_dataframe_by_manifest(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    if "split" not in df.columns:
        raise ValueError("Input dataframe must contain 'split' column.")

    return {
        "train": df[df["split"] == "train"].copy(),
        "val": df[df["split"] == "val"].copy(),
        "test": df[df["split"] == "test"].copy(),
    }


def make_dp_dataloaders(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    config: BallDriveLoaderConfig,
) -> Dict[str, DataLoader]:
    train_ds = BallDriveTabularDataset(x_train, y_train)
    val_ds = BallDriveTabularDataset(x_val, y_val)
    test_ds = BallDriveTabularDataset(x_test, y_test)

    return {
        "train": DataLoader(train_ds, batch_size=config.batch_size, shuffle=True),
        "val": DataLoader(val_ds, batch_size=config.batch_size, shuffle=False),
        "test": DataLoader(test_ds, batch_size=config.batch_size, shuffle=False),
    }


def make_de_subset(
    features_df: pd.DataFrame,
    target_column: str,
    class_filter: int,
) -> Tuple[np.ndarray, np.ndarray]:
    subset = features_df[(features_df["y_success"] == class_filter) & (features_df[target_column].notna())].copy()
    if subset.empty:
        return np.zeros((0, 0), dtype=float), np.zeros((0,), dtype=float)

    x_columns = [col for col in subset.columns if col.startswith("f_") or col == "p_ball_drive_success"]
    x = subset[x_columns].to_numpy(dtype=float)
    y = subset[target_column].to_numpy(dtype=float)
    return x, y
