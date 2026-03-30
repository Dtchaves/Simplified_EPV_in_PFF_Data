from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, TensorDataset
from tqdm import tqdm

from utils import ToSoccerMapTensor

PASS_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[3]
if str(PASS_ROOT) not in sys.path:
    sys.path.append(str(PASS_ROOT))

from reward_labels import PassRewardLabeler


class PFFDataset(Dataset):
    def __init__(
        self,
        train_directory,
        test_directory: Optional[str] = None,
        split_ratio: float = 0.8,
        pass_outcome_filter: str = "C",
        reward_event_directory: str = "data/raw/event",
        reward_horizon_seconds: float = 15.0,
        include_open_play_null: bool = True,
        pp_model_path: Optional[str] = None,
    ):
        self.train_data = []
        self.train_labels = []
        self.train_mask = []

        self.val_data = []
        self.val_labels = []
        self.val_mask = []

        self.test_data = []
        self.test_labels = []
        self.test_mask = []

        self.train_metadata = []
        self.test_metadata = []

        self.pass_outcome_filter = pass_outcome_filter

        event_root = Path(reward_event_directory)
        if not event_root.is_absolute():
            event_root = (REPO_ROOT / event_root).resolve()

        self.reward_labeler = PassRewardLabeler(
            event_root=event_root,
            horizon_seconds=reward_horizon_seconds,
            include_open_play_null=include_open_play_null,
        )

        pp_path_obj: Optional[Path] = None
        if pp_model_path:
            pp_path_obj = Path(pp_model_path)
            if not pp_path_obj.is_absolute():
                pp_path_obj = (REPO_ROOT / pp_path_obj).resolve()
        self.tensor_converter = ToSoccerMapTensor(pp_model_path=pp_path_obj)

        self._load_data(train_directory, is_train=True)
        if test_directory:
            self._load_data(test_directory, is_train=False)
        else:
            if len(self.train_data) < 2:
                self.val_data = list(self.train_data)
                self.val_labels = list(self.train_labels)
                self.val_mask = list(self.train_mask)
            else:
                (
                    self.train_data,
                    self.val_data,
                    self.train_labels,
                    self.val_labels,
                    self.train_mask,
                    self.val_mask,
                ) = train_test_split(
                    self.train_data,
                    self.train_labels,
                    self.train_mask,
                    test_size=1 - split_ratio,
                    random_state=42,
                )

    def _resolve_carrier_velocity(self, row: pd.Series, frame: pd.DataFrame) -> tuple[float, float]:
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

    def _load_data(self, directory, is_train=True):
        print(f"Temos {len(os.listdir(directory))} amostras na pasta {'treino' if is_train else 'teste'}")

        for filename in os.listdir(directory):
            if not filename.endswith(".csv"):
                continue

            filepath = os.path.join(directory, filename)
            df = pd.read_csv(filepath)
            df = df[df["pass_outcome_type"].notna()].copy()
            df = df[df["pass_outcome_type"] == self.pass_outcome_filter].copy()
            if df.empty:
                continue

            df, _ = self.reward_labeler.label_pass_dataframe(
                df,
                source_filename=filename,
                drop_unlabeled=True,
            )
            if df.empty:
                continue

            for idx, row in tqdm(df.iterrows(), total=df.shape[0], desc=f"Processando amostras do csv {filename}"):
                frame = df.loc[[idx]].copy()
                vx_carrier, vy_carrier = self._resolve_carrier_velocity(row, frame)

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

                matrix, mask, _ = self.tensor_converter(sample)
                target_value = float(int(row["reward_label"]))

                metadata_row = {
                    "source_file": filename,
                    "pass_outcome_type": row["pass_outcome_type"],
                    "reward_label": target_value,
                    "reward_status": row.get("reward_status", None),
                    "reward_join_strategy": row.get("reward_join_strategy", None),
                }

                if is_train:
                    self.train_data.append(matrix)
                    self.train_mask.append(mask)
                    self.train_labels.append(target_value)
                    self.train_metadata.append(metadata_row)
                else:
                    self.test_data.append(matrix)
                    self.test_mask.append(mask)
                    self.test_labels.append(target_value)
                    self.test_metadata.append(metadata_row)

    def __len__(self):
        return len(self.train_data)

    def __getitem__(self, idx):
        return self.train_data[idx], self.train_mask[idx], self.train_labels[idx]

    def get_validation_data(self):
        val_data = torch.stack(self.val_data)
        val_labels = torch.tensor(self.val_labels, dtype=torch.float32)
        val_mask = torch.stack(self.val_mask)
        return TensorDataset(val_data, val_mask, val_labels)

    def get_test_data(self):
        test_data = torch.stack(self.test_data)
        test_labels = torch.tensor(self.test_labels, dtype=torch.float32)
        test_mask = torch.stack(self.test_mask)
        return TensorDataset(test_data, test_mask, test_labels)


if __name__ == "__main__":
    train_directory = "passes"
    dataset = PFFDataset(train_directory, test_directory=None, split_ratio=0.8)

    train_loader = DataLoader(dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(dataset.get_validation_data(), batch_size=32, shuffle=False)

    print(len(train_loader), len(val_loader))
