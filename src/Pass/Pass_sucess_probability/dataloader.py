import os
import sys
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import numpy as np
from tqdm import tqdm

try:
    from utils import ToSoccerMapTensor
except ImportError:
    from .utils import ToSoccerMapTensor

PASS_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[3]
if str(PASS_ROOT) not in sys.path:
    sys.path.insert(0, str(PASS_ROOT))

try:
    from data_utils import (
        PP_PS_CACHE_VERSION,
        discover_pass_sources,
        split_sources_by_mode,
        load_or_build_canonical_pass_cache,
        load_tensor_cache,
        save_tensor_cache,
        build_sample_key,
    )
    from reward_labels import PassRewardLabeler
except ImportError:
    # Fallback for relative imports when package is properly structured
    from ..data_utils import (
        PP_PS_CACHE_VERSION,
        discover_pass_sources,
        split_sources_by_mode,
        load_or_build_canonical_pass_cache,
        load_tensor_cache,
        save_tensor_cache,
        build_sample_key,
    )
    from ..reward_labels import PassRewardLabeler

class PFFDataset(Dataset):
    def __init__(
        self,
        train_directory,
        test_directory=None,
        split_ratio=0.8,
        label_mode="pass_outcome",
        reward_event_directory="data/raw/event",
        reward_horizon_seconds=15.0,
        include_open_play_null=True,
        split_mode="row",
        split_manifest_path=None,
        split_seed=42,
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

        self.label_mode = label_mode
        self.split_mode = str(split_mode).strip().lower()
        self.split_manifest_path = split_manifest_path
        self.split_seed = int(split_seed)
        if self.label_mode not in {"pass_outcome", "reward"}:
            raise ValueError("label_mode must be one of: 'pass_outcome', 'reward'.")

        self.reward_labeler = None
        if self.label_mode == "reward":
            event_root = Path(reward_event_directory)
            if not event_root.is_absolute():
                event_root = (REPO_ROOT / event_root).resolve()
            self.reward_labeler = PassRewardLabeler(
                event_root=event_root,
                horizon_seconds=reward_horizon_seconds,
                include_open_play_null=include_open_play_null,
            )

        if test_directory:
            self._load_data(train_directory, is_train=True)
            self._load_data(test_directory, is_train=False)
        else:
            if self.split_mode == "match":
                sources = self._discover_sources(train_directory)
                train_sources, val_sources, _ = split_sources_by_mode(
                    sources,
                    split_ratio=split_ratio,
                    split_mode="match",
                    split_seed=self.split_seed,
                    split_manifest_path=self.split_manifest_path,
                )
                self._load_data(train_directory, is_train=True, sources=train_sources)
                self._load_data(train_directory, is_train=False, sources=val_sources)

                self.val_data = list(self.test_data)
                self.val_labels = list(self.test_labels)
                self.val_mask = list(self.test_mask)
                self.test_data = []
                self.test_labels = []
                self.test_mask = []
            else:
                self._load_data(train_directory, is_train=True)
                self.train_data, self.val_data, self.train_labels, self.val_labels, self.train_mask, self.val_mask = train_test_split(
                    self.train_data, self.train_labels, self.train_mask, test_size=1-split_ratio, random_state=42
                )

    def _discover_sources(self, directory):
        data_path = Path(directory)
        if not data_path.is_absolute():
            if directory in ["passes", "data/passes", "./passes", "./data/passes"]:
                data_path = REPO_ROOT / "data" / "passes"
            elif str(directory).startswith("data/"):
                data_path = REPO_ROOT / directory
            else:
                data_path = REPO_ROOT / directory
            data_path = data_path.resolve()

        if not data_path.exists():
            raise FileNotFoundError(f"Data directory does not exist: {data_path}")

        return discover_pass_sources(str(data_path), source_format="auto", prefer_parquet=True)

    def _resolve_carrier_velocity(self, row: pd.Series, frame: pd.DataFrame) -> tuple[float, float, float]:
        player_id = int(row["player_id"])

        for col in frame.columns:
            if not col.startswith("original_pId_player_"):
                continue
            raw_value = frame.iloc[0][col]
            if pd.isna(raw_value) or int(raw_value) != player_id:
                continue

            suffix = col.replace("original_pId_player_", "")
            vx_col = f"vx_player_{suffix}"
            vy_col = f"vy_player_{suffix}"
            vx_val = frame.iloc[0][vx_col] if vx_col in frame.columns else 0.0
            vy_val = frame.iloc[0][vy_col] if vy_col in frame.columns else 0.0
            vx = float(vx_val) if pd.notna(vx_val) else 0.0
            vy = float(vy_val) if pd.notna(vy_val) else 0.0
            return vx, vy, float(np.hypot(vx, vy))

        return 0.0, 0.0, 0.0

    def _append_payload(self, payload: Dict[str, Any], is_train: bool) -> None:
        data_list = payload.get("data", [])
        mask_list = payload.get("mask", [])
        label_list = payload.get("labels", [])

        if is_train:
            self.train_data.extend(data_list)
            self.train_mask.extend(mask_list)
            self.train_labels.extend(label_list)
        else:
            self.test_data.extend(data_list)
            self.test_mask.extend(mask_list)
            self.test_labels.extend(label_list)

    def _build_pass_outcome_tensor_payload(self, df: pd.DataFrame, filepath: Path) -> Dict[str, Any]:
        tensor_converter = ToSoccerMapTensor()
        matrices: List[torch.Tensor] = []
        masks: List[torch.Tensor] = []
        labels: List[int] = []
        sample_keys: List[str] = []
        metadata: List[Dict[str, Any]] = []

        for idx, row in tqdm(df.iterrows(), total=df.shape[0], desc=f"Processando amostras do {filepath.name}"):
            frame = df.loc[[idx]].copy()
            vx_carrier, vy_carrier, carrier_velocity = self._resolve_carrier_velocity(row, frame)

            sample = {
                "ball_x_start": row["ball_x_start"],
                "ball_y_start": row["ball_y_start"],
                "ball_x_end": row["ball_x_end"],
                "ball_y_end": row["ball_y_end"],
                "pass_outcome_type": row["pass_outcome_type"],
                "team_id": row["team_id"],
                "vx_carrier": vx_carrier,
                "vy_carrier": vy_carrier,
                "carrier_velocity": carrier_velocity,
                "frame": frame,
            }

            matrix, mask, target = tensor_converter(sample)
            matrices.append(matrix)
            masks.append(mask)
            labels.append(int(target[0]))
            sample_keys.append(build_sample_key(row, idx))
            metadata.append(
                {
                    "source_file": filepath.name,
                    "pass_outcome_type": row["pass_outcome_type"],
                }
            )

        return {
            "data": matrices,
            "mask": masks,
            "labels": labels,
            "sample_keys": sample_keys,
            "metadata": metadata,
        }

    def _load_data(self, directory, is_train=True, sources=None):
        if sources is None:
            try:
                sources = self._discover_sources(directory)
            except FileNotFoundError as exc:
                print(f"[WARNING] {exc}")
                return

        print(f"Descobertos {len(sources)} fontes de dados para {'treino' if is_train else 'teste'}")

        # Required columns that MUST exist prior to event merge
        required_cols = [
            'game_id', 'game_event_id', 'possession_event_id', 'player_id',
            'ball_x_start', 'ball_y_start',
            'ball_x_end', 'ball_y_end', 'team_id'
        ]

        cache_dependencies = {
            "label_mode": self.label_mode,
            "tensor_family": "pp_ps",
            "spatial_dim": [68, 104],
        }

        for source in sources:
            source_path = Path(source["source_path"])
            source_name = source.get("source_name", source_path.name)
            try:
                df, canonical_summary = load_or_build_canonical_pass_cache(
                    source=source,
                    required_columns=required_cols,
                    source_filename=source_name,
                    event_root="data/raw/event",
                )
            except Exception as e:
                print(f"[ERROR] Failed to load {source_name}: {e}")
                continue

            df.dropna(subset=['pass_outcome_type'], inplace=True)
            if df.empty:
                continue

            if self.label_mode == "pass_outcome":
                cached_payload = load_tensor_cache(
                    source_path=source_path,
                    cache_family="pp_ps",
                    artifact_stem="pass_outcome_tensors",
                    cache_version=PP_PS_CACHE_VERSION,
                    dependencies=cache_dependencies,
                )
                if cached_payload is not None:
                    self._append_payload(cached_payload, is_train=is_train)
                    continue

            if self.label_mode == "reward" and self.reward_labeler is not None:
                df, label_summary = self.reward_labeler.label_pass_dataframe(
                    df,
                    source_filename=source_name,
                    drop_unlabeled=True,
                    source_kind=str(source.get("source_kind", "legacy_wide")),
                    processed_events_path=source.get("events_path"),
                    processed_tracking_path=source.get("tracking_path"),
                )
                if df.empty:
                    continue

            if self.label_mode == "pass_outcome":
                payload = self._build_pass_outcome_tensor_payload(df, source_path)
                payload = save_tensor_cache(
                    source_path=source_path,
                    cache_family="pp_ps",
                    artifact_stem="pass_outcome_tensors",
                    cache_version=PP_PS_CACHE_VERSION,
                    payload=payload,
                    dependencies=cache_dependencies,
                )
                self._append_payload(payload, is_train=is_train)
                continue

            tensor_converter = ToSoccerMapTensor()
            for idx, row in tqdm(df.iterrows(), total=df.shape[0], desc=f"Processando amostras do {source_name}"):
                frame = df.loc[[idx]].copy()
                vx_carrier, vy_carrier, carrier_velocity = self._resolve_carrier_velocity(row, frame)

                sample = {
                    "ball_x_start": row["ball_x_start"],
                    "ball_y_start": row["ball_y_start"],
                    "ball_x_end": row["ball_x_end"],
                    "ball_y_end": row["ball_y_end"],
                    "pass_outcome_type": row["pass_outcome_type"],
                    "team_id": row["team_id"],
                    "vx_carrier": vx_carrier,
                    "vy_carrier": vy_carrier,
                    "carrier_velocity": carrier_velocity,
                    "frame": frame,
                }

                matrix, mask, target = tensor_converter(sample)
                target_value = int(row["reward_label"]) if self.label_mode == "reward" else int(target[0])

                if is_train:
                    self.train_data.append(matrix)
                    self.train_mask.append(mask)
                    self.train_labels.append(target_value)
                else:
                    self.test_data.append(matrix)
                    self.test_mask.append(mask)
                    self.test_labels.append(target_value)

    def __len__(self):
        return len(self.train_data)

    def __getitem__(self, idx):
        matrix = self.train_data[idx]
        mask = self.train_mask[idx]
        target = self.train_labels[idx]
        return matrix, mask, target

    def get_validation_data(self):
        val_data = torch.stack(self.val_data)
        val_labels = torch.tensor(self.val_labels, dtype=torch.long)
        val_mask = torch.stack(self.val_mask)
        val_dataset = TensorDataset(
            val_data,
            val_mask,
            val_labels
        )
        return val_dataset

    def get_test_data(self):
        # When no explicit test_directory is provided, reuse validation split for test-time checks.
        if len(self.test_data) == 0:
            if len(self.val_data) > 0:
                return self.get_validation_data()
            raise ValueError("No test samples available. Provide test_directory or ensure data is loaded.")

        test_data = torch.stack(self.test_data)
        test_labels = torch.tensor(self.test_labels, dtype=torch.long)
        test_mask = torch.stack(self.test_mask)
        test_dataset = TensorDataset(
            test_data,
            test_mask,
            test_labels
        )
        return test_dataset



if __name__ == "__main__":
    train_directory = 'passes'
    # teste_directory = '/home_cerberus/disk2/diogochaves/FUTEBOL/Simplified_EPV_in_PFF_Data/data/Test_Pass'
    dataset = PFFDataset(train_directory,test_directory=None, split_ratio=0.8)


    train_loader = DataLoader(dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(dataset.get_validation_data(), batch_size=32, shuffle=False)
    test_loader = DataLoader(dataset.get_test_data(), batch_size=32, shuffle=False)


    for batch_idx, (data, mask, target) in enumerate(test_loader):
        print(f'Batch {batch_idx + 1}:')
        print('Data:')
        print(data)
        print('Mask:')
        print(mask)
        print('Target:')
        print(target)
        print('---')
goal_x_left, goal_y_left = -52.5, 0
goal_x_right, goal_y_right = 52.5, 0


