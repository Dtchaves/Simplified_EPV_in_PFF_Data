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

try:
    from .utils import ToSoccerMapTensor
except ImportError:
    from utils import ToSoccerMapTensor

PASS_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[3]
if str(PASS_ROOT) not in sys.path:
    sys.path.insert(0, str(PASS_ROOT))

try:
    from data_utils import (
        EPV_CACHE_VERSION,
        build_sample_key,
        discover_pass_sources,
        split_sources_train_val_test_by_mode,
        get_optional_fingerprint,
        load_or_build_canonical_pass_cache,
        load_tensor_cache,
        save_tensor_cache,
    )
    from reward_labels import PassRewardLabeler
except ImportError:
    # Fallback for relative imports when package is properly structured
    from ..data_utils import (
        EPV_CACHE_VERSION,
        build_sample_key,
        discover_pass_sources,
        split_sources_train_val_test_by_mode,
        get_optional_fingerprint,
        load_or_build_canonical_pass_cache,
        load_tensor_cache,
        save_tensor_cache,
    )
    from ..reward_labels import PassRewardLabeler


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
        split_mode: str = "row",
        split_manifest_path: Optional[str] = None,
        split_seed: int = 42,
        season_sample_ratio: Optional[float] = None,
        cache_enabled: Optional[bool] = None,
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
        self.val_metadata = []
        self.test_metadata = []

        self.pass_outcome_filter = pass_outcome_filter
        self.split_mode = str(split_mode).strip().lower()
        self.split_manifest_path = split_manifest_path
        self.split_seed = int(split_seed)
        self.season_sample_ratio = season_sample_ratio
        self.cache_enabled = cache_enabled

        event_root = Path(reward_event_directory)
        if not event_root.is_absolute():
            event_root = (REPO_ROOT / event_root).resolve()

        self.event_root_str = str(event_root)  # Store resolved path for later use

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
        self.pp_model_path = str(pp_path_obj) if pp_path_obj is not None else None
        self.tensor_converter = ToSoccerMapTensor(pp_model_path=pp_path_obj)

        if test_directory:
            self._load_data(train_directory, split_name="train")
            self._load_data(test_directory, split_name="test")
        else:
            if self.split_mode == "match":
                sources = self._discover_sources(train_directory)
                train_sources, val_sources, test_sources, _ = split_sources_train_val_test_by_mode(
                    sources,
                    train_ratio=split_ratio,
                    split_mode="match",
                    split_seed=self.split_seed,
                    split_manifest_path=self.split_manifest_path,
                    season_sample_ratio=self.season_sample_ratio,
                )
                self._load_data(train_directory, split_name="train", sources=train_sources)
                self._load_data(train_directory, split_name="val", sources=val_sources)
                self._load_data(train_directory, split_name="test", sources=test_sources)
            else:
                self._load_data(train_directory, split_name="train")
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


    def _append_payload(self, payload: dict, split_name: str) -> None:
        data_list = payload.get("data", [])
        mask_list = payload.get("mask", [])
        label_list = payload.get("labels", [])
        metadata_list = payload.get("metadata", [])

        if split_name == "train":
            self.train_data.extend(data_list)
            self.train_mask.extend(mask_list)
            self.train_labels.extend(label_list)
            self.train_metadata.extend(metadata_list)
        elif split_name == "val":
            self.val_data.extend(data_list)
            self.val_mask.extend(mask_list)
            self.val_labels.extend(label_list)
            self.val_metadata.extend(metadata_list)
        else:
            self.test_data.extend(data_list)
            self.test_mask.extend(mask_list)
            self.test_labels.extend(label_list)
            self.test_metadata.extend(metadata_list)

    def _build_tensor_payload(self, df: pd.DataFrame, filepath: Path) -> dict:
        matrices = []
        masks = []
        labels = []
        sample_keys = []
        metadata = []

        for idx, row in tqdm(df.iterrows(), total=df.shape[0], desc=f"Processando amostras do {filepath.name}"):
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

            matrices.append(matrix)
            masks.append(mask)
            labels.append(target_value)
            sample_keys.append(build_sample_key(row, idx))
            metadata.append(
                {
                    "source_file": filepath.name,
                    "pass_outcome_type": row["pass_outcome_type"],
                    "reward_label": target_value,
                    "reward_status": row.get("reward_status", None),
                    "reward_join_strategy": row.get("reward_join_strategy", None),
                }
            )

        return {
            "data": matrices,
            "mask": masks,
            "labels": labels,
            "sample_keys": sample_keys,
            "metadata": metadata,
        }

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

    def _load_data(self, directory, split_name="train", sources=None):
        if sources is None:
            try:
                sources = self._discover_sources(directory)
            except FileNotFoundError as exc:
                print(f"[WARNING] {exc}")
                return

        split_label = {"train": "treino", "val": "validacao", "test": "teste"}.get(split_name, split_name)
        print(f"Descobertos {len(sources)} fontes de dados para {split_label}")

        # Required columns that MUST exist prior to event merge
        required_cols = [
            'game_id', 'game_event_id', 'possession_event_id', 'player_id',
            'ball_x_start', 'ball_y_start',
            'ball_x_end', 'ball_y_end', 'team_id'
        ]

        cache_dependencies = {
            "pass_outcome_filter": self.pass_outcome_filter,
            "tensor_family": "epv",
            "spatial_dim": [68, 104],
            "reward_horizon_seconds": self.reward_labeler.horizon_seconds,
            "include_open_play_null": self.reward_labeler.include_open_play_null,
            "pp_model_fingerprint": get_optional_fingerprint(self.pp_model_path),
        }

        for source in sources:
            source_path = Path(source["source_path"])
            source_name = source.get("source_name", source_path.name)
            try:
                df, canonical_summary = load_or_build_canonical_pass_cache(
                    source=source,
                    required_columns=required_cols,
                    source_filename=source_name,
                    event_root=self.event_root_str,
                    cache_enabled=self.cache_enabled,
                )
            except Exception as e:
                print(f"[ERROR] Failed to load {source_name}: {e}")
                continue

            df = df[df["pass_outcome_type"].notna()].copy()
            df = df[df["pass_outcome_type"] == self.pass_outcome_filter].copy()
            if df.empty:
                continue

            df, _ = self.reward_labeler.label_pass_dataframe(
                df,
                source_filename=source_name,
                drop_unlabeled=True,
                source_kind=str(source.get("source_kind", "legacy_wide")),
                processed_events_path=source.get("events_path"),
                processed_tracking_path=source.get("tracking_path"),
            )
            if df.empty:
                continue

            cached_payload = load_tensor_cache(
                source_path=source_path,
                cache_family="epv",
                artifact_stem="success_reward_tensors",
                cache_version=EPV_CACHE_VERSION,
                dependencies=cache_dependencies,
                cache_enabled=self.cache_enabled,
            )
            if cached_payload is not None:
                self._append_payload(cached_payload, split_name=split_name)
                continue

            payload = self._build_tensor_payload(df, source_path)
            payload = save_tensor_cache(
                source_path=source_path,
                cache_family="epv",
                artifact_stem="success_reward_tensors",
                cache_version=EPV_CACHE_VERSION,
                payload=payload,
                dependencies=cache_dependencies,
                cache_enabled=self.cache_enabled,
            )
            self._append_payload(payload, split_name=split_name)

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
        # When no explicit test_directory is provided, reuse validation split for test-time checks.
        if len(self.test_data) == 0:
            if len(self.val_data) > 0:
                return self.get_validation_data()
            raise ValueError("No test samples available. Provide test_directory or ensure data is loaded.")

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
