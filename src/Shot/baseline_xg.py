from __future__ import annotations

import importlib
import json
import math
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler

from Pass.data_utils import REPO_ROOT, discover_action_sources, sample_sources_by_season


PITCH_LENGTH = 105.0
PITCH_WIDTH = 68.0
HALF_LENGTH = PITCH_LENGTH / 2.0
HALF_WIDTH = PITCH_WIDTH / 2.0
GOAL_X = HALF_LENGTH
GOAL_Y = 0.0
GOAL_POST_Y = 3.66


@dataclass
class BaselineXGConfig:
    source_root: str = "data/processed"
    source_format: str = "pff_match_triplets"
    orientation_mode: str = "attack_right"
    flip_away_team_coordinates: bool = True
    split_seed: int = 42
    season_sample_ratio: Optional[float] = None


@dataclass
class BaselineXGArtifacts:
    model: Any
    scaler: StandardScaler
    feature_columns: Tuple[str, ...]

    def predict_proba(self, features: pd.DataFrame) -> np.ndarray:
        matrix = features[list(self.feature_columns)].to_numpy(dtype=float)
        scaled = self.scaler.transform(matrix)
        return self.model.predict_proba(scaled)


def _normalize_token(value: Any) -> Optional[str]:
    if value is None:
        return None
    token = str(value).strip().lower()
    if not token or token in {"nan", "none", "null", "na", "n/a", "nat"}:
        return None
    return token


def _normalize_set_piece(value: Any) -> Optional[str]:
    token = _normalize_token(value)
    if token is None:
        return None
    mapping = {
        "open_play": "open_play",
        "corner": "corner",
        "free_kick": "free_kick",
        "throw_in": "throw_in",
        "penalty": "penalty",
        "goal_kick": "goal_kick",
        "kick_off": "kick_off",
        "o": "open_play",
        "c": "corner",
        "f": "free_kick",
        "t": "throw_in",
        "p": "penalty",
        "g": "goal_kick",
        "k": "kick_off",
    }
    return mapping.get(token, token)


def _resolve_ball_xy(row: pd.Series) -> Tuple[Optional[float], Optional[float]]:
    for x_key, y_key in (
        ("ball_x", "ball_y"),
        ("ball_x_start", "ball_y_start"),
        ("x", "y"),
    ):
        if x_key in row.index and y_key in row.index:
            x_val = pd.to_numeric(row.get(x_key), errors="coerce")
            y_val = pd.to_numeric(row.get(y_key), errors="coerce")
            if pd.notna(x_val) and pd.notna(y_val):
                return float(x_val), float(y_val)
    return None, None


def _orient_attack_right(x: float, y: float, team_side: Any, flip_away: bool) -> Tuple[float, float]:
    side = str(team_side).strip().lower()
    if flip_away and side == "away":
        return -float(x), -float(y)
    return float(x), float(y)


def _shot_outcome_goal(shot_outcome: Any) -> int:
    token = _normalize_token(shot_outcome)
    if token in {"goal", "g"}:
        return 1
    return 0


def _header_flag(body_part: Any) -> int:
    token = _normalize_token(body_part)
    if token in {"head", "he", "header"}:
        return 1
    return 0


def build_baseline_xg_dataset(
    config: Optional[BaselineXGConfig] = None,
    output_path: Optional[Path] = None,
) -> pd.DataFrame:
    config = config or BaselineXGConfig()
    sources = discover_action_sources(config.source_root, source_format=config.source_format, prefer_parquet=True)
    sources, _ = sample_sources_by_season(
        sources,
        season_sample_ratio=config.season_sample_ratio,
        split_seed=config.split_seed,
    )

    rows: List[Dict[str, Any]] = []
    for source in sources:
        events_path = source.get("events_path") or source.get("source_path")
        if not events_path:
            continue

        if str(events_path).lower().endswith(".csv"):
            events_df = pd.read_csv(events_path)
        else:
            events_df = pd.read_parquet(events_path)

        if "possession_type" not in events_df.columns:
            continue

        events = events_df.copy()
        events["possession_type"] = events["possession_type"].astype(str).str.lower().str.strip()
        shot_events = events[events["possession_type"] == "shot"].copy()
        if shot_events.empty:
            continue

        for _, row in shot_events.iterrows():
            x_raw, y_raw = _resolve_ball_xy(row)
            if x_raw is None or y_raw is None:
                continue

            team_side = row.get("team_side")
            x_oriented, y_oriented = _orient_attack_right(
                x_raw, y_raw, team_side, flip_away=config.flip_away_team_coordinates
            )

            dx = GOAL_X - x_oriented
            dy = GOAL_Y - y_oriented
            distance = float(math.hypot(dx, dy))
            angle = float(abs(math.atan2(dy, dx)))

            set_piece = _normalize_set_piece(row.get("set_piece"))
            is_open_play = 1 if (set_piece in (None, "open_play")) else 0
            is_set_piece = 1 if (set_piece not in (None, "open_play")) else 0
            is_free_kick = 1 if set_piece == "free_kick" else 0
            is_corner = 1 if set_piece == "corner" else 0
            is_penalty = 1 if set_piece == "penalty" else 0

            rows.append(
                {
                    "match_id": row.get("match_id"),
                    "event_id": row.get("event_id"),
                    "frame_id": row.get("frame_id"),
                    "team_id": row.get("team_id"),
                    "player_id": row.get("player_id"),
                    "shot_x": x_oriented,
                    "shot_y": y_oriented,
                    "distance_to_goal": distance,
                    "angle_to_goal": angle,
                    "is_open_play": is_open_play,
                    "is_set_piece": is_set_piece,
                    "is_free_kick": is_free_kick,
                    "is_corner": is_corner,
                    "is_penalty": is_penalty,
                    "is_header": _header_flag(row.get("body_part")),
                    "is_goal": _shot_outcome_goal(row.get("shot_outcome")),
                }
            )

    dataset = pd.DataFrame(rows)
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        dataset.to_parquet(output_path, index=False)

    return dataset


def train_baseline_xg(
    dataset: pd.DataFrame,
    output_dir: Optional[Path] = None,
) -> BaselineXGArtifacts:
    try:
        xgb_module = importlib.import_module("xgboost")
    except Exception as exc:  # pragma: no cover - depends on local environment
        raise RuntimeError("xgboost is required to train baseline xG model.") from exc

    feature_columns = (
        "shot_x",
        "shot_y",
        "distance_to_goal",
        "angle_to_goal",
        "is_open_play",
        "is_set_piece",
        "is_free_kick",
        "is_corner",
        "is_penalty",
        "is_header",
    )

    clean = dataset.dropna(subset=list(feature_columns) + ["is_goal"]).copy()
    if clean.empty:
        raise ValueError("Baseline xG dataset is empty after filtering.")

    x = clean[list(feature_columns)].to_numpy(dtype=float)
    y = clean["is_goal"].to_numpy(dtype=int)

    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x)

    model_kwargs = {
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "use_label_encoder": False,
    }
    if torch.cuda.is_available():
        model_kwargs["device"] = "cuda"
        model_kwargs["tree_method"] = "hist"

    model = xgb_module.XGBClassifier(**model_kwargs)

    grid = GridSearchCV(
        model,
        param_grid={
            "n_estimators": [50, 100, 250],
            "learning_rate": [1e-3, 1e-2, 1e-1],
            "max_depth": [3, 5, 10],
        },
        cv=10,
        scoring="neg_log_loss",
        n_jobs=-1,
    )
    grid.fit(x_scaled, y)

    best_model = grid.best_estimator_
    artifacts = BaselineXGArtifacts(model=best_model, scaler=scaler, feature_columns=feature_columns)

    resolved_dir = output_dir or (REPO_ROOT / "results" / "models" / "shot")
    resolved_dir.mkdir(parents=True, exist_ok=True)

    model_path = resolved_dir / "baseline_xg_model.pkl"
    scaler_path = resolved_dir / "baseline_xg_feature_scaler.pkl"
    meta_path = resolved_dir / "baseline_xg_metadata.json"

    with model_path.open("wb") as handle:
        pickle.dump(best_model, handle)
    with scaler_path.open("wb") as handle:
        pickle.dump(scaler, handle)
    with meta_path.open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "feature_columns": list(feature_columns),
                "best_params": grid.best_params_,
            },
            handle,
            indent=2,
            sort_keys=True,
        )

    return artifacts


def load_baseline_xg(
    model_path: Optional[Path] = None,
    scaler_path: Optional[Path] = None,
) -> BaselineXGArtifacts:
    resolved_dir = REPO_ROOT / "results" / "models" / "shot"
    model_path = model_path or (resolved_dir / "baseline_xg_model.pkl")
    scaler_path = scaler_path or (resolved_dir / "baseline_xg_feature_scaler.pkl")

    if not model_path.exists() or not scaler_path.exists():
        raise FileNotFoundError("Baseline xG artifacts not found. Train baseline xG first.")

    with model_path.open("rb") as handle:
        model = pickle.load(handle)
    with scaler_path.open("rb") as handle:
        scaler = pickle.load(handle)

    feature_columns = (
        "shot_x",
        "shot_y",
        "distance_to_goal",
        "angle_to_goal",
        "is_open_play",
        "is_set_piece",
        "is_free_kick",
        "is_corner",
        "is_penalty",
        "is_header",
    )

    return BaselineXGArtifacts(model=model, scaler=scaler, feature_columns=feature_columns)
