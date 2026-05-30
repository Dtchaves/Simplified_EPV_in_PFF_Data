from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from .features import ActionSelectionFeatureBuilder
from Shot.baseline_xg import BaselineXGArtifacts


def build_action_selection_dataset(
    actions_df: pd.DataFrame,
    tracking_df: Optional[pd.DataFrame] = None,
    baseline_xg_artifacts: Optional[BaselineXGArtifacts] = None,
    output_path: Optional[Path] = None,
) -> pd.DataFrame:
    """Build dataset for action-selection model.

    Expects `actions_df` to contain an `action_label` column with values
    in {"pass", "ball_drive", "shot"} (case-insensitive). Returns a
    DataFrame with feature columns and integer `label` column.
    """
    if actions_df is None or actions_df.empty:
        raise ValueError("actions_df is empty")

    builder = ActionSelectionFeatureBuilder()
    features = builder.build_feature_frame(actions_df, tracking_df, baseline_xg_artifacts)

    # Normalize label column
    if "action_label" not in actions_df.columns:
        raise ValueError("actions_df must contain 'action_label' column")

    label_map = {"pass": 0, "ball_drive": 1, "shot": 2}

    def _map_label(v: Any) -> int:
        if pd.isna(v):
            raise ValueError("Found NaN in action_label column")
        token = str(v).strip().lower()
        if token in label_map:
            return label_map[token]
        # allow common synonyms
        if token in {"carry", "dribble"}:
            return label_map["ball_drive"]
        if token in {"cross", "through_ball"}:
            return label_map["pass"]
        raise ValueError(f"Unknown action label: {v}")

    labels = actions_df["action_label"].apply(_map_label).astype(int)

    dataset = features.copy()
    dataset["label"] = labels.values

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        dataset.to_parquet(output_path, index=False)

    return dataset
