from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import pickle
import json

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from .models import ActionSelectionNet
from .data import build_action_selection_dataset


@dataclass
class ActionSelectionTrainConfig:
    epochs: int = 30
    batch_size: int = 64
    lr: float = 1e-3
    test_size: float = 0.2


def train_action_selection(
    actions_df: pd.DataFrame,
    tracking_df: Optional[pd.DataFrame] = None,
    baseline_xg_artifacts=None,
    output_dir: Optional[Path] = None,
    config: Optional[ActionSelectionTrainConfig] = None,
) -> Path:
    config = config or ActionSelectionTrainConfig()
    dataset = build_action_selection_dataset(actions_df, tracking_df, baseline_xg_artifacts)
    resolved_dir = output_dir or (Path.cwd() / "results" / "models" / "action_selection")

    feature_columns = [c for c in dataset.columns if c != "label"]
    X = dataset[feature_columns].to_numpy(dtype=float)
    y = dataset["label"].to_numpy(dtype=int)

    if X.size == 0:
        raise ValueError("Empty feature matrix")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_val, y_train, y_val = train_test_split(X_scaled, y, test_size=config.test_size, random_state=42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_features = X_train.shape[1]
    n_actions = int(max(y_train.max(), y_val.max()) + 1) if y_train.size and y_val.size else int(y.max() + 1)

    model = ActionSelectionNet(n_features=n_features, n_actions=n_actions).to(device)
    optimizer = optim.Adam(model.parameters(), lr=config.lr)
    criterion = nn.CrossEntropyLoss()

    train_tensor_X = torch.from_numpy(X_train).float().to(device)
    train_tensor_y = torch.from_numpy(y_train).long().to(device)
    val_tensor_X = torch.from_numpy(X_val).float().to(device)
    val_tensor_y = torch.from_numpy(y_val).long().to(device)

    best_val_loss = float("inf")
    patience = 5
    wait = 0

    for epoch in range(1, config.epochs + 1):
        model.train()
        optimizer.zero_grad()
        logits = model(train_tensor_X)
        loss = criterion(logits, train_tensor_y)
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_logits = model(val_tensor_X)
            val_loss = float(criterion(val_logits, val_tensor_y).item())

        if val_loss < best_val_loss - 1e-6:
            best_val_loss = val_loss
            wait = 0
            # save best model
            resolved_dir.mkdir(parents=True, exist_ok=True)
            model_path = resolved_dir / "action_selection_model.pt"
            scaler_path = resolved_dir / "action_selection_feature_scaler.pkl"
            meta_path = resolved_dir / "action_selection_metadata.json"
            torch.save(model.state_dict(), model_path)
            with scaler_path.open("wb") as fh:
                pickle.dump(scaler, fh)
            with meta_path.open("w", encoding="utf-8") as fh:
                json.dump({"feature_columns": feature_columns}, fh, indent=2)
        else:
            wait += 1
            if wait >= patience:
                break

    return resolved_dir
