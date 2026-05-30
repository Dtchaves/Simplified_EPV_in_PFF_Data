from __future__ import annotations

from typing import Dict

import numpy as np
from sklearn.metrics import log_loss, mean_squared_error


def evaluate_dp(y_true: np.ndarray, p_success: np.ndarray) -> Dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    p_success = np.asarray(p_success, dtype=float)
    p_success = np.clip(p_success, 1e-6, 1.0 - 1e-6)

    return {
        "count": float(y_true.size),
        "bce": float(log_loss(y_true, p_success, labels=[0, 1])) if y_true.size > 0 else float("nan"),
        "accuracy": float(((p_success >= 0.5).astype(int) == y_true.astype(int)).mean()) if y_true.size > 0 else float("nan"),
    }


def evaluate_de(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return {
        "count": float(y_true.size),
        "mse": float(mean_squared_error(y_true, y_pred)) if y_true.size > 0 else float("nan"),
    }
