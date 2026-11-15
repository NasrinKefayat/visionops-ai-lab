"""Regression summaries; independent of training and video quality filters."""

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def regression_metrics(y, prediction, groups) -> dict[str, float | None]:
    y, prediction = np.asarray(y, dtype=float), np.asarray(prediction, dtype=float)
    if (
        len(groups) != len(y)
        or y.shape != prediction.shape
        or not y.size
        or not np.isfinite(prediction).all()
        or not np.isfinite(y).all()
    ):
        raise ValueError("Predictions must be finite and match nonempty targets")
    return {
        "mae": float(mean_absolute_error(y, prediction)),
        "rmse": float(np.sqrt(mean_squared_error(y, prediction))),
        "r2": float(r2_score(y, prediction))
        if len(set(groups)) >= 5 and np.var(y) > 1e-15
        else None,
    }
