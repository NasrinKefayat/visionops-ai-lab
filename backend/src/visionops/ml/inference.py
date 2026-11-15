"""Trusted model loading and inference; never imports the training pipeline."""

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import joblib
import numpy as np


class Regressor(Protocol):
    def fit(self, x: np.ndarray, y: np.ndarray): ...
    def predict(self, x: np.ndarray) -> np.ndarray: ...


@dataclass
class ModelBundle:
    models: dict[str, Regressor]
    feature_names: tuple[str, ...]
    metadata: dict


def load_bundle(path: Path) -> ModelBundle:
    # Pickle/joblib can execute code: only load an artifact you trust.
    bundle = joblib.load(path)
    if not isinstance(bundle, ModelBundle):
        raise ValueError("Not a VisionOps model bundle")
    return bundle


def predict_values(bundle: ModelBundle, x: np.ndarray) -> dict[str, np.ndarray]:
    x = np.asarray(x, dtype=float)
    if x.ndim != 2 or x.shape[1] != len(bundle.feature_names) or not np.isfinite(x).all():
        raise ValueError("Prediction input does not match the model feature schema")
    predictions = {
        name: np.asarray(model.predict(x), dtype=float) for name, model in bundle.models.items()
    }
    if any(p.shape != (len(x),) or not np.isfinite(p).all() for p in predictions.values()):
        raise ValueError("Model produced invalid predictions")
    return predictions
