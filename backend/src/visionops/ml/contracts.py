"""Framework-independent boundaries for future inference and experiment tracking."""

from typing import Protocol

from pydantic import BaseModel, Field


class Prediction(BaseModel):
    label: str
    score: float = Field(ge=0, le=1)
    model_version: str


class Predictor(Protocol):
    def predict(self, inputs: list[float]) -> Prediction:
        """Return a versioned prediction; implementations own preprocessing."""
        ...


class ExperimentTracker(Protocol):
    def log_metrics(self, metrics: dict[str, float], *, step: int = 0) -> None:
        """Adapter boundary for MLflow or another tracking backend."""
        ...
