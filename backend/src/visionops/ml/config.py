"""Strict YAML study configuration; paths resolve relative to the YAML file."""

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from visionops.video.features import FeatureSettings


class Options(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ForestSettings(Options):
    n_estimators: int = Field(default=64, ge=1, le=500)
    max_depth: int = Field(default=8, ge=1, le=32)
    min_samples_leaf: int = Field(default=1, ge=1)


class StudyConfig(Options):
    manifest: Path
    database: Path
    output: Path
    tracking_directory: Path
    tracking: bool = True
    experiment_name: Literal["encoding-optimization"] = "encoding-optimization"
    seed: Literal[42] = 42
    candidates: list[int] = Field(default_factory=lambda: [18, 23, 28])
    fixed_crf: int = 23
    minimum_ssim: float = Field(default=0.99, gt=0, le=1)
    validation_fraction: float = Field(default=0.2, gt=0, lt=1)
    test_fraction: float = Field(default=0.2, gt=0, lt=1)
    features: FeatureSettings = Field(default_factory=FeatureSettings)
    forest: ForestSettings = Field(default_factory=ForestSettings)
    preset: Literal["medium"] = "medium"

    @model_validator(mode="after")
    def validate_protocol(self):
        if sorted(self.candidates) != [18, 23, 28]:
            raise ValueError("Phase 3 uses the measured Phase 2 candidate grid [18, 23, 28]")
        if self.fixed_crf not in self.candidates:
            raise ValueError("Fixed CRF must be in the candidate grid")
        if self.validation_fraction + self.test_fraction >= 1:
            raise ValueError("Leave training data after validation/test splits")
        return self


def load_config(path: Path) -> StudyConfig:
    path = path.resolve()
    config = StudyConfig.model_validate(yaml.safe_load(path.read_text()))
    for name in ("manifest", "database", "output", "tracking_directory"):
        setattr(config, name, (path.parent / getattr(config, name)).resolve())
    return config
