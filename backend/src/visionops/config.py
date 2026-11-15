"""Central runtime settings; environment variables use the VISIONOPS_ prefix."""

from functools import lru_cache
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="VISIONOPS_", extra="ignore")

    app_name: str = "VisionOps-AI"
    database_url: str = "sqlite:///./visionops.db"
    random_seed: int = 42
    device: Literal["cpu", "cuda"] = "cpu"
    mlflow_tracking_uri: str = "sqlite:///./mlflow.db"
    model_version: str = "untrained"
    api_port: int = Field(default=8000, ge=1, le=65535)


@lru_cache
def get_settings() -> Settings:
    return Settings()
