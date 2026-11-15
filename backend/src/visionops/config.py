"""Central runtime settings; environment variables use the VISIONOPS_ prefix."""

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="VISIONOPS_", extra="ignore")

    ffmpeg_binary: str = "ffmpeg"
    ffprobe_binary: str = "ffprobe"
    video_timeout_seconds: int = Field(default=300, ge=1, le=86400)

    # Read-only dashboard publication is opt-in. No paths are accepted from API clients.
    dashboard_manifest: Path | None = None
    dashboard_reports: Path | None = None
    dashboard_mlflow_directory: Path | None = None
    dashboard_trusted_models: dict[str, str] = Field(default_factory=dict)

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
