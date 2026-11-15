"""Validated video pipeline boundaries, independent of HTTP and storage."""

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class Record(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class VideoMetadata(Record):
    filename: str
    source_path: Path
    duration_seconds: float = Field(gt=0)
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    fps: float = Field(gt=0)
    codec: str = Field(min_length=1)
    bitrate_bps: int = Field(gt=0)
    bitrate_source: Literal["container", "estimated"]
    file_size_bytes: int = Field(gt=0)
    pixel_format: str


class EncodingConfig(Record):
    codec: Literal["libx264"] = "libx264"
    crf: int = Field(default=23, ge=0, le=51, strict=True)
    preset: Literal[
        "ultrafast",
        "superfast",
        "veryfast",
        "faster",
        "fast",
        "medium",
        "slow",
        "slower",
        "veryslow",
    ] = "medium"
    width: int | None = Field(default=None, gt=0, le=16384, strict=True)
    height: int | None = Field(default=None, gt=0, le=16384, strict=True)
    threads: int = Field(default=1, ge=1, le=64, strict=True)
    pixel_format: Literal["yuv420p"] = "yuv420p"

    @model_validator(mode="after")
    def validate_resolution(self) -> "EncodingConfig":
        if (self.width is None) != (self.height is None):
            raise ValueError("Provide both width and height, or neither")
        if self.width is not None and (self.width % 2 or self.height % 2):
            raise ValueError("H.264 yuv420p requires even dimensions")
        return self


class EncodingResult(Record):
    output_path: Path
    elapsed_seconds: float = Field(ge=0)
    metadata: VideoMetadata


class QualityMetrics(Record):
    psnr_db: float | None = None
    psnr_infinite: bool = False
    ssim: float = Field(ge=-1, le=1)
    vmaf: float | None = Field(default=None, ge=0, le=100)
    frame_count: int = Field(gt=0)

    @model_validator(mode="after")
    def validate_psnr(self) -> "QualityMetrics":
        if self.psnr_infinite != (self.psnr_db is None):
            raise ValueError("Infinite PSNR must use null psnr_db and psnr_infinite=true")
        return self
