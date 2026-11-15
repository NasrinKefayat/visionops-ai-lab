"""Path-free, finite response models for the versioned analytics API."""

from datetime import datetime
from typing import Annotated, Generic, Literal, TypeVar
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

T = TypeVar("T")


class Response(BaseModel):
    model_config = ConfigDict(extra="ignore", allow_inf_nan=False)


class Page(Response, Generic[T]):
    items: list[T]
    total: int
    offset: int
    limit: int
    available: bool = True
    message: str | None = None


class VideoResponse(Response):
    id: int
    display_name: str
    dataset_kind: Literal["synthetic_video", "public_video"]
    duration_seconds: float
    width: int
    height: int
    fps: float
    codec: str = Field(pattern=r"^[a-zA-Z0-9_.-]{1,40}$")
    bitrate_bps: int
    file_size_bytes: int
    available_crfs: list[int] = Field(default_factory=list)
    available_run_ids: list[str] = Field(default_factory=list)


class EncodingResponse(Response):
    experiment_id: UUID
    run_id: UUID
    video_id: int
    crf: int
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
    ]
    status: Literal["completed", "running", "failed"]
    width: int | None = None
    height: int | None = None
    output_bitrate_bps: int | None = None
    output_file_size_bytes: int | None = None
    encoding_seconds: float | None = None
    compression_ratio: float | None = None
    bitrate_reduction_percent: float | None = None
    measured_ssim: float | None = None
    measured_psnr_db: float | None = None
    psnr_infinite: bool = False
    started_at: datetime
    completed_at: datetime | None = None
    seed: int
    code_revision: str | None = None
    error_code: str | None = None


class MonitoringResponse(Response):
    database_status: Literal["ready", "unavailable"]
    publication_status: Literal["ready", "not_configured", "unavailable"]
    source_videos: int | None
    completed_encodes: int | None
    average_measured_ssim: float | None
    average_measured_psnr_db: float | None
    infinite_psnr_count: int | None
    average_output_bitrate_bps: float | None
    average_encoding_seconds: float | None
    available_ml_experiments: int
    status_counts: dict[Literal["completed", "running", "failed"], int]
    message: str | None = None


class RegressionScores(Response):
    mae: float | None = None
    rmse: float | None = None
    r2: float | None = None


class PolicyScores(Response):
    test_sources: int = Field(ge=0)
    selected_sources: int = Field(ge=0)
    abstention_rate: float = Field(ge=0, le=1)
    violation_rate_among_selected: float | None = Field(default=None, ge=0, le=1)
    mean_bitrate_reduction_percent_vs_fixed: float | None = None
    mean_absolute_distance_from_oracle_percent: float | None = Field(default=None, ge=0)
    oracle_comparable_sources: int = Field(ge=0)


class Importance(Response):
    mean: float
    std: float = Field(ge=0)


class MLflowReference(Response):
    run_id: str | None = None
    status: str = "unavailable"


class RecommendationResponse(Response):
    source_name: str
    crf: int | None = Field(default=None, ge=0, le=51)
    predicted_ssim: float | None = Field(default=None, ge=-1, le=1)
    predicted_bitrate_bps: float | None = Field(default=None, gt=0)
    constraint_status: Literal["predicted_feasible", "no_predicted_candidate"]
    explanation: str
    measured_ssim: float | None = Field(default=None, ge=-1, le=1)
    measured_bitrate_bps: float | None = Field(default=None, gt=0)


class StudySummary(Response):
    id: str
    dataset_kind: Literal["synthetic_only", "public_video"]
    source_count: int = Field(ge=0)
    row_count: int = Field(ge=0)
    seed: int
    code_revision: str | None = None
    selected_models: dict[Literal["ssim", "bitrate_bps"], Literal["dummy", "random_forest"]]
    extractor_version: str
    model_status: Literal["missing", "metadata_only", "configured"]
    minimum_ssim: float | None = Field(default=None, gt=0, le=1)
    mlflow: MLflowReference


class StudyResponse(StudySummary):
    split_groups: dict[Literal["train", "validation", "test"], int]
    test_metrics: dict[Literal["ssim", "bitrate_bps"], RegressionScores]
    policies: dict[Literal["fixed", "content_aware", "oracle"], PolicyScores]
    feature_importance: dict[str, dict[str, Importance]]
    recommendations: list[RecommendationResponse]


class InferenceRequest(Response):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    video_id: int = Field(ge=1, strict=True)
    minimum_ssim: float = Field(gt=0, le=1)
    candidates: list[Annotated[int, Field(strict=True, ge=0, le=51)]] | None = Field(
        default=None, min_length=1, max_length=52
    )


class ErrorDetail(Response):
    code: str
    message: str
    fields: list[str] | None = None


class ErrorResponse(Response):
    error: ErrorDetail
