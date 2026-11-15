"""Versioned analytics routes. Reads never trigger encoding, training or model loading."""

from typing import Annotated, Literal
from uuid import UUID

from fastapi import APIRouter, Path, Query
from sqlalchemy import select

from visionops.api.artifacts import studies
from visionops.api.dependencies import DatabaseDependency, PublicationDependency, SettingsDependency
from visionops.api.errors import APIError
from visionops.api.queries import (
    encoding_response,
    encoding_statement,
    list_encodings,
    list_videos,
    require_database,
    require_video,
    video_response,
)
from visionops.api.schemas import (
    EncodingResponse,
    ErrorResponse,
    InferenceRequest,
    MonitoringResponse,
    Page,
    RecommendationResponse,
    StudyResponse,
    StudySummary,
    VideoResponse,
)
from visionops.api.services import find_study, monitoring, recommend
from visionops.db.models import EncodingConfiguration, Experiment

router = APIRouter(
    prefix="/api/v1",
    tags=["analytics"],
    responses={status: {"model": ErrorResponse} for status in (404, 422, 503)},
)
Offset = Annotated[int, Query(ge=0, le=1_000_000)]
Limit = Annotated[int, Query(ge=1, le=100)]
CRF = Annotated[int | None, Query(ge=0, le=51)]
VideoID = Annotated[int, Path(ge=1, le=2**63 - 1)]


@router.get("/monitoring", response_model=MonitoringResponse)
def get_monitoring(
    db: DatabaseDependency, published: PublicationDependency, settings: SettingsDependency
):
    return monitoring(db, published, settings)


@router.get("/videos", response_model=Page[VideoResponse])
def get_videos(
    db: DatabaseDependency,
    published: PublicationDependency,
    offset: Offset = 0,
    limit: Limit = 20,
    dataset: Literal["synthetic_video", "public_video"] | None = None,
    search: Annotated[str | None, Query(max_length=80)] = None,
):
    return list_videos(db, published, offset, limit, dataset, search)


@router.get("/videos/{video_id}", response_model=VideoResponse)
def get_video(video_id: VideoID, db: DatabaseDependency, published: PublicationDependency):
    video = require_video(db, published, video_id)
    rows = db.execute(
        select(EncodingConfiguration.parameters, Experiment.run_id)
        .join(Experiment, Experiment.config_id == EncodingConfiguration.fingerprint)
        .where(Experiment.video_id == video_id)
    ).all()
    return video_response(
        video, published, sorted({row[0]["crf"] for row in rows}), sorted({row[1] for row in rows})
    )


@router.get("/videos/{video_id}/encodings", response_model=Page[EncodingResponse])
@router.get("/videos/{video_id}/quality", response_model=Page[EncodingResponse])
@router.get("/videos/{video_id}/comparison", response_model=Page[EncodingResponse])
def get_encodings(
    video_id: VideoID,
    db: DatabaseDependency,
    published: PublicationDependency,
    offset: Offset = 0,
    limit: Limit = 20,
    crf: CRF = None,
    run_id: UUID | None = None,
):
    require_video(db, published, video_id)
    return list_encodings(
        db, published, offset, limit, video_id, crf, str(run_id) if run_id else None
    )


@router.get("/experiments", response_model=Page[EncodingResponse])
def get_experiments(
    db: DatabaseDependency,
    published: PublicationDependency,
    offset: Offset = 0,
    limit: Limit = 20,
    video_id: int | None = Query(default=None, ge=1, le=2**63 - 1),
    crf: CRF = None,
    status: Literal["completed", "running", "failed"] = "completed",
):
    return list_encodings(db, published, offset, limit, video_id=video_id, crf=crf, status=status)


@router.get("/experiments/{experiment_id}", response_model=EncodingResponse)
def get_experiment(experiment_id: UUID, db: DatabaseDependency, published: PublicationDependency):
    session = require_database(db)
    row = session.execute(
        encoding_statement(published).where(Experiment.id == str(experiment_id))
    ).first()
    if row is None:
        raise APIError(404, "not_found", "Experiment not found.")
    return encoding_response(row)


def study_page(settings, published, offset, limit):
    records = studies(settings, published)
    return Page[StudySummary](
        items=[r.response for r in records[offset : offset + limit]],
        total=len(records),
        offset=offset,
        limit=limit,
        available=settings.dashboard_reports is not None and published.status == "ready",
        message=None if records else "No published research experiments are available.",
    )


@router.get("/studies", response_model=Page[StudySummary])
@router.get("/models", response_model=Page[StudySummary])
def get_studies(
    settings: SettingsDependency,
    published: PublicationDependency,
    offset: Offset = 0,
    limit: Limit = 20,
):
    return study_page(settings, published, offset, limit)


@router.get("/studies/{study_id}", response_model=StudyResponse)
@router.get("/models/{study_id}", response_model=StudyResponse)
def get_study(study_id: UUID, settings: SettingsDependency, published: PublicationDependency):
    return find_study(str(study_id), settings, published).response


@router.post("/models/{study_id}/recommend", response_model=RecommendationResponse)
def post_recommendation(
    study_id: UUID,
    request: InferenceRequest,
    settings: SettingsDependency,
    published: PublicationDependency,
    db: DatabaseDependency,
):
    return recommend(
        find_study(str(study_id), settings, published), request, db, published, settings
    )
