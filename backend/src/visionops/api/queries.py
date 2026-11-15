"""SQLAlchemy read queries and explicit, path-free encoding projections."""

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from visionops.api.errors import APIError
from visionops.api.publication import Publication, safe_revision
from visionops.api.schemas import EncodingResponse, Page, VideoResponse
from visionops.db.models import (
    EncodingConfiguration,
    EncodingRecord,
    Experiment,
    QualityRecord,
    Video,
)


def require_database(session: Session | None) -> Session:
    if session is None:
        raise APIError(503, "database_unavailable", "Experiment data is unavailable.")
    return session


def require_video(session: Session | None, published: Publication, video_id: int) -> Video:
    session = require_database(session)
    video = session.get(Video, video_id)
    if video is None or video.sha256 not in published.sources:
        raise APIError(404, "not_found", "Video not found.")
    return video


def video_response(
    video: Video,
    published: Publication,
    crfs: list[int] | None = None,
    runs: list[str] | None = None,
) -> VideoResponse:
    source = published.sources[video.sha256]
    return VideoResponse(
        id=video.id,
        display_name=source.name,
        dataset_kind=source.domain,
        duration_seconds=video.duration_seconds,
        width=video.width,
        height=video.height,
        fps=video.fps,
        codec=video.codec,
        bitrate_bps=video.bitrate_bps,
        file_size_bytes=video.file_size_bytes,
        available_crfs=crfs or [],
        available_run_ids=runs or [],
    )


def list_videos(
    session: Session | None,
    published: Publication,
    offset: int,
    limit: int,
    dataset: str | None,
    search: str | None,
) -> Page[VideoResponse]:
    if session is None or published.status != "ready":
        return Page(
            items=[],
            total=0,
            offset=offset,
            limit=limit,
            available=False,
            message="No published experiment data is available.",
        )
    allowed = [
        digest
        for digest, source in published.sources.items()
        if (dataset is None or source.domain == dataset)
        and (not search or search.casefold() in source.name.casefold())
    ]
    statement = select(Video).where(Video.sha256.in_(allowed))
    total = session.scalar(select(func.count()).select_from(statement.subquery()))
    videos = session.scalars(statement.order_by(Video.id).offset(offset).limit(limit)).all()
    return Page(
        items=[video_response(v, published) for v in videos],
        total=total,
        offset=offset,
        limit=limit,
    )


def encoding_statement(published: Publication):
    return (
        select(Experiment, EncodingConfiguration, EncodingRecord, QualityRecord, Video)
        .join(Video, Experiment.video_id == Video.id)
        .join(EncodingConfiguration, Experiment.config_id == EncodingConfiguration.fingerprint)
        .outerjoin(EncodingRecord, Experiment.id == EncodingRecord.experiment_id)
        .outerjoin(QualityRecord, Experiment.id == QualityRecord.experiment_id)
        .where(Video.sha256.in_(list(published.sources)))
    )


def encoding_response(row) -> EncodingResponse:
    experiment, config, encoded, quality, source = row
    return EncodingResponse(
        experiment_id=experiment.id,
        run_id=experiment.run_id,
        video_id=source.id,
        crf=config.parameters["crf"],
        preset=config.parameters["preset"],
        status=experiment.status,
        width=config.parameters.get("width") or source.width,
        height=config.parameters.get("height") or source.height,
        output_bitrate_bps=encoded.bitrate_bps if encoded else None,
        output_file_size_bytes=encoded.file_size_bytes if encoded else None,
        encoding_seconds=encoded.elapsed_seconds if encoded else None,
        compression_ratio=source.file_size_bytes / encoded.file_size_bytes
        if encoded and encoded.file_size_bytes > 0
        else None,
        bitrate_reduction_percent=100 * (1 - encoded.bitrate_bps / source.bitrate_bps)
        if encoded and source.bitrate_bps > 0
        else None,
        measured_ssim=quality.ssim if quality else None,
        measured_psnr_db=quality.psnr_db if quality else None,
        psnr_infinite=quality.psnr_infinite if quality else False,
        started_at=experiment.started_at,
        completed_at=experiment.completed_at,
        seed=experiment.seed,
        code_revision=safe_revision(experiment.code_revision),
        error_code="processing_failed" if experiment.error else None,
    )


def list_encodings(
    session: Session | None,
    published: Publication,
    offset: int,
    limit: int,
    video_id: int | None = None,
    crf: int | None = None,
    run_id: str | None = None,
    status: str | None = None,
) -> Page[EncodingResponse]:
    if session is None or published.status != "ready":
        return Page(
            items=[],
            total=0,
            offset=offset,
            limit=limit,
            available=False,
            message="No published experiment data is available.",
        )
    statement = encoding_statement(published)
    if video_id is not None:
        statement = statement.where(Experiment.video_id == video_id)
    if status is not None:
        statement = statement.where(Experiment.status == status)
    if run_id is not None:
        statement = statement.where(Experiment.run_id == run_id)
    if crf is not None:
        # JSON access is portable across SQLite and future PostgreSQL.
        statement = statement.where(EncodingConfiguration.parameters["crf"].as_integer() == crf)
    total = session.scalar(select(func.count()).select_from(statement.subquery()))
    rows = session.execute(
        statement.order_by(Experiment.started_at.desc(), Experiment.id).offset(offset).limit(limit)
    ).all()
    return Page(
        items=[encoding_response(row) for row in rows], total=total, offset=offset, limit=limit
    )
