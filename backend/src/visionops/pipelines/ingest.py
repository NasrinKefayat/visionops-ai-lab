"""Content-addressed, per-file transactional ETL, independent of FastAPI."""

import hashlib
import json
import logging
from datetime import UTC, datetime
from pathlib import Path

from pydantic import BaseModel, Field
from sqlalchemy import Engine, select
from sqlalchemy.exc import IntegrityError

from visionops.db.models import Video
from visionops.db.session import Base, build_session_factory
from visionops.video.metadata import extract_metadata
from visionops.video.process import VideoError, validate_video_file

logger = logging.getLogger(__name__)


def log_event(event: str, **fields: object) -> None:
    logger.info(json.dumps({"event": event, **fields}, default=str, sort_keys=True))


def content_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def initialize_database(engine: Engine) -> None:
    """Create missing MVP tables explicitly, never at API startup; not a migration tool."""
    Base.metadata.create_all(engine)


class IngestionResult(BaseModel):
    video_id: int
    sha256: str
    duplicate: bool


class BatchResult(BaseModel):
    processed: list[IngestionResult] = Field(default_factory=list)
    errors: dict[str, str] = Field(default_factory=dict)


def ingest_video(path: Path | str, engine: Engine) -> IngestionResult:
    path = validate_video_file(path)
    before = path.stat()
    digest = content_hash(path)
    sessions = build_session_factory(engine)
    with sessions() as session:
        existing = session.scalar(select(Video).where(Video.sha256 == digest))
        if existing:
            log_event("video_duplicate", path=path, video_id=existing.id, sha256=digest)
            return IngestionResult(video_id=existing.id, sha256=digest, duplicate=True)
        metadata = extract_metadata(path)
        after = path.stat()
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise VideoError(f"Video changed during ingestion: {path}")
        record = Video(
            sha256=digest, **metadata.model_dump(mode="json"), ingested_at=datetime.now(UTC)
        )
        session.add(record)
        try:
            session.commit()
        except IntegrityError:
            session.rollback()
            existing = session.scalar(select(Video).where(Video.sha256 == digest))
            if existing is None:
                raise
            return IngestionResult(video_id=existing.id, sha256=digest, duplicate=True)
        log_event("video_ingested", path=path, video_id=record.id, sha256=digest)
        return IngestionResult(video_id=record.id, sha256=digest, duplicate=False)


def ingest_batch(paths: list[Path], engine: Engine) -> BatchResult:
    result = BatchResult()
    for path in sorted(paths, key=lambda p: str(p.resolve())):
        try:
            result.processed.append(ingest_video(path, engine))
        except (VideoError, OSError) as exc:
            result.errors[str(path)] = str(exc)
            log_event("video_ingestion_failed", path=path, error=str(exc))
    return result
