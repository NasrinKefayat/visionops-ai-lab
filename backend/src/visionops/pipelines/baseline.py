"""Reproducible CRF comparison; independent of FastAPI and ML frameworks."""

import hashlib
import json
import uuid
from datetime import UTC, datetime
from pathlib import Path

from sqlalchemy import Engine

from visionops.config import get_settings
from visionops.db.models import EncodingConfiguration, EncodingRecord, Experiment, QualityRecord
from visionops.db.session import build_session_factory
from visionops.pipelines.ingest import content_hash, ingest_video, initialize_database, log_event
from visionops.video.encoding import encode_video
from visionops.video.process import VideoError, tool_version, validate_video_file
from visionops.video.quality import evaluate_quality
from visionops.video.schemas import EncodingConfig


def run_baseline(
    source: Path,
    output_directory: Path,
    engine: Engine,
    *,
    preset: str = "medium",
    code_revision: str | None = None,
    vmaf: bool = False,
) -> dict:
    configs = [EncodingConfig(crf=crf, preset=preset) for crf in (18, 23, 28)]
    source = validate_video_file(source)
    settings = get_settings()
    ffmpeg_version = tool_version(settings.ffmpeg_binary)
    ffprobe_version = tool_version(settings.ffprobe_binary)
    initialize_database(engine)
    video = ingest_video(source, engine)
    run_id = str(uuid.uuid4())
    directory = output_directory.resolve() / run_id
    directory.mkdir(parents=True, exist_ok=False)
    sessions = build_session_factory(engine)
    report = {
        "run_id": run_id,
        "source": str(source),
        "source_sha256": video.sha256,
        "seed": settings.random_seed,
        "ffmpeg_version": ffmpeg_version,
        "ffprobe_version": ffprobe_version,
        "code_revision": code_revision,
        "vmaf_requested": vmaf,
        "results": [],
    }
    for config in configs:
        parameters = config.model_dump()
        fingerprint = hashlib.sha256(json.dumps(parameters, sort_keys=True).encode()).hexdigest()
        experiment_id = str(uuid.uuid4())
        started = datetime.now(UTC)
        with sessions.begin() as session:
            if session.get(EncodingConfiguration, fingerprint) is None:
                session.add(EncodingConfiguration(fingerprint=fingerprint, parameters=parameters))
                session.flush()
            session.add(
                Experiment(
                    id=experiment_id,
                    run_id=run_id,
                    video_id=video.video_id,
                    config_id=fingerprint,
                    status="running",
                    started_at=started,
                    ffmpeg_version=ffmpeg_version,
                    ffprobe_version=ffprobe_version,
                    seed=settings.random_seed,
                    code_revision=code_revision,
                )
            )
        item = {
            "experiment_id": experiment_id,
            "config": parameters,
            "started_at": started.isoformat(),
        }
        try:
            if content_hash(source) != video.sha256:
                raise VideoError("Source changed since ingestion")
            encoding = encode_video(source, directory / f"crf-{config.crf}.mp4", config)
            with sessions.begin() as session:
                session.add(
                    EncodingRecord(
                        experiment_id=experiment_id,
                        output_path=str(encoding.output_path),
                        output_sha256=content_hash(encoding.output_path),
                        elapsed_seconds=encoding.elapsed_seconds,
                        bitrate_bps=encoding.metadata.bitrate_bps,
                        file_size_bytes=encoding.metadata.file_size_bytes,
                        duration_seconds=encoding.metadata.duration_seconds,
                    )
                )
            item["encoding"] = encoding.model_dump(mode="json")
            metrics = evaluate_quality(source, encoding.output_path, vmaf=vmaf)
            if content_hash(source) != video.sha256:
                raise VideoError("Source changed during experiment")
            with sessions.begin() as session:
                session.add(QualityRecord(experiment_id=experiment_id, **metrics.model_dump()))
            item.update(
                status="completed",
                encoding=encoding.model_dump(mode="json"),
                quality=metrics.model_dump(mode="json"),
            )
        except (VideoError, OSError) as exc:
            item.update(status="failed", error=str(exc))
        completed = datetime.now(UTC)
        item["completed_at"] = completed.isoformat()
        with sessions.begin() as session:
            experiment = session.get(Experiment, experiment_id)
            experiment.status = item["status"]
            experiment.completed_at = completed
            experiment.error = item.get("error")
        report["results"].append(item)
        log_event("experiment_finished", run_id=run_id, crf=config.crf, status=item["status"])
        checkpoint = directory / "report.json.tmp"
        checkpoint.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        checkpoint.replace(directory / "report.json")
    return report
