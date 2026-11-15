"""Dashboard aggregation and optional, digest-pinned cached-feature inference."""

import hashlib
import hmac
import math
import re
from io import BytesIO

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from visionops.api.artifacts import FEATURE_NAMES, StudyRecord, studies
from visionops.api.errors import APIError
from visionops.api.publication import Publication, read_json
from visionops.api.queries import require_video
from visionops.api.schemas import InferenceRequest, MonitoringResponse, RecommendationResponse
from visionops.config import Settings
from visionops.db.models import EncodingRecord, Experiment, QualityRecord, Video


def monitoring(
    session: Session | None, published: Publication, settings: Settings
) -> MonitoringResponse:
    counts, values = {}, {}
    if session is not None and published.status == "ready":
        allowed = list(published.sources)
        counts = dict(
            session.execute(
                select(Experiment.status, func.count())
                .join(Video, Experiment.video_id == Video.id)
                .where(Video.sha256.in_(allowed))
                .group_by(Experiment.status)
            ).all()
        )
        encoded = (
            select(EncodingRecord)
            .join(Experiment, Experiment.id == EncodingRecord.experiment_id)
            .join(Video, Experiment.video_id == Video.id)
            .where(Video.sha256.in_(allowed), Experiment.status == "completed")
        )
        enc = encoded.subquery()
        quality = (
            select(QualityRecord)
            .join(Experiment, Experiment.id == QualityRecord.experiment_id)
            .join(Video, Experiment.video_id == Video.id)
            .where(Video.sha256.in_(allowed), Experiment.status == "completed")
        ).subquery()
        values = {
            "source_videos": session.scalar(
                select(func.count()).select_from(Video).where(Video.sha256.in_(allowed))
            ),
            "completed_encodes": session.scalar(select(func.count()).select_from(enc)),
            "average_measured_ssim": session.scalar(select(func.avg(quality.c.ssim))),
            "average_measured_psnr_db": session.scalar(select(func.avg(quality.c.psnr_db))),
            "infinite_psnr_count": session.scalar(
                select(func.count()).select_from(quality).where(quality.c.psnr_infinite.is_(True))
            ),
            "average_output_bitrate_bps": session.scalar(select(func.avg(enc.c.bitrate_bps))),
            "average_encoding_seconds": session.scalar(select(func.avg(enc.c.elapsed_seconds))),
        }
    return MonitoringResponse(
        database_status="ready" if session is not None else "unavailable",
        publication_status=published.status,
        available_ml_experiments=len(studies(settings, published)),
        status_counts=counts,
        message=None
        if session is not None and published.status == "ready"
        else "No published experiment data is available.",
        **{
            name: values.get(name)
            for name in (
                "source_videos",
                "completed_encodes",
                "average_measured_ssim",
                "average_measured_psnr_db",
                "infinite_psnr_count",
                "average_output_bitrate_bps",
                "average_encoding_seconds",
            )
        },
    )


def find_study(identity: str, settings: Settings, published: Publication) -> StudyRecord:
    for record in studies(settings, published):
        if record.response.id == identity:
            return record
    raise APIError(404, "not_found", "Published research experiment not found.")


def recommend(
    record: StudyRecord,
    request: InferenceRequest,
    session: Session | None,
    published: Publication,
    settings: Settings,
) -> RecommendationResponse:
    expected = settings.dashboard_trusted_models.get(record.response.id, "")
    if not re.fullmatch(r"[a-f0-9]{64}", expected):
        raise APIError(503, "inference_unavailable", "Inference is not enabled for this model.")
    if not re.fullmatch(r"[a-f0-9]{64}", record.digest):
        raise APIError(503, "inference_unavailable", "Model provenance is unavailable.")
    video = require_video(session, published, request.video_id)
    source_name = published.sources[video.sha256].name
    try:
        path = record.directory / "model.joblib"
        if (
            path.is_symlink()
            or not path.resolve().is_relative_to(settings.dashboard_reports.resolve())
            or path.stat().st_size > 16_000_000
        ):
            raise ValueError("Untrusted artifact")
        with path.open("rb") as file:
            payload = file.read(16_000_001)
        if len(payload) > 16_000_000:
            raise ValueError("Untrusted artifact")
        if not hmac.compare_digest(hashlib.sha256(payload).hexdigest(), expected):
            raise ValueError("Untrusted artifact")
        rows = read_json(record.directory / "dataset.json", settings.dashboard_reports)
        matches = [
            row
            for row in rows
            if row.get("source_id") == source_name and row.get("source_hash") == video.sha256
        ]
        if not matches:
            raise APIError(
                422,
                "features_unavailable",
                "This model has no cached source features for the selected video.",
            )
        features = {
            name: value for name, value in matches[0]["features"].items() if name != "condition"
        }
        if set(features) != FEATURE_NAMES - {"condition"} or not all(
            math.isfinite(float(v)) for v in features.values()
        ):
            raise ValueError("Feature schema mismatch")
        if any(
            {k: v for k, v in row["features"].items() if k != "condition"} != features
            for row in matches
        ):
            raise ValueError("Feature schema mismatch")
        # Only verified, administrator-trusted bytes; no decoding or training.
        from visionops.ml.inference import load_bundle
        from visionops.ml.optimization import select_crf

        bundle = load_bundle(BytesIO(payload))
        if (
            bundle.metadata.get("sensitive") is not False
            or bundle.metadata.get("dataset_digest") != record.digest
            or bundle.metadata.get("extractor_version") != record.response.extractor_version
        ):
            raise ValueError("Model provenance mismatch")
        candidates = (
            request.candidates if request.candidates is not None else bundle.metadata["candidates"]
        )
        result = select_crf(bundle, features, request.minimum_ssim, candidates)
        return RecommendationResponse(
            source_name=source_name,
            crf=result.crf,
            predicted_ssim=result.predicted_ssim,
            predicted_bitrate_bps=result.predicted_bitrate_bps,
            constraint_status="predicted_feasible"
            if result.crf is not None
            else "no_predicted_candidate",
            explanation="Predicted feasibility requires measurement."
            if result.crf is not None
            else "No candidate is predicted to meet the quality constraint.",
        )
    except APIError:
        raise
    except ImportError as exc:
        raise APIError(
            503, "inference_unavailable", "Optional ML dependencies are unavailable."
        ) from exc
    except Exception as exc:
        raise APIError(
            422, "inference_unavailable", "Model or candidate inputs could not be verified."
        ) from exc
