"""Explicit adapter from Phase 2 measured records to encoding regression targets."""

import json
from dataclasses import dataclass
from pathlib import Path

from sqlalchemy import select

from visionops.db.models import (
    EncodingConfiguration,
    EncodingRecord,
    Experiment,
    QualityRecord,
    Video,
)
from visionops.db.session import build_engine, build_session_factory
from visionops.ml.data import Dataset, Observation, SourceSample
from visionops.pipelines.ingest import content_hash


def load_manifest(path: Path) -> tuple[SourceSample, ...]:
    records = json.loads(path.read_text())["sources"]
    samples = []
    for record in records:
        sample = SourceSample(
            source_id=record["source_id"],
            content_hash=record["content_hash"],
            path=(path.parent / record["path"]).resolve(),
            domain=record.get("domain", "restricted"),
            patient_group=record.get("patient_group"),
            study_group=record.get("study_group"),
            sensitive=record.get("sensitive", True),
        )
        samples.append(sample)
    if not samples or len({s.source_id for s in samples}) != len(samples):
        raise ValueError("Manifest must contain unique sources")
    return tuple(samples)


@dataclass
class Phase2VideoAdapter:
    manifest: Path
    database: Path
    candidates: tuple[int, ...] = (18, 23, 28)

    def load(self) -> Dataset:
        samples = load_manifest(self.manifest)
        # Existing Phase 2 logs/DB paths are not a medical ingestion path.
        if any(s.sensitive or s.domain not in {"public_video", "synthetic_video"} for s in samples):
            raise ValueError("Phase 2 adapter cannot ingest medical/restricted sources")
        if not self.database.is_file():
            raise ValueError("Phase 2 database does not exist")
        engine = build_engine(f"sqlite:///{self.database.resolve()}")
        observations = []
        try:
            with build_session_factory(engine)() as session:
                for sample in samples:
                    if content_hash(sample.path) != sample.content_hash:
                        raise ValueError("Source content identity mismatch")
                    video = session.scalar(select(Video).where(Video.sha256 == sample.content_hash))
                    if video is None:
                        raise ValueError("Source has no Phase 2 measurements")
                    for crf in self.candidates:
                        rows = session.execute(
                            select(Experiment, EncodingConfiguration, EncodingRecord, QualityRecord)
                            .join(
                                EncodingConfiguration,
                                Experiment.config_id == EncodingConfiguration.fingerprint,
                            )
                            .join(EncodingRecord, Experiment.id == EncodingRecord.experiment_id)
                            .join(QualityRecord, Experiment.id == QualityRecord.experiment_id)
                            .where(
                                Experiment.video_id == video.id, Experiment.status == "completed"
                            )
                            .order_by(Experiment.completed_at.desc(), Experiment.id.desc())
                        ).all()
                        compatible = [
                            r
                            for r in rows
                            if r[1].parameters
                            == {
                                "codec": "libx264",
                                "crf": crf,
                                "preset": "medium",
                                "width": None,
                                "height": None,
                                "threads": 1,
                                "pixel_format": "yuv420p",
                            }
                        ]
                        if not compatible:
                            raise ValueError(
                                "Missing compatible medium/native/one-thread CRF measurements"
                            )
                        # A reproducible snapshot selects the most recent completed compatible row.
                        experiment, _, encoding, quality = compatible[0]
                        observations.append(
                            Observation(
                                sample.source_id,
                                crf,
                                {"ssim": quality.ssim, "bitrate_bps": float(encoding.bitrate_bps)},
                                {
                                    "psnr_db": quality.psnr_db,
                                    "psnr_infinite": quality.psnr_infinite,
                                    "experiment_id": experiment.id,
                                    "frame_count": quality.frame_count,
                                    "ffmpeg_version": experiment.ffmpeg_version,
                                    "ffprobe_version": experiment.ffprobe_version,
                                },
                            )
                        )
            return Dataset(samples, tuple(observations))
        finally:
            engine.dispose()
