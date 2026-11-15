"""Portable relational records; no database is opened at import time."""

from datetime import datetime

from sqlalchemy import JSON, BigInteger, DateTime, Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from visionops.db.session import Base


class Video(Base):
    __tablename__ = "videos"
    id: Mapped[int] = mapped_column(primary_key=True)
    sha256: Mapped[str] = mapped_column(String(64), unique=True)
    source_path: Mapped[str] = mapped_column(Text)
    filename: Mapped[str] = mapped_column(Text)
    duration_seconds: Mapped[float] = mapped_column(Float)
    width: Mapped[int] = mapped_column(Integer)
    height: Mapped[int] = mapped_column(Integer)
    fps: Mapped[float] = mapped_column(Float)
    codec: Mapped[str] = mapped_column(String(100))
    bitrate_bps: Mapped[int] = mapped_column(BigInteger)
    bitrate_source: Mapped[str] = mapped_column(String(20))
    file_size_bytes: Mapped[int] = mapped_column(BigInteger)
    pixel_format: Mapped[str] = mapped_column(String(100))
    ingested_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))


class EncodingConfiguration(Base):
    __tablename__ = "encoding_configurations"
    fingerprint: Mapped[str] = mapped_column(String(64), primary_key=True)
    parameters: Mapped[dict] = mapped_column(JSON)


class Experiment(Base):
    __tablename__ = "experiments"
    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    run_id: Mapped[str] = mapped_column(String(36), index=True)
    video_id: Mapped[int] = mapped_column(ForeignKey("videos.id"))
    config_id: Mapped[str] = mapped_column(ForeignKey("encoding_configurations.fingerprint"))
    status: Mapped[str] = mapped_column(String(20))
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    ffmpeg_version: Mapped[str] = mapped_column(Text)
    ffprobe_version: Mapped[str] = mapped_column(Text)
    seed: Mapped[int] = mapped_column(Integer)
    code_revision: Mapped[str | None] = mapped_column(String(100))
    error: Mapped[str | None] = mapped_column(Text)


class EncodingRecord(Base):
    __tablename__ = "encoding_results"
    experiment_id: Mapped[str] = mapped_column(ForeignKey("experiments.id"), primary_key=True)
    output_path: Mapped[str] = mapped_column(Text)
    output_sha256: Mapped[str] = mapped_column(String(64))
    elapsed_seconds: Mapped[float] = mapped_column(Float)
    bitrate_bps: Mapped[int] = mapped_column(BigInteger)
    file_size_bytes: Mapped[int] = mapped_column(BigInteger)
    duration_seconds: Mapped[float] = mapped_column(Float)


class QualityRecord(Base):
    __tablename__ = "quality_metrics"
    experiment_id: Mapped[str] = mapped_column(ForeignKey("experiments.id"), primary_key=True)
    psnr_db: Mapped[float | None] = mapped_column(Float)
    psnr_infinite: Mapped[bool]
    ssim: Mapped[float] = mapped_column(Float)
    vmaf: Mapped[float | None] = mapped_column(Float)
    frame_count: Mapped[int] = mapped_column(Integer)
