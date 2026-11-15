"""Deterministic, source-only video features; not diagnostic medical features."""

from dataclasses import dataclass

import cv2
import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from visionops.ml.data import SourceSample
from visionops.pipelines.ingest import content_hash
from visionops.video.process import VideoError, validate_video_file


class FeatureSettings(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    strategy: str = "uniform"
    max_frames: int = Field(default=12, ge=2, le=256)
    stride: int = Field(default=4, ge=1)
    resize_width: int = Field(default=160, ge=32, le=640)

    def model_post_init(self, _context):
        if self.strategy not in {"uniform", "stride"}:
            raise ValueError("Feature strategy must be uniform or stride")


def sample_indices(frame_count: int, settings: FeatureSettings) -> list[int]:
    if frame_count <= 0:
        raise VideoError("Video has no decodable frames")
    if settings.strategy == "uniform":
        return np.unique(
            np.linspace(0, frame_count - 1, min(frame_count, settings.max_frames), dtype=int)
        ).tolist()
    return list(range(0, frame_count, settings.stride))[: settings.max_frames]


@dataclass
class VideoFeatureExtractor:
    settings: FeatureSettings
    version: str = "opencv-gray-v1"
    feature_names: tuple[str, ...] = (
        "spatial_complexity",
        "temporal_complexity",
        "motion_intensity",
        "edge_density",
        "texture_complexity",
        "mean_brightness",
        "brightness_variation",
    )

    def extract(self, source: SourceSample) -> dict[str, float]:
        if source.domain not in {"synthetic_video", "public_video"} or source.sensitive:
            raise ValueError("Default video extractor accepts only explicitly nonsensitive video")
        path = validate_video_file(source.path)
        if content_hash(path) != source.content_hash:
            raise VideoError("Source content no longer matches its identity")
        cv2.setNumThreads(1)
        capture = cv2.VideoCapture(str(path))
        try:
            count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
            indices = sample_indices(count, self.settings)
            frames = []
            for index in indices:
                capture.set(cv2.CAP_PROP_POS_FRAMES, index)
                success, frame = capture.read()
                if not success:
                    raise VideoError("Cannot decode a requested source frame")
                height = max(2, round(frame.shape[0] * self.settings.resize_width / frame.shape[1]))
                gray = cv2.cvtColor(
                    cv2.resize(
                        frame, (self.settings.resize_width, height), interpolation=cv2.INTER_AREA
                    ),
                    cv2.COLOR_BGR2GRAY,
                )
                frames.append(gray)
        finally:
            capture.release()
        if content_hash(path) != source.content_hash:
            raise VideoError("Source changed during feature extraction")
        return self.summarize(frames)

    def summarize(self, frames: list[np.ndarray]) -> dict[str, float]:
        if not frames or any(f.shape != frames[0].shape for f in frames):
            raise VideoError("Expected at least one frame with consistent dimensions")
        spatial, edge, texture, brightness, temporal, motion = [], [], [], [], [], []
        previous = None
        for frame in frames:
            normalized = frame.astype(np.float32) / 255
            dx = cv2.Sobel(normalized, cv2.CV_32F, 1, 0)
            dy = cv2.Sobel(normalized, cv2.CV_32F, 0, 1)
            spatial.append(float(np.std(np.hypot(dx, dy))))
            edge.append(float(np.mean(cv2.Canny(frame, 100, 200) > 0)))
            counts = np.bincount(frame.ravel(), minlength=256).astype(float)
            probability = counts[counts > 0] / frame.size
            texture.append(float(-np.sum(probability * np.log2(probability))))
            brightness.append(float(normalized.mean()))
            if previous is not None:
                temporal.append(float(np.std(normalized - previous)))
                motion.append(float(np.mean(np.abs(normalized - previous))))
            previous = normalized
        values = [
            np.mean(spatial),
            np.mean(temporal) if temporal else 0.0,
            np.mean(motion) if motion else 0.0,
            np.mean(edge),
            np.mean(texture),
            np.mean(brightness),
            np.std(brightness),
        ]
        return dict(zip(self.feature_names, map(float, values), strict=True))
