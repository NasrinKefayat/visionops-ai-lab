"""Opt-in local study preparation using the existing Phase 2 baseline."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory

import cv2
import numpy as np

from visionops.config import get_settings
from visionops.db.session import build_engine
from visionops.ml.video_dataset import load_manifest
from visionops.pipelines.baseline import run_baseline
from visionops.pipelines.ingest import content_hash
from visionops.video.process import run_tool


def generate_sources(directory: Path, *, count: int = 12, seed: int = 42) -> Path:
    if count < 5 or count > 50:
        raise ValueError("Choose 5..50 small synthetic sources")
    directory.mkdir(parents=True, exist_ok=False)
    sources = []
    for index in range(count):
        rng = np.random.default_rng(seed + index)
        background = rng.integers(20, 200, size=(120, 160, 3), dtype=np.uint8)
        # Independent random scene per source, not variants of a shared parent clip.
        if index % 3 == 0:
            background = cv2.GaussianBlur(background, (15, 15), 0)
        elif index % 3 == 1:
            background = cv2.GaussianBlur(background, (5, 5), 0)
        with TemporaryDirectory(dir=directory) as temporary:
            frames = Path(temporary)
            for frame_index in range(12):
                frame = background.copy()
                x = (frame_index * (index + 1) * 2) % 120
                color = tuple(int(v) for v in rng.integers(0, 256, size=3))
                cv2.rectangle(frame, (x, 20 + index % 40), (x + 30, 60 + index % 40), color, -1)
                if not cv2.imwrite(str(frames / f"frame-{frame_index:03d}.ppm"), frame):
                    raise ValueError("Cannot write synthetic frame")
            output = directory / f"source-{index:02d}.mp4"
            run_tool(
                [
                    get_settings().ffmpeg_binary,
                    "-hide_banner",
                    "-nostdin",
                    "-v",
                    "error",
                    "-n",
                    "-framerate",
                    "12",
                    "-i",
                    str(frames / "frame-%03d.ppm"),
                    "-c:v",
                    "libx264",
                    "-crf",
                    "0",
                    "-preset",
                    "medium",
                    "-threads",
                    "1",
                    "-pix_fmt",
                    "yuv420p",
                    str(output),
                ]
            )
        sources.append(
            {
                "source_id": f"synthetic-{index:02d}",
                "content_hash": content_hash(output),
                "path": output.name,
                "domain": "synthetic_video",
                "sensitive": False,
            }
        )
    manifest = directory / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "seed": seed,
                "license": "Procedurally generated; no third-party footage",
                "sources": sources,
            },
            indent=2,
        )
        + "\n"
    )
    return manifest


def prepare_measurements(
    manifest: Path, database: Path, output: Path, code_revision: str | None = None
) -> None:
    samples = load_manifest(manifest)
    # Do not send medical paths through the Phase 2 logger or FFmpeg ingestion path.
    if any(s.sensitive or s.domain not in {"synthetic_video", "public_video"} for s in samples):
        raise ValueError(
            "Use a dedicated restricted-data adapter; this command is for public/synthetic video"
        )
    database.parent.mkdir(parents=True, exist_ok=True)
    engine = build_engine(f"sqlite:///{database.resolve()}")
    try:
        for sample in samples:
            if content_hash(sample.path) != sample.content_hash:
                raise ValueError("Source content identity mismatch")
            report = run_baseline(sample.path, output, engine, code_revision=code_revision)
            if any(r["status"] != "completed" for r in report["results"]):
                raise ValueError("A Phase 2 measurement failed; inspect the local baseline report")
    finally:
        engine.dispose()
