import shutil

import pytest

from visionops.db.session import build_engine
from visionops.pipelines.ingest import initialize_database
from visionops.video.dataset import prepare_dataset


@pytest.fixture
def engine(tmp_path):
    engine = build_engine(f"sqlite:///{tmp_path / 'test.db'}")
    initialize_database(engine)
    yield engine
    engine.dispose()


@pytest.fixture
def synthetic_video(tmp_path):
    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        pytest.skip("FFmpeg/FFprobe not installed; CPU integration test cannot run")
    return prepare_dataset(tmp_path / "dataset")
