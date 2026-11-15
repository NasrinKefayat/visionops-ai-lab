import pytest
from pydantic import ValidationError
from sqlalchemy import text

from visionops.config import Settings
from visionops.db.session import build_engine, build_session_factory
from visionops.ml.contracts import Prediction


def test_settings_environment_override(monkeypatch) -> None:
    monkeypatch.setenv("VISIONOPS_RANDOM_SEED", "42")
    monkeypatch.setenv("VISIONOPS_DEVICE", "cpu")
    assert Settings().random_seed == 42
    assert Settings().device == "cpu"


def test_invalid_device() -> None:
    with pytest.raises(ValidationError):
        Settings(device="unsupported")


def test_sqlite_session(tmp_path) -> None:
    engine = build_engine(f"sqlite:///{tmp_path / 'test.db'}")
    try:
        with build_session_factory(engine)() as session:
            assert session.scalar(text("SELECT 42")) == 42
    finally:
        engine.dispose()


def test_prediction_validation() -> None:
    with pytest.raises(ValidationError):
        Prediction(label="example", score=1.5, model_version="test")
