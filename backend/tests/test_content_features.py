import numpy as np
import pytest

from visionops.ml.data import SourceSample
from visionops.pipelines.ingest import content_hash
from visionops.video.features import FeatureSettings, VideoFeatureExtractor, sample_indices
from visionops.video.process import VideoError


def test_uniform_stride_and_short_sampling():
    assert sample_indices(1, FeatureSettings()) == [0]
    assert sample_indices(3, FeatureSettings(max_frames=10)) == [0, 1, 2]
    assert sample_indices(10, FeatureSettings(max_frames=3)) == [0, 4, 9]
    assert sample_indices(10, FeatureSettings(strategy="stride", stride=3, max_frames=3)) == [
        0,
        3,
        6,
    ]
    with pytest.raises(VideoError):
        sample_indices(0, FeatureSettings())
    with pytest.raises(ValueError):
        FeatureSettings(strategy="random-unseeded")


def test_static_and_changing_frame_definitions():
    extractor = VideoFeatureExtractor(FeatureSettings())
    black = np.zeros((32, 32), dtype=np.uint8)
    white = np.full((32, 32), 255, dtype=np.uint8)
    static = extractor.summarize([black])
    assert all(value == 0 for value in static.values())
    changing = extractor.summarize([black, white])
    assert changing["motion_intensity"] == 1
    assert changing["mean_brightness"] == 0.5
    assert changing["brightness_variation"] == 0.5
    assert (
        changing["temporal_complexity"] == 0
    )  # Global illumination change, not spatial variation.


def test_real_features_repeatable_and_source_only(synthetic_video):
    source = SourceSample(
        "source",
        content_hash(synthetic_video),
        synthetic_video,
        domain="synthetic_video",
        sensitive=False,
    )
    extractor = VideoFeatureExtractor(FeatureSettings(max_frames=4))
    first = extractor.extract(source)
    assert first == extractor.extract(source)
    assert set(first) == set(extractor.feature_names)
    assert all(np.isfinite(value) for value in first.values())
    assert first["spatial_complexity"] > 0
    assert 0 <= first["edge_density"] <= 1


def test_medical_default_extractor_refuses_before_opening(tmp_path):
    source = SourceSample(
        "opaque",
        "opaquehash",
        tmp_path / "private.dcm",
        domain="medical",
        patient_group="opaque-patient",
    )
    assert "private.dcm" not in repr(source)
    assert "opaque-patient" not in repr(source)
    with pytest.raises(ValueError, match="nonsensitive"):
        VideoFeatureExtractor(FeatureSettings()).extract(source)
    with pytest.raises(ValueError, match="patient"):
        SourceSample("opaque", "hash", tmp_path / "a", domain="medical")
