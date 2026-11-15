import json

import pytest

from visionops.ml.video_dataset import Phase2VideoAdapter
from visionops.pipelines.baseline import run_baseline
from visionops.pipelines.ingest import content_hash


def test_adapter_reuses_phase2_measured_results(synthetic_video, engine, tmp_path):
    run_baseline(synthetic_video, tmp_path / "baselines", engine)
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "sources": [
                    {
                        "source_id": "original",
                        "content_hash": content_hash(synthetic_video),
                        "path": str(synthetic_video),
                        "domain": "synthetic_video",
                        "sensitive": False,
                    }
                ]
            }
        )
    )
    dataset = Phase2VideoAdapter(manifest, tmp_path / "test.db").load()
    assert len(dataset.sources) == 1 and len(dataset.observations) == 3
    assert {o.condition for o in dataset.observations} == {18, 23, 28}
    assert all(o.targets["ssim"] > 0 and o.targets["bitrate_bps"] > 0 for o in dataset.observations)
    assert all(
        "psnr_db" in o.measurements and "ffmpeg_version" in o.measurements
        for o in dataset.observations
    )
    payload = json.loads(manifest.read_text())
    payload["sources"][0].update(domain="medical", sensitive=True, patient_group="opaque")
    manifest.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="restricted"):
        Phase2VideoAdapter(manifest, tmp_path / "test.db").load()
