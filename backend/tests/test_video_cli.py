import json
import sys
from unittest.mock import patch

from visionops.cli import main


def test_cli_ingestion_failure_nonzero(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("VISIONOPS_DATABASE_URL", f"sqlite:///{tmp_path / 'cli.db'}")
    monkeypatch.setattr(sys, "argv", ["visionops", "ingest", str(tmp_path / "missing.mp4")])
    assert main() == 1
    result = json.loads(capsys.readouterr().out)
    assert not result["processed"]
    assert str(tmp_path / "missing.mp4") in result["errors"]


def test_cli_configuration_error_returns_nonzero(tmp_path, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["visionops", "prepare-dataset", "--output", str(tmp_path)])
    with patch("visionops.cli.prepare_dataset", side_effect=ValueError("bad configuration")):
        assert main() == 1
