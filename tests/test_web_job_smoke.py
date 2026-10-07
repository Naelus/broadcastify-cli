import json
import os
from datetime import date
from pathlib import Path

import pytest

from broadcastify_cli.storage import AnalysisStore
from scripts.web_job_smoke import run_web_job


pytestmark = pytest.mark.usefixtures("fast_server_shutdown")


@pytest.fixture(autouse=True)
def isolated_worker_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in os.environ:
        if name.startswith("BROADCASTIFY_") or name in {
            "HF_TOKEN", "HUGGINGFACE_TOKEN", "HUGGINGFACE_SECURE_TOKEN", "OPENAI_API_KEY",
        }:
            monkeypatch.delenv(name)
    monkeypatch.setenv("PYTHONPATH", str(Path(__file__).resolve().parents[1]))


def test_web_job_smoke_runs_real_diagnostics_worker(tmp_path: Path) -> None:
    result = run_web_job(
        "diagnostics",
        {},
        output_dir=tmp_path / "archives",
        working_dir=tmp_path,
        timeout=30,
        poll_interval=0.02,
    )

    assert result["status"] == "completed"
    assert result["command"] == "diagnostics"
    assert result["error"] == ""
    assert any(event.get("type") == "diagnostics" for event in result["events"])
    assert result["result"]["type"] == "diagnostics"


def test_web_local_job_finishes_with_unicode_paths_and_preserves_search_evidence(
    tmp_path: Path,
) -> None:
    output = tmp_path / "archives — café"
    database = output / "broadcastify-analysis.sqlite3"
    day = output / "99001" / "20260716"
    transcript = day / "transcripts" / "combined_99001_20260716.json"
    transcript.parent.mkdir(parents=True)
    audio = day / "combined_99001_20260716.mp3"
    audio.write_bytes(b"retained")
    quote = "Dispatch — unité responding near the café."
    transcript.write_text(
        json.dumps({
            "segments": [{"start": 1.0, "end": 2.0, "text": quote}],
            "diarization_completed": False,
        }),
        encoding="utf-8",
    )
    with AnalysisStore(database) as store:
        imported = store.import_transcript("99001", date(2026, 7, 16), transcript, audio)
    retained = (audio.read_bytes(), transcript.read_bytes())

    result = run_web_job(
        "continue-local",
        {
            "feed_id": "99001",
            "archive_date": "2026-07-16",
            "output_dir": str(output),
            "diarize": False,
            "analyze": False,
        },
        output_dir=output,
        working_dir=tmp_path,
        timeout=30,
        poll_interval=0.02,
    )

    assert result["status"] == "completed", result["error"]
    assert result["error"] == ""
    assert result["result"]["type"] == "local_complete"
    prepared = result["result"]["prepared"]
    assert prepared["operation"] == "reused"
    assert prepared["transcript_path"] == str(transcript.resolve())
    first_message = str(result["events"][0]["message"])
    assert first_message.endswith("files…")
    assert "\ufffd" not in first_message
    assert (audio.read_bytes(), transcript.read_bytes()) == retained
    with AnalysisStore(database) as store:
        assert store.get_segments(imported.day_id)[0]["text"] == quote
        matches = store.search_passages("99001", date(2026, 7, 16), date(2026, 7, 16), "café")
        assert len(matches) == 1
        assert quote in matches[0]["text"]
