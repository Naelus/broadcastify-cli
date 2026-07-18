import json
from pathlib import Path

from scripts.web_job_smoke import run_web_job


def test_web_job_smoke_runs_real_diagnostics_worker(tmp_path: Path) -> None:
    repository = Path(__file__).resolve().parents[1]
    result = run_web_job(
        "diagnostics",
        {},
        output_dir=tmp_path / "archives",
        working_dir=repository,
        timeout=30,
        poll_interval=0.02,
    )

    assert result["status"] == "completed"
    assert result["command"] == "diagnostics"
    assert result["error"] == ""
    assert any(event.get("type") == "diagnostics" for event in result["events"])
    assert result["result"]["type"] == "diagnostics"


def test_web_worker_pipe_preserves_unicode_on_windows(tmp_path: Path) -> None:
    repository = Path(__file__).resolve().parents[1]
    output = tmp_path / "archives"
    day = output / "99001" / "20260716"
    transcript = day / "transcripts" / "combined_99001_20260716.json"
    transcript.parent.mkdir(parents=True)
    (day / "combined_99001_20260716.mp3").write_bytes(b"retained")
    transcript.write_text(
        json.dumps({"segments": [], "diarization_completed": False}),
        encoding="utf-8",
    )

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
        working_dir=repository,
        timeout=30,
        poll_interval=0.02,
    )

    first_message = str(result["events"][0]["message"])
    assert first_message.endswith("files…")
    assert "\ufffd" not in first_message
