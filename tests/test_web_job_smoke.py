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
