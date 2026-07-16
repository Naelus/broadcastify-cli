from datetime import date
from pathlib import Path

from broadcastify_cli.jobs import JobRunner
from broadcastify_cli.broadcastify import DownloadLimitExceeded
from broadcastify_cli.models import JobRequest


class FakeClient:
    def __init__(self, files: list[Path], calls: list[str]) -> None:
        self.files = files
        self.calls = calls

    def authenticate(self) -> None:
        self.calls.append("authenticate")

    def download_day(self, *_args: object, **_kwargs: object) -> list[Path]:
        self.calls.append("download")
        return self.files


def test_combined_audio_is_created_before_one_transcription_pass(
    monkeypatch, tmp_path: Path
) -> None:
    calls: list[str] = []
    archive_date = date.today()
    day_dir = tmp_path / "5318" / archive_date.strftime("%Y%m%d")
    day_dir.mkdir(parents=True)
    source_files = [day_dir / "one.mp3", day_dir / "two.mp3"]
    for source in source_files:
        source.write_bytes(b"audio")
    combined_file = day_dir / f"combined_5318_{archive_date:%Y%m%d}.mp3"

    def fake_combine(*_args: object, **_kwargs: object) -> Path:
        calls.append("combine")
        combined_file.write_bytes(b"combined")
        return combined_file

    class FakeTranscriber:
        device = "cuda"
        device_index = 0
        compute_type = "float16"

        def __init__(self, **_kwargs: object) -> None:
            calls.append("load_model")

        def transcribe_files(self, files: list[Path], **_kwargs: object) -> list[Path]:
            calls.append("transcribe")
            assert files == [combined_file]
            return [combined_file.with_suffix(".json")]

    monkeypatch.setattr("broadcastify_cli.jobs.combine_mp3_files", fake_combine)
    monkeypatch.setattr("broadcastify_cli.jobs.LocalTranscriber", FakeTranscriber)

    request = JobRequest(
        feed_id="5318",
        start_date=archive_date,
        end_date=archive_date,
        output_dir=tmp_path,
        combine=True,
        transcribe=True,
        diarize=True,
    )
    JobRunner(request, client=FakeClient(source_files, calls)).run()

    assert calls.index("download") < calls.index("load_model")
    assert calls.index("combine") < calls.index("transcribe")
    assert calls.count("transcribe") == 1


def test_quota_stops_new_requests_but_keeps_complete_cached_days(tmp_path: Path) -> None:
    calls: list[str] = []
    first_day = date(2026, 7, 3)
    second_day = date(2026, 7, 4)
    third_day = date(2026, 7, 5)

    def source_for(archive_date: date) -> Path:
        day_dir = tmp_path / "90001" / archive_date.strftime("%Y%m%d")
        day_dir.mkdir(parents=True, exist_ok=True)
        source = day_dir / f"{archive_date:%Y%m%d}0000-cached-90001.mp3"
        source.write_bytes(b"audio")
        return source

    class QuotaClient:
        def authenticate(self) -> None:
            calls.append("authenticate")

        def download_day(
            self, _feed_id: str, archive_date: date, *_args: object, **_kwargs: object
        ) -> list[Path]:
            calls.append(f"download:{archive_date}")
            if archive_date == second_day:
                raise DownloadLimitExceeded("archive download quota is exhausted")
            return [source_for(archive_date)]

        def cached_day(
            self, _feed_id: str, archive_date: date, *_args: object
        ) -> tuple[list[Path], int]:
            calls.append(f"cache:{archive_date}")
            assert archive_date == third_day
            return [source_for(archive_date)], 1

    events: list[dict[str, object]] = []
    request = JobRequest(
        feed_id="90001",
        start_date=first_day,
        end_date=third_day,
        output_dir=tmp_path,
    )

    result = JobRunner(request, emit=events.append, client=QuotaClient()).run()  # type: ignore[arg-type]

    assert [day["date"] for day in result["days"]] == [
        first_day.isoformat(),
        third_day.isoformat(),
    ]
    assert result["download_limited"] is True
    assert result["missing_days"] == [second_day.isoformat()]
    assert f"download:{third_day}" not in calls
    assert f"cache:{third_day}" in calls
    assert "Completed 2/3 requested days" in str(events[-1]["message"])
