from datetime import date
from pathlib import Path

from broadcastify_cli.jobs import JobRunner
from broadcastify_cli.broadcastify import DownloadLimitExceeded
from broadcastify_cli.models import JobRequest
from broadcastify_cli.lan_sync import LanDownloadTurn, LanSyncResult


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
    combined_feed_names: list[str] = []

    def fake_combine(*_args: object, **kwargs: object) -> Path:
        calls.append("combine")
        combined_feed_names.append(str(kwargs.get("feed_name") or ""))
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
        feed_name="Example Public Safety",
        start_date=archive_date,
        end_date=archive_date,
        output_dir=tmp_path,
        combine=True,
        transcribe=True,
        diarize=True,
    )
    JobRunner(request, client=FakeClient(source_files, calls)).run()

    assert calls.index("load_model") < calls.index("download")
    assert calls.index("combine") < calls.index("transcribe")
    assert calls.count("transcribe") == 1
    assert combined_feed_names == ["Example Public Safety"]


def test_local_audio_failure_happens_before_archive_requests(
    tmp_path: Path,
    monkeypatch,
) -> None:
    calls: list[str] = []

    class FailingTranscriber:
        def __init__(self, **_kwargs: object) -> None:
            calls.append("load_model")
            raise RuntimeError("audio runtime missing")

    monkeypatch.setattr(
        "broadcastify_cli.jobs.LocalTranscriber",
        FailingTranscriber,
    )
    request = JobRequest(
        feed_id="5318",
        start_date=date(2026, 7, 1),
        end_date=date(2026, 7, 1),
        output_dir=tmp_path,
        transcribe=True,
    )

    try:
        JobRunner(request, client=FakeClient([], calls)).run()
    except RuntimeError as exc:
        assert str(exc) == "audio runtime missing"
    else:
        raise AssertionError("The local audio preflight should have failed.")

    assert calls == ["load_model"]


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


def test_lan_source_reuse_runs_before_any_broadcastify_request(
    tmp_path: Path,
) -> None:
    calls: list[str] = []
    archive_date = date(2026, 7, 12)

    class FakeLanSync:
        enabled = True

        def sync_day(
            self,
            output_dir: Path,
            feed_id: str,
            requested_date: date,
            **_kwargs: object,
        ) -> LanSyncResult:
            calls.append("lan")
            day = output_dir / feed_id / requested_date.strftime("%Y%m%d")
            day.mkdir(parents=True)
            (day / "202607120000-123456-90001.mp3").write_bytes(b"peer audio")
            return LanSyncResult(enabled=True, peers_reached=1, blocks_copied=1)

    class OrderedClient:
        def authenticate(self) -> None:
            calls.append("authenticate")

        def download_day(
            self,
            feed_id: str,
            requested_date: date,
            output_dir: Path,
            **_kwargs: object,
        ) -> list[Path]:
            calls.append("website")
            return sorted(
                (output_dir / feed_id / requested_date.strftime("%Y%m%d")).glob(
                    "*.mp3"
                )
            )

    request = JobRequest(
        feed_id="90001",
        start_date=archive_date,
        end_date=archive_date,
        output_dir=tmp_path,
        lan_sync_enabled=True,
    )
    result = JobRunner(
        request,
        client=OrderedClient(),  # type: ignore[arg-type]
        lan_sync=FakeLanSync(),  # type: ignore[arg-type]
    ).run()

    assert calls == ["lan", "authenticate", "website"]
    assert result["lan_sync"]["blocks_copied"] == 1


def test_completed_lan_queue_day_skips_every_broadcastify_request(
    tmp_path: Path,
) -> None:
    archive_date = date(2026, 7, 12)
    day = tmp_path / "90001" / "20260712"
    day.mkdir(parents=True)
    source = day / "202607120000-123456-90001.mp3"
    source.write_bytes(b"peer-completed audio")

    class CompletedLanQueue:
        enabled = True

        def sync_day(self, *_args: object, **_kwargs: object) -> LanSyncResult:
            return LanSyncResult(enabled=True, peers_reached=1)

        def wait_for_download_turn(
            self,
            *_args: object,
            **_kwargs: object,
        ) -> LanDownloadTurn:
            return LanDownloadTurn(
                role="completed",
                feed_id="90001",
                archive_date=archive_date.isoformat(),
                block_count=1,
                audio_files=(source,),
            )

    class ForbiddenWebsiteClient:
        def authenticate(self) -> None:
            raise AssertionError("A follower must not authenticate.")

        def download_day(self, *_args: object, **_kwargs: object) -> list[Path]:
            raise AssertionError("A follower must not use archive endpoints.")

    request = JobRequest(
        feed_id="90001",
        start_date=archive_date,
        end_date=archive_date,
        output_dir=tmp_path,
        lan_sync_enabled=True,
    )
    result = JobRunner(
        request,
        client=ForbiddenWebsiteClient(),  # type: ignore[arg-type]
        lan_sync=CompletedLanQueue(),  # type: ignore[arg-type]
    ).run()

    assert result["days"][0]["audio_files"] == [str(source)]
    assert result["lan_sync"]["acquisition_queue"]["completed"] == 1


def test_lan_queue_leader_publishes_completion_after_one_upstream_download(
    tmp_path: Path,
) -> None:
    archive_date = date(2026, 7, 12)
    calls: list[str] = []

    class Heartbeat:
        warnings: tuple[str, ...] = ()

        def __enter__(self) -> "Heartbeat":
            calls.append("heartbeat:start")
            return self

        def __exit__(self, *_args: object) -> None:
            calls.append("heartbeat:stop")

        def assert_active(self) -> None:
            calls.append("heartbeat:active")

    class LeaderLanQueue:
        enabled = True

        def sync_day(self, *_args: object, **_kwargs: object) -> LanSyncResult:
            calls.append("sync")
            return LanSyncResult(enabled=True, peers_reached=1)

        def wait_for_download_turn(
            self,
            *_args: object,
            **_kwargs: object,
        ) -> LanDownloadTurn:
            calls.append("claim")
            return LanDownloadTurn(
                role="leader",
                feed_id="90001",
                archive_date=archive_date.isoformat(),
                coordinator_url="http://127.0.0.1:8765",
                producer_url="http://127.0.0.1:8766",
                owner_node_id="producer_one",
                lease_token="lease_token_value_that_is_long_enough",
                lease_seconds=90.0,
            )

        def maintain_download_lease(self, _turn: LanDownloadTurn) -> Heartbeat:
            return Heartbeat()

        def finish_download_turn(
            self,
            _turn: LanDownloadTurn,
            *,
            outcome: str,
            block_count: int = 0,
            source_files: tuple[Path, ...] | list[Path] = (),
        ) -> str:
            if outcome == "complete":
                assert len(source_files) == block_count
            calls.append(f"finish:{outcome}:{block_count}")
            return ""

    class UpstreamClient:
        def authenticate(self) -> None:
            calls.append("authenticate")

        def download_day(
            self,
            feed_id: str,
            requested_date: date,
            output_dir: Path,
            **_kwargs: object,
        ) -> list[Path]:
            calls.append("download")
            day = output_dir / feed_id / requested_date.strftime("%Y%m%d")
            day.mkdir(parents=True)
            source = day / "202607120000-123456-90001.mp3"
            source.write_bytes(b"one upstream response")
            return [source]

    request = JobRequest(
        feed_id="90001",
        start_date=archive_date,
        end_date=archive_date,
        output_dir=tmp_path,
        lan_sync_enabled=True,
    )
    JobRunner(
        request,
        client=UpstreamClient(),  # type: ignore[arg-type]
        lan_sync=LeaderLanQueue(),  # type: ignore[arg-type]
    ).run()

    assert calls == [
        "sync",
        "claim",
        "heartbeat:start",
        "authenticate",
        "download",
        "finish:complete:1",
        "heartbeat:stop",
    ]


def test_lan_queue_quota_result_uses_local_next_safe_delay(
    tmp_path: Path,
) -> None:
    archive_date = date(2026, 7, 12)
    published: list[tuple[str, float | None]] = []

    class Heartbeat:
        warnings: tuple[str, ...] = ()

        def __enter__(self) -> "Heartbeat":
            return self

        def __exit__(self, *_args: object) -> None:
            return None

        def assert_active(self) -> None:
            return None

    class LeaderLanQueue:
        enabled = True

        def sync_day(self, *_args: object, **_kwargs: object) -> LanSyncResult:
            return LanSyncResult(enabled=True, peers_reached=1)

        def wait_for_download_turn(
            self,
            *_args: object,
            **_kwargs: object,
        ) -> LanDownloadTurn:
            return LanDownloadTurn(
                role="leader",
                feed_id="90001",
                archive_date=archive_date.isoformat(),
                coordinator_url="http://127.0.0.1:8765",
                producer_url="http://127.0.0.1:8766",
                owner_node_id="producer_one",
                lease_token="lease_token_value_that_is_long_enough",
                lease_seconds=90.0,
            )

        def maintain_download_lease(self, _turn: LanDownloadTurn) -> Heartbeat:
            return Heartbeat()

        def local_source_files(
            self,
            *_args: object,
            **_kwargs: object,
        ) -> list[Path]:
            return []

        def finish_download_turn(
            self,
            _turn: LanDownloadTurn,
            *,
            outcome: str,
            retry_after_seconds: float | None = None,
            **_kwargs: object,
        ) -> str:
            published.append((outcome, retry_after_seconds))
            return ""

    class LimitedClient:
        def authenticate(self) -> None:
            return None

        def download_day(self, *_args: object, **_kwargs: object) -> list[Path]:
            raise DownloadLimitExceeded("rolling quota reached")

        def archive_quota_status(self) -> dict[str, object]:
            return {"next_request_seconds": 37}

    request = JobRequest(
        feed_id="90001",
        start_date=archive_date,
        end_date=archive_date,
        output_dir=tmp_path,
        lan_sync_enabled=True,
    )
    result = JobRunner(
        request,
        client=LimitedClient(),  # type: ignore[arg-type]
        lan_sync=LeaderLanQueue(),  # type: ignore[arg-type]
    ).run()

    assert published == [("quota_limited", 37.0)]
    assert result["download_limited"] is True
