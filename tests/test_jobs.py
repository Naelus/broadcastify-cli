from datetime import date
from pathlib import Path

from broadcastify_cli.jobs import JobRunner
from broadcastify_cli.archive_cache import (
    remember_archive_identity,
    remember_complete_archive_day,
)
from broadcastify_cli.broadcastify import BroadcastifyClient, DownloadLimitExceeded
from broadcastify_cli.models import JobRequest
from broadcastify_cli.lan_sync import (
    LanDownloadTurn,
    LanFeedSyncResult,
    LanProcessingTurn,
    LanSyncResult,
    LanTranscriptSyncResult,
)
from broadcastify_cli.quota import ArchiveRequestLedger


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


def test_scheduled_processing_limit_queues_remaining_local_days(
    monkeypatch,
    tmp_path: Path,
) -> None:
    calls: list[str] = []
    first_day = date(2026, 7, 10)
    last_day = date(2026, 7, 12)

    def source_for(archive_date: date) -> Path:
        day_dir = tmp_path / "5318" / archive_date.strftime("%Y%m%d")
        day_dir.mkdir(parents=True, exist_ok=True)
        source = day_dir / f"{archive_date:%Y%m%d}0000-source-5318.mp3"
        source.write_bytes(b"audio")
        return source

    class ProcessingClient:
        def authenticate(self) -> None:
            calls.append("authenticate")

        def download_day(
            self,
            _feed_id: str,
            archive_date: date,
            *_args: object,
            **_kwargs: object,
        ) -> list[Path]:
            calls.append(f"download:{archive_date}")
            return [source_for(archive_date)]

    class ProcessingTranscriber:
        device = "cpu"
        device_index = 0
        compute_type = "float32"

        def __init__(self, **_kwargs: object) -> None:
            calls.append("load")

        def current_transcripts(self, _inputs: list[Path]) -> list[Path]:
            return []

        def transcribe_files(
            self,
            inputs: list[Path],
            **_kwargs: object,
        ) -> list[Path]:
            calls.append(f"transcribe:{inputs[0].parent.name}")
            transcript = inputs[0].with_suffix(".json")
            transcript.write_text("{}", encoding="utf-8")
            return [transcript]

    monkeypatch.setattr("broadcastify_cli.jobs.LocalTranscriber", ProcessingTranscriber)
    events: list[dict[str, object]] = []
    request = JobRequest(
        feed_id="5318",
        start_date=first_day,
        end_date=last_day,
        output_dir=tmp_path,
        transcribe=True,
        max_processing_days=1,
    )

    result = JobRunner(
        request,
        emit=events.append,
        client=ProcessingClient(),  # type: ignore[arg-type]
    ).run()

    assert [value for value in calls if value.startswith("transcribe:")] == [
        "transcribe:20260712"
    ]
    assert result["pending_processing_days"] == [
        "2026-07-11",
        "2026-07-10",
    ]
    assert result["missing_days"] == []
    assert any(
        "return to archive acquisition" in str(event.get("message") or "")
        for event in events
    )


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

        def cached_day_local(
            self, _feed_id: str, archive_date: date, *_args: object
        ) -> tuple[list[Path], int] | None:
            calls.append(f"cache:{archive_date}")
            assert archive_date == third_day
            return [source_for(archive_date)], 1

    events: list[dict[str, object]] = []
    request = JobRequest(
        feed_id="90001",
        start_date=first_day,
        end_date=third_day,
        output_dir=tmp_path,
        newest_first=False,
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


def test_full_rolling_guard_never_authenticates_and_still_uses_cache(
    tmp_path: Path,
) -> None:
    first_day = date(2026, 7, 3)
    second_day = date(2026, 7, 4)
    cached_dir = tmp_path / "90001" / first_day.strftime("%Y%m%d")
    cached_dir.mkdir(parents=True)
    cached = cached_dir / "202607030000-provider-90001.mp3"
    cached.write_bytes(b"retained audio")
    calls: list[str] = []

    class GuardedClient:
        def archive_quota_status(self) -> dict[str, object]:
            calls.append("quota")
            return {"available": False, "next_request_at": "2026-07-05T07:00:00+00:00"}

        def cached_day_local(
            self, _feed_id: str, archive_date: date, *_args: object
        ) -> tuple[list[Path], int] | None:
            calls.append(f"cache:{archive_date}")
            return ([cached], 1) if archive_date == first_day else None

        def cached_day(self, *_args: object) -> tuple[list[Path], int]:
            raise AssertionError("The network-capable cache check must not run.")

        def authenticate(self) -> None:
            raise AssertionError("A full local guard must prevent authentication.")

        def download_day(self, *_args: object, **_kwargs: object) -> list[Path]:
            raise AssertionError("A full local guard must prevent archive requests.")

    events: list[dict[str, object]] = []
    result = JobRunner(
        JobRequest(
            feed_id="90001",
            start_date=first_day,
            end_date=second_day,
            output_dir=tmp_path,
        ),
        emit=events.append,
        client=GuardedClient(),  # type: ignore[arg-type]
    ).run()

    assert calls == ["quota", f"cache:{second_day}", f"cache:{first_day}"]
    assert result["completed_days"] == 1
    assert result["missing_days"] == [second_day.isoformat()]
    assert result["download_limited"] is True
    assert any("will not be contacted" in str(event.get("message")) for event in events)


def test_full_rolling_guard_with_complete_cache_finishes_without_false_limit(
    tmp_path: Path,
) -> None:
    archive_date = date(2026, 7, 3)
    cached_dir = tmp_path / "90001" / archive_date.strftime("%Y%m%d")
    cached_dir.mkdir(parents=True)
    cached = cached_dir / "202607030000-provider-90001.mp3"
    cached.write_bytes(b"retained audio")

    class GuardedClient:
        def archive_quota_status(self) -> dict[str, object]:
            return {"available": False}

        def cached_day_local(
            self, *_args: object
        ) -> tuple[list[Path], int] | None:
            return [cached], 1

        def cached_day(self, *_args: object) -> tuple[list[Path], int]:
            raise AssertionError("The network-capable cache check must not run.")

        def authenticate(self) -> None:
            raise AssertionError("A complete cache must not authenticate.")

    result = JobRunner(
        JobRequest(
            feed_id="90001",
            start_date=archive_date,
            end_date=archive_date,
            output_dir=tmp_path,
        ),
        client=GuardedClient(),  # type: ignore[arg-type]
    ).run()

    assert result["completed_days"] == 1
    assert result["missing_days"] == []
    assert result["download_limited"] is False


def test_real_client_full_guard_uses_completion_snapshot_without_session_calls(
    tmp_path: Path,
) -> None:
    archive_date = date(2026, 7, 3)
    feed_id = "90001"
    day_dir = tmp_path / feed_id / archive_date.strftime("%Y%m%d")
    day_dir.mkdir(parents=True)
    cached = day_dir / "202607030000-provider-90001.mp3"
    cached.write_bytes(b"retained audio")
    remember_archive_identity(
        day_dir,
        feed_id,
        archive_date,
        "90001-provider-id",
        cached,
        listing_prefix="202607030000",
    )
    assert remember_complete_archive_day(
        day_dir,
        feed_id,
        archive_date,
        ["90001-provider-id"],
    )

    ledger = ArchiveRequestLedger(
        tmp_path / "quota.sqlite3",
        limit=1,
        provider_limit=2,
    )
    request_id = ledger.reserve(
        feed_id=feed_id,
        archive_date=archive_date.isoformat(),
        archive_id="already-used",
    )
    ledger.finish(request_id, outcome="http_200", http_status=200)
    session_calls: list[str] = []

    class NoNetworkSession:
        def get(self, *_args: object, **_kwargs: object) -> None:
            session_calls.append("get")
            raise AssertionError("A closed guard must not make a GET request.")

        def post(self, *_args: object, **_kwargs: object) -> None:
            session_calls.append("post")
            raise AssertionError("A closed guard must not make a POST request.")

        def close(self) -> None:
            session_calls.append("close")

    client = BroadcastifyClient(quota_ledger=ledger)
    client.session = NoNetworkSession()  # type: ignore[assignment]
    result = JobRunner(
        JobRequest(
            feed_id=feed_id,
            start_date=archive_date,
            end_date=archive_date,
            output_dir=tmp_path,
        ),
        client=client,
    ).run()

    assert session_calls == []
    assert result["completed_days"] == 1
    assert result["missing_days"] == []
    assert result["download_limited"] is False


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


def test_lan_source_reuse_is_interleaved_with_each_backlog_day(
    tmp_path: Path,
) -> None:
    calls: list[str] = []
    first_date = date(2026, 7, 12)
    second_date = date(2026, 7, 13)

    class FakeLanSync:
        enabled = True

        def sync_day(
            self,
            _output_dir: Path,
            _feed_id: str,
            requested_date: date,
            **_kwargs: object,
        ) -> LanSyncResult:
            calls.append(f"lan:{requested_date.isoformat()}")
            return LanSyncResult(enabled=True, peers_reached=1)

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
            calls.append(f"website:{requested_date.isoformat()}")
            day = output_dir / feed_id / requested_date.strftime("%Y%m%d")
            day.mkdir(parents=True)
            source = day / f"{requested_date:%Y%m%d}0000-123456-{feed_id}.mp3"
            source.write_bytes(b"provider audio")
            return [source]

    JobRunner(
        JobRequest(
            feed_id="90001",
            start_date=first_date,
            end_date=second_date,
            output_dir=tmp_path,
            lan_sync_enabled=True,
            newest_first=False,
        ),
        client=OrderedClient(),  # type: ignore[arg-type]
        lan_sync=FakeLanSync(),  # type: ignore[arg-type]
    ).run()

    assert calls == [
        "lan:2026-07-12",
        "authenticate",
        "website:2026-07-12",
        "lan:2026-07-13",
        "website:2026-07-13",
    ]


def test_completed_lan_queue_day_skips_every_broadcastify_request(
    tmp_path: Path,
) -> None:
    archive_date = date(2026, 7, 12)
    day = tmp_path / "90001" / "20260712"
    day.mkdir(parents=True)
    source = day / "202607120000-123456-90001.mp3"
    source.write_bytes(b"peer-completed audio")
    completion_calls: list[tuple[str, date, Path, tuple[Path, ...]]] = []

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
        def remember_cached_day_complete(
            self,
            feed_id: str,
            requested_date: date,
            output_dir: Path,
            audio_files: tuple[Path, ...],
        ) -> bool:
            completion_calls.append(
                (feed_id, requested_date, output_dir, audio_files)
            )
            return True

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
    assert completion_calls == [
        ("90001", archive_date, tmp_path, (source,))
    ]


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


def test_job_claims_and_publishes_model_specific_processing_turn(
    tmp_path: Path,
    monkeypatch,
) -> None:
    archive_date = date(2026, 7, 12)
    calls: list[str] = []
    fingerprint = "d" * 64
    day = tmp_path / "91059" / "20260712"
    day.mkdir(parents=True)
    source = day / "202607120000-123456-91059.mp3"
    source.write_bytes(b"source")
    combined = day / "combined_91059_20260712.mp3"

    class ProcessingTranscriber:
        device = "cpu"
        device_index = 0
        compute_type = "float32"
        processing_fingerprint = fingerprint

        def __init__(self, **_kwargs: object) -> None:
            calls.append("load")

        def current_transcripts(self, _inputs: list[Path]) -> list[Path]:
            return []

        def transcribe_files(
            self,
            _inputs: list[Path],
            **_kwargs: object,
        ) -> list[Path]:
            calls.append("transcribe")
            transcript = day / "transcripts" / f"{combined.stem}.json"
            transcript.parent.mkdir()
            transcript.write_text("{}", encoding="utf-8")
            return [transcript]

    class Heartbeat:
        warnings: tuple[str, ...] = ()

        def __enter__(self) -> "Heartbeat":
            calls.append("heartbeat:start")
            return self

        def __exit__(self, *_args: object) -> None:
            calls.append("heartbeat:stop")

        def assert_active(self) -> None:
            calls.append("heartbeat:active")

    class ProcessingLan:
        enabled = True

        def sync_feed(self, *_args: object, **_kwargs: object) -> LanFeedSyncResult:
            return LanFeedSyncResult(enabled=True)

        def sync_day(self, *_args: object, **_kwargs: object) -> LanSyncResult:
            return LanSyncResult(enabled=True)

        def wait_for_download_turn(
            self,
            *_args: object,
            **_kwargs: object,
        ) -> LanDownloadTurn:
            return LanDownloadTurn(role="uncoordinated")

        def sync_transcripts(
            self,
            *_args: object,
            **_kwargs: object,
        ) -> LanTranscriptSyncResult:
            return LanTranscriptSyncResult(enabled=True)

        def claim_processing_turn(
            self,
            *_args: object,
            **_kwargs: object,
        ) -> LanProcessingTurn:
            calls.append("claim:processing")
            return LanProcessingTurn(
                role="leader",
                feed_id="91059",
                archive_date=archive_date.isoformat(),
                processing_fingerprint=fingerprint,
                coordinator_url="http://127.0.0.1:8765",
                lease_token="processing_lease_token_long_enough",
                lease_seconds=90.0,
            )

        def maintain_processing_lease(
            self,
            _turn: LanProcessingTurn,
        ) -> Heartbeat:
            return Heartbeat()

        def finish_processing_turn(
            self,
            _turn: LanProcessingTurn,
            *,
            outcome: str,
            artifact_count: int = 0,
        ) -> str:
            calls.append(f"finish:processing:{outcome}:{artifact_count}")
            return ""

    class Client:
        def authenticate(self) -> None:
            calls.append("authenticate")

        def download_day(self, *_args: object, **_kwargs: object) -> list[Path]:
            calls.append("download")
            return [source]

    def fake_combine(*_args: object, **_kwargs: object) -> Path:
        combined.write_bytes(b"combined")
        return combined

    class ArtifactCatalog:
        def __init__(self, *_args: object, **_kwargs: object) -> None:
            pass

        def transcript_inventory(self, *_args: object, **_kwargs: object) -> list[int]:
            return [1, 2, 3, 4]

    monkeypatch.setattr("broadcastify_cli.jobs.LocalTranscriber", ProcessingTranscriber)
    monkeypatch.setattr("broadcastify_cli.jobs.combine_mp3_files", fake_combine)
    monkeypatch.setattr("broadcastify_cli.jobs.LanArchiveCatalog", ArtifactCatalog)
    request = JobRequest(
        feed_id="91059",
        start_date=archive_date,
        end_date=archive_date,
        output_dir=tmp_path,
        combine=True,
        transcribe=True,
        lan_sync_enabled=True,
    )

    result = JobRunner(
        request,
        client=Client(),  # type: ignore[arg-type]
        lan_sync=ProcessingLan(),  # type: ignore[arg-type]
    ).run()

    assert calls == ["load", "authenticate", "download", "transcribe"]
    assert result["pending_processing_days"] == []
    assert result["lan_sync"]["processing_queue"]["uncoordinated"] == 1


def test_job_defers_duplicate_model_day_and_keeps_it_resumable(
    tmp_path: Path,
    monkeypatch,
) -> None:
    archive_date = date(2026, 7, 13)
    fingerprint = "e" * 64
    calls: list[str] = []
    day = tmp_path / "91059" / "20260713"
    day.mkdir(parents=True)
    source = day / "202607130000-123457-91059.mp3"
    source.write_bytes(b"source")

    class DeferredTranscriber:
        device = "cpu"
        device_index = 0
        compute_type = "float32"
        processing_fingerprint = fingerprint

        def __init__(self, **_kwargs: object) -> None:
            calls.append("load")

        def current_transcripts(self, _inputs: list[Path]) -> list[Path]:
            return []

        def transcribe_files(self, *_args: object, **_kwargs: object) -> list[Path]:
            calls.append("transcribe")
            transcript = day / "transcripts" / f"{source.stem}.json"
            transcript.parent.mkdir()
            transcript.write_text("{}", encoding="utf-8")
            return [transcript]

    class DeferredLan:
        enabled = True

        def sync_feed(self, *_args: object, **_kwargs: object) -> LanFeedSyncResult:
            return LanFeedSyncResult(enabled=True)

        def sync_day(self, *_args: object, **_kwargs: object) -> LanSyncResult:
            return LanSyncResult(enabled=True)

        def wait_for_download_turn(
            self,
            *_args: object,
            **_kwargs: object,
        ) -> LanDownloadTurn:
            return LanDownloadTurn(role="uncoordinated")

        def sync_transcripts(
            self,
            *_args: object,
            **_kwargs: object,
        ) -> LanTranscriptSyncResult:
            calls.append("reconcile")
            return LanTranscriptSyncResult(enabled=True)

        def claim_processing_turn(
            self,
            *_args: object,
            **_kwargs: object,
        ) -> LanProcessingTurn:
            calls.append("claim:deferred")
            return LanProcessingTurn(
                role="deferred",
                feed_id="91059",
                archive_date=archive_date.isoformat(),
                processing_fingerprint=fingerprint,
                coordinator_url="http://127.0.0.1:8765",
                owner_node_id="peer_node",
            )

    class Client:
        def authenticate(self) -> None:
            calls.append("authenticate")

        def download_day(self, *_args: object, **_kwargs: object) -> list[Path]:
            calls.append("download")
            return [source]

    monkeypatch.setattr("broadcastify_cli.jobs.LocalTranscriber", DeferredTranscriber)
    request = JobRequest(
        feed_id="91059",
        start_date=archive_date,
        end_date=archive_date,
        output_dir=tmp_path,
        transcribe=True,
        lan_sync_enabled=True,
    )

    result = JobRunner(
        request,
        client=Client(),  # type: ignore[arg-type]
        lan_sync=DeferredLan(),  # type: ignore[arg-type]
    ).run()

    assert result["pending_processing_days"] == []
    assert result["missing_days"] == []
    assert result["lan_sync"]["processing_queue"]["uncoordinated"] == 1
    assert calls == ["load", "authenticate", "download", "transcribe"]


def test_job_reuses_a_reconciled_variant_transcript_without_running_model(
    tmp_path: Path,
    monkeypatch,
) -> None:
    archive_date = date(2026, 7, 14)
    fingerprint = "f" * 64
    calls: list[str] = []
    day = tmp_path / "91059" / "20260714"
    day.mkdir(parents=True)
    source = day / "202607140000-123458-91059.mp3"
    source.write_bytes(b"source")
    variant = (
        day
        / ".broadcastify-derived"
        / fingerprint
        / ("1" * 64)
        / "transcripts"
        / "202607140000-123458-91059.json"
    )
    variant.parent.mkdir(parents=True)
    variant.write_text("{}", encoding="utf-8")

    class VariantTranscriber:
        device = "cpu"
        device_index = 0
        compute_type = "float32"
        processing_fingerprint = fingerprint

        def __init__(self, **_kwargs: object) -> None:
            calls.append("load")

        def current_transcripts(self, _inputs: list[Path]) -> list[Path]:
            return []

        def transcribe_files(self, *_args: object, **_kwargs: object) -> list[Path]:
            calls.append("transcribe")
            transcript = day / "transcripts" / f"{source.stem}.json"
            transcript.parent.mkdir(exist_ok=True)
            transcript.write_text("{}", encoding="utf-8")
            return [transcript]

    class VariantLan:
        enabled = True

        def sync_feed(self, *_args: object, **_kwargs: object) -> LanFeedSyncResult:
            return LanFeedSyncResult(enabled=True)

        def sync_day(self, *_args: object, **_kwargs: object) -> LanSyncResult:
            return LanSyncResult(enabled=True)

        def wait_for_download_turn(
            self,
            *_args: object,
            **_kwargs: object,
        ) -> LanDownloadTurn:
            return LanDownloadTurn(role="uncoordinated")

        def sync_transcripts(
            self,
            *_args: object,
            **_kwargs: object,
        ) -> LanTranscriptSyncResult:
            calls.append("reconcile")
            return LanTranscriptSyncResult(
                enabled=True,
                artifacts_already_local=4,
                transcripts=(variant,),
            )

        def claim_processing_turn(self, *_args: object, **_kwargs: object) -> None:
            raise AssertionError("reconciled work must not request a processing lease")

    class Client:
        def authenticate(self) -> None:
            calls.append("authenticate")

        def download_day(self, *_args: object, **_kwargs: object) -> list[Path]:
            calls.append("download")
            return [source]

    monkeypatch.setattr("broadcastify_cli.jobs.LocalTranscriber", VariantTranscriber)
    result = JobRunner(
        JobRequest(
            feed_id="91059",
            start_date=archive_date,
            end_date=archive_date,
            output_dir=tmp_path,
            transcribe=True,
            lan_sync_enabled=True,
        ),
        client=Client(),  # type: ignore[arg-type]
        lan_sync=VariantLan(),  # type: ignore[arg-type]
    ).run()

    assert result["days"][0]["transcripts"] != [str(variant)]
    assert result["pending_processing_days"] == []
    assert result["lan_sync"]["processing_queue"]["uncoordinated"] == 1
    assert calls == ["load", "authenticate", "download", "transcribe"]
