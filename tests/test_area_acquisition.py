from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Any

from broadcastify_cli.area_acquisition import AreaAcquisitionRunner
from broadcastify_cli.models import JobRequest
from broadcastify_cli.storage import AnalysisStore


class _FakeJobRunner:
    def __init__(self, request: JobRequest, result: dict[str, Any]) -> None:
        self.request = request
        self.result = result

    def run(self) -> dict[str, Any]:
        return {"feed_id": self.request.feed_id, **self.result}


def _result(*, complete: bool, limited: bool = False) -> dict[str, Any]:
    return {
        "days": [
            {
                "date": "2026-07-12",
                "audio_files": [],
                "transcripts": ["transcript.json"] if complete else [],
                "combined_file": "combined.mp3" if complete else None,
            }
        ]
        if complete
        else [],
        "requested_days": 1,
        "completed_days": 1 if complete else 0,
        "download_limited": limited,
        "missing_days": [] if complete else ["2026-07-12"],
    }


def _payload() -> dict[str, Any]:
    return {
        "profile_name": "Regional desk",
        "job": {
            "start_date": "2026-07-12",
            "end_date": "2026-07-12",
            "output_dir": "archives",
            "combine": True,
            "keep_originals": True,
            "transcribe": True,
            "diarize": False,
            "model": "turbo",
            "download_jobs": 8,
            "huggingface_token": "must-not-persist",
        },
    }


def _profile(store: AnalysisStore) -> None:
    store.save_area_profile(
        "Regional desk",
        ["12345", "12346"],
        [
            {"feed_id": "100", "name": "Near", "priority_rank": 1, "distance_miles": 0},
            {"feed_id": "200", "name": "Far", "priority_rank": 2, "distance_miles": 4.5},
        ],
    )


def test_area_queue_stops_on_quota_and_resumes_in_place(tmp_path: Path) -> None:
    events: list[dict[str, Any]] = []
    calls: list[str] = []
    outcomes = {"100": [_result(complete=False, limited=True)], "200": []}

    def factory(request: JobRequest, _emit: Any, _client: Any) -> _FakeJobRunner:
        calls.append(request.feed_id)
        return _FakeJobRunner(request, outcomes[request.feed_id].pop(0))

    with AnalysisStore(tmp_path / "analysis.sqlite3") as store:
        _profile(store)
        first = AreaAcquisitionRunner(
            _payload(), store, emit=events.append, client=object(), job_runner_factory=factory
        ).run()

        assert calls == ["100"]
        assert first["status"] == "quota_limited"
        assert [item["status"] for item in first["items"]] == ["partial", "pending"]
        assert first["processing"]["download_jobs"] == 1
        assert "huggingface" not in str(first["processing"]).lower()

        outcomes["100"] = [_result(complete=True)]
        outcomes["200"] = [_result(complete=True)]
        second = AreaAcquisitionRunner(
            _payload(), store, emit=events.append, client=object(), job_runner_factory=factory
        ).run()

        assert second["id"] == first["id"]
        assert calls == ["100", "100", "200"]
        assert second["status"] == "complete"
        assert [item["attempt_count"] for item in second["items"]] == [2, 1]

        third_calls = len(calls)
        third = AreaAcquisitionRunner(
            _payload(), store, emit=events.append, client=object(), job_runner_factory=factory
        ).run()
        assert third["status"] == "complete"
        assert len(calls) == third_calls


def test_area_queue_rejects_feed_outside_saved_profile(tmp_path: Path) -> None:
    payload = _payload()
    payload["feed_ids"] = ["999"]
    with AnalysisStore(tmp_path / "analysis.sqlite3") as store:
        _profile(store)
        try:
            AreaAcquisitionRunner(payload, store, client=object()).run()
        except ValueError as exc:
            assert "not in the saved profile" in str(exc)
        else:
            raise AssertionError("Ad hoc feed IDs must not enter a persisted profile queue.")


def test_legacy_area_profile_preserves_saved_priority_order(tmp_path: Path) -> None:
    with AnalysisStore(tmp_path / "analysis.sqlite3") as store:
        runner = AreaAcquisitionRunner(
            {"profile_name": "Legacy desk"}, store, client=object()
        )
        selected = runner._selected_feeds(
            [
                {"feed_id": "100", "name": "Zulu nearest"},
                {"feed_id": "200", "name": "Alpha farther"},
            ]
        )

    assert [feed["feed_id"] for feed in selected] == ["100", "200"]
    assert [feed["priority_rank"] for feed in selected] == [1, 2]


def test_area_profile_without_ranks_still_prefers_known_distance(tmp_path: Path) -> None:
    with AnalysisStore(tmp_path / "analysis.sqlite3") as store:
        runner = AreaAcquisitionRunner(
            {"profile_name": "Distance desk"}, store, client=object()
        )
        selected = runner._selected_feeds(
            [
                {"feed_id": "100", "name": "Far", "distance_miles": 8.0},
                {"feed_id": "200", "name": "Near", "distance_miles": 2.0},
            ]
        )

    assert [feed["feed_id"] for feed in selected] == ["200", "100"]


def test_interrupted_running_item_returns_to_pending(tmp_path: Path) -> None:
    with AnalysisStore(tmp_path / "analysis.sqlite3") as store:
        _profile(store)
        profile = store.get_area_profile("Regional desk")
        assert profile is not None
        run = store.ensure_area_acquisition_run(
            int(profile["id"]),
            date(2026, 7, 12),
            date(2026, 7, 12),
            {"download_jobs": 1},
            profile["feeds"],
        )
        store.start_area_acquisition_item(int(run["id"]), "100")

        recovered = store.ensure_area_acquisition_run(
            int(profile["id"]),
            date(2026, 7, 12),
            date(2026, 7, 12),
            {"download_jobs": 1},
            profile["feeds"],
        )

        assert recovered["items"][0]["status"] == "pending"
        assert "Recovered" in recovered["items"][0]["message"]
