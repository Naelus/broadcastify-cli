from __future__ import annotations

from collections.abc import Callable, Sequence
from datetime import date
from typing import Any

from .broadcastify import BroadcastifyClient
from .jobs import JobRunner
from .models import JobRequest
from .storage import AnalysisStore


EventCallback = Callable[[dict[str, Any]], None]
JobRunnerFactory = Callable[[JobRequest, EventCallback, BroadcastifyClient], JobRunner]


PERSISTED_JOB_FIELDS = {
    "output_dir",
    "combine",
    "keep_originals",
    "transcribe",
    "diarize",
    "model",
    "asr_engine",
    "device",
    "device_index",
    "compute_type",
    "asr_model_path",
    "diarization_engine",
    "diarization_device",
    "download_jobs",
    "batch_size",
    "min_speakers",
    "max_speakers",
    "lan_sync_enabled",
    "lan_discovery_enabled",
    "lan_peer_urls",
}


def _default_job_runner(
    request: JobRequest, emit: EventCallback, client: BroadcastifyClient
) -> JobRunner:
    return JobRunner(request, emit=emit, client=client)


class AreaAcquisitionRunner:
    """Persist and execute one explicit area profile in stable priority order."""

    def __init__(
        self,
        payload: dict[str, Any],
        store: AnalysisStore,
        *,
        emit: EventCallback | None = None,
        client: BroadcastifyClient | None = None,
        job_runner_factory: JobRunnerFactory = _default_job_runner,
    ) -> None:
        self.payload = payload
        self.store = store
        self.emit = emit or (lambda _: None)
        self.client = client or BroadcastifyClient()
        self.job_runner_factory = job_runner_factory

    def run(self) -> dict[str, Any]:
        profile_name = str(self.payload.get("profile_name") or "").strip()
        if not profile_name:
            raise ValueError("A saved area profile is required for resumable acquisition.")
        profile = self.store.get_area_profile(profile_name)
        if profile is None:
            raise ValueError(f"Area profile {profile_name!r} was not found.")

        job_payload = dict(self.payload.get("job") or {})
        start_date = date.fromisoformat(str(job_payload.get("start_date") or ""))
        end_date = date.fromisoformat(str(job_payload.get("end_date") or ""))
        selected_feeds = self._selected_feeds(profile["feeds"])
        processing = {
            key: job_payload[key]
            for key in sorted(PERSISTED_JOB_FIELDS)
            if key in job_payload and key != "huggingface_token"
        }
        # Area acquisition is always serialized even if a caller submits an old
        # multi-worker preference. This is a quota boundary, not only a UI default.
        processing["download_jobs"] = 1
        run = self.store.ensure_area_acquisition_run(
            int(profile["id"]),
            start_date,
            end_date,
            processing,
            selected_feeds,
        )
        run_id = int(run["id"])
        items_by_feed = {str(item["feed_id"]): item for item in run["items"]}
        quota_limited = False
        completed_results: list[dict[str, Any]] = []

        for index, feed in enumerate(selected_feeds, start=1):
            feed_id = str(feed["feed_id"])
            item = items_by_feed[feed_id]
            if item["status"] == "complete":
                self.emit(
                    {
                        "type": "area_feed",
                        "state": "cached",
                        "run_id": run_id,
                        "feed_id": feed_id,
                        "message": (
                            f"Priority {index}/{len(selected_feeds)} already complete: "
                            f"{feed['name']} ({feed_id})."
                        ),
                    }
                )
                if item["result"]:
                    completed_results.append(
                        {"feed": feed, "result": item["result"], "status": "complete"}
                    )
                continue

            self.store.start_area_acquisition_item(run_id, feed_id)
            self.emit(
                {
                    "type": "area_feed",
                    "state": "running",
                    "run_id": run_id,
                    "feed_id": feed_id,
                    "priority_rank": index,
                    "message": (
                        f"Priority {index}/{len(selected_feeds)}: "
                        f"{feed['name']} ({feed_id})."
                    ),
                }
            )
            request_payload = {
                **job_payload,
                "feed_id": feed_id,
                "feed_name": str(feed.get("name") or ""),
                "start_date": start_date.isoformat(),
                "end_date": end_date.isoformat(),
                "download_jobs": 1,
            }
            request = JobRequest.from_dict(request_payload)
            try:
                result = self.job_runner_factory(
                    request, self.emit, self.client
                ).run()
            except Exception as exc:
                self.store.finish_area_acquisition_item(
                    run_id,
                    feed_id,
                    status="failed",
                    message=str(exc),
                )
                self.store.finish_area_acquisition_run(run_id, "failed", str(exc))
                raise

            item_complete = (
                not bool(result.get("download_limited"))
                and int(result.get("completed_days") or 0)
                == int(result.get("requested_days") or 0)
            )
            item_status = "complete" if item_complete else "partial"
            self.store.finish_area_acquisition_item(
                run_id,
                feed_id,
                status=item_status,
                result=result,
                message=(
                    "Complete."
                    if item_complete
                    else "Incomplete and safely resumable from the local cache."
                ),
            )
            completed_results.append(
                {"feed": feed, "result": result, "status": item_status}
            )
            if bool(result.get("download_limited")):
                quota_limited = True
                self.emit(
                    {
                        "type": "log",
                        "stage": "download",
                        "message": (
                            "The area queue stopped on the first explicit Broadcastify "
                            "quota response. No lower-priority feed will make an archive request."
                        ),
                    }
                )
                break

        current = self.store.get_area_acquisition_run(run_id)
        assert current is not None
        all_complete = all(item["status"] == "complete" for item in current["items"])
        if all_complete:
            status = "complete"
            stop_reason = ""
            message = f"All {len(selected_feeds)} area feeds are complete."
        elif quota_limited:
            status = "quota_limited"
            stop_reason = "Broadcastify archive quota exhausted; resume this saved queue later."
            message = (
                "Area acquisition paused at the quota boundary. Completed work is retained "
                "and the first incomplete feed remains next."
            )
        else:
            status = "partial"
            stop_reason = "One or more selected feeds remain incomplete."
            message = "Area acquisition ended with resumable coverage gaps."
        persisted = self.store.finish_area_acquisition_run(run_id, status, stop_reason)
        result = {
            **persisted,
            "feed_results": completed_results,
            "download_limited": quota_limited,
        }
        self.emit(
            {
                "type": "area_complete",
                "message": message,
                "result": result,
            }
        )
        return result

    def _selected_feeds(
        self, profile_feeds: Sequence[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        requested_ids = list(
            dict.fromkeys(str(value).strip() for value in self.payload.get("feed_ids", []))
        )
        available = {str(feed["feed_id"]): dict(feed) for feed in profile_feeds}
        saved_order = {
            str(feed["feed_id"]): index for index, feed in enumerate(profile_feeds)
        }
        if requested_ids:
            unknown = [value for value in requested_ids if value not in available]
            if unknown:
                raise ValueError(
                    "Selected area feeds are not in the saved profile: " + ", ".join(unknown)
                )
            feeds = [available[value] for value in requested_ids]
        else:
            feeds = list(available.values())
        has_explicit_priority = any(feed.get("priority_rank") is not None for feed in feeds)
        if has_explicit_priority:
            feeds.sort(
                key=lambda feed: (
                    int(feed["priority_rank"])
                    if feed.get("priority_rank") is not None
                    else len(profile_feeds) + saved_order[str(feed["feed_id"])] + 1,
                    float(feed["distance_miles"])
                    if feed.get("distance_miles") is not None
                    else float("inf"),
                    saved_order[str(feed["feed_id"])],
                )
            )
        elif any(feed.get("distance_miles") is not None for feed in feeds):
            feeds.sort(
                key=lambda feed: (
                    float(feed["distance_miles"])
                    if feed.get("distance_miles") is not None
                    else float("inf"),
                    saved_order[str(feed["feed_id"])],
                )
            )
        else:
            # Hand-curated and legacy profiles may have neither a calculated
            # distance nor an explicit rank. Their saved order is intentional;
            # alphabetical resorting can send the scarce upstream allowance to
            # a lower-priority feed first.
            feeds.sort(key=lambda feed: saved_order[str(feed["feed_id"])])
        for index, feed in enumerate(feeds, start=1):
            feed["priority_rank"] = index
        return feeds
