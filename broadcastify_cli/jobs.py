from __future__ import annotations

import os
from collections.abc import Callable
from datetime import date
from pathlib import Path
from typing import Any

from .audio import combine_mp3_files
from .broadcastify import BroadcastifyClient, DownloadLimitExceeded
from .lan_sync import (
    LanArchiveCatalog,
    LanArchiveSyncClient,
    LanDownloadTurn,
    LanFeedSyncResult,
    LanProcessingTurn,
    LanSyncResult,
    LanTranscriptSyncResult,
    LanSyncError,
    merge_lan_sync_results,
)
from .models import JobRequest
from .transcription import LocalTranscriber


EventCallback = Callable[[dict[str, Any]], None]


class JobRunner:
    def __init__(
        self,
        request: JobRequest,
        emit: EventCallback | None = None,
        client: BroadcastifyClient | None = None,
        lan_sync: LanArchiveSyncClient | None = None,
    ) -> None:
        request.validate()
        self.request = request
        self.emit = emit or (lambda _: None)
        self.client = client or BroadcastifyClient()
        self.lan_sync = lan_sync or LanArchiveSyncClient.from_settings(
            enabled=request.lan_sync_enabled,
            peer_urls=request.lan_peer_urls,
            discovery_enabled=request.lan_discovery_enabled,
        )

    def run(self) -> dict[str, Any]:
        dates = list(self.request.dates())
        transcriber = None
        if self.request.transcribe:
            self.emit(
                {
                    "type": "log",
                    "message": (
                        f"Loading local model {self.request.model} before "
                        "archive acquisition..."
                    ),
                }
            )
            transcriber = LocalTranscriber(
                model_name=self.request.model,
                asr_engine=self.request.asr_engine,
                device=self.request.device,
                device_index=self.request.device_index,
                compute_type=self.request.compute_type,
                asr_model_path=self.request.asr_model_path,
                diarization_engine=self.request.diarization_engine,
                diarization_device=self.request.diarization_device,
                diarize=self.request.diarize,
                huggingface_token=(
                    self.request.huggingface_token
                    or os.getenv("HUGGINGFACE_TOKEN")
                    or os.getenv("HF_TOKEN")
                ),
                batch_size=self.request.batch_size,
                min_speakers=self.request.min_speakers,
                max_speakers=self.request.max_speakers,
            )
            self.emit(
                {
                    "type": "log",
                    "message": (
                        "Transcription: "
                        f"{getattr(transcriber, 'backend_description', f'{transcriber.device}:{transcriber.device_index} ({transcriber.compute_type})')}. "
                        "Diarization: "
                        f"{getattr(transcriber, 'diarization_engine', 'community-1')} "
                        f"on {getattr(transcriber, 'diarization_device', 'auto')}."
                    ),
                }
            )
        processing_fingerprint = str(
            getattr(transcriber, "processing_fingerprint", "") or ""
        )
        feed_reconciliation = LanFeedSyncResult(enabled=False)
        lan_results: dict[str, LanSyncResult] = {}
        lan_transcript_results: dict[str, LanTranscriptSyncResult] = {}
        if self.lan_sync.enabled:
            self.emit(
                {
                    "type": "stage",
                    "stage": "lan_sync",
                    "message": (
                        "Checking the trusted LAN for retained archive blocks "
                        "before contacting Broadcastify"
                    ),
                }
            )
            reconcile_feed = getattr(self.lan_sync, "sync_feed", None)
            if callable(reconcile_feed):
                try:
                    feed_reconciliation = reconcile_feed(
                        self.request.output_dir,
                        self.request.feed_id,
                        processing_fingerprint=processing_fingerprint,
                        progress=lambda message: self.emit(
                            {
                                "type": "progress",
                                "stage": "lan_reconcile",
                                "current": 0,
                                "total": 0,
                                "message": message,
                            }
                        ),
                    )
                except Exception as exc:
                    feed_reconciliation = LanFeedSyncResult(
                        enabled=True,
                        failures=(str(exc),),
                    )
            for day_number, archive_date in enumerate(dates, start=1):
                day_label = archive_date.isoformat()

                def lan_progress(message: str) -> None:
                    self.emit(
                        {
                            "type": "progress",
                            "stage": "lan_sync",
                            "current": day_number,
                            "total": len(dates),
                            "message": message,
                        }
                    )

                try:
                    sync_result = self.lan_sync.sync_day(
                        self.request.output_dir,
                        self.request.feed_id,
                        archive_date,
                        progress=lan_progress,
                    )
                except Exception as exc:
                    sync_result = LanSyncResult(
                        enabled=True,
                        failures=(str(exc),),
                    )
                lan_results[day_label] = sync_result
            copied = sum(value.blocks_copied for value in lan_results.values())
            copied_bytes = sum(value.bytes_copied for value in lan_results.values())
            failures = [
                failure
                for value in lan_results.values()
                for failure in value.failures
            ]
            reached = max(
                (value.peers_reached for value in lan_results.values()),
                default=0,
            )
            if copied:
                self.emit(
                    {
                        "type": "log",
                        "stage": "lan_sync",
                        "message": (
                            f"LAN archive reuse supplied {copied} source block"
                            f"{'s' if copied != 1 else ''} "
                            f"({copied_bytes / (1024 * 1024):.1f} MiB). "
                            "Those blocks will not consume Broadcastify download quota."
                        ),
                    }
                )
            elif reached:
                self.emit(
                    {
                        "type": "log",
                        "stage": "lan_sync",
                        "message": (
                            "Trusted-LAN peers were reachable, but they had no missing "
                            "source blocks for this request."
                        ),
                    }
                )
            else:
                self.emit(
                    {
                        "type": "log",
                        "stage": "lan_sync",
                        "message": (
                            "No trusted-LAN archive peer answered; continuing with the "
                            "local cache and quota-safe website fallback."
                        ),
                    }
                )
            if failures:
                self.emit(
                    {
                        "type": "log",
                        "stage": "lan_sync",
                        "message": (
                            f"LAN reuse reported {len(failures)} peer warning"
                            f"{'s' if len(failures) != 1 else ''}; website fallback "
                            f"remains available. First warning: {failures[0]}"
                        ),
                    }
                )

        downloaded_days: list[tuple[Any, list[Path]]] = []
        quota_message: str | None = None
        quota_status = getattr(self.client, "archive_quota_status", None)
        initial_quota = quota_status() if callable(quota_status) else {}
        if initial_quota and not bool(initial_quota.get("available", True)):
            quota_message = (
                "The local rolling archive-request guard has no automated "
                "slot available."
            )
            self.emit(
                {
                    "type": "log",
                    "stage": "download",
                    "message": (
                        "The rolling archive-request guard is full. This job "
                        "will process complete local or trusted-LAN days only; "
                        "Broadcastify will not be contacted until the ledger's "
                        "next safe slot."
                    ),
                }
            )
        authenticated = False
        queue_roles = {
            "leader": 0,
            "completed": 0,
            "quota_limited": 0,
            "deferred": 0,
            "uncoordinated": 0,
        }

        def ensure_authenticated() -> None:
            nonlocal authenticated
            if authenticated:
                return
            self.emit(
                {"type": "log", "message": "Authenticating with Broadcastify..."}
            )
            self.client.authenticate()
            authenticated = True

        for day_number, archive_date in enumerate(dates, start=1):
            day_label = archive_date.isoformat()
            if quota_message is not None:
                local_cache = getattr(self.client, "cached_day_local", None)
                cache_state = (
                    local_cache(
                        self.request.feed_id,
                        archive_date,
                        self.request.output_dir,
                    )
                    if callable(local_cache)
                    else None
                )
                if cache_state is not None:
                    cached, expected = cache_state
                    downloaded_days.append((archive_date, cached))
                    self.emit(
                        {
                            "type": "log",
                            "message": (
                                f"Reusing a locally proven completion snapshot for "
                                f"{day_label} ({expected} archive identities in "
                                f"{len(cached)} retained source files)."
                            ),
                        }
                    )
                else:
                    self.emit(
                        {
                            "type": "log",
                            "message": (
                                f"Skipping {day_label}: its local cache is incomplete, "
                                "and the Broadcastify download quota is already exhausted."
                            ),
                        }
                    )
                continue

            def queue_progress(message: str) -> None:
                self.emit(
                    {
                        "type": "progress",
                        "stage": "lan_queue",
                        "current": day_number,
                        "total": len(dates),
                        "message": message,
                    }
                )

            coordinate = getattr(self.lan_sync, "wait_for_download_turn", None)
            turn = (
                coordinate(
                    self.request.output_dir,
                    self.request.feed_id,
                    archive_date,
                    progress=queue_progress,
                    defer_active=True,
                )
                if self.lan_sync.enabled and callable(coordinate)
                else LanDownloadTurn(role="uncoordinated")
            )
            queue_roles[turn.role] = queue_roles.get(turn.role, 0) + 1
            if turn.sync_result.enabled:
                lan_results[day_label] = merge_lan_sync_results(
                    (
                        lan_results.get(
                            day_label,
                            LanSyncResult(enabled=True),
                        ),
                        turn.sync_result,
                    )
                )
            for warning in turn.warnings:
                self.emit(
                    {
                        "type": "log",
                        "stage": "lan_queue",
                        "message": warning,
                    }
                )
            if turn.role == "completed":
                downloaded_days.append((archive_date, list(turn.audio_files)))
                remember_complete = getattr(
                    self.client,
                    "remember_cached_day_complete",
                    None,
                )
                if callable(remember_complete) and not remember_complete(
                    self.request.feed_id,
                    archive_date,
                    self.request.output_dir,
                    turn.audio_files,
                ):
                    self.emit(
                        {
                            "type": "log",
                            "stage": "lan_queue",
                            "message": (
                                f"The LAN completion for {day_label} is usable now, "
                                "but its local completion snapshot could not be "
                                "persisted for an offline retry."
                            ),
                        }
                    )
                completion_kind = (
                    "rolling current-day snapshot"
                    if turn.rolling
                    else "LAN acquisition"
                )
                self.emit(
                    {
                        "type": "log",
                        "stage": "lan_queue",
                        "message": (
                            f"Reused the completed {completion_kind} for {day_label} "
                            f"({len(turn.audio_files)} verified source blocks); "
                            "Broadcastify was not contacted for this day."
                        ),
                    }
                )
                continue
            if turn.role == "quota_limited":
                quota_message = (
                    "A trusted-LAN producer reached Broadcastify's shared "
                    "archive download quota."
                )
                self.emit(
                    {
                        "type": "log",
                        "stage": "lan_queue",
                        "message": (
                            f"Skipping upstream acquisition for {day_label}: "
                            "another LAN producer already reached the shared "
                            "download limit. Its completed partial blocks remain "
                            "available for a later resume."
                        ),
                    }
                )
                continue
            if turn.role == "deferred":
                self.emit(
                    {
                        "type": "log",
                        "stage": "lan_queue",
                        "message": (
                            f"Deferred {day_label}: its LAN producer is still "
                            "active or temporarily unreachable, so this client "
                            "will not duplicate the upstream archive requests."
                        ),
                    }
                )
                continue
            if turn.role == "leader":
                self.emit(
                    {
                        "type": "log",
                        "stage": "lan_queue",
                        "message": (
                            f"This client is the sole LAN upstream producer for "
                            f"{day_label}; peers can copy blocks as they complete."
                        ),
                    }
                )

            self.emit(
                {
                    "type": "stage",
                    "stage": "download",
                    "message": f"Downloading {day_label} ({day_number}/{len(dates)})",
                }
            )

            def download_progress(current: int, total: int, message: str) -> None:
                self.emit(
                    {
                        "type": "progress",
                        "stage": "download",
                        "current": current,
                        "total": total,
                        "message": message,
                    }
                )

            try:
                if turn.is_leader:
                    with self.lan_sync.maintain_download_lease(turn) as heartbeat:
                        try:
                            ensure_authenticated()
                            audio_files = self.client.download_day(
                                self.request.feed_id,
                                archive_date,
                                self.request.output_dir,
                                jobs=self.request.download_jobs,
                                progress=download_progress,
                                admit_download=heartbeat.assert_active,
                            )
                        except DownloadLimitExceeded:
                            quota = self.client.archive_quota_status()
                            warning = self.lan_sync.finish_download_turn(
                                turn,
                                outcome="quota_limited",
                                block_count=len(
                                    self.lan_sync.local_source_files(
                                        self.request.output_dir,
                                        self.request.feed_id,
                                        archive_date,
                                    )
                                ),
                                retry_after_seconds=max(
                                    5.0,
                                    float(
                                        quota.get("next_request_seconds")
                                        or 5.0
                                    ),
                                ),
                            )
                            if warning:
                                self.emit(
                                    {
                                        "type": "log",
                                        "stage": "lan_queue",
                                        "message": warning,
                                    }
                                )
                            raise
                        except Exception:
                            warning = self.lan_sync.finish_download_turn(
                                turn,
                                outcome="failed",
                            )
                            if warning:
                                self.emit(
                                    {
                                        "type": "log",
                                        "stage": "lan_queue",
                                        "message": warning,
                                    }
                                )
                            raise
                        else:
                            warning = self.lan_sync.finish_download_turn(
                                turn,
                                outcome="complete",
                                block_count=len(audio_files),
                                source_files=audio_files,
                            )
                            if warning:
                                self.emit(
                                    {
                                        "type": "log",
                                        "stage": "lan_queue",
                                        "message": warning,
                                    }
                                )
                    for warning in heartbeat.warnings:
                        self.emit(
                            {
                                "type": "log",
                                "stage": "lan_queue",
                                "message": warning,
                            }
                        )
                else:
                    ensure_authenticated()
                    audio_files = self.client.download_day(
                        self.request.feed_id,
                        archive_date,
                        self.request.output_dir,
                        jobs=self.request.download_jobs,
                        progress=download_progress,
                    )
            except DownloadLimitExceeded as exc:
                quota_message = str(exc)
                self.emit(
                    {
                        "type": "log",
                        "stage": "download",
                        "message": (
                            f"Download quota reached while acquiring {day_label}. "
                            "No more archive download requests will be made in this job; "
                            "complete cached days will still be processed."
                        ),
                    }
                )
                continue
            downloaded_days.append((archive_date, audio_files))

        day_results: list[dict[str, Any]] = []
        pending_processing_days: list[str] = []
        processing_roles = {
            "leader": 0,
            "completed": 0,
            "deferred": 0,
            "uncoordinated": 0,
            "local_cache": 0,
        }

        def matching_transcripts(inputs: list[Path]) -> list[Path]:
            current = getattr(transcriber, "current_transcripts", None)
            if transcriber is None or not callable(current):
                return []
            return list(current(inputs))

        def sync_matching_transcripts(
            archive_date: Any,
            inputs: list[Path],
        ) -> list[Path]:
            if (
                not self.lan_sync.enabled
                or not processing_fingerprint
                or not inputs
            ):
                return matching_transcripts(inputs)
            sync_transcripts = getattr(self.lan_sync, "sync_transcripts", None)
            if callable(sync_transcripts):
                try:
                    value = sync_transcripts(
                        self.request.output_dir,
                        self.request.feed_id,
                        archive_date,
                        processing_fingerprint,
                    )
                except Exception as exc:
                    value = LanTranscriptSyncResult(
                        enabled=True,
                        failures=(str(exc),),
                    )
                lan_transcript_results[archive_date.isoformat()] = value
                for warning in value.failures:
                    self.emit(
                        {
                            "type": "log",
                            "stage": "lan_processing",
                            "message": warning,
                        }
                    )
            return matching_transcripts(inputs)

        for archive_date, audio_files in downloaded_days:
            day_label = archive_date.isoformat()
            day_dir = (
                Path(self.request.output_dir)
                / self.request.feed_id
                / archive_date.strftime("%Y%m%d")
            )

            combined = None
            if self.request.combine and audio_files:
                self.emit(
                    {
                        "type": "stage",
                        "stage": "combine",
                        "message": f"Combining {day_label}",
                    }
                )
                combined = combine_mp3_files(
                    day_dir,
                    self.request.feed_id,
                    archive_date,
                    source_files=audio_files,
                    delete_sources=not self.request.keep_originals,
                    feed_name=self.request.feed_name,
                )

            # A combined job must be transcribed after concatenation. This gives
            # pyannote one continuous timeline and prevents speaker labels from
            # restarting independently in every source archive.
            transcription_inputs = [combined] if combined else audio_files
            transcripts: list[Path] = []
            if transcriber and transcription_inputs:
                transcripts = sync_matching_transcripts(
                    archive_date,
                    transcription_inputs,
                )
                if transcripts:
                    processing_roles["local_cache"] += 1
                    self.emit(
                        {
                            "type": "log",
                            "stage": "lan_processing",
                            "message": (
                                f"Reused matching local or LAN transcript artifacts "
                                f"for {day_label}; the model was not run again."
                            ),
                        }
                    )

                claim_processing = getattr(
                    self.lan_sync,
                    "claim_processing_turn",
                    None,
                )
                processing_turn = (
                    claim_processing(
                        self.request.feed_id,
                        archive_date,
                        processing_fingerprint,
                        progress=lambda message: self.emit(
                            {
                                "type": "progress",
                                "stage": "lan_processing",
                                "current": 0,
                                "total": 0,
                                "message": message,
                            }
                        ),
                    )
                    if not transcripts
                    and self.lan_sync.enabled
                    and processing_fingerprint
                    and callable(claim_processing)
                    else LanProcessingTurn(role="uncoordinated")
                )
                if not transcripts:
                    processing_roles[processing_turn.role] = (
                        processing_roles.get(processing_turn.role, 0) + 1
                    )
                for warning in processing_turn.warnings:
                    self.emit(
                        {
                            "type": "log",
                            "stage": "lan_processing",
                            "message": warning,
                        }
                    )

                if not transcripts and processing_turn.role in {
                    "completed",
                    "deferred",
                }:
                    transcripts = sync_matching_transcripts(
                        archive_date,
                        transcription_inputs,
                    )
                    if not transcripts:
                        pending_processing_days.append(day_label)
                        self.emit(
                            {
                                "type": "log",
                                "stage": "lan_processing",
                                "message": (
                                    f"Deferred model work for {day_label}; another "
                                    "node owns or just completed the identical model/day. "
                                    "This job will reconcile its artifacts again before exit."
                                ),
                            }
                        )
                elif not transcripts:
                    self.emit(
                        {
                            "type": "stage",
                            "stage": "transcribe",
                            "message": f"Transcribing {day_label}",
                        }
                    )

                    def transcription_progress(
                        current: int,
                        total: int,
                        message: str,
                    ) -> None:
                        self.emit(
                            {
                                "type": "progress",
                                "stage": "transcribe",
                                "current": current,
                                "total": total,
                                "message": message,
                            }
                        )

                    if processing_turn.is_leader:
                        heartbeat_manager = self.lan_sync.maintain_processing_lease(
                            processing_turn
                        )
                        with heartbeat_manager as heartbeat:
                            try:
                                heartbeat.assert_active()
                                transcripts = transcriber.transcribe_files(
                                    transcription_inputs,
                                    progress=transcription_progress,
                                )
                                heartbeat.assert_active()
                                artifact_count = len(
                                    LanArchiveCatalog(
                                        self.request.output_dir,
                                        enabled=True,
                                        queue_enabled=False,
                                    ).transcript_inventory(
                                        self.request.feed_id,
                                        archive_date,
                                        processing_fingerprint,
                                    )
                                )
                                minimum_artifacts = 3 * len(transcription_inputs)
                                if artifact_count < minimum_artifacts:
                                    raise LanSyncError(
                                        "The completed model run did not publish a "
                                        "complete hash-verified transcript artifact set."
                                    )
                            except Exception:
                                warning = self.lan_sync.finish_processing_turn(
                                    processing_turn,
                                    outcome="failed",
                                )
                                if warning:
                                    self.emit(
                                        {
                                            "type": "log",
                                            "stage": "lan_processing",
                                            "message": warning,
                                        }
                                    )
                                raise
                            else:
                                warning = self.lan_sync.finish_processing_turn(
                                    processing_turn,
                                    outcome="complete",
                                    artifact_count=artifact_count,
                                )
                                if warning:
                                    self.emit(
                                        {
                                            "type": "log",
                                            "stage": "lan_processing",
                                            "message": warning,
                                        }
                                    )
                        for warning in heartbeat.warnings:
                            self.emit(
                                {
                                    "type": "log",
                                    "stage": "lan_processing",
                                    "message": warning,
                                }
                            )
                    else:
                        transcripts = transcriber.transcribe_files(
                            transcription_inputs,
                            progress=transcription_progress,
                        )

            day_results.append(
                {
                    "date": day_label,
                    "audio_files": [str(path) for path in audio_files],
                    "transcripts": [str(path) for path in transcripts],
                    "combined_file": str(combined) if combined else None,
                }
            )

        # A peer may finish while this node processes another day. Pull those
        # artifacts once more without waiting; anything still active remains a
        # durable scheduled retry instead of being duplicated here.
        if transcriber and pending_processing_days:
            still_pending: list[str] = []
            by_date = {value["date"]: value for value in day_results}
            for day_label in pending_processing_days:
                value = by_date[day_label]
                inputs = (
                    [Path(value["combined_file"])]
                    if value["combined_file"]
                    else [Path(path) for path in value["audio_files"]]
                )
                transcripts = sync_matching_transcripts(
                    date.fromisoformat(day_label),
                    inputs,
                )
                if transcripts:
                    value["transcripts"] = [str(path) for path in transcripts]
                else:
                    still_pending.append(day_label)
            pending_processing_days = still_pending

        completed_dates = {day for day, _files in downloaded_days}
        missing_days = [
            value.isoformat()
            for value in dates
            if value not in completed_dates
        ]
        download_limited = quota_message is not None and bool(missing_days)
        result = {
            "feed_id": self.request.feed_id,
            "feed_name": self.request.feed_name,
            "output_dir": str(self.request.output_dir),
            "days": day_results,
            "requested_days": len(dates),
            "completed_days": len(day_results),
            "download_limited": download_limited,
            "missing_days": missing_days,
            "pending_processing_days": pending_processing_days,
            "lan_sync": {
                "enabled": self.lan_sync.enabled,
                "blocks_copied": sum(
                    value.blocks_copied for value in lan_results.values()
                ),
                "bytes_copied": sum(
                    value.bytes_copied for value in lan_results.values()
                ),
                "days": {
                    day: value.to_dict() for day, value in lan_results.items()
                },
                "acquisition_queue": queue_roles,
                "processing_queue": processing_roles,
                "transcripts": {
                    day: value.to_dict()
                    for day, value in lan_transcript_results.items()
                },
                "feed_reconciliation": feed_reconciliation.to_dict(),
            },
        }
        if pending_processing_days:
            message = (
                f"Archive acquisition completed; {len(pending_processing_days)} "
                "model/day result(s) remain assigned to another LAN node and "
                "will reconcile on the next scheduled pass."
            )
        elif not download_limited:
            message = "All operations completed."
        else:
            message = (
                f"Completed {len(day_results)}/{len(dates)} requested days. "
                "Broadcastify's download quota stopped new archive requests; "
                "incomplete days remain safely resumable."
            )
        self.emit({"type": "complete", "message": message, "result": result})
        return result
