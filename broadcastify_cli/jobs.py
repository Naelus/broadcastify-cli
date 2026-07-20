from __future__ import annotations

import os
from collections.abc import Callable
from pathlib import Path
from typing import Any

from .audio import combine_mp3_files
from .broadcastify import BroadcastifyClient, DownloadLimitExceeded
from .lan_sync import (
    LanArchiveSyncClient,
    LanDownloadTurn,
    LanSyncResult,
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
        lan_results: dict[str, LanSyncResult] = {}
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
                cached, expected = self.client.cached_day(
                    self.request.feed_id,
                    archive_date,
                    self.request.output_dir,
                )
                if cached or expected == 0:
                    downloaded_days.append((archive_date, cached))
                    self.emit(
                        {
                            "type": "log",
                            "message": (
                                f"Reusing complete local cache for {day_label} "
                                f"({len(cached)}/{expected} archives)."
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
                self.emit(
                    {
                        "type": "log",
                        "stage": "lan_queue",
                        "message": (
                            f"Reused the completed LAN acquisition for {day_label} "
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

        transcriber = None
        if self.request.transcribe and downloaded_days:
            self.emit(
                {
                    "type": "log",
                    "message": f"Loading local model {self.request.model}...",
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

        day_results: list[dict[str, Any]] = []
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
                self.emit(
                    {
                        "type": "stage",
                        "stage": "transcribe",
                        "message": f"Transcribing {day_label}",
                    }
                )

                def transcription_progress(current: int, total: int, message: str) -> None:
                    self.emit(
                        {
                            "type": "progress",
                            "stage": "transcribe",
                            "current": current,
                            "total": total,
                            "message": message,
                        }
                    )

                transcripts = transcriber.transcribe_files(
                    transcription_inputs, progress=transcription_progress
                )

            day_results.append(
                {
                    "date": day_label,
                    "audio_files": [str(path) for path in audio_files],
                    "transcripts": [str(path) for path in transcripts],
                    "combined_file": str(combined) if combined else None,
                }
            )

        result = {
            "feed_id": self.request.feed_id,
            "feed_name": self.request.feed_name,
            "output_dir": str(self.request.output_dir),
            "days": day_results,
            "requested_days": len(dates),
            "completed_days": len(day_results),
            "download_limited": quota_message is not None,
            "missing_days": [
                value.isoformat()
                for value in dates
                if value not in {day for day, _files in downloaded_days}
            ],
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
            },
        }
        if quota_message is None:
            message = "All operations completed."
        else:
            message = (
                f"Completed {len(day_results)}/{len(dates)} requested days. "
                "Broadcastify's download quota stopped new archive requests; "
                "incomplete days remain safely resumable."
            )
        self.emit({"type": "complete", "message": message, "result": result})
        return result
