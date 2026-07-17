from __future__ import annotations

import os
from collections.abc import Callable
from pathlib import Path
from typing import Any

from .audio import combine_mp3_files
from .broadcastify import BroadcastifyClient, DownloadLimitExceeded
from .models import JobRequest
from .transcription import LocalTranscriber


EventCallback = Callable[[dict[str, Any]], None]


class JobRunner:
    def __init__(
        self,
        request: JobRequest,
        emit: EventCallback | None = None,
        client: BroadcastifyClient | None = None,
    ) -> None:
        request.validate()
        self.request = request
        self.emit = emit or (lambda _: None)
        self.client = client or BroadcastifyClient()

    def run(self) -> dict[str, Any]:
        self.emit({"type": "log", "message": "Authenticating with Broadcastify..."})
        self.client.authenticate()

        dates = list(self.request.dates())
        downloaded_days: list[tuple[Any, list[Path]]] = []
        quota_message: str | None = None
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
                        f"Diarization: {getattr(transcriber, 'diarization_device', 'auto')}."
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
