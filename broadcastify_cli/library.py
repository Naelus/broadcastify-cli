from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, Callable

from .analysis import PROMPT_VERSION
from .audio import combined_output_is_current
from .portable_diarization import (
    COMMUNITY_DIARIZATION_ENGINE,
    COMMUNITY_DIARIZATION_QUALITY,
    PORTABLE_DIARIZATION_ENGINE,
    diarization_engine_satisfies,
    normalize_diarization_engine,
)
from .storage import AnalysisStore
from .transcription import LocalTranscriber
from .workfiles import (
    cleanup_orphaned_audio_work_files,
    directory_storage_usage,
)


RAW_ARCHIVE_PATTERN = re.compile(r"^\d{12}-\d+-(\d+)\.mp3$", re.IGNORECASE)
DAY_DIRECTORY_PATTERN = re.compile(r"^\d{8}$")


def _transcript_tail(path: str | Path) -> str:
    transcript = Path(path)
    if not transcript.is_file():
        return ""
    try:
        with transcript.open("rb") as handle:
            handle.seek(max(0, transcript.stat().st_size - 512 * 1024))
            return handle.read().decode("utf-8", errors="ignore").lower()
    except OSError:
        return ""


def transcript_has_diarization(path: str | Path) -> bool:
    """Confirm that diarization finished, rather than merely being requested."""

    tail = _transcript_tail(path)
    if not tail:
        return False
    if '"diarization_completed": true' in tail:
        return True
    requested = '"diarization_requested": true' in tail
    model_recorded = '"diarization_model"' in tail
    completion_evidence = (
        '"speaker_turns"' in tail
        or '"speaker": "speaker_' in tail
        or '"speaker":"speaker_' in tail
    )
    return requested and model_recorded and completion_evidence


def transcript_diarization_engine(path: str | Path) -> str:
    if not transcript_has_diarization(path):
        return ""
    tail = _transcript_tail(path)
    match = re.search(r'"diarization_engine"\s*:\s*"([^"]+)"', tail)
    if match:
        try:
            return normalize_diarization_engine(match.group(1))
        except ValueError:
            return match.group(1)
    if "pyannote/speaker-diarization-community-1" in tail:
        return COMMUNITY_DIARIZATION_ENGINE
    if "pyannote-segmentation-3.0-int8+nemo-titanet-small" in tail:
        return PORTABLE_DIARIZATION_ENGINE
    # Completed transcripts written before engine metadata existed used
    # Community-1 exclusively.
    return COMMUNITY_DIARIZATION_ENGINE


def transcript_satisfies_diarization(
    path: str | Path, requested_engine: str
) -> bool:
    actual = transcript_diarization_engine(path)
    return diarization_engine_satisfies(actual, requested_engine)


def _friendly_feed_names(store: AnalysisStore) -> dict[str, str]:
    names: dict[str, str] = {}
    if hasattr(store, "list_feed_catalog"):
        for feed in store.list_feed_catalog():
            feed_id = str(feed.get("feed_id") or "")
            name = str(feed.get("name") or "")
            if feed_id and name:
                names[feed_id] = name
    for profile in store.list_area_profiles():
        for feed in profile.get("feeds", []):
            feed_id = str(feed.get("feed_id") or "")
            name = str(feed.get("name") or "")
            if feed_id and name:
                names.setdefault(feed_id, name)
    return names


def _manifest_feed_name(path: Path) -> str:
    if not path.is_file():
        return ""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        return ""
    if not isinstance(payload, dict):
        return ""
    return str(payload.get("feed_name") or "").strip()[:200]


def _transcript_is_current(audio: Path, transcript: Path) -> bool:
    """Reject results created before the combined recording was refreshed."""

    try:
        return (
            audio.is_file()
            and transcript.is_file()
            and transcript.stat().st_size > 0
            and transcript.stat().st_mtime_ns >= audio.stat().st_mtime_ns
        )
    except OSError:
        return False


def _state_for_day(
    output_root: Path,
    feed_id: str,
    archive_date: date,
    stored: dict[str, Any] | None,
    feed_name: str,
) -> dict[str, Any]:
    day_directory = output_root / feed_id / archive_date.strftime("%Y%m%d")
    stem = f"combined_{feed_id}_{archive_date:%Y%m%d}"
    combined = day_directory / f"{stem}.mp3"
    manifest = day_directory / f"{stem}.manifest.json"
    transcript = day_directory / "transcripts" / f"{stem}.json"

    if stored:
        stored_audio = Path(str(stored.get("audio_path") or ""))
        stored_transcript = Path(str(stored.get("transcript_path") or ""))
        stored_manifest = Path(str(stored.get("manifest_path") or ""))
        if not combined.is_file() and stored_audio.is_file():
            combined = stored_audio
            day_directory = combined.parent
        if not transcript.is_file() and stored_transcript.is_file():
            transcript = stored_transcript
        if not manifest.is_file() and stored_manifest.is_file():
            manifest = stored_manifest

    raw_files = []
    if day_directory.is_dir():
        raw_files = sorted(
            path
            for path in day_directory.glob("*.mp3")
            if RAW_ARCHIVE_PATTERN.match(path.name)
        )
    has_combined_file = combined.is_file() and combined.stat().st_size > 0
    # Imported/legacy combined recordings may legitimately have no retained raw
    # blocks.  When raw blocks are present, however, the manifest must describe
    # that exact set.  Otherwise an interrupted refresh can leave an older MP3,
    # transcript, and analysis that look complete even though newer blocks are
    # waiting to be combined.
    has_stale_combined = bool(
        has_combined_file
        and raw_files
        and not combined_output_is_current(combined, manifest, raw_files)
    )
    has_combined = has_combined_file and not has_stale_combined
    has_transcript = bool(
        has_combined and _transcript_is_current(combined, transcript)
    )
    has_diarization = has_transcript and (
        transcript_has_diarization(transcript)
        or bool(stored and stored.get("has_diarization"))
    )
    diarization_engine = (
        transcript_diarization_engine(transcript) if has_diarization else ""
    )
    speaker_upgrade_available = (
        diarization_engine == PORTABLE_DIARIZATION_ENGINE
    )
    has_saved_analysis = bool(stored and stored.get("has_summary"))
    analysis_prompt_version = (
        str(stored.get("summary_prompt_version") or "") if stored else ""
    )
    has_analysis = bool(
        has_transcript
        and has_saved_analysis
        and analysis_prompt_version == PROMPT_VERSION
    )
    has_stale_analysis = bool(
        has_transcript and has_saved_analysis and not has_analysis
    )
    incident_count = int(stored.get("incident_count") or 0) if stored else 0
    segment_count = int(stored.get("segment_count") or 0) if stored else 0

    if has_stale_combined:
        next_step = "Refresh archive day"
        action = "resume_download"
        status = "New audio pending combine"
        status_detail = (
            f"{len(raw_files)} retained source segment"
            f"{'s' if len(raw_files) != 1 else ''} do not match the older "
            "combined timeline; the older recording is preserved"
        )
    elif not has_combined:
        next_step = "Resume archive download"
        action = "resume_download"
        status = "Needs download"
        status_detail = (
            f"{len(raw_files)} local segment{'s' if len(raw_files) != 1 else ''}; "
            "completeness has not been verified"
            if raw_files
            else "No usable combined audio was found"
        )
    elif not has_transcript:
        next_step = "Transcribe locally"
        action = "continue_local"
        status = "Audio ready"
        status_detail = "Combined audio is ready for local transcription"
    elif not has_diarization:
        next_step = "Add speaker labels"
        action = "continue_local"
        status = "Transcript ready"
        status_detail = (
            "Transcript exists; speaker labels can run without repeating transcription"
        )
    elif has_stale_analysis:
        next_step = "Re-run evidence analysis"
        action = "continue_local"
        status = "Analysis update available"
        status_detail = (
            "Saved results predate the current evidence rules; retained audio, "
            "transcript, and speaker labels will be reused"
        )
    elif not has_analysis:
        next_step = "Extract and summarize incidents"
        action = "continue_local"
        status = "Speakers ready"
        status_detail = "Diarized transcript is ready for local analysis"
    else:
        next_step = "Open review"
        action = "open_review"
        status = "Ready to review"
        status_detail = f"{incident_count} extracted incident{'s' if incident_count != 1 else ''}"

    completed_stages = sum(
        [bool(raw_files or has_combined), has_combined, has_transcript, has_diarization, has_analysis]
    )
    stage_parts = [
        f"Audio {len(raw_files)} segment{'s' if len(raw_files) != 1 else ''}"
        if raw_files
        else "Audio retained" if has_combined else "Audio missing",
        "Combined" if has_combined else "Not combined",
        f"Transcript {segment_count:,} segments" if has_transcript and segment_count else "Transcribed" if has_transcript else "Not transcribed",
        "Diarized" if has_diarization else "Not diarized",
        f"Analyzed {incident_count} incidents"
        if has_analysis
        else "Analysis update required"
        if has_stale_analysis
        else "Not analyzed",
    ]
    resolved_feed_name = feed_name or _manifest_feed_name(manifest)
    storage_bytes, working_storage_bytes = directory_storage_usage(day_directory)
    return {
        "feed_id": feed_id,
        "feed_name": resolved_feed_name or f"Feed {feed_id}",
        "archive_date": archive_date.isoformat(),
        "day_directory": str(day_directory.resolve()),
        "raw_file_count": len(raw_files),
        "combined_path": str(combined.resolve()) if has_combined else "",
        "transcript_path": str(transcript.resolve()) if has_transcript else "",
        "manifest_path": str(manifest.resolve()) if manifest.is_file() else "",
        "has_combined": has_combined,
        "has_stale_combined": has_stale_combined,
        "has_transcript": has_transcript,
        "has_diarization": has_diarization,
        "diarization_engine": diarization_engine,
        "diarization_quality": (
            "preview"
            if diarization_engine == PORTABLE_DIARIZATION_ENGINE
            else COMMUNITY_DIARIZATION_QUALITY
            if diarization_engine == COMMUNITY_DIARIZATION_ENGINE
            else ""
        ),
        "speaker_upgrade_available": speaker_upgrade_available,
        "has_analysis": has_analysis,
        "has_stale_analysis": has_stale_analysis,
        "analysis_prompt_version": analysis_prompt_version,
        "expected_analysis_prompt_version": PROMPT_VERSION,
        "incident_count": incident_count,
        "segment_count": segment_count,
        "storage_bytes": storage_bytes,
        "working_storage_bytes": working_storage_bytes,
        "pipeline_percent": completed_stages * 20,
        "pipeline_summary": "  ·  ".join(stage_parts),
        "status": status,
        "status_detail": status_detail,
        "next_step": next_step,
        "primary_action": action,
        "can_open_review": has_analysis,
        "is_complete": has_diarization and has_analysis,
        "needs_network": not has_combined,
    }


def scan_local_library(
    output_dir: str | Path = "archives",
    database_path: str | Path | None = None,
) -> list[dict[str, Any]]:
    output_root = Path(output_dir)
    database = Path(database_path) if database_path else output_root / "broadcastify-analysis.sqlite3"
    with AnalysisStore(database) as store:
        stored_days = {
            (str(value["feed_id"]), date.fromisoformat(str(value["archive_date"]))): value
            for value in store.list_days()
        }
        feed_names = _friendly_feed_names(store)

    keys = set(stored_days)
    if output_root.is_dir():
        for feed_directory in output_root.iterdir():
            if not feed_directory.is_dir() or not feed_directory.name.isdigit():
                continue
            for day_directory in feed_directory.iterdir():
                if not day_directory.is_dir() or not DAY_DIRECTORY_PATTERN.match(day_directory.name):
                    continue
                try:
                    archive_date = date.fromisoformat(
                        f"{day_directory.name[:4]}-{day_directory.name[4:6]}-{day_directory.name[6:]}"
                    )
                except ValueError:
                    continue
                keys.add((feed_directory.name, archive_date))

    results = []
    for feed_id, archive_date in keys:
        day_directory = (
            output_root / feed_id / archive_date.strftime("%Y%m%d")
        )
        cleanup_orphaned_audio_work_files(day_directory)
        results.append(
            _state_for_day(
                output_root,
                feed_id,
                archive_date,
                stored_days.get((feed_id, archive_date)),
                feed_names.get(feed_id, ""),
            )
        )
    return sorted(results, key=lambda value: (value["archive_date"], value["feed_id"]), reverse=True)


@dataclass(frozen=True)
class LocalProcessingRequest:
    feed_id: str
    archive_date: date
    output_dir: Path = Path("archives")
    model: str = "turbo"
    asr_engine: str = "auto"
    device: str = "auto"
    device_index: int = 0
    compute_type: str = "auto"
    asr_model_path: str | None = None
    diarization_engine: str = COMMUNITY_DIARIZATION_ENGINE
    diarization_device: str = "auto"
    batch_size: int = 8
    diarize: bool = True
    min_speakers: int | None = None
    max_speakers: int | None = None
    huggingface_token: str | None = None

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "LocalProcessingRequest":
        return cls(
            feed_id=str(value["feed_id"]).strip(),
            archive_date=date.fromisoformat(str(value["archive_date"])),
            output_dir=Path(value.get("output_dir") or "archives"),
            model=str(value.get("model") or "turbo"),
            asr_engine=str(value.get("asr_engine") or "auto"),
            device=str(value.get("device") or "auto"),
            device_index=max(0, int(value.get("device_index", 0))),
            compute_type=str(value.get("compute_type") or "auto"),
            asr_model_path=(
                str(value["asr_model_path"]).strip()
                if value.get("asr_model_path")
                else None
            ),
            diarization_engine=normalize_diarization_engine(
                str(
                    value.get("diarization_engine")
                    or COMMUNITY_DIARIZATION_ENGINE
                )
            ),
            diarization_device=str(value.get("diarization_device") or "auto"),
            batch_size=max(1, int(value.get("batch_size", 8))),
            diarize=bool(value.get("diarize", True)),
            min_speakers=(
                int(value["min_speakers"])
                if value.get("min_speakers") not in {None, ""}
                else None
            ),
            max_speakers=(
                int(value["max_speakers"])
                if value.get("max_speakers") not in {None, ""}
                else None
            ),
            huggingface_token=(
                str(value["huggingface_token"]).strip()
                if value.get("huggingface_token")
                else None
            ),
        )


def prepare_local_day(
    request: LocalProcessingRequest,
    progress: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    day_directory = request.output_dir / request.feed_id / request.archive_date.strftime("%Y%m%d")
    stem = f"combined_{request.feed_id}_{request.archive_date:%Y%m%d}"
    audio = day_directory / f"{stem}.mp3"
    transcript = day_directory / "transcripts" / f"{stem}.json"
    if not audio.is_file():
        raise FileNotFoundError(
            "This day has no combined local audio. Resume its archive download first."
        )

    shared = {
        "model_name": request.model,
        "asr_engine": request.asr_engine,
        "device": request.device,
        "device_index": request.device_index,
        "compute_type": request.compute_type,
        "asr_model_path": request.asr_model_path,
        "diarization_engine": request.diarization_engine,
        "diarization_device": request.diarization_device,
        "diarize": request.diarize,
        "huggingface_token": request.huggingface_token,
        "batch_size": request.batch_size,
        "min_speakers": request.min_speakers,
        "max_speakers": request.max_speakers,
    }
    operation = "reused"
    if not transcript.is_file():
        if progress:
            progress(
                f"Loading local transcription {request.model} "
                f"for {request.archive_date}…"
            )
        transcriber = LocalTranscriber(**shared)
        transcript = transcriber.transcribe_file(audio, progress=progress)
        operation = "transcribed"
    elif request.diarize and not transcript_satisfies_diarization(
        transcript, request.diarization_engine
    ):
        previous_engine = transcript_diarization_engine(transcript)
        if progress:
            progress(
                "Loading the selected speaker model without reloading transcription…"
            )
        transcriber = LocalTranscriber(**shared, load_asr=False)
        transcript = transcriber.diarize_existing_transcript(
            audio, transcript, progress=progress
        )
        operation = (
            "upgraded_diarization"
            if previous_engine == PORTABLE_DIARIZATION_ENGINE
            and request.diarization_engine == COMMUNITY_DIARIZATION_ENGINE
            else "diarized"
        )
    elif progress:
        progress("The existing local transcript already satisfies the selected stages.")

    return {
        "feed_id": request.feed_id,
        "archive_date": request.archive_date.isoformat(),
        "audio_path": str(audio.resolve()),
        "transcript_path": str(transcript.resolve()),
        "diarization_engine": transcript_diarization_engine(transcript),
        "operation": operation,
    }
