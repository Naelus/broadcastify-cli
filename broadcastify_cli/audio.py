from __future__ import annotations

import shutil
import subprocess
import tempfile
import os
import json
import re
import math
from datetime import date, datetime
from pathlib import Path
from typing import Any, Iterable, Mapping


class AudioCombineError(RuntimeError):
    pass


class AudioClipError(RuntimeError):
    pass


_EVIDENCE_WORD = re.compile(r"[a-z0-9]+")
_EVIDENCE_STOP_WORDS = {
    "about",
    "after",
    "again",
    "been",
    "before",
    "being",
    "from",
    "have",
    "into",
    "near",
    "report",
    "reported",
    "that",
    "their",
    "there",
    "they",
    "this",
    "units",
    "were",
    "with",
}


_DLL_DIRECTORY_HANDLES: list[object] = []
MANIFEST_TIMELINE_VERSION = 2


def _winget_ffmpeg_bin_directories() -> list[Path]:
    local_app_data = os.getenv("LOCALAPPDATA")
    if not local_app_data:
        return []
    package_root = Path(local_app_data) / "Microsoft" / "WinGet" / "Packages"
    patterns = (
        "Gyan.FFmpeg.Shared_*/*/bin",
        "Gyan.FFmpeg_*/*/bin",
    )
    directories: list[Path] = []
    for pattern in patterns:
        try:
            candidates = package_root.glob(pattern)
            for candidate in candidates:
                try:
                    if candidate.is_dir():
                        directories.append(candidate)
                except OSError:
                    continue
        except OSError:
            # Sandboxed workers and locked-down service accounts may not be
            # allowed to enumerate another process's WinGet package cache.
            continue
    return directories


def configure_ffmpeg_runtime() -> None:
    """Expose FFmpeg shared DLLs to TorchCodec in an existing process."""
    if os.name != "nt":
        return
    for bin_directory in _winget_ffmpeg_bin_directories():
        if not any(bin_directory.glob("avcodec-*.dll")):
            continue
        value = str(bin_directory)
        path_parts = os.environ.get("PATH", "").split(os.pathsep)
        if value not in path_parts:
            os.environ["PATH"] = value + os.pathsep + os.environ.get("PATH", "")
        if hasattr(os, "add_dll_directory"):
            _DLL_DIRECTORY_HANDLES.append(os.add_dll_directory(value))
        break


def find_ffmpeg() -> str | None:
    configured = os.getenv("FFMPEG_PATH")
    if configured and Path(configured).is_file():
        return configured

    discovered = shutil.which("ffmpeg")
    if discovered:
        return discovered

    # WinGet updates the user's PATH, but an already-running desktop app does
    # not see that change until restart. Find Gyan.FFmpeg in its stable WinGet
    # package root so combining works immediately after setup.
    for bin_directory in _winget_ffmpeg_bin_directories():
        candidate = bin_directory / "ffmpeg.exe"
        if candidate.is_file():
            return str(candidate)
    return None


def _evidence_keywords(value: object) -> set[str]:
    return {
        word
        for word in _EVIDENCE_WORD.findall(str(value or "").lower())
        if len(word) >= 3 and word not in _EVIDENCE_STOP_WORDS
    }


def select_incident_evidence_window(
    incident: Mapping[str, Any],
    *,
    context_before_seconds: float = 8.0,
    context_after_seconds: float = 12.0,
    maximum_duration_seconds: float = 120.0,
) -> tuple[float, float]:
    """Choose a compact window around the citation that best explains an incident.

    Incident extraction can cite follow-up traffic many minutes after the first
    dispatch. Using the full minimum-to-maximum citation span either creates an
    oversized clip or leaves the useful citation outside it. Prefer the cited
    transcript segment with the strongest title/summary/location overlap, then
    add a small amount of radio context around that exact segment.
    """

    evidence: list[tuple[float, float, str]] = []
    for raw in incident.get("evidence", []):
        if not isinstance(raw, Mapping):
            continue
        try:
            start = max(0.0, float(raw.get("start_seconds", 0.0)))
            end = max(start + 0.001, float(raw.get("end_seconds", start)))
        except (TypeError, ValueError):
            continue
        if not math.isfinite(start) or not math.isfinite(end):
            continue
        evidence.append((start, end, str(raw.get("text") or "").strip()))

    if evidence:
        title_words = _evidence_keywords(incident.get("title"))
        summary_words = _evidence_keywords(incident.get("summary"))
        location_words = _evidence_keywords(incident.get("location"))

        def relevance(segment: tuple[float, float, str]) -> tuple[int, int, int, float]:
            words = _evidence_keywords(segment[2])
            weighted_overlap = (
                3 * len(words & title_words)
                + len(words & summary_words)
                + 4 * len(words & location_words)
            )
            all_overlap = len(words & (title_words | summary_words | location_words))
            # Prefer the earlier citation only when its descriptive value ties.
            return weighted_overlap, all_overlap, len(words), -segment[0]

        anchor_start, anchor_end, _ = max(evidence, key=relevance)
    else:
        try:
            anchor_start = max(0.0, float(incident.get("start_seconds", 0.0)))
            anchor_end = max(
                anchor_start + 1.0,
                float(incident.get("end_seconds", anchor_start + 1.0)),
            )
        except (TypeError, ValueError):
            anchor_start, anchor_end = 0.0, 1.0

    start = max(0.0, anchor_start - max(0.0, context_before_seconds))
    end = max(start + 1.0, anchor_end + max(0.0, context_after_seconds))
    return start, min(end, start + max(1.0, maximum_duration_seconds))


def select_incident_context_window(
    incident: Mapping[str, Any],
    *,
    preceding_seconds: float = 300.0,
    following_seconds: float = 60.0,
    maximum_duration_seconds: float = 480.0,
) -> tuple[float, float]:
    """Return optional surrounding radio traffic without calling it cited evidence.

    The compact evidence clip remains the auditable citation. This wider window
    is useful when an extraction cites a recovery or disposition several minutes
    after the initial dispatch, as often happens in scanner traffic.
    """

    evidence_start, evidence_end = select_incident_evidence_window(incident)
    start = max(0.0, evidence_start - max(0.0, preceding_seconds))
    end = max(evidence_end, evidence_end + max(0.0, following_seconds))
    if end - start > max(1.0, maximum_duration_seconds):
        start = max(0.0, end - max(1.0, maximum_duration_seconds))
    return start, end


def list_source_mp3s(directory: str | Path) -> list[Path]:
    root = Path(directory)
    return sorted(
        path
        for path in root.glob("*.mp3")
        if not path.name.lower().startswith("combined_")
    )


def archive_start_from_filename(path: str | Path) -> datetime | None:
    match = re.match(r"^(\d{12})-", Path(path).name)
    if not match:
        return None
    try:
        return datetime.strptime(match.group(1), "%Y%m%d%H%M")
    except ValueError:
        return None


def combined_output_is_current(
    output: Path, manifest_path: Path, source_files: list[Path]
) -> bool:
    if not output.is_file() or output.stat().st_size == 0 or not manifest_path.is_file():
        return False
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        recorded = [str(value["source_file"]) for value in manifest["sources"]]
    except (OSError, KeyError, TypeError, json.JSONDecodeError):
        return False
    if recorded != [value.name for value in source_files]:
        return False
    output_mtime = min(output.stat().st_mtime, manifest_path.stat().st_mtime)
    return all(value.stat().st_mtime <= output_mtime for value in source_files)


def _probe_audio_duration(source: Path, ffmpeg: str) -> float:
    ffmpeg_path = Path(ffmpeg)
    sibling_name = "ffprobe.exe" if ffmpeg_path.suffix.lower() == ".exe" else "ffprobe"
    sibling = ffmpeg_path.with_name(sibling_name)
    ffprobe = str(sibling) if sibling.is_file() else shutil.which("ffprobe")
    if not ffprobe:
        raise AudioCombineError(
            f"FFprobe was not found; the timeline duration of {source.name} cannot be measured."
        )
    process = subprocess.run(
        [
            ffprobe,
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(source),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    try:
        duration = float(process.stdout.strip())
    except (AttributeError, TypeError, ValueError) as exc:
        raise AudioCombineError(
            process.stderr.strip() or f"FFprobe could not measure {source.name}."
        ) from exc
    if process.returncode != 0 or not math.isfinite(duration) or duration <= 0:
        raise AudioCombineError(
            process.stderr.strip() or f"FFprobe returned an invalid duration for {source.name}."
        )
    return duration


def _source_durations(
    files: list[Path], source_starts: list[datetime | None], ffmpeg: str
) -> tuple[list[float | None], list[float]]:
    trim_durations: list[float | None] = []
    timeline_durations: list[float] = []
    for index, (source, source_start) in enumerate(zip(files, source_starts)):
        next_start = source_starts[index + 1] if index + 1 < len(files) else None
        interval = (
            (next_start - source_start).total_seconds()
            if source_start is not None and next_start is not None
            else None
        )
        # Normal archives are approximately 30 minutes. Trim small encoder
        # overlaps, but do not stretch a file across a genuine feed outage.
        trim_duration = interval if interval is not None and 0 < interval <= 2_700 else None
        trim_durations.append(trim_duration)
        timeline_durations.append(
            trim_duration
            if trim_duration is not None
            else _probe_audio_duration(source, ffmpeg)
        )
    return trim_durations, timeline_durations


def _write_manifest(
    manifest_path: Path,
    output: Path,
    feed_id: str,
    archive_date: date,
    files: list[Path],
    source_starts: list[datetime | None],
    trim_durations: list[float | None],
    timeline_durations: list[float],
    feed_name: str = "",
) -> None:
    combined_offset = 0.0
    manifest_sources: list[dict[str, object]] = []
    for source, source_start, trim_duration, timeline_duration in zip(
        files, source_starts, trim_durations, timeline_durations
    ):
        manifest_sources.append(
            {
                "source_file": source.name,
                "archive_start": source_start.isoformat() if source_start else None,
                "combined_start_seconds": combined_offset,
                "trimmed_duration_seconds": timeline_duration,
                "duration_source": (
                    "archive_interval" if trim_duration is not None else "media_probe"
                ),
            }
        )
        combined_offset += timeline_duration
    payload: dict[str, object] = {
        "timeline_version": MANIFEST_TIMELINE_VERSION,
        "feed_id": feed_id,
        "archive_date": archive_date.isoformat(),
        "combined_file": output.name,
        "sources": manifest_sources,
    }
    if normalized_name := str(feed_name or "").strip():
        payload["feed_name"] = normalized_name
    _write_manifest_payload(manifest_path, payload)


def _write_manifest_payload(manifest_path: Path, payload: dict[str, object]) -> None:
    partial = manifest_path.with_suffix(manifest_path.suffix + ".tmp")
    partial.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(partial, manifest_path)


def combine_mp3_files(
    directory: str | Path,
    feed_id: str,
    archive_date: date,
    source_files: Iterable[str | Path] | None = None,
    delete_sources: bool = False,
    feed_name: str = "",
) -> Path | None:
    root = Path(directory)
    files = sorted(Path(path) for path in (source_files or list_source_mp3s(root)))
    files = [path for path in files if path.exists() and path.suffix.lower() == ".mp3"]
    if not files:
        return None

    output = root / f"combined_{feed_id}_{archive_date.strftime('%Y%m%d')}.mp3"
    manifest_path = output.with_suffix(".manifest.json")
    source_starts = [archive_start_from_filename(path) for path in files]
    current = combined_output_is_current(output, manifest_path, files)
    if current:
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if not isinstance(manifest, dict):
                raise ValueError("Combined manifest must be a JSON object.")
            if int(manifest.get("timeline_version", 0)) >= MANIFEST_TIMELINE_VERSION:
                normalized_name = str(feed_name or "").strip()
                if normalized_name and str(manifest.get("feed_name") or "") != normalized_name:
                    manifest["feed_name"] = normalized_name
                    _write_manifest_payload(manifest_path, manifest)
                return output
        except (OSError, TypeError, ValueError, json.JSONDecodeError):
            pass

    ffmpeg = find_ffmpeg()
    if not ffmpeg:
        raise AudioCombineError("FFmpeg was not found on PATH.")
    trim_durations, timeline_durations = _source_durations(
        files, source_starts, ffmpeg
    )
    if current:
        # Older manifests omitted the real duration of the last block and of
        # blocks followed by a feed outage. Repair only the metadata; the
        # continuously re-encoded MP3 itself is already correct.
        _write_manifest(
            manifest_path,
            output,
            feed_id,
            archive_date,
            files,
            source_starts,
            trim_durations,
            timeline_durations,
            feed_name,
        )
        return output

    list_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".ffconcat", delete=False, encoding="utf-8", newline="\n"
        ) as concat_file:
            concat_file.write("ffconcat version 1.0\n")
            for source, expected_duration in zip(files, trim_durations):
                normalized = str(source.resolve()).replace("\\", "/")
                escaped = normalized.replace("'", "'\\''")
                concat_file.write(f"file '{escaped}'\n")
                if expected_duration is not None:
                    concat_file.write(f"outpoint {expected_duration:.3f}\n")
                    concat_file.write(f"duration {expected_duration:.3f}\n")
            list_path = Path(concat_file.name)

        process = subprocess.run(
            [
                ffmpeg,
                "-hide_banner",
                "-loglevel",
                "error",
                "-f",
                "concat",
                "-safe",
                "0",
                "-i",
                str(list_path),
                # Archive MP3s each start their own timestamp clock. A stream
                # copy can therefore look correct to ffprobe while decoders
                # stop at the first 30-minute boundary. Decode and re-encode
                # once to create a genuinely continuous ASR/diarization input.
                "-map",
                "0:a:0",
                "-af",
                "asetpts=N/SR/TB",
                "-ar",
                "16000",
                "-ac",
                "1",
                "-c:a",
                "libmp3lame",
                "-b:a",
                "16k",
                "-y",
                str(output),
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        if process.returncode != 0:
            raise AudioCombineError(process.stderr.strip() or "FFmpeg failed to combine audio.")
        if not output.exists() or output.stat().st_size == 0:
            raise AudioCombineError("FFmpeg did not produce a valid combined MP3.")

        _write_manifest(
            manifest_path,
            output,
            feed_id,
            archive_date,
            files,
            source_starts,
            trim_durations,
            timeline_durations,
            feed_name,
        )

        if delete_sources:
            for source in files:
                if source.resolve() != output.resolve():
                    source.unlink()
        return output
    finally:
        if list_path and list_path.exists():
            list_path.unlink()


def extract_audio_clip(
    source_file: str | Path,
    output_file: str | Path,
    start_seconds: float,
    end_seconds: float,
) -> Path:
    """Create a small, seekable MP3 evidence clip from retained combined audio."""
    source = Path(source_file)
    output = Path(output_file)
    if not source.is_file():
        raise AudioClipError(f"Source audio does not exist: {source}")
    start = max(0.0, float(start_seconds))
    end = max(start + 1.0, float(end_seconds))
    duration = min(120.0, end - start)
    if output.is_file() and output.stat().st_size > 0:
        if output.stat().st_mtime_ns >= source.stat().st_mtime_ns:
            return output

    ffmpeg = find_ffmpeg()
    if not ffmpeg:
        raise AudioClipError("FFmpeg was not found; the transcript quote is still available.")
    output.parent.mkdir(parents=True, exist_ok=True)
    partial = output.with_suffix(f".part{output.suffix}")
    try:
        process = subprocess.run(
            [
                ffmpeg,
                "-hide_banner",
                "-loglevel",
                "error",
                "-ss",
                f"{start:.3f}",
                "-i",
                str(source),
                "-t",
                f"{duration:.3f}",
                "-map",
                "0:a:0",
                "-ar",
                "16000",
                "-ac",
                "1",
                "-c:a",
                "libmp3lame",
                "-b:a",
                "32k",
                "-y",
                str(partial),
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        if process.returncode != 0 or not partial.is_file() or partial.stat().st_size == 0:
            raise AudioClipError(process.stderr.strip() or "FFmpeg did not create the evidence clip.")
        os.replace(partial, output)
        return output
    finally:
        if partial.exists():
            partial.unlink()
