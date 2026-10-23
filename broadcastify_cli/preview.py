"""Create isolated, synthetic data for the native and browser workflow previews."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import tempfile
import time
from dataclasses import asdict, dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Sequence

from .analysis import DEFAULT_EMBEDDING_MODEL, PROMPT_VERSION
from .storage import AnalysisStore


@dataclass(frozen=True)
class PreviewSeedPaths:
    data_root: Path
    library_root: Path
    database_path: Path
    quota_ledger_path: Path


def _fresh_temporary_root(root: Path) -> Path:
    requested = root.expanduser().absolute()
    temp_root = Path(tempfile.gettempdir()).resolve()
    try:
        parent = requested.parent.resolve(strict=True)
        parent.relative_to(temp_root)
    except (OSError, ValueError) as exc:
        raise ValueError("Preview data must be created in a directory under the system temporary folder.") from exc
    if requested.exists() or requested.is_symlink():
        raise FileExistsError("Preview refuses to use an existing data directory.")
    requested.mkdir()
    resolved = requested.resolve(strict=True)
    try:
        resolved.relative_to(temp_root)
    except ValueError as exc:
        raise ValueError("Preview data must remain inside the system temporary folder.") from exc
    if resolved == temp_root:
        raise ValueError("The system temporary folder cannot be used as preview data.")
    return resolved


def _within(root: Path, relative: str) -> Path:
    candidate = (root / relative).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise ValueError("A preview path escaped its isolated data directory.") from exc
    return candidate


def _resolve_ffmpeg(executable: str | Path | None) -> Path:
    candidates = [Path(executable).expanduser() if executable else None]
    configured = shutil.which("ffmpeg")
    candidates.append(Path(configured) if configured else None)
    for candidate in candidates:
        if candidate is not None and candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError("FFmpeg is required to generate the preview's short playable sample audio.")


def _synthetic_incident(day: date) -> dict[str, object]:
    spoken = f"Synthetic dispatch sample for {day:%B} {day.day}: traffic incident near Main and First."
    return {
        "fingerprint": f"synthetic-preview-traffic-{day.isoformat()}",
        "event_type": "traffic",
        "title": "Preview traffic report",
        "summary": "Synthetic sample for reviewing the interface. This is not a real incident.",
        "location": "Main and First",
        "start_seconds": 0.0,
        "end_seconds": 2.0,
        "priority": 2,
        "confidence": 1.0,
        "evidence": [
            {
                "segment_index": 0,
                "start_seconds": 0.0,
                "end_seconds": 2.0,
                "text": spoken,
            }
        ],
        "attributes": {"preview_data": True},
    }


def seed_preview(
    root: Path,
    *,
    ffmpeg: str | Path | None = None,
    python: str | Path = "",
    feed_id: str = "999991",
    today: date | None = None,
) -> PreviewSeedPaths:
    """Create synthetic local workflow evidence under a new system-temp directory.

    ``root`` must not exist and its parent must be inside ``tempfile.gettempdir()``.
    The function does not access network services, credentials, models, or an
    existing app library. Callers can use the returned paths to launch an
    isolated app process or to configure an isolated web preview.
    """

    if not str(feed_id).isdigit():
        raise ValueError("Preview feed IDs must contain only digits.")
    isolated_root = _fresh_temporary_root(Path(root))
    current = today or date.today()
    data_root = _within(isolated_root, "app-data")
    library_root = _within(isolated_root, "library")
    database_path = _within(library_root, "broadcastify-analysis.sqlite3")
    quota_ledger_path = _within(data_root, "archive-quota.sqlite3")
    data_root.mkdir()
    library_root.mkdir()
    configured_ffmpeg = _resolve_ffmpeg(ffmpeg)
    sample_audio = _within(isolated_root, "preview-silence.mp3")
    subprocess.run(
        [
            str(configured_ffmpeg), "-hide_banner", "-loglevel", "error", "-nostdin",
            "-f", "lavfi", "-i", "anullsrc=r=16000:cl=mono", "-t", "2",
            "-codec:a", "libmp3lame", "-q:a", "9", str(sample_audio),
        ],
        check=True,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )
    audio_bytes = sample_audio.read_bytes()
    sample_audio.unlink()
    if not audio_bytes:
        raise RuntimeError("FFmpeg produced an empty preview sample.")

    secondary_feed_ids = [
        candidate for candidate in ("999992", "999993", "999994", "999995")
        if candidate != str(feed_id)
    ]
    audio_feed_id, no_audio_feed_id = secondary_feed_ids[:2]
    feed_names = {
        str(feed_id): "Preview — ready and index-pending days",
        audio_feed_id: "Preview — audio awaiting transcription",
        no_audio_feed_id: "Preview — no retained source audio",
    }
    first_day = current - timedelta(days=3)
    second_day = current - timedelta(days=2)
    unfinished_day = current - timedelta(days=1)
    with AnalysisStore(database_path) as store:
        store.save_feed_catalog(
            [
                {"feed_id": item_id, "name": name, "status": "Synthetic preview data"}
                for item_id, name in feed_names.items()
            ]
        )
        for day, indexed in ((first_day, True), (second_day, False)):
            transcript_root = _within(
                isolated_root,
                f"library/{feed_id}/{day:%Y%m%d}/transcripts",
            )
            transcript_root.mkdir(parents=True)
            stem = f"combined_{feed_id}_{day:%Y%m%d}"
            audio_path = _within(isolated_root, f"library/{feed_id}/{day:%Y%m%d}/{stem}.mp3")
            audio_path.write_bytes(audio_bytes)
            transcript_path = _within(
                isolated_root,
                f"library/{feed_id}/{day:%Y%m%d}/transcripts/{stem}.json",
            )
            spoken = (
                f"Synthetic dispatch sample for {day:%B} {day.day}: "
                "traffic incident near Main and First."
            )
            transcript_path.write_text(
                json.dumps(
                    {
                        "duration": 2.0,
                        "model": "synthetic-preview",
                        "diarization_completed": True,
                        "segments": [
                            {"start": 0.0, "end": 2.0, "text": spoken, "speaker": "SPEAKER_00"}
                        ],
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            imported = store.import_transcript(feed_id, day, transcript_path, audio_path)
            incident_ids = store.replace_incidents(
                imported.day_id,
                [_synthetic_incident(day)],
                model="synthetic-preview",
                prompt_version=PROMPT_VERSION,
            )
            store.save_daily_summary(
                imported.day_id,
                "Synthetic preview report; not an actual incident.",
                incident_ids,
                "synthetic-preview",
                PROMPT_VERSION,
                imported.transcript_sha256,
            )
            if indexed:
                passages = store.passages_missing_embeddings(
                    DEFAULT_EMBEDDING_MODEL,
                    feed_id=feed_id,
                    archive_dates=[day.isoformat()],
                )
                store.save_embeddings(
                    "passage",
                    [
                        (
                            int(passage["id"]),
                            hashlib.sha256(f"passage: {passage['text']}".encode("utf-8")).hexdigest(),
                            1,
                            b"\0" * 4,
                        )
                        for passage in passages
                    ],
                    DEFAULT_EMBEDDING_MODEL,
                )

        unfinished_feed = audio_feed_id
        unfinished_root = _within(
            isolated_root,
            f"library/{unfinished_feed}/{unfinished_day:%Y%m%d}",
        )
        unfinished_root.mkdir(parents=True)
        unfinished_audio = _within(
            isolated_root,
            f"library/{unfinished_feed}/{unfinished_day:%Y%m%d}/combined_{unfinished_feed}_{unfinished_day:%Y%m%d}.mp3",
        )
        unfinished_audio.write_bytes(audio_bytes)
        store.save_library_catchup(
            {
                "feed_id": str(feed_id),
                "feed_name": feed_names[str(feed_id)],
                "start_date": first_day.isoformat(),
                "through_current": True,
            }
        )

    today_root = _within(isolated_root, f"library/{feed_id}/{current:%Y%m%d}")
    today_root.mkdir(parents=True)
    completion = {
        "schema_version": 1,
        "feed_id": str(feed_id),
        "archive_date": current.isoformat(),
        "archive_ids": [],
        "completed_at_unix": time.time(),
    }
    _within(isolated_root, f"library/{feed_id}/{current:%Y%m%d}/.broadcastify-archive-complete.json").write_text(
        json.dumps(completion),
        encoding="utf-8",
    )
    settings = {
        "Version": 9,
        "PythonRuntimePath": str(Path(python).expanduser().resolve()) if python else "",
        "OutputDirectory": str(library_root),
        "LanSyncEnabled": False,
        "LanDiscoveryEnabled": False,
        "LanShareEnabled": False,
        "LastReviewFeedId": str(feed_id),
        "LastReviewDate": first_day.isoformat(),
    }
    _within(isolated_root, "app-data/settings.json").write_text(
        json.dumps(settings, indent=2),
        encoding="utf-8",
    )
    return PreviewSeedPaths(data_root, library_root, database_path, quota_ledger_path)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Seed an isolated Broadcastify Desktop preview.")
    parser.add_argument("--root", type=Path, required=True, help="New directory under the system temporary folder.")
    parser.add_argument("--ffmpeg", type=Path, default=None, help="FFmpeg executable shipped with the desktop app.")
    parser.add_argument("--python", default="", help="Python path recorded in the isolated preview settings.")
    parser.add_argument("--feed-id", default="999991")
    arguments = parser.parse_args(argv)
    paths = seed_preview(
        arguments.root,
        ffmpeg=arguments.ffmpeg,
        python=arguments.python,
        feed_id=arguments.feed_id,
    )
    print(json.dumps({key: str(value) for key, value in asdict(paths).items()}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
