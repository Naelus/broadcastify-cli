import json
import math
import os
import struct
import subprocess
import sys
import wave
from array import array
from datetime import date
from pathlib import Path
from types import SimpleNamespace

import pytest

from broadcastify_cli.audio import (
    AudioCombineError,
    combine_mp3_files,
    extract_audio_clip,
    find_ffmpeg,
    select_incident_context_window,
    select_incident_evidence_window,
)


def test_combined_recording_and_evidence_clip_decode_the_right_audio(tmp_path: Path) -> None:
    ffmpeg = find_ffmpeg()
    assert ffmpeg, "The media verification gate requires FFmpeg and FFprobe."
    sources = [
        tmp_path / "202607120000-1-90001.mp3",
        tmp_path / "202607120200-2-90001.mp3",
    ]
    # Distinct signals expose truncation, reversed blocks, and incorrect seeking.
    # A real feed outage must not insert hours of silence into combined audio.
    for source, frequency in zip(sources, (440, 880)):
        pcm = source.with_suffix(".wav")
        with wave.open(str(pcm), "wb") as writer:
            writer.setparams((1, 2, 16_000, 0, "NONE", "not compressed"))
            writer.writeframes(
                b"".join(
                    struct.pack(
                        "<h", int(12_000 * math.sin(2 * math.pi * frequency * i / 16_000))
                    )
                    for i in range(32_000)
                )
            )
        subprocess.run(
            [ffmpeg, "-v", "error", "-i", str(pcm), "-c:a", "libmp3lame", "-y", str(source)],
            check=True,
            capture_output=True,
            timeout=15,
        )
    retained_bytes = [source.read_bytes() for source in sources]

    def decode(path: Path) -> array:
        result = subprocess.run(
            [
                ffmpeg, "-v", "error", "-i", str(path),
                "-f", "s16le", "-ar", "8000", "-ac", "1", "pipe:1",
            ],
            check=True,
            capture_output=True,
            timeout=15,
        )
        samples = array("h", result.stdout)
        if sys.byteorder != "little":
            samples.byteswap()
        return samples

    def signal_power(samples: array, frequency: int) -> float:
        real = sum(
            value * math.cos(2 * math.pi * frequency * i / 8000)
            for i, value in enumerate(samples)
        )
        imaginary = sum(
            value * math.sin(2 * math.pi * frequency * i / 8000)
            for i, value in enumerate(samples)
        )
        return real * real + imaginary * imaginary

    output = combine_mp3_files(
        tmp_path,
        "90001",
        date(2026, 7, 12),
        source_files=sources,
    )

    assert output == tmp_path / "combined_90001_20260712.mp3"
    samples = decode(output)
    assert len(samples) / 8000 == pytest.approx(4.0, abs=0.2)
    first = samples[4000:8000]
    last = samples[-8000:-4000]
    assert signal_power(first, 440) > 10 * signal_power(first, 880)
    assert signal_power(last, 880) > 10 * signal_power(last, 440)
    assert [source.read_bytes() for source in sources] == retained_bytes
    assert not list(tmp_path.glob("*.part.mp3"))

    manifest = output.with_suffix(".manifest.json")
    assert manifest.exists()
    timeline = json.loads(manifest.read_text(encoding="utf-8"))["sources"]
    assert timeline[1]["combined_start_seconds"] == pytest.approx(2.0, abs=0.1)
    assert timeline[1]["archive_start"].startswith("2026-07-12T02:00:00")
    published = (output.stat().st_mtime_ns, output.read_bytes())

    second = combine_mp3_files(
        tmp_path,
        "90001",
        date(2026, 7, 12),
        source_files=sources,
        feed_name="Example City Public Safety",
    )
    assert second == output
    assert (output.stat().st_mtime_ns, output.read_bytes()) == published
    assert json.loads(manifest.read_text(encoding="utf-8"))["feed_name"] == (
        "Example City Public Safety"
    )

    clip = tmp_path / "evidence-clips" / "incident.mp3"
    assert extract_audio_clip(output, clip, 2.6, 3.6) == clip
    clip_samples = decode(clip)
    assert len(clip_samples) / 8000 == pytest.approx(1.0, abs=0.08)
    assert signal_power(clip_samples, 880) > 10 * signal_power(clip_samples, 440)
    cached = (clip.stat().st_mtime_ns, clip.read_bytes())
    assert extract_audio_clip(output, clip, 2.6, 3.6) == clip
    assert (clip.stat().st_mtime_ns, clip.read_bytes()) == cached
    assert not list(tmp_path.rglob("*.part.mp3"))


def test_combiner_retries_atomic_publish_after_player_releases(
    monkeypatch, tmp_path: Path
) -> None:
    source = tmp_path / "202607120000-1-90001.mp3"
    source.write_bytes(b"audio")
    output = tmp_path / "combined_90001_20260712.mp3"
    output.write_bytes(b"previous combined recording")

    def fake_run(arguments: list[str], **_kwargs: object) -> SimpleNamespace:
        Path(arguments[-1]).write_bytes(b"refreshed combined recording")
        return SimpleNamespace(returncode=0, stderr="")

    original_replace = os.replace
    attempts = 0

    def briefly_locked_replace(source_path: str | Path, destination: str | Path) -> None:
        nonlocal attempts
        if Path(destination) == output:
            attempts += 1
            if attempts < 3:
                raise PermissionError("player is releasing the file")
        original_replace(source_path, destination)

    monkeypatch.setattr("broadcastify_cli.audio.find_ffmpeg", lambda: "ffmpeg")
    monkeypatch.setattr(
        "broadcastify_cli.audio._probe_audio_duration", lambda *_args: 1_800.0
    )
    monkeypatch.setattr("broadcastify_cli.audio.subprocess.run", fake_run)
    monkeypatch.setattr("broadcastify_cli.audio.os.replace", briefly_locked_replace)
    monkeypatch.setattr("broadcastify_cli.audio.time.sleep", lambda _seconds: None)

    result = combine_mp3_files(
        tmp_path, "90001", date(2026, 7, 12), source_files=[source]
    )

    assert result == output
    assert attempts == 3
    assert output.read_bytes() == b"refreshed combined recording"
    assert not list(tmp_path.glob("*.part.mp3"))


def test_combiner_preserves_previous_recording_when_player_stays_open(
    monkeypatch, tmp_path: Path
) -> None:
    source = tmp_path / "202607120000-1-90001.mp3"
    source.write_bytes(b"audio")
    output = tmp_path / "combined_90001_20260712.mp3"
    output.write_bytes(b"previous combined recording")

    def fake_run(arguments: list[str], **_kwargs: object) -> SimpleNamespace:
        Path(arguments[-1]).write_bytes(b"unpublished refresh")
        return SimpleNamespace(returncode=0, stderr="")

    def locked_replace(_source: str | Path, destination: str | Path) -> None:
        if Path(destination) == output:
            raise PermissionError("player still owns the file")
        raise AssertionError("No other file should be published in this path.")

    monkeypatch.setattr("broadcastify_cli.audio.find_ffmpeg", lambda: "ffmpeg")
    monkeypatch.setattr(
        "broadcastify_cli.audio._probe_audio_duration", lambda *_args: 1_800.0
    )
    monkeypatch.setattr("broadcastify_cli.audio.subprocess.run", fake_run)
    monkeypatch.setattr("broadcastify_cli.audio.os.replace", locked_replace)
    monkeypatch.setattr("broadcastify_cli.audio.time.sleep", lambda _seconds: None)

    with pytest.raises(AudioCombineError, match="still open in another player"):
        combine_mp3_files(
            tmp_path, "90001", date(2026, 7, 12), source_files=[source]
        )

    assert output.read_bytes() == b"previous combined recording"
    assert not list(tmp_path.glob("*.part.mp3"))


def test_combiner_manifest_counts_media_duration_across_feed_gaps(
    monkeypatch, tmp_path: Path
) -> None:
    sources = [
        tmp_path / "202607120000-1-90001.mp3",
        tmp_path / "202607120030-2-90001.mp3",
        tmp_path / "202607120200-3-90001.mp3",
        tmp_path / "202607120230-4-90001.mp3",
    ]
    for source in sources:
        source.write_bytes(b"audio")
    combine_calls = 0

    def fake_run(arguments: list[str], **_kwargs: object) -> SimpleNamespace:
        nonlocal combine_calls
        combine_calls += 1
        Path(arguments[-1]).write_bytes(b"combined")
        return SimpleNamespace(returncode=0, stderr="")

    probed = {
        sources[1].name: 1_797.5,
        sources[3].name: 1_801.25,
    }
    monkeypatch.setattr("broadcastify_cli.audio.find_ffmpeg", lambda: "ffmpeg")
    monkeypatch.setattr(
        "broadcastify_cli.audio._probe_audio_duration",
        lambda source, _ffmpeg: probed[source.name],
    )
    monkeypatch.setattr("broadcastify_cli.audio.subprocess.run", fake_run)

    output = combine_mp3_files(
        tmp_path, "90001", date(2026, 7, 12), source_files=sources
    )
    manifest_path = output.with_suffix(".manifest.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert manifest["timeline_version"] == 2
    assert [value["combined_start_seconds"] for value in manifest["sources"]] == [
        0.0,
        1_800.0,
        3_597.5,
        5_397.5,
    ]
    assert manifest["sources"][1]["duration_source"] == "media_probe"

    # A legacy manifest can be repaired without re-encoding the correct audio.
    manifest.pop("timeline_version")
    manifest["sources"][2]["combined_start_seconds"] = 1_800.0
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    repaired = combine_mp3_files(
        tmp_path, "90001", date(2026, 7, 12), source_files=sources
    )
    repaired_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert repaired == output
    assert repaired_manifest["sources"][2]["combined_start_seconds"] == 3_597.5
    assert combine_calls == 1


def test_incident_context_window_reaches_the_initial_dispatch_before_a_disposition() -> None:
    incident = {
        "start_seconds": 64709.45,
        "end_seconds": 64714.65,
        "evidence": [
            {
                "start_seconds": 64709.45,
                "end_seconds": 64714.65,
                "text": "The stolen squad car was located unoccupied.",
            }
        ],
    }

    assert select_incident_evidence_window(incident) == (64701.45, 64726.65)
    assert select_incident_context_window(incident) == (64401.45, 64786.65)
