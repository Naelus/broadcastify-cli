import json
from datetime import date
from pathlib import Path
from types import SimpleNamespace

from broadcastify_cli.audio import (
    _winget_ffmpeg_bin_directories,
    combine_mp3_files,
    extract_audio_clip,
    select_incident_evidence_window,
)


def test_winget_ffmpeg_lookup_tolerates_inaccessible_package_cache(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path))

    def denied(_self: Path, _pattern: str):
        raise PermissionError("package cache is not readable")

    monkeypatch.setattr(Path, "glob", denied)

    assert _winget_ffmpeg_bin_directories() == []


def test_combiner_reencodes_with_continuous_timestamps(monkeypatch, tmp_path: Path) -> None:
    sources = [
        tmp_path / "202607120000-1-90001.mp3",
        tmp_path / "202607120030-2-90001.mp3",
    ]
    for source in sources:
        source.write_bytes(b"audio")

    captured: list[str] = []
    concat_contents = ""

    def fake_run(arguments: list[str], **_kwargs: object) -> SimpleNamespace:
        nonlocal concat_contents
        captured.extend(arguments)
        concat_contents = Path(arguments[arguments.index("-i") + 1]).read_text(
            encoding="utf-8"
        )
        Path(arguments[-1]).write_bytes(b"combined")
        return SimpleNamespace(returncode=0, stderr="")

    monkeypatch.setattr("broadcastify_cli.audio.find_ffmpeg", lambda: "ffmpeg")
    monkeypatch.setattr(
        "broadcastify_cli.audio._probe_audio_duration", lambda _source, _ffmpeg: 1_800.0
    )
    monkeypatch.setattr("broadcastify_cli.audio.subprocess.run", fake_run)

    output = combine_mp3_files(
        tmp_path,
        "90001",
        date(2026, 7, 12),
        source_files=sources,
    )

    assert output == tmp_path / "combined_90001_20260712.mp3"
    assert "asetpts=N/SR/TB" in captured
    assert "libmp3lame" in captured
    assert "16000" in captured
    assert "copy" not in captured
    assert "outpoint 1800.000" in concat_contents

    manifest = output.with_suffix(".manifest.json")
    assert manifest.exists()
    assert '"combined_start_seconds": 1800.0' in manifest.read_text(encoding="utf-8")

    second = combine_mp3_files(
        tmp_path,
        "90001",
        date(2026, 7, 12),
        source_files=sources,
        feed_name="Example City Public Safety",
    )
    assert second == output
    assert captured.count("ffmpeg") == 1
    assert json.loads(manifest.read_text(encoding="utf-8"))["feed_name"] == (
        "Example City Public Safety"
    )


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


def test_evidence_clip_is_timestamped_and_cached(monkeypatch, tmp_path: Path) -> None:
    source = tmp_path / "combined.mp3"
    output = tmp_path / "evidence-clips" / "incident.mp3"
    source.write_bytes(b"source audio")
    calls: list[list[str]] = []

    def fake_run(arguments: list[str], **_kwargs: object) -> SimpleNamespace:
        calls.append(arguments)
        Path(arguments[-1]).write_bytes(b"clip")
        return SimpleNamespace(returncode=0, stderr="")

    monkeypatch.setattr("broadcastify_cli.audio.find_ffmpeg", lambda: "ffmpeg")
    monkeypatch.setattr("broadcastify_cli.audio.subprocess.run", fake_run)

    first = extract_audio_clip(source, output, 92.0, 128.0)
    second = extract_audio_clip(source, output, 92.0, 128.0)

    assert first == output
    assert second == output
    assert len(calls) == 1
    assert calls[0][calls[0].index("-ss") + 1] == "92.000"
    assert calls[0][calls[0].index("-t") + 1] == "36.000"
    assert "libmp3lame" in calls[0]


def test_incident_window_prefers_the_most_descriptive_cited_segment() -> None:
    start, end = select_incident_evidence_window(
        {
            "title": "Intrusion alarm at CMS building",
            "summary": "An alarm was dispatched at the Example State CMS building.",
            "location": "100 Example Road",
            "start_seconds": 4_993.14,
            "end_seconds": 6_198.2,
            "evidence": [
                {
                    "start_seconds": 4_993.14,
                    "end_seconds": 4_996.09,
                    "text": "All right, intrusion alarm, 6510, left 150.",
                },
                {
                    "start_seconds": 6_192.6,
                    "end_seconds": 6_198.2,
                    "text": "100 Example Road, intrusion alarm, state of Example State CMS building.",
                },
            ],
        }
    )

    assert start == 6_184.6
    assert end == 6_210.2


def test_incident_window_falls_back_to_a_bounded_incident_start() -> None:
    assert select_incident_evidence_window(
        {"start_seconds": 500.0, "end_seconds": 900.0, "evidence": []}
    ) == (492.0, 612.0)
