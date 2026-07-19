import json
from datetime import date
from pathlib import Path

from broadcastify_cli.storage import AnalysisStore


def make_transcript(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "model": "turbo",
                "duration": 600.0,
                "segments": [
                    {
                        "start": 10.0,
                        "end": 15.0,
                        "text": "Dispatch reports shots fired near Main and First.",
                        "speaker": "SPEAKER_00",
                    },
                    {
                        "start": 20.0,
                        "end": 24.0,
                        "text": "Units are responding to the area.",
                        "speaker": "SPEAKER_01",
                    },
                ],
            }
        ),
        encoding="utf-8",
    )


def test_import_is_idempotent_and_searchable(tmp_path: Path) -> None:
    transcript = tmp_path / "transcript.json"
    audio = tmp_path / "audio.mp3"
    make_transcript(transcript)
    audio.write_bytes(b"audio")

    with AnalysisStore(tmp_path / "analysis.sqlite3") as store:
        first = store.import_transcript("90001", date(2026, 7, 12), transcript, audio)
        passage_id = store.get_passages(
            "90001", date(2026, 7, 12), date(2026, 7, 12)
        )[0]["id"]
        second = store.import_transcript("90001", date(2026, 7, 12), transcript, audio)

        assert first.day_id == second.day_id
        assert store.stats()["feed_days"] == 1
        assert store.stats()["transcript_segments"] == 2
        assert store.stats()["passages"] == 1
        assert store.get_passages(
            "90001", date(2026, 7, 12), date(2026, 7, 12)
        )[0]["id"] == passage_id

        results = store.search_passages(
            "90001", date(2026, 7, 12), date(2026, 7, 12), "shots fired"
        )
        assert len(results) == 1
        assert "Main and First" in results[0]["text"]


def test_idempotent_import_refreshes_paths_after_library_move(
    tmp_path: Path,
) -> None:
    archive_date = date(2026, 7, 12)
    legacy = tmp_path / "windows-export"
    old_audio = legacy / "combined_90001_20260712.mp3"
    old_transcript = legacy / "combined_90001_20260712.json"
    old_manifest = legacy / "combined_90001_20260712.manifest.json"
    old_audio.parent.mkdir(parents=True)
    old_audio.write_bytes(b"retained audio")
    make_transcript(old_transcript)
    old_manifest.write_text("{}", encoding="utf-8")

    library = tmp_path / "nas-library"
    database = library / "broadcastify-analysis.sqlite3"
    with AnalysisStore(database) as store:
        first = store.import_transcript(
            "90001",
            archive_date,
            old_transcript,
            old_audio,
            old_manifest,
        )

        day_directory = library / "90001" / "20260712"
        new_audio = day_directory / "combined_90001_20260712.mp3"
        new_transcript = (
            day_directory / "transcripts" / "combined_90001_20260712.json"
        )
        new_manifest = day_directory / "combined_90001_20260712.manifest.json"
        new_audio.parent.mkdir(parents=True)
        new_transcript.parent.mkdir(parents=True)
        new_audio.write_bytes(old_audio.read_bytes())
        new_transcript.write_bytes(old_transcript.read_bytes())
        new_manifest.write_bytes(old_manifest.read_bytes())

        second = store.import_transcript(
            "90001",
            archive_date,
            new_transcript,
            new_audio,
            new_manifest,
        )
        raw = store.connection.execute(
            """
            SELECT audio_path, transcript_path, manifest_path
            FROM feed_days WHERE id=?
            """,
            (first.day_id,),
        ).fetchone()

        assert second.day_id == first.day_id
        assert raw["audio_path"] == str(new_audio.resolve())
        assert raw["transcript_path"] == str(new_transcript.resolve())
        assert raw["manifest_path"] == str(new_manifest.resolve())
        assert store.stats()["transcript_segments"] == 2


def test_store_rebases_stale_paths_to_conventional_library_layout(
    tmp_path: Path,
) -> None:
    archive_date = date(2026, 7, 12)
    legacy = tmp_path / "old-host"
    old_audio = legacy / "day.mp3"
    old_transcript = legacy / "day.json"
    old_audio.parent.mkdir()
    old_audio.write_bytes(b"retained audio")
    make_transcript(old_transcript)

    library = tmp_path / "portable-library"
    database = library / "broadcastify-analysis.sqlite3"
    with AnalysisStore(database) as store:
        imported = store.import_transcript(
            "90001",
            archive_date,
            old_transcript,
            old_audio,
        )
        incident_id = store.replace_incidents(
            imported.day_id,
            [
                {
                    "fingerprint": "portable-path",
                    "event_type": "shots_fired",
                    "title": "Reported shots fired",
                    "summary": "A shots-fired report was retained.",
                    "location": "Main and First",
                    "start_seconds": 10.0,
                    "end_seconds": 15.0,
                    "priority": 4,
                    "confidence": 0.9,
                    "evidence": [],
                    "attributes": {},
                }
            ],
            model="test-model",
            prompt_version="test-prompt",
        )[0]

        day_directory = library / "90001" / "20260712"
        new_audio = day_directory / "combined_90001_20260712.mp3"
        new_transcript = (
            day_directory / "transcripts" / "combined_90001_20260712.json"
        )
        new_audio.parent.mkdir(parents=True)
        new_transcript.parent.mkdir(parents=True)
        old_audio.replace(new_audio)
        old_transcript.replace(new_transcript)

        day = store.get_day("90001", archive_date)
        incident = store.get_incident(incident_id)

        assert day is not None
        assert incident is not None
        assert day["audio_path"] == str(new_audio.resolve())
        assert day["transcript_path"] == str(new_transcript.resolve())
        assert incident["audio_path"] == str(new_audio.resolve())


def test_analysis_window_checkpoint_reuses_exact_window_across_transcript_revisions(
    tmp_path: Path,
) -> None:
    transcript = tmp_path / "transcript.json"
    make_transcript(transcript)

    with AnalysisStore(tmp_path / "analysis.sqlite3") as store:
        imported = store.import_transcript(
            "90001",
            date(2026, 7, 12),
            transcript,
        )
        old_incidents = [{"title": "old exact window"}]
        store.save_analysis_window_checkpoint(
            imported.day_id,
            "local-model",
            "prompt-v1",
            "old-transcript",
            0,
            "same-window",
            old_incidents,
        )

        transcript.write_text(
            transcript.read_text(encoding="utf-8").replace(
                '"duration": 600.0',
                '"duration": 900.0',
            ),
            encoding="utf-8",
        )
        reimported = store.import_transcript(
            "90001",
            date(2026, 7, 12),
            transcript,
        )

        assert reimported.day_id == imported.day_id
        assert store.stats()["analysis_window_checkpoints"] == 1
        assert store.get_analysis_window_checkpoint(
            imported.day_id,
            "local-model",
            "prompt-v1",
            "new-transcript",
            0,
            "same-window",
        ) == old_incidents
        assert (
            store.get_analysis_window_checkpoint(
                imported.day_id,
                "local-model",
                "prompt-v1",
                "new-transcript",
                0,
                "changed-window",
            )
            is None
        )

        new_incidents = [{"title": "new exact window"}]
        store.save_analysis_window_checkpoint(
            imported.day_id,
            "local-model",
            "prompt-v1",
            "new-transcript",
            0,
            "same-window",
            new_incidents,
        )
        assert store.get_analysis_window_checkpoint(
            imported.day_id,
            "local-model",
            "prompt-v1",
            "new-transcript",
            0,
            "same-window",
        ) == new_incidents

        store.prune_analysis_window_checkpoints(
            imported.day_id,
            "local-model",
            "prompt-v1",
            "new-transcript",
        )
        assert store.stats()["analysis_window_checkpoints"] == 1


def test_area_profiles_are_persisted_and_updated(tmp_path: Path) -> None:
    with AnalysisStore(tmp_path / "analysis.sqlite3") as store:
        first = store.save_area_profile(
            "Dallas desk",
            ["75201", "75202"],
            [{"feed_id": "90001", "name": "Dallas Police"}],
        )
        updated = store.save_area_profile(
            "Dallas desk",
            ["75201"],
            [
                {"feed_id": "90001", "name": "Dallas Police"},
                {"feed_id": "5318", "name": "Dallas Central"},
            ],
        )

        assert first["feed_ids"] == ["90001"]
        assert updated["feed_ids"] == ["90001", "5318"]
        assert store.list_area_profiles()[0]["zip_codes"] == ["75201"]
        assert store.list_area_profiles()[0]["coverage"]["mode"] == "zip-list"
        assert store.stats()["area_profiles"] == 1


def test_radius_area_profile_retains_nearest_first_metadata(tmp_path: Path) -> None:
    with AnalysisStore(tmp_path / "analysis.sqlite3") as store:
        profile = store.save_area_profile(
            "Example radius",
            ["12345", "12346"],
            [
                {
                    "feed_id": "200",
                    "name": "Outer feed",
                    "distance_miles": 4.5,
                    "priority_rank": 2,
                    "nearest_zip_code": "12346",
                },
                {
                    "feed_id": "100",
                    "name": "Center feed",
                    "distance_miles": 0,
                    "priority_rank": 1,
                    "nearest_zip_code": "12345",
                },
            ],
            {
                "mode": "radius",
                "center_zip": "12345",
                "radius_miles": 25,
                "max_zip_codes": 12,
                "distance_basis": "test centroids",
            },
        )

        assert profile["coverage"]["center_zip"] == "12345"
        assert profile["coverage"]["radius_miles"] == 25
        assert [value["feed_id"] for value in profile["feeds"]] == ["100", "200"]
        assert profile["feeds"][1]["distance_miles"] == 4.5
