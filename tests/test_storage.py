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
        assert store.stats()["area_profiles"] == 1
