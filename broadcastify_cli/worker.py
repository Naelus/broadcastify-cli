from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from datetime import date
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from .accelerators import collect_accelerator_diagnostics
from .audio import (
    extract_audio_clip,
    find_ffmpeg,
    select_incident_evidence_window,
)
from .analysis import (
    DEFAULT_EMBEDDING_MODEL,
    PROMPT_VERSION,
    IncidentAnalyzer,
    RangeQuestionAnswerer,
    SemanticIndexer,
    WeeklySummaryAnalyzer,
    discover_day_paths,
    find_llama_server,
    format_archive_time,
)
from .analysis_providers import AnalysisProviderConfig, open_analysis_client
from .area_watch import AreaStoryAnalyzer
from .broadcastify import BroadcastifyClient
from .jobs import JobRunner
from .library import LocalProcessingRequest, prepare_local_day, scan_local_library
from .models import JobRequest
from .storage import AnalysisStore, sha256_file


DEFAULT_DATABASE = Path("archives/broadcastify-analysis.sqlite3")
LOADED_ENVIRONMENT_FILE: Path | None = None


def emit(value: dict[str, Any]) -> None:
    print(json.dumps(value, ensure_ascii=False), flush=True)


def search_feeds(query: str) -> int:
    with BroadcastifyClient() as client:
        results = client.search_feeds(query)
    serialized = [result.to_dict() for result in results]
    with AnalysisStore(DEFAULT_DATABASE) as store:
        store.save_feed_catalog(serialized)
    emit({"type": "result", "results": serialized})
    return 0


def search_area(zip_codes: list[str]) -> int:
    with BroadcastifyClient() as client:
        results = client.search_area_feeds(zip_codes)
    with AnalysisStore(DEFAULT_DATABASE) as store:
        store.save_feed_catalog(results)
    emit({"type": "area_search", "results": results})
    return 0


def list_area_profiles() -> int:
    with AnalysisStore(DEFAULT_DATABASE) as store:
        profiles = store.list_area_profiles()
    emit({"type": "area_profiles", "profiles": profiles})
    return 0


def save_area_profile() -> int:
    payload = json.load(sys.stdin)
    with AnalysisStore(DEFAULT_DATABASE) as store:
        store.save_feed_catalog(list(payload.get("feeds") or []))
        profile = store.save_area_profile(
            str(payload.get("name") or ""),
            list(payload.get("zip_codes") or []),
            list(payload.get("feeds") or []),
        )
    emit(
        {
            "type": "area_profile_saved",
            "message": f"Saved area profile {profile['name']} with {len(profile['feed_ids'])} feeds.",
            "profile": profile,
        }
    )
    return 0


def run_job() -> int:
    payload = json.load(sys.stdin)
    request = JobRequest.from_dict(payload)
    with BroadcastifyClient() as client:
        JobRunner(request, emit=emit, client=client).run()
    return 0


def authenticate() -> int:
    payload = json.load(sys.stdin)
    username = str(payload.get("username") or "").strip()
    password = str(payload.get("password") or "")
    if not username or not password:
        raise ValueError("Broadcastify username and password are required.")
    with BroadcastifyClient(username=username, password=password) as client:
        client.authenticate(force=True)
    emit({"type": "authenticated", "message": "Broadcastify sign-in succeeded."})
    return 0


def diagnostics() -> int:
    llama_server = find_llama_server()
    payload: dict[str, Any] = {
        "type": "diagnostics",
        "python": sys.version.split()[0],
        "python_executable": sys.executable,
        "ffmpeg": find_ffmpeg(),
        "cuda_available": False,
        "cuda_devices": [],
        "llama_server": llama_server,
        "huggingface_token_configured": bool(
            os.getenv("HUGGINGFACE_TOKEN") or os.getenv("HF_TOKEN")
        ),
        "broadcastify_credentials_configured": bool(
            (
                os.getenv("BROADCASTIFY_USERNAME")
                and os.getenv("BROADCASTIFY_PASSWORD")
            )
            or (os.getenv("USERNAME") and os.getenv("PASSWORD"))
        ),
        "saved_session_available": Path("cookies.json").is_file(),
        "environment_file": str(LOADED_ENVIRONMENT_FILE.resolve())
        if LOADED_ENVIRONMENT_FILE
        else "",
        "analysis_database": str(DEFAULT_DATABASE.resolve()),
        "analysis_stats": {},
    }
    try:
        import torch

        payload["cuda_available"] = bool(torch.cuda.is_available())
        payload["cuda_devices"] = [
            torch.cuda.get_device_name(index)
            for index in range(torch.cuda.device_count())
        ]
    except ModuleNotFoundError:
        payload["transcription_dependencies"] = "not installed"
    payload["accelerators"] = collect_accelerator_diagnostics(llama_server)
    if DEFAULT_DATABASE.exists():
        with AnalysisStore(DEFAULT_DATABASE) as store:
            payload["analysis_stats"] = store.stats()
    emit(payload)
    return 0


def analysis_days(feed_id: str | None) -> int:
    with AnalysisStore(DEFAULT_DATABASE) as store:
        days = store.list_days(feed_id)
    emit({"type": "analysis_days", "days": days})
    return 0


def library_days(output_dir: str) -> int:
    days = scan_local_library(Path(output_dir), DEFAULT_DATABASE)
    emit(
        {
            "type": "library_days",
            "days": days,
            "summary": {
                "feed_count": len({value["feed_id"] for value in days}),
                "day_count": len(days),
                "complete_count": sum(bool(value["is_complete"]) for value in days),
                "attention_count": sum(not bool(value["is_complete"]) for value in days),
                "storage_bytes": sum(int(value["storage_bytes"]) for value in days),
            },
        }
    )
    return 0


def _day_report(store: AnalysisStore, feed_id: str, archive_date: date) -> dict[str, Any]:
    day = store.get_day(feed_id, archive_date)
    if day is None:
        raise ValueError(f"No imported transcript for feed {feed_id} on {archive_date}.")
    summary = store.get_latest_daily_summary(int(day["id"]))
    incidents = []
    for stored in store.get_incidents(feed_id, archive_date, archive_date):
        # Keep the routine list/report response compact. Raw transcript evidence
        # stays in the local database for retrieval and question answering.
        incidents.append(
            {
                "id": stored["id"],
                "event_type": stored["event_type"],
                "title": stored["title"],
                "summary": stored["summary"],
                "location": stored["location"],
                "priority": stored["priority"],
                "confidence": stored["confidence"],
                "start_seconds": stored["start_seconds"],
                "end_seconds": stored["end_seconds"],
                "archive_time": format_archive_time(
                    stored, float(stored["start_seconds"])
                ),
            }
        )
    return {
        "feed_id": feed_id,
        "archive_date": archive_date.isoformat(),
        "summary": str(summary["summary"]) if summary else "",
        "incidents": incidents,
        "audio_path": str(day["audio_path"] or ""),
        "has_diarization": bool(day["has_diarization"]),
    }


def report_day(feed_id: str, date_value: str) -> int:
    archive_date = date.fromisoformat(date_value)
    with AnalysisStore(DEFAULT_DATABASE) as store:
        report = _day_report(store, feed_id, archive_date)
    emit({"type": "day_report", "report": report})
    return 0


def _incident_clip(store: AnalysisStore, incident_id: int) -> dict[str, Any]:
    incident = store.get_incident(incident_id)
    if incident is None:
        raise ValueError(f"Incident I{incident_id} was not found in the local analysis database.")
    source_value = str(incident.get("audio_path") or "")
    source = Path(source_value)
    if not source.is_file():
        raise ValueError(
            f"The retained combined audio for incident I{incident_id} could not be found."
        )

    start_seconds, end_seconds = select_incident_evidence_window(incident)
    start_milliseconds = round(start_seconds * 1_000)
    end_milliseconds = round(end_seconds * 1_000)
    output = (
        source.parent
        / "evidence-clips"
        / (
            f"{incident['feed_id']}_{incident['archive_date']}_I{incident_id}_"
            f"{start_milliseconds}-{end_milliseconds}.mp3"
        )
    )
    clip = extract_audio_clip(source, output, start_seconds, end_seconds).resolve()
    return {
        "incident_id": int(incident_id),
        "feed_id": str(incident["feed_id"]),
        "archive_date": str(incident["archive_date"]),
        "archive_time": format_archive_time(incident, start_seconds),
        "start_seconds": start_seconds,
        "end_seconds": end_seconds,
        "duration_seconds": end_seconds - start_seconds,
        "path": str(clip),
        "sha256": sha256_file(clip),
    }


def incident_clip(incident_id: int) -> int:
    with AnalysisStore(DEFAULT_DATABASE) as store:
        result = _incident_clip(store, incident_id)
    emit(
        {
            "type": "incident_clip",
            "message": f"Exact local evidence clip ready for incident I{incident_id}.",
            "clip": result,
        }
    )
    return 0


def _analyze_day_payload(payload: dict[str, Any]) -> tuple[dict[str, Any], int, int]:
    feed_id = str(payload["feed_id"])
    archive_date = date.fromisoformat(str(payload["archive_date"]))
    output_dir = Path(payload.get("output_dir") or "archives")
    provider = AnalysisProviderConfig.from_mapping(payload)
    force = bool(payload.get("force", False))
    force_summary = bool(payload.get("force_summary", False))
    audio, transcript, manifest = discover_day_paths(output_dir, feed_id, archive_date)
    with AnalysisStore(DEFAULT_DATABASE) as store:
        imported = store.import_transcript(
            feed_id, archive_date, transcript, audio, manifest
        )
        emit(
            {
                "type": "stage",
                "stage": "analysis_import",
                "message": f"Imported {imported.segment_count} transcript segments.",
            }
        )
        day = store.get_day(feed_id, archive_date)
        complete = bool(
            day is not None
            and store.get_daily_summary(
                int(day["id"]),
                provider.cache_model,
                PROMPT_VERSION,
                str(day["transcript_sha256"]),
            )
        )
        emit(
            {
                "type": "stage",
                "stage": "analysis_provider",
                "message": (
                    f"Analysis provider: {provider.provider} / {provider.cache_model}"
                    + (" (external transcript excerpts allowed)." if provider.is_external else ".")
                ),
            }
        )
        with open_analysis_client(
            provider,
            launch_local_server=not (complete and not force and not force_summary),
        ) as client:
            result = IncidentAnalyzer(
                store,
                client,
                progress=lambda message: emit(
                    {"type": "stage", "stage": "analysis", "message": message}
                ),
            ).analyze_day(
                feed_id,
                archive_date,
                force=force,
                force_summary=force_summary,
            )
        emit(
            {
                "type": "stage",
                "stage": "embeddings",
                "message": "Updating semantic search index…",
            }
        )
        indexed = SemanticIndexer(store, model=DEFAULT_EMBEDDING_MODEL).index_missing()
        report = _day_report(store, feed_id, archive_date)
    return report, int(result["incidents"]), indexed


def analyze_day() -> int:
    payload = json.load(sys.stdin)
    report, incident_count, indexed = _analyze_day_payload(payload)
    emit(
        {
            "type": "analysis_complete",
            "message": f"Analysis complete: {incident_count} incidents, {indexed} new passages indexed.",
            "report": report,
        }
    )
    return 0


def continue_local_day() -> int:
    payload = json.load(sys.stdin)
    request = LocalProcessingRequest.from_dict(payload)
    emit(
        {
            "type": "stage",
            "stage": "local_prepare",
            "message": (
                f"Continuing feed {request.feed_id} for {request.archive_date} "
                "from retained local files…"
            ),
        }
    )
    prepared = prepare_local_day(
        request,
        progress=lambda message: emit(
            {"type": "stage", "stage": "local_prepare", "message": message}
        ),
    )
    report = None
    if bool(payload.get("analyze", True)):
        provider_fields = {
            key: payload[key]
            for key in (
                "analysis_provider",
                "analysis_model",
                "analysis_endpoint",
                "analysis_api_key",
                "analysis_api_key_env",
                "codex_cli_path",
                "allow_external_analysis",
                "analysis_timeout",
            )
            if key in payload
        }
        report, incident_count, indexed = _analyze_day_payload(
            {
                "feed_id": request.feed_id,
                "archive_date": request.archive_date.isoformat(),
                "output_dir": str(request.output_dir),
                **provider_fields,
            }
        )
        emit(
            {
                "type": "stage",
                "stage": "analysis_complete",
                "message": (
                    f"Analysis complete: {incident_count} incidents, "
                    f"{indexed} new passages indexed."
                ),
            }
        )
    state = next(
        (
            value
            for value in scan_local_library(request.output_dir, DEFAULT_DATABASE)
            if value["feed_id"] == request.feed_id
            and value["archive_date"] == request.archive_date.isoformat()
        ),
        None,
    )
    emit(
        {
            "type": "local_complete",
            "message": f"Local processing finished for {request.archive_date}.",
            "prepared": prepared,
            "state": state,
            "report": report,
        }
    )
    return 0


def ask_archive() -> int:
    payload = json.load(sys.stdin)
    feed_id = str(payload["feed_id"])
    start_date = date.fromisoformat(str(payload["start_date"]))
    end_date = date.fromisoformat(str(payload["end_date"]))
    question = str(payload["question"]).strip()
    if not question:
        raise ValueError("A question is required.")
    provider = AnalysisProviderConfig.from_mapping(payload)
    emit(
        {
            "type": "stage",
            "stage": "retrieval",
            "message": "Retrieving transcript evidence…",
        }
    )
    with AnalysisStore(DEFAULT_DATABASE) as store:
        indexer = SemanticIndexer(store, model=DEFAULT_EMBEDDING_MODEL)
        indexer.index_missing()
        with open_analysis_client(provider) as client:
            result = RangeQuestionAnswerer(
                store,
                client,
                indexer=indexer,
            ).ask(feed_id, start_date, end_date, question)
    emit({"type": "answer", "message": "Question answered.", "result": result})
    return 0


def summarize_week() -> int:
    payload = json.load(sys.stdin)
    feed_id = str(payload["feed_id"])
    week_ending = date.fromisoformat(str(payload["week_ending"]))
    provider = AnalysisProviderConfig.from_mapping(payload)
    force = bool(payload.get("force", False))
    emit(
        {
            "type": "stage",
            "stage": "weekly_summary",
            "message": f"Building seven-day brief ending {week_ending}…",
        }
    )
    with AnalysisStore(DEFAULT_DATABASE) as store:
        with open_analysis_client(provider) as client:
            result = WeeklySummaryAnalyzer(
                store,
                client,
                progress=lambda message: emit(
                    {
                        "type": "stage",
                        "stage": "weekly_summary",
                        "message": message,
                    }
                ),
            ).summarize(feed_id, week_ending, force=force)
    emit(
        {
            "type": "weekly_summary",
            "message": (
                f"Weekly brief ready: {result['days_available']}/7 days, "
                f"{result['incident_count']} incidents."
            ),
            "result": result,
        }
    )
    return 0


def summarize_area() -> int:
    payload = json.load(sys.stdin)
    profile_name = str(payload["profile_name"])
    start_date = date.fromisoformat(str(payload["start_date"]))
    end_date = date.fromisoformat(str(payload["end_date"]))
    provider = AnalysisProviderConfig.from_mapping(payload)
    force = bool(payload.get("force", False))
    emit(
        {
            "type": "stage",
            "stage": "area_digest",
            "message": f"Building area story brief for {start_date} through {end_date}…",
        }
    )
    with AnalysisStore(DEFAULT_DATABASE) as store:
        with open_analysis_client(provider) as client:
            result = AreaStoryAnalyzer(
                store,
                client,
                progress=lambda message: emit(
                    {"type": "stage", "stage": "area_digest", "message": message}
                ),
            ).summarize(profile_name, start_date, end_date, force=force)
    emit(
        {
            "type": "area_digest",
            "message": f"Area brief ready: {len(result['stories'])} ranked story leads.",
            "result": result,
        }
    )
    return 0


def latest_area_digest(profile_name: str) -> int:
    with AnalysisStore(DEFAULT_DATABASE) as store:
        row = store.get_latest_area_story_digest(profile_name)
    result = None
    if row is not None:
        result = {
            "profile_name": str(row["profile_name"]),
            "start_date": str(row["start_date"]),
            "end_date": str(row["end_date"]),
            "summary": str(row["summary"]),
            "stories": json.loads(str(row["stories_json"])),
            "coverage": json.loads(str(row["coverage_json"])),
            "cached": True,
        }
    emit({"type": "saved_area_digest", "result": result})
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="JSON worker for the Windows desktop UI")
    subparsers = parser.add_subparsers(dest="command", required=True)
    search = subparsers.add_parser("search")
    search.add_argument("--query", required=True)
    area_search = subparsers.add_parser("area-search")
    area_search.add_argument("--zip", action="append", dest="zip_codes", required=True)
    subparsers.add_parser("area-profiles")
    subparsers.add_parser("save-area-profile")
    subparsers.add_parser("summarize-area")
    saved_area = subparsers.add_parser("saved-area-digest")
    saved_area.add_argument("--profile-name", required=True)
    subparsers.add_parser("run")
    subparsers.add_parser("authenticate")
    subparsers.add_parser("diagnostics")
    library = subparsers.add_parser("library")
    library.add_argument("--output-dir", default="archives")
    subparsers.add_parser("continue-local")
    days = subparsers.add_parser("analysis-days")
    days.add_argument("--feed-id")
    report = subparsers.add_parser("report-day")
    report.add_argument("--feed-id", required=True)
    report.add_argument("--date", required=True)
    clip = subparsers.add_parser("incident-clip")
    clip.add_argument("--incident-id", required=True, type=int)
    subparsers.add_parser("analyze-day")
    subparsers.add_parser("summarize-week")
    subparsers.add_parser("ask")
    return parser


def load_worker_environment() -> Path | None:
    """Load repository defaults, then an explicitly bundled private env file."""

    loaded: Path | None = None
    repository_env = Path.cwd() / ".env"
    if repository_env.is_file():
        load_dotenv(repository_env, override=True)
        loaded = repository_env
    configured = os.getenv("BROADCASTIFY_ENV_FILE")
    if configured:
        bundled_env = Path(configured)
        if bundled_env.is_file():
            load_dotenv(bundled_env, override=True)
            loaded = bundled_env
    return loaded


def main() -> int:
    global LOADED_ENVIRONMENT_FILE
    # override=True preserves compatibility with the original USERNAME setting
    # on Windows, where USERNAME already exists in the parent environment.
    LOADED_ENVIRONMENT_FILE = load_worker_environment()
    arguments = build_parser().parse_args()
    try:
        if arguments.command == "search":
            return search_feeds(arguments.query)
        if arguments.command == "area-search":
            return search_area(arguments.zip_codes)
        if arguments.command == "area-profiles":
            return list_area_profiles()
        if arguments.command == "save-area-profile":
            return save_area_profile()
        if arguments.command == "summarize-area":
            return summarize_area()
        if arguments.command == "saved-area-digest":
            return latest_area_digest(arguments.profile_name)
        if arguments.command == "run":
            return run_job()
        if arguments.command == "authenticate":
            return authenticate()
        if arguments.command == "diagnostics":
            return diagnostics()
        if arguments.command == "library":
            return library_days(arguments.output_dir)
        if arguments.command == "continue-local":
            return continue_local_day()
        if arguments.command == "analysis-days":
            return analysis_days(arguments.feed_id)
        if arguments.command == "report-day":
            return report_day(arguments.feed_id, arguments.date)
        if arguments.command == "incident-clip":
            return incident_clip(arguments.incident_id)
        if arguments.command == "analyze-day":
            return analyze_day()
        if arguments.command == "summarize-week":
            return summarize_week()
        if arguments.command == "ask":
            return ask_archive()
        raise ValueError(f"Unknown command: {arguments.command}")
    except Exception as exc:
        emit(
            {
                "type": "error",
                "message": str(exc),
                "exception": type(exc).__name__,
                "traceback": traceback.format_exc(),
            }
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
