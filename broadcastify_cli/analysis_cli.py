from __future__ import annotations

import json
from datetime import date, datetime, timedelta
from pathlib import Path

import click
from dotenv import load_dotenv
from rich.console import Console

from .analysis import (
    DEFAULT_EMBEDDING_MODEL,
    PROMPT_VERSION,
    IncidentAnalyzer,
    RangeQuestionAnswerer,
    SemanticIndexer,
    WeeklySummaryAnalyzer,
    discover_day_paths,
    format_archive_time,
)
from .analysis_providers import (
    PROVIDER_CHOICES,
    AnalysisProviderConfig,
    open_analysis_client,
)
from .library import (
    build_archive_question_coverage,
    entire_archive_feed_range,
    require_current_range_evidence,
    scan_local_library,
)
from .storage import AnalysisStore


console = Console()


def parse_date(value: str) -> date:
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise click.BadParameter("Use YYYY-MM-DD.") from exc


def provider_config(
    provider: str,
    model: str | None,
    server_url: str | None,
    api_key_env: str,
    codex_path: Path | None,
    allow_external_analysis: bool,
    analysis_device: str | None = None,
) -> AnalysisProviderConfig:
    value: dict[str, object] = {
        "analysis_provider": provider,
        "analysis_api_key_env": api_key_env,
        "allow_external_analysis": allow_external_analysis,
    }
    if analysis_device:
        value["analysis_device"] = analysis_device
    if model:
        value["analysis_model"] = model
    if server_url:
        value["analysis_endpoint"] = server_url
    if codex_path:
        value["codex_cli_path"] = str(codex_path)
    return AnalysisProviderConfig.from_mapping(value)


@click.group(context_settings={"help_option_names": ["-h", "--help"]})
def cli() -> None:
    """Persist, classify, summarize, and query local scanner transcripts."""
    load_dotenv(Path.cwd() / ".env", override=True)


@cli.command("import-day")
@click.option("--feed-id", required=True)
@click.option("--date", "date_value", required=True, callback=lambda _c, _p, v: parse_date(v))
@click.option("--output-dir", type=click.Path(path_type=Path), default=Path("archives"))
@click.option("--db", type=click.Path(path_type=Path), default=Path("archives/broadcastify-analysis.sqlite3"))
def import_day(feed_id: str, date_value: date, output_dir: Path, db: Path) -> None:
    """Import one existing combined transcript into the persistent database."""
    audio, transcript, manifest = discover_day_paths(output_dir, feed_id, date_value)
    with AnalysisStore(db) as store:
        imported = store.import_transcript(
            feed_id, date_value, transcript, audio, manifest
        )
        stats = store.stats()
    console.print(
        f"Imported feed {feed_id} {date_value}: {imported.segment_count} segments."
    )
    console.print_json(json.dumps(stats))


@cli.command("analyze-day")
@click.option("--feed-id", required=True)
@click.option("--date", "date_value", required=True, callback=lambda _c, _p, v: parse_date(v))
@click.option("--output-dir", type=click.Path(path_type=Path), default=Path("archives"))
@click.option("--db", type=click.Path(path_type=Path), default=Path("archives/broadcastify-analysis.sqlite3"))
@click.option("--provider", type=click.Choice(PROVIDER_CHOICES), default="local", show_default=True)
@click.option("--model", help="Provider model; local defaults to quantized Gemma")
@click.option("--analysis-device", type=click.Choice(["auto", "cpu"]), help="Managed local llama.cpp device; defaults to ANALYSIS_DEVICE or auto")
@click.option("--server-url", help="Provider base URL ending in /v1, or an existing local llama.cpp endpoint")
@click.option("--api-key-env", default="OPENAI_API_KEY", show_default=True, help="Environment variable containing the provider key")
@click.option("--codex-path", type=click.Path(path_type=Path), help="Optional Codex CLI executable")
@click.option("--allow-external-analysis", is_flag=True, help="Acknowledge that transcript excerpts may leave this computer")
@click.option("--embedding-model", default=DEFAULT_EMBEDDING_MODEL, show_default=True)
@click.option("--embeddings/--no-embeddings", default=True, show_default=True)
@click.option("--force", is_flag=True, help="Re-run extraction and summary even if cached")
@click.option("--force-summary", is_flag=True, help="Refresh only the daily summary")
def analyze_day(
    feed_id: str,
    date_value: date,
    output_dir: Path,
    db: Path,
    provider: str,
    model: str | None,
    analysis_device: str | None,
    server_url: str | None,
    api_key_env: str,
    codex_path: Path | None,
    allow_external_analysis: bool,
    embedding_model: str,
    embeddings: bool,
    force: bool,
    force_summary: bool,
) -> None:
    """Import and classify one day, then create its end-of-day summary."""
    provider_settings = provider_config(
        provider,
        model,
        server_url,
        api_key_env,
        codex_path,
        allow_external_analysis,
        analysis_device,
    )
    audio, transcript, manifest = discover_day_paths(output_dir, feed_id, date_value)
    with AnalysisStore(db) as store:
        imported = store.import_transcript(
            feed_id, date_value, transcript, audio, manifest
        )
        console.print(f"Imported {imported.segment_count} transcript segments.")
        day = store.get_day(feed_id, date_value)
        complete = bool(
            day is not None
            and store.get_daily_summary(
                int(day["id"]),
                provider_settings.cache_model,
                PROMPT_VERSION,
                str(day["transcript_sha256"]),
            )
        )
        with open_analysis_client(
            provider_settings,
            launch_local_server=not (complete and not force and not force_summary),
        ) as client:
            result = IncidentAnalyzer(
                store, client, progress=console.print
            ).analyze_day(
                feed_id,
                date_value,
                force=force,
                force_summary=force_summary,
            )
            console.print(f"Extracted {result['incidents']} supported incidents.")
            console.print(result["summary"])
        if embeddings:
            indexer = SemanticIndexer(store, model=embedding_model)
            indexed = indexer.index_missing()
            console.print(f"Indexed {indexed} new transcript passages for semantic search.")


@cli.command("ask")
@click.option("--feed-id", required=True)
@click.option("--start-date", callback=lambda _c, _p, v: parse_date(v) if v else None)
@click.option("--end-date", callback=lambda _c, _p, v: parse_date(v) if v else None)
@click.option(
    "--entire-feed",
    is_flag=True,
    help="Use the earliest-to-latest locally retained span for this feed.",
)
@click.option("--question", required=True)
@click.option("--db", type=click.Path(path_type=Path), default=Path("archives/broadcastify-analysis.sqlite3"))
@click.option("--provider", type=click.Choice(PROVIDER_CHOICES), default="local", show_default=True)
@click.option("--model", help="Provider model; local defaults to quantized Gemma")
@click.option("--analysis-device", type=click.Choice(["auto", "cpu"]), help="Managed local llama.cpp device; defaults to ANALYSIS_DEVICE or auto")
@click.option("--server-url", help="Provider base URL ending in /v1, or an existing local llama.cpp endpoint")
@click.option("--api-key-env", default="OPENAI_API_KEY", show_default=True)
@click.option("--codex-path", type=click.Path(path_type=Path))
@click.option("--allow-external-analysis", is_flag=True, help="Acknowledge that transcript excerpts may leave this computer")
@click.option("--embedding-model", default=DEFAULT_EMBEDDING_MODEL, show_default=True)
@click.option("--semantic/--keyword-only", default=True, show_default=True)
def ask(
    feed_id: str,
    start_date: date | None,
    end_date: date | None,
    entire_feed: bool,
    question: str,
    db: Path,
    provider: str,
    model: str | None,
    analysis_device: str | None,
    server_url: str | None,
    api_key_env: str,
    codex_path: Path | None,
    allow_external_analysis: bool,
    embedding_model: str,
    semantic: bool,
) -> None:
    """Ask an evidence-grounded question over a range or entire retained feed."""
    days = scan_local_library(db.parent, db)
    if entire_feed:
        if start_date is not None or end_date is not None:
            raise click.UsageError(
                "Use --entire-feed or --start-date/--end-date, not both."
            )
        try:
            start_date, end_date = entire_archive_feed_range(days, feed_id)
        except ValueError as exc:
            raise click.ClickException(str(exc)) from exc
    elif start_date is None or end_date is None:
        raise click.UsageError(
            "Provide both --start-date and --end-date, or use --entire-feed."
        )
    if start_date > end_date:
        raise click.BadParameter("Start date must be on or before end date.")
    provider_settings = provider_config(
        provider,
        model,
        server_url,
        api_key_env,
        codex_path,
        allow_external_analysis,
        analysis_device,
    )
    with AnalysisStore(db) as store:
        coverage = build_archive_question_coverage(
            days,
            feed_id,
            start_date,
            end_date,
        )
        if int(coverage["question_ready_day_count"]) == 0:
            raise click.ClickException(
                "No question-ready retained transcripts exist for this feed and range."
            )
        try:
            require_current_range_evidence(
                store,
                [feed_id],
                start_date,
                end_date,
                require_analysis=False,
                purpose="Archive question answering",
                archive_dates=coverage["question_ready_dates"],
            )
        except ValueError as exc:
            raise click.ClickException(str(exc)) from exc
        indexer = None
        if semantic:
            indexer = SemanticIndexer(store, model=embedding_model)
            indexed = indexer.index_missing()
            if indexed:
                console.print(f"Indexed {indexed} new transcript passages.")
        with open_analysis_client(provider_settings) as client:
            result = RangeQuestionAnswerer(
                store,
                client,
                indexer=indexer,
            ).ask(
                feed_id,
                start_date,
                end_date,
                question,
                coverage=coverage,
            )
    console.print(result["answer"])
    console.print("[dim]Coverage: " + str(coverage["summary"]) + "[/dim]")
    if result["limitations"]:
        console.print("[dim]Limitations: " + "; ".join(result["limitations"]) + "[/dim]")


@cli.command("summarize-week")
@click.option("--feed-id", required=True)
@click.option("--week-ending", required=True, callback=lambda _c, _p, v: parse_date(v))
@click.option("--db", type=click.Path(path_type=Path), default=Path("archives/broadcastify-analysis.sqlite3"))
@click.option("--provider", type=click.Choice(PROVIDER_CHOICES), default="local", show_default=True)
@click.option("--model", help="Provider model; local defaults to quantized Gemma")
@click.option("--analysis-device", type=click.Choice(["auto", "cpu"]), help="Managed local llama.cpp device; defaults to ANALYSIS_DEVICE or auto")
@click.option("--server-url", help="Provider base URL ending in /v1, or an existing local llama.cpp endpoint")
@click.option("--api-key-env", default="OPENAI_API_KEY", show_default=True)
@click.option("--codex-path", type=click.Path(path_type=Path))
@click.option("--allow-external-analysis", is_flag=True, help="Acknowledge that transcript excerpts may leave this computer")
@click.option("--force", is_flag=True, help="Regenerate even when the source data is unchanged")
@click.option("--json-output", is_flag=True, help="Emit the complete weekly report as JSON")
def summarize_week(
    feed_id: str,
    week_ending: date,
    db: Path,
    provider: str,
    model: str | None,
    analysis_device: str | None,
    server_url: str | None,
    api_key_env: str,
    codex_path: Path | None,
    allow_external_analysis: bool,
    force: bool,
    json_output: bool,
) -> None:
    """Summarize the seven-day period ending on the selected date."""
    provider_settings = provider_config(
        provider,
        model,
        server_url,
        api_key_env,
        codex_path,
        allow_external_analysis,
        analysis_device,
    )
    with AnalysisStore(db) as store:
        try:
            require_current_range_evidence(
                store,
                [feed_id],
                week_ending - timedelta(days=6),
                week_ending,
                require_analysis=True,
                purpose="Weekly summary",
            )
        except ValueError as exc:
            raise click.ClickException(str(exc)) from exc
        with open_analysis_client(provider_settings) as client:
            result = WeeklySummaryAnalyzer(
                store,
                client,
                progress=console.print,
            ).summarize(feed_id, week_ending, force=force)
    if json_output:
        console.print_json(json.dumps(result, ensure_ascii=False))
        return
    console.print(
        f"Coverage: {result['start_date']} through {result['end_date']} · "
        f"{result['days_available']}/7 days · {result['incident_count']} incidents"
    )
    console.print(result["summary"])


@cli.command("stats")
@click.option("--db", type=click.Path(path_type=Path), default=Path("archives/broadcastify-analysis.sqlite3"))
def stats(db: Path) -> None:
    """Show persistent database counts."""
    with AnalysisStore(db) as store:
        console.print_json(json.dumps(store.stats()))


@cli.command("report-day")
@click.option("--feed-id", required=True)
@click.option("--date", "date_value", required=True, callback=lambda _c, _p, v: parse_date(v))
@click.option("--min-priority", type=click.IntRange(1, 5), default=3, show_default=True)
@click.option("--db", type=click.Path(path_type=Path), default=Path("archives/broadcastify-analysis.sqlite3"))
@click.option("--json-output", is_flag=True, help="Emit machine-readable incident JSON")
def report_day(
    feed_id: str,
    date_value: date,
    min_priority: int,
    db: Path,
    json_output: bool,
) -> None:
    """List every persisted eventful incident for one day without starting a model."""
    with AnalysisStore(db) as store:
        day = store.get_day(feed_id, date_value)
        if day is None:
            raise click.ClickException(
                f"No imported transcript for feed {feed_id} on {date_value}."
            )
        try:
            require_current_range_evidence(
                store,
                [feed_id],
                date_value,
                date_value,
                require_analysis=True,
                purpose="Day report",
            )
        except ValueError as exc:
            raise click.ClickException(str(exc)) from exc
        incidents = [
            value
            for value in store.get_incidents(
                feed_id,
                date_value,
                date_value,
                prompt_version=PROMPT_VERSION,
            )
            if int(value["priority"]) >= min_priority
        ]
        summary = store.get_latest_daily_summary(int(day["id"]))
        if summary is not None and str(summary["prompt_version"]) != PROMPT_VERSION:
            summary = None
    if json_output:
        console.print_json(json.dumps(incidents, ensure_ascii=False))
        return
    if summary is not None:
        console.print(str(summary["summary"]))
        console.print()
    console.print(f"{len(incidents)} incidents at priority {min_priority} or higher:")
    for value in incidents:
        location = f" — {value['location']}" if value.get("location") else ""
        console.print(
            f"[{format_archive_time(value, float(value['start_seconds']))}] "
            f"P{value['priority']} {value['event_type']} — {value['title']}"
            f"{location} [I{value['id']}]"
        )


if __name__ == "__main__":
    cli()
