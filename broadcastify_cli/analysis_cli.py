from __future__ import annotations

import json
from contextlib import nullcontext
from datetime import date, datetime
from pathlib import Path

import click
from dotenv import load_dotenv
from rich.console import Console

from .analysis import (
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_LLM_MODEL,
    PROMPT_VERSION,
    IncidentAnalyzer,
    LlamaCppClient,
    LlamaServerProcess,
    RangeQuestionAnswerer,
    SemanticIndexer,
    WeeklySummaryAnalyzer,
    discover_day_paths,
    format_archive_time,
)
from .storage import AnalysisStore


console = Console()


def parse_date(value: str) -> date:
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise click.BadParameter("Use YYYY-MM-DD.") from exc


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
@click.option("--model", default=DEFAULT_LLM_MODEL, show_default=True)
@click.option("--server-url", help="Use an already-running llama.cpp /v1 endpoint")
@click.option("--embedding-model", default=DEFAULT_EMBEDDING_MODEL, show_default=True)
@click.option("--embeddings/--no-embeddings", default=True, show_default=True)
@click.option("--force", is_flag=True, help="Re-run extraction and summary even if cached")
@click.option("--force-summary", is_flag=True, help="Refresh only the daily summary")
def analyze_day(
    feed_id: str,
    date_value: date,
    output_dir: Path,
    db: Path,
    model: str,
    server_url: str | None,
    embedding_model: str,
    embeddings: bool,
    force: bool,
    force_summary: bool,
) -> None:
    """Import and classify one day, then create its end-of-day summary."""
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
                int(day["id"]), model, PROMPT_VERSION, str(day["transcript_sha256"])
            )
        )
        server_context = (
            nullcontext(None)
            if server_url or (complete and not force and not force_summary)
            else LlamaServerProcess(model=model)
        )
        with server_context as server:
            base_url = server_url or (server.base_url if server else "http://127.0.0.1:8088/v1")
            client = LlamaCppClient(base_url=base_url, model=model)
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
@click.option("--start-date", required=True, callback=lambda _c, _p, v: parse_date(v))
@click.option("--end-date", required=True, callback=lambda _c, _p, v: parse_date(v))
@click.option("--question", required=True)
@click.option("--db", type=click.Path(path_type=Path), default=Path("archives/broadcastify-analysis.sqlite3"))
@click.option("--model", default=DEFAULT_LLM_MODEL, show_default=True)
@click.option("--server-url", help="Use an already-running llama.cpp /v1 endpoint")
@click.option("--embedding-model", default=DEFAULT_EMBEDDING_MODEL, show_default=True)
@click.option("--semantic/--keyword-only", default=True, show_default=True)
def ask(
    feed_id: str,
    start_date: date,
    end_date: date,
    question: str,
    db: Path,
    model: str,
    server_url: str | None,
    embedding_model: str,
    semantic: bool,
) -> None:
    """Ask an evidence-grounded question over an imported date range."""
    if start_date > end_date:
        raise click.BadParameter("Start date must be on or before end date.")
    with AnalysisStore(db) as store:
        indexer = None
        if semantic:
            indexer = SemanticIndexer(store, model=embedding_model)
            indexed = indexer.index_missing()
            if indexed:
                console.print(f"Indexed {indexed} new transcript passages.")
        server_context = (
            nullcontext(None)
            if server_url
            else LlamaServerProcess(model=model)
        )
        with server_context as server:
            base_url = server_url or server.base_url
            result = RangeQuestionAnswerer(
                store,
                LlamaCppClient(base_url=base_url, model=model),
                indexer=indexer,
            ).ask(feed_id, start_date, end_date, question)
    console.print(result["answer"])
    if result["limitations"]:
        console.print("[dim]Limitations: " + "; ".join(result["limitations"]) + "[/dim]")


@cli.command("summarize-week")
@click.option("--feed-id", required=True)
@click.option("--week-ending", required=True, callback=lambda _c, _p, v: parse_date(v))
@click.option("--db", type=click.Path(path_type=Path), default=Path("archives/broadcastify-analysis.sqlite3"))
@click.option("--model", default=DEFAULT_LLM_MODEL, show_default=True)
@click.option("--server-url", help="Use an already-running llama.cpp /v1 endpoint")
@click.option("--force", is_flag=True, help="Regenerate even when the source data is unchanged")
@click.option("--json-output", is_flag=True, help="Emit the complete weekly report as JSON")
def summarize_week(
    feed_id: str,
    week_ending: date,
    db: Path,
    model: str,
    server_url: str | None,
    force: bool,
    json_output: bool,
) -> None:
    """Summarize the seven-day period ending on the selected date."""
    with AnalysisStore(db) as store:
        server_context = nullcontext(None) if server_url else LlamaServerProcess(model=model)
        with server_context as server:
            base_url = server_url or server.base_url
            result = WeeklySummaryAnalyzer(
                store,
                LlamaCppClient(base_url=base_url, model=model),
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
        incidents = [
            value
            for value in store.get_incidents(feed_id, date_value, date_value)
            if int(value["priority"]) >= min_priority
        ]
        summary = store.get_daily_summary(
            int(day["id"]),
            DEFAULT_LLM_MODEL,
            PROMPT_VERSION,
            str(day["transcript_sha256"]),
        )
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
