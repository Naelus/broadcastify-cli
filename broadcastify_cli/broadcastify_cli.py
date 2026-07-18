from __future__ import annotations

import datetime as dt
from pathlib import Path

import click
from dotenv import load_dotenv
from rich.console import Console

from .broadcastify import BroadcastifyClient
from .jobs import JobRunner
from .models import JobRequest
from .transcription import LocalTranscriber


VERSION = "0.2.0"
console = Console()


def parse_date(value: str) -> dt.date:
    for date_format in ("%Y-%m-%d", "%Y/%m/%d", "%m/%d/%Y"):
        try:
            return dt.datetime.strptime(value, date_format).date()
        except ValueError:
            continue
    raise click.BadParameter("Use YYYY-MM-DD, YYYY/MM/DD, or MM/DD/YYYY.")


def parse_range(value: str) -> tuple[dt.date, dt.date]:
    if ":" in value:
        start, end = value.split(":", 1)
    elif "-" in value and "/" in value:
        start, end = value.split("-", 1)
    else:
        raise click.BadParameter(
            "Use START:END, for example 2026-07-01:2026-07-07."
        )
    return parse_date(start), parse_date(end)


@click.group(context_settings={"help_option_names": ["-h", "--help"]})
@click.version_option(VERSION, message="broadcastify-cli %(version)s")
def cli() -> None:
    """Download and locally transcribe Broadcastify archives."""
    load_dotenv(Path.cwd() / ".env", override=True)


@cli.command("search")
@click.argument("query")
def search(query: str) -> None:
    """Search the Broadcastify website feed directory."""
    with BroadcastifyClient() as client:
        results = client.search_feeds(query)
    for result in results:
        console.print(
            f"[cyan]{result.feed_id:>6}[/cyan]  {result.name}  "
            f"[dim]{result.location} · {result.genre} · {result.listeners} listeners[/dim]"
        )


@cli.command("download", help="Download archives by feed and date selection")
@click.option("--feed-id", "-id", required=True, help="Broadcastify feed ID")
@click.option("--date", "date_value", help="Single date")
@click.option("--range", "range_value", help="Date range as START:END")
@click.option("--past-days", "-p", type=click.IntRange(min=1), help="Include today and the previous N-1 days")
@click.option("--combine", is_flag=True, help="Combine each day before transcription")
@click.option("--keep-originals/--delete-originals", default=True, show_default=True)
@click.option("--transcribe", "-t", is_flag=True, help="Create local timestamped transcripts")
@click.option("--diarize", is_flag=True, help="Assign local pyannote speaker labels")
@click.option(
    "--asr-engine",
    type=click.Choice(
        [
            "auto",
            "faster-whisper",
            "whisper.cpp",
            "openvino",
            "windows-ml",
            "qwen3-asr",
        ]
    ),
    default="auto",
    show_default=True,
)
@click.option(
    "--device",
    type=click.Choice(
        [
            "auto", "cpu", "cuda", "vulkan", "metal", "openvino-auto",
            "openvino-cpu", "openvino-gpu", "openvino-npu", "windows-ml", "directml",
        ]
    ),
    default="auto",
    show_default=True,
)
@click.option("--device-index", type=click.IntRange(min=0), default=0, show_default=True)
@click.option("--compute-type", default="auto", show_default=True)
@click.option("--asr-model-path", type=click.Path(path_type=Path, dir_okay=True))
@click.option(
    "--diarization-device",
    type=click.Choice(["auto", "cpu", "cuda"]),
    default="auto",
    show_default=True,
)
@click.option(
    "--model",
    "model_name",
    type=click.Choice(
        [
            "tiny",
            "base",
            "small",
            "medium",
            "large-v3",
            "turbo",
            "distil-large-v3",
            "qwen3-asr-0.6b-int8",
        ]
    ),
    default="turbo",
    show_default=True,
)
@click.option("--jobs", "download_jobs", type=click.IntRange(min=1, max=32), default=1, show_default=True)
@click.option("--batch-size", type=click.IntRange(min=1, max=64), default=8, show_default=True)
@click.option("--min-speakers", type=click.IntRange(min=1), help="Minimum diarization speakers")
@click.option("--max-speakers", type=click.IntRange(min=1), help="Maximum diarization speakers")
@click.option("--output-dir", "-o", type=click.Path(path_type=Path, file_okay=False), default=Path("archives"), show_default=True)
def download(
    feed_id: str,
    date_value: str | None,
    range_value: str | None,
    past_days: int | None,
    combine: bool,
    keep_originals: bool,
    transcribe: bool,
    diarize: bool,
    asr_engine: str,
    device: str,
    device_index: int,
    compute_type: str,
    asr_model_path: Path | None,
    diarization_device: str,
    model_name: str,
    download_jobs: int,
    batch_size: int,
    min_speakers: int | None,
    max_speakers: int | None,
    output_dir: Path,
) -> None:
    today = dt.date.today()
    if date_value:
        start_date = end_date = parse_date(date_value)
    elif range_value:
        start_date, end_date = parse_range(range_value)
    elif past_days:
        start_date = today - dt.timedelta(days=past_days - 1)
        end_date = today
    else:
        start_date = today - dt.timedelta(days=364)
        end_date = today

    request = JobRequest(
        feed_id=feed_id,
        start_date=start_date,
        end_date=end_date,
        output_dir=output_dir,
        combine=combine,
        keep_originals=keep_originals,
        transcribe=transcribe,
        diarize=diarize,
        model=model_name,
        asr_engine=asr_engine,
        device=device,
        device_index=device_index,
        compute_type=compute_type,
        asr_model_path=str(asr_model_path) if asr_model_path else None,
        diarization_device=diarization_device,
        download_jobs=download_jobs,
        batch_size=batch_size,
        min_speakers=min_speakers,
        max_speakers=max_speakers,
    )

    def show_event(event: dict[str, object]) -> None:
        if event.get("type") in {"log", "stage", "progress", "complete"}:
            console.print(str(event.get("message", "")))

    with BroadcastifyClient() as client:
        JobRunner(request, emit=show_event, client=client).run()


@cli.command("transcribe", help="Transcribe existing MP3 files in a directory")
@click.option("--directory", "-d", type=click.Path(path_type=Path, exists=True, file_okay=False), required=True)
@click.option("--diarize", is_flag=True)
@click.option(
    "--asr-engine",
    type=click.Choice(
        [
            "auto",
            "faster-whisper",
            "whisper.cpp",
            "openvino",
            "windows-ml",
            "qwen3-asr",
        ]
    ),
    default="auto",
)
@click.option(
    "--device",
    type=click.Choice(
        [
            "auto", "cpu", "cuda", "vulkan", "metal", "openvino-auto",
            "openvino-cpu", "openvino-gpu", "openvino-npu", "windows-ml", "directml",
        ]
    ),
    default="auto",
)
@click.option("--device-index", type=click.IntRange(min=0), default=0)
@click.option("--compute-type", default="auto")
@click.option("--asr-model-path", type=click.Path(path_type=Path, dir_okay=True))
@click.option(
    "--diarization-device",
    type=click.Choice(["auto", "cpu", "cuda"]),
    default="auto",
)
@click.option("--model", "model_name", default="turbo")
@click.option("--batch-size", type=click.IntRange(min=1, max=64), default=8)
@click.option("--min-speakers", type=click.IntRange(min=1))
@click.option("--max-speakers", type=click.IntRange(min=1))
@click.option("--combined", is_flag=True, help="Transcribe combined MP3s instead of source blocks")
def transcribe(
    directory: Path,
    diarize: bool,
    asr_engine: str,
    device: str,
    device_index: int,
    compute_type: str,
    asr_model_path: Path | None,
    diarization_device: str,
    model_name: str,
    batch_size: int,
    min_speakers: int | None,
    max_speakers: int | None,
    combined: bool,
) -> None:
    if min_speakers is not None and max_speakers is not None and min_speakers > max_speakers:
        raise click.BadParameter("Minimum speakers cannot exceed maximum speakers.")
    files = sorted(
        path
        for path in directory.glob("*.mp3")
        if path.name.lower().startswith("combined_") == combined
    )
    transcriber = LocalTranscriber(
        model_name=model_name,
        asr_engine=asr_engine,
        device=device,
        device_index=device_index,
        compute_type=compute_type,
        asr_model_path=asr_model_path,
        diarization_device=diarization_device,
        diarize=diarize,
        batch_size=batch_size,
        min_speakers=min_speakers,
        max_speakers=max_speakers,
    )

    def progress(current: int, total: int, message: str) -> None:
        console.print(message)

    transcriber.transcribe_files(files, progress=progress)


if __name__ == "__main__":
    cli()
