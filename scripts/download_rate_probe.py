from __future__ import annotations

import argparse
import json
import threading
import time
import uuid
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from urllib.parse import urlparse

import requests

from broadcastify_cli.broadcastify import BroadcastifyClient


RATE_HEADERS = (
    "Retry-After",
    "RateLimit-Limit",
    "RateLimit-Remaining",
    "RateLimit-Reset",
    "X-RateLimit-Limit",
    "X-RateLimit-Remaining",
    "X-RateLimit-Reset",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a paced, resumable archive download with JSONL HTTP telemetry."
    )
    parser.add_argument("--feed-id", required=True)
    parser.add_argument("--start-date", type=date.fromisoformat, required=True)
    parser.add_argument("--end-date", type=date.fromisoformat, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("archives"))
    parser.add_argument("--interval", type=float, default=5.0)
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--log", type=Path)
    return parser.parse_args()


def dates(start: date, end: date):
    current = start
    while current <= end:
        yield current
        current += timedelta(days=1)


def main() -> int:
    args = parse_args()
    if args.start_date > args.end_date:
        raise ValueError("start-date must be on or before end-date")
    log_path = args.log or (
        args.output_dir
        / args.feed_id
        / f"rate-probe-{args.start_date:%Y%m%d}-{args.end_date:%Y%m%d}.jsonl"
    )
    log_path.parent.mkdir(parents=True, exist_ok=True)
    session_id = uuid.uuid4().hex
    started = time.monotonic()
    write_lock = threading.Lock()

    with log_path.open("a", encoding="utf-8", buffering=1) as log_handle:
        def emit(event: str, **values: object) -> None:
            payload = {
                "at": datetime.now(timezone.utc).isoformat(),
                "elapsed_seconds": round(time.monotonic() - started, 3),
                "session_id": session_id,
                "event": event,
                **values,
            }
            line = json.dumps(payload, ensure_ascii=False, sort_keys=True)
            with write_lock:
                log_handle.write(line + "\n")
                print(line, flush=True)

        emit(
            "probe_start",
            feed_id=args.feed_id,
            start_date=args.start_date.isoformat(),
            end_date=args.end_date.isoformat(),
            interval_seconds=args.interval,
            jobs=args.jobs,
        )

        with BroadcastifyClient(
            download_request_interval=args.interval
        ) as client:
            def record_response(
                response: requests.Response, *_: object, **__: object
            ) -> None:
                emit(
                    "http_response",
                    method=response.request.method if response.request else None,
                    path=urlparse(response.url).path,
                    status=response.status_code,
                    response_elapsed_seconds=round(
                        response.elapsed.total_seconds(), 3
                    ),
                    rate_headers={
                        name: response.headers[name]
                        for name in RATE_HEADERS
                        if name in response.headers
                    },
                    error_body=(
                        response.text[:500]
                        if response.status_code >= 400
                        and (
                            "text" in response.headers.get("Content-Type", "").lower()
                            or "html"
                            in response.headers.get("Content-Type", "").lower()
                        )
                        else None
                    ),
                )

            client.session.hooks["response"].append(record_response)
            client.authenticate()

            total_files = 0
            for archive_date in dates(args.start_date, args.end_date):
                emit("day_start", archive_date=archive_date.isoformat())

                def progress(current: int, total: int, message: str) -> None:
                    emit(
                        "progress",
                        archive_date=archive_date.isoformat(),
                        current=current,
                        total=total,
                        message=message,
                    )

                try:
                    downloaded = client.download_day(
                        args.feed_id,
                        archive_date,
                        args.output_dir,
                        jobs=args.jobs,
                        progress=progress,
                    )
                except Exception as exc:
                    emit(
                        "probe_error",
                        archive_date=archive_date.isoformat(),
                        error_type=type(exc).__name__,
                        message=str(exc),
                    )
                    raise
                total_files += len(downloaded)
                emit(
                    "day_complete",
                    archive_date=archive_date.isoformat(),
                    files=len(downloaded),
                    bytes=sum(path.stat().st_size for path in downloaded),
                )

        emit("probe_complete", files=total_files, log=str(log_path.resolve()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
