from __future__ import annotations

import argparse
import http.client
import json
import re
import sys
import threading
import time
from pathlib import Path
from typing import Any

from broadcastify_cli.web_app import create_server


TERMINAL_STATUSES = {"completed", "failed", "canceled"}
TOKEN_PATTERN = re.compile(rb'<meta name="app-token" content="([^"]+)">')


def _json_request(
    connection: http.client.HTTPConnection,
    method: str,
    path: str,
    *,
    cookie: str = "",
    token: str = "",
    origin: str = "",
    body: dict[str, Any] | None = None,
) -> tuple[http.client.HTTPResponse, bytes]:
    headers: dict[str, str] = {}
    encoded = None
    if cookie:
        headers["Cookie"] = cookie
    if token:
        headers["X-Radio-Archive-Token"] = token
    if origin:
        headers["Origin"] = origin
    if body is not None:
        headers["Content-Type"] = "application/json"
        encoded = json.dumps(body, ensure_ascii=False)
    connection.request(method, path, body=encoded, headers=headers)
    response = connection.getresponse()
    return response, response.read()


def _decode_response(response: http.client.HTTPResponse, body: bytes) -> dict[str, Any]:
    try:
        value = json.loads(body)
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise RuntimeError(
            f"Local Web API returned HTTP {response.status} with invalid JSON."
        ) from exc
    if not isinstance(value, dict):
        raise RuntimeError(f"Local Web API returned HTTP {response.status} with a non-object body.")
    if response.status >= 400:
        raise RuntimeError(str(value.get("error") or f"Local Web API returned HTTP {response.status}."))
    return value


def run_web_job(
    command: str,
    payload: dict[str, Any],
    *,
    output_dir: str | Path,
    database_path: str | Path | None = None,
    working_dir: str | Path | None = None,
    timeout: float = 300.0,
    poll_interval: float = 0.25,
) -> dict[str, Any]:
    """Submit one job through the real loopback HTTP boundary and wait for it."""

    if timeout <= 0:
        raise ValueError("timeout must be greater than zero")
    if poll_interval <= 0:
        raise ValueError("poll_interval must be greater than zero")
    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    database = (
        Path(database_path).expanduser().resolve()
        if database_path
        else output / "broadcastify-analysis.sqlite3"
    )
    work = Path(working_dir or Path.cwd()).expanduser().resolve()
    server = create_server(output, database, port=0, working_dir=work)
    server.quiet = True  # type: ignore[attr-defined]
    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()
    connection = http.client.HTTPConnection("127.0.0.1", server.server_port, timeout=10)
    cookie = ""
    token = ""
    job_id = ""
    origin = f"http://127.0.0.1:{server.server_port}"
    try:
        response, body = _json_request(connection, "GET", "/")
        if response.status != 200:
            raise RuntimeError(f"Local Web app returned HTTP {response.status} for its start page.")
        cookie = response.getheader("Set-Cookie", "").split(";", 1)[0]
        token_match = TOKEN_PATTERN.search(body)
        if not cookie or token_match is None:
            raise RuntimeError("Local Web app did not establish its protected session.")
        token = token_match.group(1).decode("utf-8")

        response, body = _json_request(
            connection,
            "POST",
            "/api/jobs",
            cookie=cookie,
            token=token,
            origin=origin,
            body={"command": command, "payload": payload},
        )
        job = _decode_response(response, body)
        job_id = str(job.get("id") or "")
        if not job_id:
            raise RuntimeError("Local Web API accepted the job without returning an ID.")

        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            response, body = _json_request(
                connection,
                "GET",
                f"/api/jobs/{job_id}",
                cookie=cookie,
            )
            job = _decode_response(response, body)
            if str(job.get("status") or "") in TERMINAL_STATUSES:
                return job
            time.sleep(poll_interval)

        try:
            _json_request(
                connection,
                "POST",
                f"/api/jobs/{job_id}/cancel",
                cookie=cookie,
                token=token,
                origin=origin,
                body={},
            )
        except (OSError, RuntimeError):
            pass
        raise TimeoutError(f"Local Web job {job_id} did not finish within {timeout:g} seconds.")
    finally:
        connection.close()
        server.shutdown()
        server.server_close()
        server_thread.join(timeout=3)


def _payload(arguments: argparse.Namespace) -> dict[str, Any]:
    if arguments.payload_file:
        value = json.loads(Path(arguments.payload_file).read_text(encoding="utf-8"))
    elif arguments.payload_json:
        value = json.loads(arguments.payload_json)
    else:
        value = {}
    if not isinstance(value, dict):
        raise ValueError("The job payload must be a JSON object.")
    return value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run one worker command through the loopback Web API, including its "
            "cookie/action-token boundary, and emit the final job snapshot."
        )
    )
    parser.add_argument("command", help="Supported Web job command, such as diagnostics or continue-local.")
    payload = parser.add_mutually_exclusive_group()
    payload.add_argument("--payload-file", help="Path to a UTF-8 JSON object used as the job payload.")
    payload.add_argument("--payload-json", help="Inline JSON object used as the job payload.")
    parser.add_argument("--output-dir", default="archives")
    parser.add_argument("--database")
    parser.add_argument("--working-dir")
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--poll-interval", type=float, default=0.25)
    parser.add_argument("--result-file", help="Optional path that receives the final JSON snapshot.")
    return parser


def main() -> int:
    arguments = build_parser().parse_args()
    try:
        result = run_web_job(
            arguments.command,
            _payload(arguments),
            output_dir=arguments.output_dir,
            database_path=arguments.database,
            working_dir=arguments.working_dir,
            timeout=arguments.timeout,
            poll_interval=arguments.poll_interval,
        )
        rendered = json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True)
        if arguments.result_file:
            result_path = Path(arguments.result_file).expanduser().resolve()
            result_path.parent.mkdir(parents=True, exist_ok=True)
            result_path.write_text(rendered + "\n", encoding="utf-8")
        print(rendered)
        return 0 if result.get("status") == "completed" else 1
    except Exception as exc:
        print(
            json.dumps(
                {"status": "harness-error", "error": str(exc)},
                ensure_ascii=False,
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
