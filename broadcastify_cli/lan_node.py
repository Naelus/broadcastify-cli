from __future__ import annotations

import argparse
import hmac
import ipaddress
import json
import os
import re
import socket
import threading
import time
from datetime import date
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

from dotenv import load_dotenv

from .lan_sync import (
    LAN_PROTOCOL,
    LAN_QUEUE_REQUEST_BYTES,
    LanArchiveCatalog,
    LanArchiveReconciler,
    LanArchiveSyncClient,
    LanDiscoveryResponder,
    LanSyncError,
    environment_flag,
    environment_float,
    normalize_peer_url,
    normalize_peer_urls,
)
from .quota import (
    ArchiveRequestBudgetExceeded,
    ArchiveRequestLedger,
    normalize_account_profile_id,
    normalize_archive_request_id,
)
from .pipeline_sync import PipelineSyncStore, normalize_pipeline_role
from .storage import AnalysisStore
from .web_app import (
    JobManager,
    WebRequestError,
    _account_pool_profiles,
    _select_account_profile,
)


def validate_lan_host(value: str) -> str:
    host = str(value or "").strip()
    try:
        address = ipaddress.ip_address(host)
    except ValueError as exc:
        raise ValueError(
            "The LAN node host must be a numeric loopback, private, link-local, "
            "or wildcard address."
        ) from exc
    if address.is_multicast or not (
        address.is_loopback
        or address.is_private
        or address.is_link_local
        or address.is_unspecified
    ):
        raise ValueError("The LAN node refuses a public or multicast bind address.")
    return host


def _date_value(value: str) -> date:
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise ValueError("Use an ISO archive date such as 2026-07-19.") from exc


def _load_environment() -> None:
    load_dotenv(Path.cwd() / ".env", override=True)
    load_dotenv(Path.cwd() / ".env.accounts", override=True)
    configured = os.getenv("BROADCASTIFY_ENV_FILE")
    if configured:
        path = Path(configured).expanduser()
        if path.is_file():
            load_dotenv(path, override=True)


def _windows_wait_for_process_exit(parent_pid: int) -> None:
    import ctypes
    from ctypes import wintypes

    synchronize = 0x00100000
    infinite = 0xFFFFFFFF
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    kernel32.OpenProcess.restype = wintypes.HANDLE
    kernel32.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
    kernel32.WaitForSingleObject.restype = wintypes.DWORD
    kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
    kernel32.CloseHandle.restype = wintypes.BOOL
    handle = kernel32.OpenProcess(synchronize, False, int(parent_pid))
    if not handle:
        return
    try:
        kernel32.WaitForSingleObject(handle, infinite)
    finally:
        kernel32.CloseHandle(handle)


def _wait_for_parent_exit(parent_pid: int) -> None:
    if os.name == "nt":
        _windows_wait_for_process_exit(parent_pid)
        return
    while True:
        try:
            os.kill(parent_pid, 0)
        except ProcessLookupError:
            return
        except PermissionError:
            return
        time.sleep(2.0)


def _start_parent_watchdog(
    server: ThreadingHTTPServer,
    parent_pid: int | None,
) -> None:
    if parent_pid is None or parent_pid <= 0 or parent_pid == os.getpid():
        return

    def watch() -> None:
        _wait_for_parent_exit(parent_pid)
        server.shutdown()

    threading.Thread(
        target=watch,
        name="lan-node-parent-watchdog",
        daemon=True,
    ).start()


def create_lan_node_server(
    output_dir: str | Path,
    *,
    host: str = "0.0.0.0",
    port: int = 8_766,
    sync_key: str | None = None,
    peer_urls: str | tuple[str, ...] | list[str] | None = None,
    discovery_enabled: bool = True,
    advertise_url: str = "",
    background_sync_enabled: bool = False,
    role: str | None = None,
    working_dir: str | Path | None = None,
    database_path: str | Path | None = None,
    job_relay_enabled: bool = False,
) -> ThreadingHTTPServer:
    host = validate_lan_host(host)
    pipeline_role = normalize_pipeline_role(
        role if role is not None else os.getenv("BROADCASTIFY_LAN_ROLE") or "master"
    )
    with PipelineSyncStore(output_dir) as pipeline_store:
        pipeline_node_id = pipeline_store.node_id()
    catalog = LanArchiveCatalog(
        output_dir,
        enabled=True,
        sync_key=(
            str(sync_key)
            if sync_key is not None
            else os.getenv("BROADCASTIFY_LAN_SYNC_KEY") or ""
        ),
        peer_urls=normalize_peer_urls(
            peer_urls
            if peer_urls is not None
            else os.getenv("BROADCASTIFY_LAN_PEERS")
        ),
        queue_enabled=environment_flag(
            "BROADCASTIFY_LAN_QUEUE_ENABLED",
            default=True,
        ),
        role=pipeline_role,
        node_id=pipeline_node_id,
    )
    if job_relay_enabled and pipeline_role != "master":
        raise ValueError("Only the Windows master may host relayed Web jobs.")
    relay_working_dir = Path(working_dir or Path.cwd()).expanduser().resolve()
    relay_database_path = (
        Path(database_path).expanduser().resolve()
        if database_path
        else (catalog.output_dir / "broadcastify-analysis.sqlite3").resolve()
    )
    job_manager = (
        JobManager(
            catalog.output_dir,
            relay_database_path,
            relay_working_dir,
        )
        if job_relay_enabled
        else None
    )

    class Handler(BaseHTTPRequestHandler):
        server_version = "RadioArchiveLAN/1"

        def log_message(self, format_value: str, *args: object) -> None:
            if getattr(self.server, "quiet", False):
                return
            super().log_message(format_value, *args)

        def do_GET(self) -> None:  # noqa: N802
            try:
                self._get()
            except WebRequestError as exc:
                self._json(exc.status, {"error": str(exc)})
            except PermissionError as exc:
                self._json(HTTPStatus.FORBIDDEN, {"error": str(exc)})
            except FileNotFoundError:
                self._json(HTTPStatus.NOT_FOUND, {"error": "Archive block not found."})
            except (LanSyncError, ValueError) as exc:
                self._json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
            except (BrokenPipeError, ConnectionResetError):
                return
            except Exception:
                self._json(
                    HTTPStatus.INTERNAL_SERVER_ERROR,
                    {"error": "The read-only LAN archive node hit an unexpected error."},
                )

        def do_POST(self) -> None:  # noqa: N802
            try:
                self._authorize()
                self._post()
            except WebRequestError as exc:
                self._json(exc.status, {"error": str(exc)})
            except ArchiveRequestBudgetExceeded as exc:
                self._json(HTTPStatus.TOO_MANY_REQUESTS, {"error": str(exc)})
            except PermissionError as exc:
                self._json(HTTPStatus.FORBIDDEN, {"error": str(exc)})
            except FileNotFoundError:
                self._json(HTTPStatus.NOT_FOUND, {"error": "LAN queue not found."})
            except (LanSyncError, ValueError) as exc:
                self._json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
            except (BrokenPipeError, ConnectionResetError):
                return
            except Exception:
                self._json(
                    HTTPStatus.INTERNAL_SERVER_ERROR,
                    {"error": "The LAN acquisition coordinator hit an unexpected error."},
                )

        def _get(self) -> None:
            parsed = urlparse(self.path)
            if parsed.path == "/health":
                self._json(
                    HTTPStatus.OK,
                    {
                        "status": "ok",
                        "protocol": LAN_PROTOCOL,
                        "sharing": True,
                    },
                )
                return
            self._authorize()
            if parsed.path == "/api/lan/v1/library":
                if job_manager is None:
                    raise FileNotFoundError
                with AnalysisStore(relay_database_path) as store:
                    transcript_spans = store.list_feed_spans()
                    schedules = store.list_feed_schedules()
                    catchups = store.list_library_catchups()
                with PipelineSyncStore(catalog.output_dir) as journal:
                    source_spans = journal.list_source_spans()
                spans: dict[str, dict[str, object]] = {}
                for raw in [*source_spans, *transcript_spans]:
                    feed_id = str(raw.get("feed_id") or "")
                    if not feed_id.isdigit():
                        continue
                    current = spans.setdefault(
                        feed_id,
                        {
                            "feed_id": feed_id,
                            "feed_name": f"Feed {feed_id}",
                            "start_date": str(raw.get("start_date") or ""),
                            "end_date": str(raw.get("end_date") or ""),
                        },
                    )
                    current["start_date"] = min(
                        str(current.get("start_date") or "9999-12-31"),
                        str(raw.get("start_date") or "9999-12-31"),
                    )
                    current["end_date"] = max(
                        str(current.get("end_date") or ""),
                        str(raw.get("end_date") or ""),
                    )
                    feed_name = str(raw.get("feed_name") or "").strip()
                    if feed_name:
                        current["feed_name"] = feed_name[:200]
                library = {
                    "feed_spans": [spans[key] for key in sorted(spans)],
                    "schedules": schedules,
                    "catchups": catchups,
                    "account_pool": _account_pool_profiles(
                        relay_working_dir,
                        job_manager.credential_store,
                    ),
                }
                self._json(
                    HTTPStatus.OK,
                    {"protocol": LAN_PROTOCOL, "library": library},
                )
                return
            job_prefix = "/api/lan/v1/jobs/"
            if parsed.path.startswith(job_prefix):
                if job_manager is None:
                    raise FileNotFoundError
                job_id = unquote(parsed.path.removeprefix(job_prefix))
                if not re.fullmatch(r"[0-9a-f]{16}", job_id):
                    raise LanSyncError("The relayed job ID is not valid.")
                self._json(
                    HTTPStatus.OK,
                    {
                        "protocol": LAN_PROTOCOL,
                        "job": job_manager.get(job_id),
                    },
                )
                return
            if parsed.path == "/api/lan/v1/info":
                info = catalog.info()
                reconciler = getattr(self.server, "lan_reconciler", None)
                if reconciler is not None:
                    info["reconciliation"] = reconciler.status()
                if job_manager is not None:
                    with AnalysisStore(relay_database_path) as store:
                        schedules = store.list_feed_schedules()
                    info["job_relay"] = job_manager.status()
                    info["schedules"] = schedules[:100]
                    info["account_pool"] = _account_pool_profiles(
                        relay_working_dir,
                        job_manager.credential_store,
                    )
                else:
                    info["job_relay"] = {"enabled": False, "active": None}
                    info["schedules"] = []
                self._json(HTTPStatus.OK, info)
                return
            if parsed.path == "/api/lan/v1/changes":
                query = parse_qs(parsed.query)
                try:
                    after = max(0, int((query.get("after") or ["0"])[0]))
                    limit = min(512, max(1, int((query.get("limit") or ["128"])[0])))
                except ValueError as exc:
                    raise LanSyncError("The change cursor is not valid.") from exc
                with PipelineSyncStore(catalog.output_dir) as journal:
                    journal.seed_sources()
                    changes = journal.changes(after, limit=limit)
                self._json(
                    HTTPStatus.OK,
                    {
                        "protocol": LAN_PROTOCOL,
                        "node_id": catalog.node_id,
                        "role": catalog.role,
                        **changes,
                    },
                )
                return
            if parsed.path == "/api/lan/v1/feed-days":
                query = parse_qs(parsed.query)
                feed_id = str((query.get("feed_id") or [""])[0]).strip()
                self._json(
                    HTTPStatus.OK,
                    {
                        "protocol": LAN_PROTOCOL,
                        "node_id": catalog.node_id,
                        "feed_id": feed_id,
                        "dates": [
                            value.isoformat() for value in catalog.feed_dates(feed_id)
                        ],
                        "peers": list(catalog.peer_urls),
                    },
                )
                return
            if parsed.path == "/api/lan/v1/blocks":
                query = parse_qs(parsed.query)
                feed_id = str((query.get("feed_id") or [""])[0]).strip()
                archive_date = _date_value(
                    str((query.get("date") or [""])[0]).strip()
                )
                blocks = catalog.inventory(feed_id, archive_date)
                complete, completion_blocks = catalog.completion_inventory(
                    feed_id,
                    archive_date,
                )
                self._json(
                    HTTPStatus.OK,
                    {
                        "protocol": LAN_PROTOCOL,
                        "node_id": catalog.node_id,
                        "feed_id": feed_id,
                        "archive_date": archive_date.isoformat(),
                        "blocks": [block.to_dict() for block in blocks],
                        "complete": complete,
                        "completion_blocks": [
                            block.to_dict() for block in completion_blocks
                        ],
                        "peers": list(catalog.peer_urls),
                    },
                )
                return
            if parsed.path == "/api/lan/v1/acquisition":
                if not catalog.acquisition_queue.enabled:
                    raise FileNotFoundError
                query = parse_qs(parsed.query)
                feed_id = str((query.get("feed_id") or [""])[0]).strip()
                archive_date = _date_value(
                    str((query.get("date") or [""])[0]).strip()
                )
                quota_scope = str(
                    (query.get("quota_scope") or ["default"])[0]
                ).strip()
                self._json(
                    HTTPStatus.OK,
                    catalog.acquisition_queue.status(
                        quota_scope,
                        feed_id,
                        archive_date,
                    ),
                )
                return
            if parsed.path == "/api/lan/v1/quota":
                query = parse_qs(parsed.query)
                profile_id = normalize_account_profile_id(
                    str(
                        (query.get("account_profile_id") or ["default"])[0]
                    )
                )
                self._json(
                    HTTPStatus.OK,
                    {
                        "protocol": LAN_PROTOCOL,
                        **ArchiveRequestLedger(
                            account_profile_id=profile_id
                        ).status(),
                    },
                )
                return
            if parsed.path == "/api/lan/v1/transcript-fingerprints":
                query = parse_qs(parsed.query)
                feed_id = str((query.get("feed_id") or [""])[0]).strip()
                archive_date = _date_value(
                    str((query.get("date") or [""])[0]).strip()
                )
                fingerprints = catalog.transcript_fingerprints(
                    feed_id,
                    archive_date,
                )
                self._json(
                    HTTPStatus.OK,
                    {
                        "protocol": LAN_PROTOCOL,
                        "node_id": catalog.node_id,
                        "feed_id": feed_id,
                        "archive_date": archive_date.isoformat(),
                        "processing_fingerprints": list(fingerprints),
                        "peers": list(catalog.peer_urls),
                    },
                )
                return
            if parsed.path == "/api/lan/v1/transcripts":
                query = parse_qs(parsed.query)
                feed_id = str((query.get("feed_id") or [""])[0]).strip()
                archive_date = _date_value(
                    str((query.get("date") or [""])[0]).strip()
                )
                fingerprint = str(
                    (query.get("processing_fingerprint") or [""])[0]
                ).strip()
                artifacts = catalog.transcript_inventory(
                    feed_id,
                    archive_date,
                    fingerprint,
                )
                self._json(
                    HTTPStatus.OK,
                    {
                        "protocol": LAN_PROTOCOL,
                        "node_id": catalog.node_id,
                        "feed_id": feed_id,
                        "archive_date": archive_date.isoformat(),
                        "processing_fingerprint": fingerprint,
                        "artifacts": [value.to_dict() for value in artifacts],
                        "peers": list(catalog.peer_urls),
                    },
                )
                return
            if parsed.path == "/api/lan/v1/processing":
                if not catalog.processing_queue.enabled:
                    raise FileNotFoundError
                query = parse_qs(parsed.query)
                feed_id = str((query.get("feed_id") or [""])[0]).strip()
                archive_date = _date_value(
                    str((query.get("date") or [""])[0]).strip()
                )
                fingerprint = str(
                    (query.get("processing_fingerprint") or [""])[0]
                ).strip()
                self._json(
                    HTTPStatus.OK,
                    catalog.processing_queue.status(
                        fingerprint,
                        feed_id,
                        archive_date,
                    ),
                )
                return
            if parsed.path.startswith("/api/lan/v1/blocks/"):
                parts = parsed.path.removeprefix("/api/lan/v1/blocks/").split("/")
                if len(parts) != 3:
                    raise FileNotFoundError
                feed_id, raw_date, filename = (unquote(value) for value in parts)
                path, block = catalog.resolve_block(
                    feed_id,
                    _date_value(raw_date),
                    filename,
                )
                self._block(path, block.sha256)
                return
            if parsed.path.startswith("/api/lan/v1/transcripts/"):
                parts = parsed.path.removeprefix(
                    "/api/lan/v1/transcripts/"
                ).split("/")
                if len(parts) != 4:
                    raise FileNotFoundError
                feed_id, raw_date, fingerprint, filename = (
                    unquote(value) for value in parts
                )
                path, artifact = catalog.resolve_transcript_artifact(
                    feed_id,
                    _date_value(raw_date),
                    fingerprint,
                    filename,
                )
                self._artifact(path, artifact.sha256)
                return
            raise FileNotFoundError

        def _post(self) -> None:
            parsed = urlparse(self.path)
            if parsed.path == "/api/lan/v1/jobs":
                if job_manager is None:
                    raise FileNotFoundError
                body = self._body()
                command = str(body.get("command") or "").strip()
                payload = body.get("payload") or {}
                if not isinstance(payload, dict):
                    raise LanSyncError("The relayed job payload must be an object.")
                self._json(
                    HTTPStatus.ACCEPTED,
                    {
                        "protocol": LAN_PROTOCOL,
                        "job": job_manager.start(command, payload),
                    },
                )
                return
            job_prefix = "/api/lan/v1/jobs/"
            if parsed.path.startswith(job_prefix) and parsed.path.endswith("/cancel"):
                if job_manager is None:
                    raise FileNotFoundError
                job_id = unquote(
                    parsed.path.removeprefix(job_prefix).removesuffix("/cancel")
                )
                if not re.fullmatch(r"[0-9a-f]{16}", job_id):
                    raise LanSyncError("The relayed job ID is not valid.")
                self._body()
                self._json(
                    HTTPStatus.OK,
                    {
                        "protocol": LAN_PROTOCOL,
                        "job": job_manager.cancel(job_id),
                    },
                )
                return
            if parsed.path == "/api/lan/v1/schedules":
                if job_manager is None:
                    raise FileNotFoundError
                body = self._body()
                requested_profile_id = str(
                    body.get("account_profile_id") or "automatic"
                ).strip().lower()
                if requested_profile_id != "automatic":
                    _select_account_profile(
                        relay_working_dir,
                        job_manager.credential_store,
                        requested_profile_id,
                    )
                with AnalysisStore(relay_database_path) as store:
                    schedule = store.save_feed_schedule(body)
                self._json(
                    HTTPStatus.OK,
                    {"protocol": LAN_PROTOCOL, "schedule": schedule},
                )
                return
            schedule_prefix = "/api/lan/v1/schedules/"
            if parsed.path.startswith(schedule_prefix) and parsed.path.endswith("/delete"):
                if job_manager is None:
                    raise FileNotFoundError
                value = parsed.path.removeprefix(schedule_prefix).removesuffix("/delete")
                try:
                    schedule_id = int(value)
                except ValueError as exc:
                    raise LanSyncError("A numeric schedule ID is required.") from exc
                self._body()
                with AnalysisStore(relay_database_path) as store:
                    deleted = store.delete_feed_schedule(schedule_id)
                if not deleted:
                    raise WebRequestError(
                        HTTPStatus.NOT_FOUND,
                        "That feed schedule was not found on the Windows master.",
                    )
                self._json(
                    HTTPStatus.OK,
                    {"protocol": LAN_PROTOCOL, "deleted": True},
                )
                return
            if parsed.path == "/api/lan/v1/catchups":
                if job_manager is None:
                    raise FileNotFoundError
                body = self._body()
                action = str(body.get("action") or "save").strip().lower()
                with AnalysisStore(relay_database_path) as store:
                    if action == "save":
                        catchup = store.save_library_catchup(
                            {
                                **body,
                                "through_current": True,
                                "end_date": "",
                            }
                        )
                        result: dict[str, object] = {"catchup": catchup}
                    elif action == "clear":
                        result = {
                            "deleted": store.delete_library_catchup(
                                str(body.get("feed_id") or "")
                            )
                        }
                    else:
                        raise LanSyncError("Unknown catch-up action.")
                self._json(
                    HTTPStatus.OK,
                    {"protocol": LAN_PROTOCOL, **result},
                )
                return
            if parsed.path == "/api/lan/v1/pipeline-request":
                if catalog.role != "master":
                    raise LanSyncError("Pipeline requests must be sent to the master.")
                body = self._body()
                with PipelineSyncStore(catalog.output_dir) as journal:
                    request = journal.request(
                        body.get("feed_id"),
                        body.get("start_date"),
                        body.get("end_date"),
                        requester_node_id=str(body.get("requester_node_id") or ""),
                        feed_name=str(body.get("feed_name") or ""),
                    )
                self._json(
                    HTTPStatus.ACCEPTED,
                    {
                        "protocol": LAN_PROTOCOL,
                        "accepted": True,
                        "request_id": int(request["id"]),
                        "state": str(request["state"]),
                    },
                )
                return
            quota_prefix = "/api/lan/v1/quota/"
            if parsed.path.startswith(quota_prefix):
                self._post_quota(parsed.path.removeprefix(quota_prefix))
                return
            processing_prefix = "/api/lan/v1/processing/"
            if parsed.path.startswith(processing_prefix):
                self._post_processing(
                    parsed.path.removeprefix(processing_prefix)
                )
                return
            prefix = "/api/lan/v1/acquisition/"
            if (
                not catalog.acquisition_queue.enabled
                or not parsed.path.startswith(prefix)
            ):
                raise FileNotFoundError
            action = parsed.path.removeprefix(prefix)
            body = self._body()
            feed_id = str(body.get("feed_id") or "").strip()
            archive_date = _date_value(
                str(body.get("archive_date") or "").strip()
            )
            quota_scope = str(body.get("quota_scope") or "default").strip()
            if action == "claim":
                owner_node_id = str(body.get("owner_node_id") or "")
                producer_url = str(body.get("producer_url") or "")
                value = catalog.acquisition_queue.claim(
                    quota_scope,
                    feed_id,
                    archive_date,
                    owner_node_id=owner_node_id,
                    producer_url=producer_url,
                    requester_address=str(self.client_address[0]),
                    allow_multihomed_self=bool(
                        configured_advertisement
                        and owner_node_id == catalog.node_id
                        and normalize_peer_url(producer_url)
                        == configured_advertisement
                    ),
                )
            elif action == "renew":
                value = catalog.acquisition_queue.renew(
                    quota_scope,
                    feed_id,
                    archive_date,
                    lease_token=str(body.get("lease_token") or ""),
                )
            elif action == "finish":
                value = catalog.acquisition_queue.finish(
                    quota_scope,
                    feed_id,
                    archive_date,
                    lease_token=str(body.get("lease_token") or ""),
                    outcome=str(body.get("outcome") or ""),
                    block_count=int(body.get("block_count") or 0),
                    blocks=body.get("blocks") or (),  # type: ignore[arg-type]
                    retry_after_seconds=body.get("retry_after_seconds"),  # type: ignore[arg-type]
                )
            else:
                raise FileNotFoundError
            self._json(HTTPStatus.OK, value)

        def _post_processing(self, action: str) -> None:
            if not catalog.processing_queue.enabled:
                raise FileNotFoundError
            body = self._body()
            feed_id = str(body.get("feed_id") or "").strip()
            archive_date = _date_value(
                str(body.get("archive_date") or "").strip()
            )
            fingerprint = str(
                body.get("processing_fingerprint") or ""
            ).strip()
            if action == "claim":
                owner_node_id = str(body.get("owner_node_id") or "")
                producer_url = str(body.get("producer_url") or "")
                value = catalog.processing_queue.claim(
                    fingerprint,
                    feed_id,
                    archive_date,
                    owner_node_id=owner_node_id,
                    producer_url=producer_url,
                    requester_address=str(self.client_address[0]),
                    allow_multihomed_self=bool(
                        configured_advertisement
                        and owner_node_id == catalog.node_id
                        and normalize_peer_url(producer_url)
                        == configured_advertisement
                    ),
                )
            elif action == "renew":
                value = catalog.processing_queue.renew(
                    fingerprint,
                    feed_id,
                    archive_date,
                    lease_token=str(body.get("lease_token") or ""),
                )
            elif action == "finish":
                value = catalog.processing_queue.finish(
                    fingerprint,
                    feed_id,
                    archive_date,
                    lease_token=str(body.get("lease_token") or ""),
                    outcome=str(body.get("outcome") or ""),
                    artifact_count=int(body.get("artifact_count") or 0),
                )
            else:
                raise FileNotFoundError
            self._json(HTTPStatus.OK, value)

        def _post_quota(self, action: str) -> None:
            body = self._body()
            profile_id = normalize_account_profile_id(
                str(body.get("account_profile_id") or "default")
            )
            ledger = ArchiveRequestLedger(account_profile_id=profile_id)
            if action == "reserve":
                feed_id = str(body.get("feed_id") or "").strip()
                archive_date = str(body.get("archive_date") or "").strip()
                archive_id = normalize_archive_request_id(body.get("archive_id"))
                if not feed_id.isdigit():
                    raise ValueError("A numeric feed ID is required.")
                _date_value(archive_date)
                value: dict[str, object] = {
                    "request_id": ledger.reserve(
                        feed_id=feed_id,
                        archive_date=archive_date,
                        archive_id=archive_id,
                    )
                }
            elif action == "finish":
                request_id = int(body.get("request_id") or 0)
                if request_id < 1:
                    raise ValueError("A positive quota request ID is required.")
                raw_status = body.get("http_status")
                http_status = int(raw_status) if raw_status is not None else None
                if http_status is not None and not 100 <= http_status <= 599:
                    raise ValueError("The archive HTTP status is not valid.")
                ledger.finish(
                    request_id,
                    outcome=str(body.get("outcome") or "")[:80],
                    http_status=http_status,
                )
                value = ledger.status()
            elif action == "rate-limit":
                reason = str(body.get("reason") or "").strip()
                if not reason:
                    raise ValueError("A rate-limit reason is required.")
                value = ledger.mark_rate_limited(reason[:800])
            else:
                raise FileNotFoundError
            self._json(
                HTTPStatus.OK,
                {"protocol": LAN_PROTOCOL, **value},
            )

        def _authorize(self) -> None:
            supplied = self.headers.get("X-Radio-Archive-LAN-Key", "")
            if catalog.sync_key and not hmac.compare_digest(
                catalog.sync_key,
                supplied,
            ):
                raise PermissionError("The LAN archive key is missing or invalid.")

        def _body(self) -> dict[str, object]:
            content_type = (
                self.headers.get("Content-Type", "")
                .split(";", 1)[0]
                .strip()
                .lower()
            )
            if content_type != "application/json":
                raise LanSyncError("LAN queue actions require application/json.")
            try:
                length = int(self.headers.get("Content-Length", "0"))
            except ValueError as exc:
                raise LanSyncError("The LAN queue request length is invalid.") from exc
            if not 1 <= length <= LAN_QUEUE_REQUEST_BYTES:
                raise LanSyncError("The LAN queue request is too large.")
            try:
                value = json.loads(self.rfile.read(length))
            except (json.JSONDecodeError, UnicodeDecodeError) as exc:
                raise LanSyncError("The LAN queue request is not valid JSON.") from exc
            if not isinstance(value, dict):
                raise LanSyncError("The LAN queue request must be a JSON object.")
            return value

        def _json(self, status: int, value: dict[str, object]) -> None:
            body = json.dumps(
                value,
                ensure_ascii=False,
                default=str,
            ).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            self.wfile.write(body)

        def _block(self, path: Path, sha256: str) -> None:
            size = path.stat().st_size
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "audio/mpeg")
            self.send_header("Content-Length", str(size))
            self.send_header("X-Radio-Archive-Protocol", LAN_PROTOCOL)
            self.send_header("X-Radio-Archive-SHA256", sha256)
            self.send_header("Cache-Control", "private, no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            with path.open("rb") as handle:
                for chunk in iter(lambda: handle.read(256 * 1024), b""):
                    self.wfile.write(chunk)

        def _artifact(self, path: Path, sha256: str) -> None:
            size = path.stat().st_size
            self.send_response(HTTPStatus.OK)
            content_type = {
                ".json": "application/json; charset=utf-8",
                ".mp3": "audio/mpeg",
            }.get(path.suffix.lower(), "text/plain; charset=utf-8")
            self.send_header(
                "Content-Type",
                content_type,
            )
            self.send_header("Content-Length", str(size))
            self.send_header("X-Radio-Archive-Protocol", LAN_PROTOCOL)
            self.send_header("X-Radio-Archive-SHA256", sha256)
            self.send_header("Cache-Control", "private, no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            with path.open("rb") as handle:
                for chunk in iter(lambda: handle.read(256 * 1024), b""):
                    self.wfile.write(chunk)

    class LanNodeServer(ThreadingHTTPServer):
        address_family = (
            socket.AF_INET6 if ipaddress.ip_address(host).version == 6 else socket.AF_INET
        )
        daemon_threads = True

        def server_close(self) -> None:
            reconciler = getattr(self, "lan_reconciler", None)
            if reconciler is not None:
                reconciler.close()
            responder = getattr(self, "lan_discovery", None)
            if responder is not None:
                responder.close()
            super().server_close()

    server = LanNodeServer((host, port), Handler)
    server.catalog = catalog  # type: ignore[attr-defined]
    server.job_manager = job_manager  # type: ignore[attr-defined]
    server.relay_database_path = relay_database_path  # type: ignore[attr-defined]
    server.quiet = False  # type: ignore[attr-defined]
    server.lan_discovery = None  # type: ignore[attr-defined]
    server.lan_reconciler = None  # type: ignore[attr-defined]
    configured_advertisement = str(
        advertise_url or os.getenv("BROADCASTIFY_LAN_ADVERTISE_URL") or ""
    ).strip()
    if configured_advertisement:
        configured_advertisement = normalize_peer_url(configured_advertisement)

    if discovery_enabled:
        def url_for_remote(remote_address: str) -> str:
            if configured_advertisement:
                return configured_advertisement
            address = ipaddress.ip_address(host)
            if not address.is_unspecified:
                display_host = f"[{host}]" if address.version == 6 else host
                return f"http://{display_host}:{server.server_port}"
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as route:
                route.connect((remote_address, 9))
                local_address = str(route.getsockname()[0])
            return f"http://{local_address}:{server.server_port}"

        responder = LanDiscoveryResponder(catalog, url_for_remote)
        responder.start()
        server.lan_discovery = responder  # type: ignore[attr-defined]
    if background_sync_enabled and catalog.enabled:
        reconciler = LanArchiveReconciler(
            catalog.output_dir,
            LanArchiveSyncClient(
                enabled=True,
                peer_urls=catalog.peer_urls,
                discovery_enabled=discovery_enabled,
                sync_key=catalog.sync_key,
                queue_enabled=False,
                role=catalog.role,
                master_url=os.getenv("BROADCASTIFY_LAN_MASTER_URL") or "",
            ),
            poll_seconds=environment_float(
                "BROADCASTIFY_LAN_RECONCILE_SECONDS",
                5 * 60.0,
                minimum=30.0,
                maximum=60 * 60.0,
            ),
        )
        server.lan_reconciler = reconciler  # type: ignore[attr-defined]
        reconciler.start()
    return server


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Serve only retained original archive blocks to trusted-LAN clients."
        )
    )
    parser.add_argument("--output-dir", default="archives")
    parser.add_argument("--host", default="0.0.0.0", type=validate_lan_host)
    parser.add_argument("--port", type=int, default=8_766)
    parser.add_argument("--advertise-url", default="")
    parser.add_argument(
        "--database",
        help="Authoritative Windows analysis database for relayed Web work.",
    )
    parser.add_argument(
        "--working-dir",
        help="Authoritative Windows app-data directory for relayed Web work.",
    )
    parser.add_argument(
        "--enable-job-relay",
        action="store_true",
        help="Accept authenticated follower jobs and execute them on this master.",
    )
    parser.add_argument(
        "--role",
        choices=("master", "follower"),
        default=None,
        help="Own the post-download pipeline or follow its completed results.",
    )
    parser.add_argument(
        "--no-discovery",
        action="store_true",
        help="Serve explicit peer URLs without UDP discovery.",
    )
    parser.add_argument(
        "--parent-pid",
        type=int,
        default=0,
        help="Exit when the owning desktop process exits.",
    )
    return parser


def main() -> int:
    _load_environment()
    arguments = build_parser().parse_args()
    server = create_lan_node_server(
        arguments.output_dir,
        host=arguments.host,
        port=arguments.port,
        discovery_enabled=(
            not arguments.no_discovery
            and environment_flag(
                "BROADCASTIFY_LAN_DISCOVERY_ENABLED",
                default=True,
            )
        ),
        advertise_url=arguments.advertise_url,
        role=arguments.role,
        working_dir=arguments.working_dir,
        database_path=arguments.database,
        job_relay_enabled=arguments.enable_job_relay,
        background_sync_enabled=environment_flag(
            "BROADCASTIFY_LAN_BACKGROUND_SYNC",
            default=True,
        ),
    )
    _start_parent_watchdog(server, arguments.parent_pid)
    print(
        json.dumps(
            {
                "type": "lan_node_ready",
                "protocol": LAN_PROTOCOL,
                "host": arguments.host,
                "port": server.server_port,
                "output_dir": str(Path(arguments.output_dir).expanduser().resolve()),
                "key_required": bool(server.catalog.sync_key),  # type: ignore[attr-defined]
            }
        ),
        flush=True,
    )
    try:
        server.serve_forever(poll_interval=0.25)
    except KeyboardInterrupt:
        return 0
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
