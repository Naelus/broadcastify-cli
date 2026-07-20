from __future__ import annotations

import argparse
import hmac
import ipaddress
import json
import os
import socket
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
    LanDiscoveryResponder,
    LanSyncError,
    environment_flag,
    normalize_peer_url,
    normalize_peer_urls,
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
    configured = os.getenv("BROADCASTIFY_ENV_FILE")
    if configured:
        path = Path(configured).expanduser()
        if path.is_file():
            load_dotenv(path, override=True)


def create_lan_node_server(
    output_dir: str | Path,
    *,
    host: str = "0.0.0.0",
    port: int = 8_766,
    sync_key: str | None = None,
    peer_urls: str | tuple[str, ...] | list[str] | None = None,
    discovery_enabled: bool = True,
    advertise_url: str = "",
) -> ThreadingHTTPServer:
    host = validate_lan_host(host)
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
            if parsed.path == "/api/lan/v1/info":
                self._json(HTTPStatus.OK, catalog.info())
                return
            if parsed.path == "/api/lan/v1/blocks":
                query = parse_qs(parsed.query)
                feed_id = str((query.get("feed_id") or [""])[0]).strip()
                archive_date = _date_value(
                    str((query.get("date") or [""])[0]).strip()
                )
                blocks = catalog.inventory(feed_id, archive_date)
                self._json(
                    HTTPStatus.OK,
                    {
                        "protocol": LAN_PROTOCOL,
                        "node_id": catalog.node_id,
                        "feed_id": feed_id,
                        "archive_date": archive_date.isoformat(),
                        "blocks": [block.to_dict() for block in blocks],
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
            raise FileNotFoundError

        def _post(self) -> None:
            parsed = urlparse(self.path)
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
                value = catalog.acquisition_queue.claim(
                    quota_scope,
                    feed_id,
                    archive_date,
                    owner_node_id=str(body.get("owner_node_id") or ""),
                    producer_url=str(body.get("producer_url") or ""),
                    requester_address=str(self.client_address[0]),
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
                )
            else:
                raise FileNotFoundError
            self._json(HTTPStatus.OK, value)

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

    class LanNodeServer(ThreadingHTTPServer):
        address_family = (
            socket.AF_INET6 if ipaddress.ip_address(host).version == 6 else socket.AF_INET
        )
        daemon_threads = True

        def server_close(self) -> None:
            responder = getattr(self, "lan_discovery", None)
            if responder is not None:
                responder.close()
            super().server_close()

    server = LanNodeServer((host, port), Handler)
    server.catalog = catalog  # type: ignore[attr-defined]
    server.quiet = False  # type: ignore[attr-defined]
    server.lan_discovery = None  # type: ignore[attr-defined]
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
        "--no-discovery",
        action="store_true",
        help="Serve explicit peer URLs without UDP discovery.",
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
    )
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
