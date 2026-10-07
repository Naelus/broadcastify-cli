from __future__ import annotations

import socketserver
from collections.abc import Iterator
from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def isolated_quota_ledger(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    # Clients without an explicit ledger must never use the developer's ledger
    # or require a writable application checkout in an offline verification run.
    monkeypatch.setenv("BROADCASTIFY_QUOTA_LEDGER", str(tmp_path / "archive-quota.sqlite3"))


@pytest.fixture
def fast_server_shutdown(
    monkeypatch: pytest.MonkeyPatch,
) -> Iterator[None]:
    """Remove BaseServer's idle shutdown wait without changing requests."""

    original = socketserver.BaseServer.serve_forever

    def serve_forever(
        server: socketserver.BaseServer,
        poll_interval: float = 0.01,
    ) -> None:
        original(server, poll_interval=poll_interval)

    monkeypatch.setattr(socketserver.BaseServer, "serve_forever", serve_forever)
    yield
