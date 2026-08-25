from __future__ import annotations

import socketserver
from collections.abc import Iterator

import pytest


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
