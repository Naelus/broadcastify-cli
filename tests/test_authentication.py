import json
import os
import stat
from pathlib import Path

import pytest
import requests

from broadcastify_cli.broadcastify import AuthenticationError, BroadcastifyClient


class FakeResponse:
    def __init__(
        self,
        status_code: int,
        *,
        url: str,
        location: str = "",
        token: str = "",
    ) -> None:
        self.status_code = status_code
        self.url = url
        self.headers: dict[str, str] = {}
        if location:
            self.headers["Location"] = location
        self.cookies = requests.cookies.RequestsCookieJar()
        if token:
            self.cookies.set(
                "bcfyuser1",
                token,
                domain=".broadcastify.com",
                path="/",
            )

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise requests.HTTPError(f"HTTP {self.status_code}")


class FakeSession:
    def __init__(
        self,
        *,
        post_response: FakeResponse,
        redirect_response: FakeResponse | None = None,
    ) -> None:
        self.cookies = requests.cookies.RequestsCookieJar()
        self.post_response = post_response
        self.redirect_response = redirect_response
        self.get_calls: list[str] = []
        self.post_calls: list[str] = []

    def get(self, url: str, **_kwargs: object) -> FakeResponse:
        self.get_calls.append(url)
        if len(self.get_calls) == 1:
            return FakeResponse(200, url=url)
        if self.redirect_response is None:
            raise AssertionError("Unexpected authentication redirect request.")
        self._retain(self.redirect_response)
        return self.redirect_response

    def post(self, url: str, **_kwargs: object) -> FakeResponse:
        self.post_calls.append(url)
        self._retain(self.post_response)
        return self.post_response

    def close(self) -> None:
        pass

    def _retain(self, response: FakeResponse) -> None:
        for cookie in response.cookies:
            self.cookies.set_cookie(cookie)


def _client(
    tmp_path: Path,
    *,
    post_response: FakeResponse,
    redirect_response: FakeResponse | None = None,
) -> tuple[BroadcastifyClient, FakeSession]:
    client = BroadcastifyClient(
        username="example-user",
        password="example-password",
        cookie_path=tmp_path / "cookies.json",
    )
    session = FakeSession(
        post_response=post_response,
        redirect_response=redirect_response,
    )
    client.session = session  # type: ignore[assignment]
    return client, session


def test_account_profile_cookie_path_comes_from_isolated_worker_environment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    secondary_cookie = tmp_path / "account-sessions" / "secondary.json"
    monkeypatch.setenv("BROADCASTIFY_COOKIE_PATH", str(secondary_cookie))

    with BroadcastifyClient(username="secondary", password="secret") as client:
        assert client.cookie_path == secondary_cookie


def test_authentication_accepts_cookie_from_initial_redirect(
    tmp_path: Path,
) -> None:
    cookie_path = tmp_path / "cookies.json"
    cookie_path.write_text('{"bcfyuser1":"stale-token"}', encoding="utf-8")
    if os.name != "nt":
        cookie_path.chmod(0o644)
    client, session = _client(
        tmp_path,
        post_response=FakeResponse(
            302,
            url=BroadcastifyClient.LOGIN_URL,
            location=BroadcastifyClient.BASE_URL,
            token="initial-token",
        ),
    )

    client.authenticate(force=True)

    assert session.get_calls == [BroadcastifyClient.LOGIN_URL]
    assert session.post_calls == [BroadcastifyClient.LOGIN_URL]
    assert json.loads(cookie_path.read_text()) == {
        "bcfyuser1": "initial-token"
    }
    assert list(tmp_path.glob(".cookies.json.*.tmp")) == []
    if os.name != "nt":
        assert stat.S_IMODE(cookie_path.stat().st_mode) == 0o600


def test_authentication_accepts_cookie_from_safe_followup_redirect(
    tmp_path: Path,
) -> None:
    client, session = _client(
        tmp_path,
        post_response=FakeResponse(
            302,
            url=BroadcastifyClient.LOGIN_URL,
            location=BroadcastifyClient.BASE_URL,
        ),
        redirect_response=FakeResponse(
            200,
            url=BroadcastifyClient.BASE_URL,
            token="followup-token",
        ),
    )

    client.authenticate(force=True)

    assert session.get_calls == [
        BroadcastifyClient.LOGIN_URL,
        BroadcastifyClient.BASE_URL,
    ]
    assert json.loads((tmp_path / "cookies.json").read_text()) == {
        "bcfyuser1": "followup-token"
    }


def test_authentication_reports_rejected_credentials_instead_of_http_302(
    tmp_path: Path,
) -> None:
    client, session = _client(
        tmp_path,
        post_response=FakeResponse(
            302,
            url=BroadcastifyClient.LOGIN_URL,
            location="/login/?failed=1&redirect=https://www.broadcastify.com",
        ),
    )

    with pytest.raises(AuthenticationError, match="rejected the username or password"):
        client.authenticate(force=True)

    assert session.get_calls == [BroadcastifyClient.LOGIN_URL]


def test_authentication_does_not_follow_external_redirects(
    tmp_path: Path,
) -> None:
    client, session = _client(
        tmp_path,
        post_response=FakeResponse(
            302,
            url=BroadcastifyClient.LOGIN_URL,
            location="https://example.invalid/sign-in",
        ),
    )

    with pytest.raises(AuthenticationError, match="without issuing"):
        client.authenticate(force=True)

    assert session.get_calls == [BroadcastifyClient.LOGIN_URL]
