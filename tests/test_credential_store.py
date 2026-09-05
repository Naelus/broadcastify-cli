from __future__ import annotations

import json
import os
import stat
from pathlib import Path

import pytest

from broadcastify_cli.credential_store import (
    EncryptedCredentialStore,
    masked_secret,
)


def test_secret_preview_reveals_only_a_short_prefix() -> None:
    assert masked_secret("hf_abcdefghijk", prefix_length=8) == "hf_abcde••••"
    assert masked_secret("password", prefix_length=2) == "pa••••"
    assert masked_secret("x", prefix_length=8) == "x••••"
    assert masked_secret("", prefix_length=8) == ""


def test_portable_store_encrypts_round_trips_and_forgets(
    tmp_path: Path,
) -> None:
    cryptography = pytest.importorskip("cryptography")
    assert cryptography is not None
    path = tmp_path / "credentials.enc"
    store = EncryptedCredentialStore(path, protector="aes-gcm-local-key")

    store.save_broadcastify("example-user", "example-password")
    store.save_huggingface("hf_example_read_token_123")

    raw = path.read_text(encoding="utf-8")
    assert "example-user" not in raw
    assert "example-password" not in raw
    assert "hf_example_read_token_123" not in raw
    assert json.loads(raw)["protector"] == "aes-gcm-local-key"
    assert store.worker_environment() == {
        "BROADCASTIFY_SECURE_USERNAME": "example-user",
        "BROADCASTIFY_SECURE_PASSWORD": "example-password",
        "HUGGINGFACE_SECURE_TOKEN": "hf_example_read_token_123",
    }
    status = store.status()
    assert status["broadcastify"] == {
        "saved": True,
        "configured": True,
        "username": "example-user",
        "password_preview": "ex••••",
        "source": "encrypted-store",
    }
    assert status["huggingface"] == {
        "saved": True,
        "configured": True,
        "token_preview": "hf_examp••••",
        "source": "encrypted-store",
    }
    if os.name != "nt":
        assert stat.S_IMODE(path.stat().st_mode) == 0o600
        assert stat.S_IMODE(store.key_path.stat().st_mode) == 0o600

    reopened = EncryptedCredentialStore(path, protector="aes-gcm-local-key")
    assert reopened.worker_environment()["HUGGINGFACE_SECURE_TOKEN"].startswith("hf_")
    reopened.clear_broadcastify()
    assert reopened.status()["broadcastify"]["saved"] is False
    reopened.clear_huggingface()
    assert not path.exists()


@pytest.mark.skipif(os.name != "nt", reason="Windows DPAPI is Windows-only")
def test_windows_store_uses_current_user_dpapi(tmp_path: Path) -> None:
    path = tmp_path / "credentials.enc"
    store = EncryptedCredentialStore(path, protector="dpapi-current-user")
    store.save_broadcastify("windows-user", "windows-password")
    store.save_huggingface("hf_windows_read_token_123")

    raw = path.read_text(encoding="utf-8")
    assert "windows-user" not in raw
    assert "windows-password" not in raw
    assert "hf_windows_read_token_123" not in raw
    assert json.loads(raw)["protector"] == "dpapi-current-user"

    reopened = EncryptedCredentialStore(path, protector="dpapi-current-user")
    assert reopened.worker_environment() == {
        "BROADCASTIFY_SECURE_USERNAME": "windows-user",
        "BROADCASTIFY_SECURE_PASSWORD": "windows-password",
        "HUGGINGFACE_SECURE_TOKEN": "hf_windows_read_token_123",
    }


def test_environment_is_reported_without_returning_its_secret(tmp_path: Path) -> None:
    store = EncryptedCredentialStore(
        tmp_path / "credentials.enc",
        protector="aes-gcm-local-key",
    )
    status = store.status(
        {
            "BROADCASTIFY_USERNAME": "environment-user",
            "BROADCASTIFY_PASSWORD": "environment-password",
            "HUGGINGFACE_TOKEN": "hf_environment-token",
        }
    )

    assert status["broadcastify"]["configured"] is True
    assert status["broadcastify"]["password_preview"] == "configured in environment"
    assert status["huggingface"]["configured"] is True
    assert status["huggingface"]["token_preview"] == "configured in environment"
    assert "environment-password" not in json.dumps(status)
    assert "hf_environment-token" not in json.dumps(status)


def test_named_broadcastify_profiles_are_encrypted_and_selected_in_isolation(
    tmp_path: Path,
) -> None:
    pytest.importorskip("cryptography")
    path = tmp_path / "credentials.enc"
    store = EncryptedCredentialStore(path, protector="aes-gcm-local-key")
    store.save_broadcastify("primary-user", "primary-password")
    store.save_broadcastify(
        "secondary-user",
        "secondary-password",
        profile_id="secondary",
        label="Secondary account",
    )

    raw = path.read_text(encoding="utf-8")
    assert "primary-password" not in raw
    assert "secondary-password" not in raw
    assert store.worker_environment("secondary") == {
        "BROADCASTIFY_SECURE_USERNAME": "secondary-user",
        "BROADCASTIFY_SECURE_PASSWORD": "secondary-password",
        "BROADCASTIFY_ACCOUNT_PROFILE": "secondary",
    }
    status = store.status()
    assert [profile["id"] for profile in status["broadcastify_profiles"]] == [
        "default",
        "secondary",
    ]

    store.clear_broadcastify("secondary")
    assert store.worker_environment("secondary") == {
        "BROADCASTIFY_ACCOUNT_PROFILE": "secondary"
    }
    assert store.worker_environment()["BROADCASTIFY_SECURE_USERNAME"] == (
        "primary-user"
    )
