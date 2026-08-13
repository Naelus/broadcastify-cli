from __future__ import annotations

import base64
import ctypes
import json
import os
import secrets
import stat
import threading
from ctypes import wintypes
from pathlib import Path
from typing import Any, Mapping

from .quota import DEFAULT_ACCOUNT_PROFILE_ID, normalize_account_profile_id


_FORMAT_VERSION = 1
_ASSOCIATED_DATA = b"broadcastify-cli-credentials-v1"
_WINDOWS_ENTROPY = b"Broadcastify Desktop local credentials v1"
_MAX_USERNAME_LENGTH = 320
_MAX_SECRET_LENGTH = 16_384


class CredentialStoreError(RuntimeError):
    """Raised when local encrypted credential storage cannot be used safely."""


def masked_secret(value: str, *, prefix_length: int) -> str:
    """Return a non-reversible prefix preview without exposing the full secret."""

    secret = str(value or "")
    if not secret:
        return ""
    visible = min(max(1, prefix_length), max(1, len(secret) - 1))
    return f"{secret[:visible]}••••"


class _DataBlob(ctypes.Structure):
    _fields_ = [
        ("cbData", wintypes.DWORD),
        ("pbData", ctypes.POINTER(ctypes.c_ubyte)),
    ]


def _blob(value: bytes) -> tuple[_DataBlob, Any]:
    if not value:
        return _DataBlob(0, None), None
    buffer = (ctypes.c_ubyte * len(value)).from_buffer_copy(value)
    return _DataBlob(len(value), ctypes.cast(buffer, ctypes.POINTER(ctypes.c_ubyte))), buffer


def _windows_protect(value: bytes) -> bytes:
    if os.name != "nt":
        raise CredentialStoreError("Windows DPAPI is available only on Windows.")
    crypt32 = ctypes.WinDLL("crypt32", use_last_error=True)
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    crypt32.CryptProtectData.argtypes = [
        ctypes.POINTER(_DataBlob),
        wintypes.LPCWSTR,
        ctypes.POINTER(_DataBlob),
        wintypes.LPVOID,
        wintypes.LPVOID,
        wintypes.DWORD,
        ctypes.POINTER(_DataBlob),
    ]
    crypt32.CryptProtectData.restype = wintypes.BOOL
    kernel32.LocalFree.argtypes = [wintypes.HLOCAL]
    kernel32.LocalFree.restype = wintypes.HLOCAL
    source, source_buffer = _blob(value)
    entropy, entropy_buffer = _blob(_WINDOWS_ENTROPY)
    output = _DataBlob()
    if not crypt32.CryptProtectData(
        ctypes.byref(source),
        "Broadcastify Desktop credentials",
        ctypes.byref(entropy),
        None,
        None,
        0x1,  # CRYPTPROTECT_UI_FORBIDDEN
        ctypes.byref(output),
    ):
        raise CredentialStoreError(
            f"Windows could not encrypt the credential store (error {ctypes.get_last_error()})."
        )
    try:
        return ctypes.string_at(output.pbData, output.cbData)
    finally:
        if output.pbData:
            kernel32.LocalFree(ctypes.cast(output.pbData, wintypes.HLOCAL))
        _ = source_buffer, entropy_buffer


def _windows_unprotect(value: bytes) -> bytes:
    if os.name != "nt":
        raise CredentialStoreError("Windows DPAPI is available only on Windows.")
    crypt32 = ctypes.WinDLL("crypt32", use_last_error=True)
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    crypt32.CryptUnprotectData.argtypes = [
        ctypes.POINTER(_DataBlob),
        ctypes.POINTER(wintypes.LPWSTR),
        ctypes.POINTER(_DataBlob),
        wintypes.LPVOID,
        wintypes.LPVOID,
        wintypes.DWORD,
        ctypes.POINTER(_DataBlob),
    ]
    crypt32.CryptUnprotectData.restype = wintypes.BOOL
    kernel32.LocalFree.argtypes = [wintypes.HLOCAL]
    kernel32.LocalFree.restype = wintypes.HLOCAL
    source, source_buffer = _blob(value)
    entropy, entropy_buffer = _blob(_WINDOWS_ENTROPY)
    output = _DataBlob()
    description = wintypes.LPWSTR()
    if not crypt32.CryptUnprotectData(
        ctypes.byref(source),
        ctypes.byref(description),
        ctypes.byref(entropy),
        None,
        None,
        0x1,  # CRYPTPROTECT_UI_FORBIDDEN
        ctypes.byref(output),
    ):
        raise CredentialStoreError(
            "Windows could not decrypt the credential store for this account. "
            "Use Forget saved credentials and enter them again."
        )
    try:
        return ctypes.string_at(output.pbData, output.cbData)
    finally:
        if output.pbData:
            kernel32.LocalFree(ctypes.cast(output.pbData, wintypes.HLOCAL))
        if description:
            kernel32.LocalFree(ctypes.cast(description, wintypes.HLOCAL))
        _ = source_buffer, entropy_buffer


class EncryptedCredentialStore:
    """Small encrypted-at-rest store for the local browser/server surface.

    Windows protects the complete payload with current-user DPAPI. On POSIX
    systems (including a TrueNAS App), AES-GCM protects the payload and a
    separately created 256-bit key is restricted to the service account.
    """

    def __init__(
        self,
        path: str | Path,
        *,
        protector: str | None = None,
    ) -> None:
        self.path = Path(path).expanduser().resolve()
        self.key_path = self.path.with_suffix(self.path.suffix + ".key")
        selected = str(protector or ("dpapi-current-user" if os.name == "nt" else "aes-gcm-local-key"))
        if selected not in {"dpapi-current-user", "aes-gcm-local-key"}:
            raise ValueError(f"Unsupported credential protector: {selected}")
        self.protector = selected
        self._lock = threading.RLock()

    @classmethod
    def for_working_directory(cls, working_dir: str | Path) -> "EncryptedCredentialStore":
        configured = str(os.getenv("BROADCASTIFY_CREDENTIAL_STORE") or "").strip()
        path = Path(configured).expanduser() if configured else Path(working_dir) / ".credentials.enc"
        return cls(path)

    def status(self, environment: Mapping[str, str] | None = None) -> dict[str, Any]:
        values = self._load()
        environment = environment or {}
        broadcastify = dict(values.get("broadcastify") or {})
        profiles = {
            str(profile_id): dict(profile)
            for profile_id, profile in dict(
                values.get("broadcastify_profiles") or {}
            ).items()
            if isinstance(profile, dict)
        }
        if broadcastify:
            profiles[DEFAULT_ACCOUNT_PROFILE_ID] = broadcastify
        huggingface = dict(values.get("huggingface") or {})
        username = str(broadcastify.get("username") or "")
        password = str(broadcastify.get("password") or "")
        token = str(huggingface.get("token") or "")
        environment_username = str(
            environment.get("BROADCASTIFY_USERNAME")
            or (
                environment.get("USERNAME")
                if environment.get("PASSWORD")
                else ""
            )
            or ""
        )
        environment_password = str(
            environment.get("BROADCASTIFY_PASSWORD")
            or environment.get("PASSWORD")
            or ""
        )
        environment_token = str(
            environment.get("HUGGINGFACE_TOKEN")
            or environment.get("HF_TOKEN")
            or ""
        )
        profile_status: list[dict[str, Any]] = []
        for profile_id, profile in sorted(profiles.items()):
            profile_username = str(profile.get("username") or "")
            profile_password = str(profile.get("password") or "")
            if not profile_username or not profile_password:
                continue
            profile_status.append(
                {
                    "id": profile_id,
                    "label": str(profile.get("label") or profile_id),
                    "username": profile_username,
                    "password_preview": masked_secret(
                        profile_password, prefix_length=2
                    ),
                    "source": "encrypted-store",
                }
            )
        return {
            "storage": self.protector,
            "broadcastify": {
                "saved": bool(username and password),
                "configured": bool(
                    (username and password)
                    or (environment_username and environment_password)
                ),
                "username": username or environment_username,
                "password_preview": (
                    masked_secret(password, prefix_length=2)
                    if password
                    else "configured in environment"
                    if environment_password
                    else ""
                ),
                "source": "encrypted-store" if username and password else "environment" if environment_username and environment_password else "",
            },
            "huggingface": {
                "saved": bool(token),
                "configured": bool(token or environment_token),
                "token_preview": (
                    masked_secret(token, prefix_length=8)
                    if token
                    else "configured in environment"
                    if environment_token
                    else ""
                ),
                "source": "encrypted-store" if token else "environment" if environment_token else "",
            },
            "broadcastify_profiles": profile_status,
        }

    def worker_environment(
        self,
        account_profile_id: str = DEFAULT_ACCOUNT_PROFILE_ID,
    ) -> dict[str, str]:
        values = self._load()
        profile_id = normalize_account_profile_id(account_profile_id)
        if profile_id == DEFAULT_ACCOUNT_PROFILE_ID:
            broadcastify = dict(values.get("broadcastify") or {})
        else:
            broadcastify = dict(
                dict(values.get("broadcastify_profiles") or {}).get(profile_id)
                or {}
            )
        huggingface = dict(values.get("huggingface") or {})
        result: dict[str, str] = {}
        username = str(broadcastify.get("username") or "")
        password = str(broadcastify.get("password") or "")
        token = str(huggingface.get("token") or "")
        if username and password:
            result["BROADCASTIFY_SECURE_USERNAME"] = username
            result["BROADCASTIFY_SECURE_PASSWORD"] = password
        if profile_id != DEFAULT_ACCOUNT_PROFILE_ID:
            result["BROADCASTIFY_ACCOUNT_PROFILE"] = profile_id
        if token:
            result["HUGGINGFACE_SECURE_TOKEN"] = token
        return result

    def save_broadcastify(
        self,
        username: str,
        password: str,
        *,
        profile_id: str = DEFAULT_ACCOUNT_PROFILE_ID,
        label: str = "",
    ) -> None:
        clean_username = str(username or "").strip()
        clean_password = str(password or "")
        clean_profile_id = normalize_account_profile_id(profile_id)
        clean_label = str(label or clean_profile_id).strip()[:80]
        if not clean_username or len(clean_username) > _MAX_USERNAME_LENGTH:
            raise ValueError("Enter a valid Broadcastify username.")
        self._validate_secret(clean_password, "Broadcastify password")
        with self._lock:
            values = self._load_unlocked()
            profile = {
                "username": clean_username,
                "password": clean_password,
                "label": clean_label,
            }
            if clean_profile_id == DEFAULT_ACCOUNT_PROFILE_ID:
                values["broadcastify"] = profile
            else:
                profiles = dict(values.get("broadcastify_profiles") or {})
                profiles[clean_profile_id] = profile
                values["broadcastify_profiles"] = profiles
            self._save_unlocked(values)

    def save_huggingface(self, token: str) -> None:
        clean_token = str(token or "").strip()
        self._validate_secret(clean_token, "Hugging Face token")
        if not clean_token.startswith("hf_"):
            raise ValueError("Hugging Face tokens should start with hf_.")
        with self._lock:
            values = self._load_unlocked()
            values["huggingface"] = {"token": clean_token}
            self._save_unlocked(values)

    def clear_broadcastify(
        self,
        profile_id: str = DEFAULT_ACCOUNT_PROFILE_ID,
    ) -> None:
        clean_profile_id = normalize_account_profile_id(profile_id)
        if clean_profile_id == DEFAULT_ACCOUNT_PROFILE_ID:
            self._clear_kind("broadcastify")
            return
        with self._lock:
            values = self._load_unlocked()
            profiles = dict(values.get("broadcastify_profiles") or {})
            profiles.pop(clean_profile_id, None)
            if profiles:
                values["broadcastify_profiles"] = profiles
            else:
                values.pop("broadcastify_profiles", None)
            if any(values.values()):
                self._save_unlocked(values)
                return
            self.path.unlink(missing_ok=True)

    def clear_huggingface(self) -> None:
        self._clear_kind("huggingface")

    @staticmethod
    def _validate_secret(secret: str, label: str) -> None:
        if not secret:
            raise ValueError(f"Enter a {label}.")
        if len(secret) > _MAX_SECRET_LENGTH:
            raise ValueError(f"The {label} is unexpectedly long.")
        if "\x00" in secret:
            raise ValueError(f"The {label} contains an invalid character.")

    def _clear_kind(self, kind: str) -> None:
        with self._lock:
            values = self._load_unlocked()
            values.pop(kind, None)
            if any(values.values()):
                self._save_unlocked(values)
                return
            try:
                self.path.unlink()
            except FileNotFoundError:
                pass

    def _load(self) -> dict[str, Any]:
        with self._lock:
            return self._load_unlocked()

    def _load_unlocked(self) -> dict[str, Any]:
        if not self.path.is_file():
            return {}
        try:
            envelope = json.loads(self.path.read_text(encoding="utf-8"))
            if int(envelope.get("version", 0)) != _FORMAT_VERSION:
                raise CredentialStoreError("The credential store uses an unsupported format.")
            provider = str(envelope.get("protector") or "")
            if provider != self.protector:
                raise CredentialStoreError(
                    f"The credential store requires {provider or 'an unknown protector'}, "
                    f"but this runtime selected {self.protector}."
                )
            ciphertext = base64.b64decode(str(envelope["ciphertext"]), validate=True)
            if provider == "dpapi-current-user":
                plaintext = _windows_unprotect(ciphertext)
            else:
                nonce = base64.b64decode(str(envelope["nonce"]), validate=True)
                plaintext = self._aesgcm().decrypt(
                    nonce,
                    ciphertext,
                    _ASSOCIATED_DATA,
                )
            values = json.loads(plaintext.decode("utf-8"))
            if not isinstance(values, dict):
                raise ValueError("Credential payload is not an object.")
            return values
        except CredentialStoreError:
            raise
        except Exception as exc:
            raise CredentialStoreError(
                "The encrypted credential store could not be read. "
                "Restore its matching key or remove it and enter credentials again."
            ) from exc

    def _save_unlocked(self, values: Mapping[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._restrict(self.path.parent, directory=True)
        plaintext = json.dumps(
            dict(values),
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
        if self.protector == "dpapi-current-user":
            envelope = {
                "version": _FORMAT_VERSION,
                "protector": self.protector,
                "ciphertext": base64.b64encode(_windows_protect(plaintext)).decode("ascii"),
            }
        else:
            nonce = secrets.token_bytes(12)
            ciphertext = self._aesgcm().encrypt(
                nonce,
                plaintext,
                _ASSOCIATED_DATA,
            )
            envelope = {
                "version": _FORMAT_VERSION,
                "protector": self.protector,
                "nonce": base64.b64encode(nonce).decode("ascii"),
                "ciphertext": base64.b64encode(ciphertext).decode("ascii"),
            }
        temporary = self.path.with_suffix(self.path.suffix + ".tmp")
        temporary.write_text(
            json.dumps(envelope, indent=2) + "\n",
            encoding="utf-8",
        )
        self._restrict(temporary)
        os.replace(temporary, self.path)
        self._restrict(self.path)

    def _aesgcm(self) -> Any:
        try:
            from cryptography.hazmat.primitives.ciphers.aead import AESGCM
        except ModuleNotFoundError as exc:
            raise CredentialStoreError(
                "Encrypted browser credential persistence requires the cryptography package."
            ) from exc
        key = self._load_or_create_key()
        return AESGCM(key)

    def _load_or_create_key(self) -> bytes:
        if self.key_path.is_file():
            key = self.key_path.read_bytes()
            if len(key) != 32:
                raise CredentialStoreError("The local credential encryption key is invalid.")
            self._restrict(self.key_path)
            return key
        key = secrets.token_bytes(32)
        temporary = self.key_path.with_suffix(self.key_path.suffix + ".tmp")
        temporary.write_bytes(key)
        self._restrict(temporary)
        try:
            os.replace(temporary, self.key_path)
        except OSError:
            temporary.unlink(missing_ok=True)
            if not self.key_path.is_file():
                raise
        self._restrict(self.key_path)
        return self.key_path.read_bytes()

    @staticmethod
    def _restrict(path: Path, *, directory: bool = False) -> None:
        if os.name == "nt":
            return
        mode = stat.S_IRWXU if directory else stat.S_IRUSR | stat.S_IWUSR
        try:
            path.chmod(mode)
        except OSError as exc:
            raise CredentialStoreError(
                f"Could not restrict local credential permissions for {path}."
            ) from exc
