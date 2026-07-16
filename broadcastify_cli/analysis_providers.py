from __future__ import annotations

import hashlib
import os
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Mapping
from urllib.parse import urlparse

from .analysis import DEFAULT_LLM_MODEL, LlamaCppClient, LlamaServerProcess
from .analysis_clients import (
    AnalysisClient,
    AnalysisProviderError,
    CodexCliClient,
    OpenAICompatibleClient,
    OpenAIResponsesClient,
)


DEFAULT_OPENAI_MODEL = "gpt-5.6-luna"
PROVIDER_CHOICES = ("local", "openai-responses", "openai-compatible", "codex-cli")


def _truthy(value: object) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "on"}


def _provider_name(value: object) -> str:
    aliases = {
        "": "local",
        "local": "local",
        "llama.cpp": "local",
        "openai": "openai-responses",
        "openai-responses": "openai-responses",
        "responses": "openai-responses",
        "openai-compatible": "openai-compatible",
        "compatible": "openai-compatible",
        "codex": "codex-cli",
        "codex-cli": "codex-cli",
    }
    normalized = str(value or "").strip().lower()
    try:
        return aliases[normalized]
    except KeyError as exc:
        raise AnalysisProviderError(
            f"Unknown analysis provider {value!r}; choose {', '.join(PROVIDER_CHOICES)}."
        ) from exc


def _loopback_endpoint(value: str) -> bool:
    hostname = (urlparse(value).hostname or "").lower()
    return hostname in {"localhost", "127.0.0.1", "::1"}


@dataclass(frozen=True)
class AnalysisProviderConfig:
    provider: str = "local"
    model: str = DEFAULT_LLM_MODEL
    endpoint: str = ""
    api_key: str = ""
    api_key_env: str = "OPENAI_API_KEY"
    codex_path: str = ""
    allow_external: bool = False
    timeout: float = 600.0

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "AnalysisProviderConfig":
        provider = _provider_name(
            value.get("analysis_provider") or os.getenv("ANALYSIS_PROVIDER") or "local"
        )
        explicit_model = str(
            value.get("analysis_model") or os.getenv("ANALYSIS_MODEL") or ""
        ).strip()
        if not explicit_model and provider == "local":
            explicit_model = str(value.get("model") or DEFAULT_LLM_MODEL)
        if not explicit_model and provider == "openai-responses":
            explicit_model = DEFAULT_OPENAI_MODEL
        endpoint = str(
            value.get("analysis_endpoint") or os.getenv("ANALYSIS_ENDPOINT") or ""
        ).strip()
        return cls(
            provider=provider,
            model=explicit_model,
            endpoint=endpoint,
            api_key=str(value.get("analysis_api_key") or "").strip(),
            api_key_env=str(
                value.get("analysis_api_key_env")
                or os.getenv("ANALYSIS_API_KEY_ENV")
                or "OPENAI_API_KEY"
            ).strip(),
            codex_path=str(
                value.get("codex_cli_path") or os.getenv("CODEX_CLI_PATH") or ""
            ).strip(),
            allow_external=bool(value.get("allow_external_analysis", False))
            or _truthy(os.getenv("ALLOW_EXTERNAL_ANALYSIS")),
            timeout=float(value.get("analysis_timeout") or 600.0),
        )

    @property
    def is_external(self) -> bool:
        if self.provider in {"openai-responses", "codex-cli"}:
            return True
        return self.provider == "openai-compatible" and not _loopback_endpoint(
            self.endpoint
        )

    @property
    def cache_model(self) -> str:
        if self.provider == "local":
            return self.model
        if self.provider == "openai-responses":
            return f"openai-responses:{self.model}"
        if self.provider == "codex-cli":
            return f"codex-cli:{self.model or 'account-default'}"
        endpoint_hash = hashlib.sha256(self.endpoint.encode("utf-8")).hexdigest()[:10]
        return f"openai-compatible:{self.model}@{endpoint_hash}"

    def validate(self) -> None:
        if self.is_external and not self.allow_external:
            raise AnalysisProviderError(
                "External analysis is off. Explicitly allow transcript excerpts to leave this "
                "computer before using this provider."
            )
        if self.provider == "openai-compatible" and not self.endpoint:
            raise AnalysisProviderError(
                "The OpenAI-compatible provider needs an endpoint ending in /v1."
            )
        if self.provider in {"local", "openai-compatible", "openai-responses"} and not self.model:
            raise AnalysisProviderError("The selected analysis provider needs a model name.")


def _api_key(config: AnalysisProviderConfig) -> str:
    return config.api_key or os.getenv(config.api_key_env, "")


@contextmanager
def open_analysis_client(
    config: AnalysisProviderConfig,
    *,
    launch_local_server: bool = True,
) -> Iterator[AnalysisClient]:
    config.validate()
    if config.provider == "local":
        if config.endpoint:
            yield LlamaCppClient(
                base_url=config.endpoint,
                model=config.model,
                timeout=config.timeout,
            )
            return
        if not launch_local_server:
            yield LlamaCppClient(model=config.model, timeout=config.timeout)
            return
        with LlamaServerProcess(model=config.model) as server:
            yield LlamaCppClient(
                base_url=server.base_url,
                model=config.model,
                timeout=config.timeout,
            )
        return
    if config.provider == "openai-responses":
        yield OpenAIResponsesClient(
            api_key=_api_key(config),
            model=config.model,
            base_url=config.endpoint or "https://api.openai.com/v1",
            timeout=config.timeout,
        )
        return
    if config.provider == "openai-compatible":
        yield OpenAICompatibleClient(
            base_url=config.endpoint,
            model=config.model,
            api_key=_api_key(config) or None,
            cache_model=config.cache_model,
            timeout=config.timeout,
        )
        return
    yield CodexCliClient(
        executable=Path(config.codex_path) if config.codex_path else None,
        model=config.model or None,
        timeout=config.timeout,
    )
