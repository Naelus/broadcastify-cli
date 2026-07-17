from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from broadcastify_cli.analysis_clients import (
    AnalysisProviderError,
    CodexCliClient,
    OpenAICompatibleClient,
    OpenAIResponsesClient,
)
from broadcastify_cli.analysis_providers import (
    AnalysisProviderConfig,
    diagnose_analysis_provider,
)


class FakeResponse:
    def __init__(self, value: dict[str, object], status_code: int = 200) -> None:
        self.value = value
        self.status_code = status_code
        self.headers: dict[str, str] = {}

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")

    def json(self) -> dict[str, object]:
        return self.value


def test_external_provider_requires_explicit_transcript_consent() -> None:
    with pytest.raises(AnalysisProviderError, match="Explicitly allow"):
        AnalysisProviderConfig(
            provider="openai-responses",
            model="test-model",
        ).validate()

    AnalysisProviderConfig(
        provider="openai-compatible",
        model="local-model",
        endpoint="http://127.0.0.1:1234/v1",
    ).validate()


def test_provider_config_has_distinct_persistence_identity() -> None:
    first = AnalysisProviderConfig(
        provider="openai-compatible",
        model="gemma",
        endpoint="http://127.0.0.1:1234/v1",
    )
    second = AnalysisProviderConfig(
        provider="openai-compatible",
        model="gemma",
        endpoint="http://127.0.0.1:5678/v1",
    )
    assert first.cache_model != second.cache_model
    assert AnalysisProviderConfig(
        provider="codex-cli", model=""
    ).cache_model == "codex-cli:account-default"


def test_explicit_ui_provider_values_override_environment(monkeypatch) -> None:
    monkeypatch.setenv("ANALYSIS_MODEL", "environment-model")
    monkeypatch.setenv("ANALYSIS_ENDPOINT", "https://environment.invalid/v1")
    monkeypatch.setenv("ALLOW_EXTERNAL_ANALYSIS", "true")

    config = AnalysisProviderConfig.from_mapping(
        {
            "analysis_provider": "codex-cli",
            "analysis_model": "",
            "analysis_endpoint": "",
            "allow_external_analysis": False,
        }
    )

    assert config.model == ""
    assert config.endpoint == ""
    assert config.allow_external is False


def test_removed_local_model_selector_migrates_when_not_cached(monkeypatch) -> None:
    monkeypatch.setattr(
        "broadcastify_cli.analysis.find_cached_huggingface_gguf",
        lambda *_args, **_kwargs: None,
    )

    config = AnalysisProviderConfig.from_mapping(
        {
            "analysis_provider": "local",
            "analysis_model": "ggml-org/gemma-4-12B-it-GGUF:Q4_K_M",
        }
    )

    assert config.model == "ggml-org/gemma-4-12B-it-GGUF:Q4_0"
    assert config.cache_model == "ggml-org/gemma-4-12B-it-GGUF:Q4_0"


def test_openai_responses_uses_structured_output_without_storage(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def fake_post(url: str, **kwargs: object) -> FakeResponse:
        captured["url"] = url
        captured.update(kwargs)
        return FakeResponse(
            {
                "status": "completed",
                "output": [
                    {
                        "type": "message",
                        "content": [
                            {
                                "type": "output_text",
                                "text": '{"incidents": []}',
                            }
                        ],
                    }
                ],
            }
        )

    monkeypatch.setattr("broadcastify_cli.analysis_clients.requests.post", fake_post)
    client = OpenAIResponsesClient(api_key="test-key", model="fast-model")
    result = client.chat_json(
        "system",
        "evidence",
        "incidents",
        {
            "type": "object",
            "properties": {"incidents": {"type": "array"}},
            "required": ["incidents"],
            "additionalProperties": False,
        },
    )

    payload = captured["json"]
    headers = captured["headers"]
    assert result == {"incidents": []}
    assert captured["url"] == "https://api.openai.com/v1/responses"
    assert isinstance(payload, dict) and payload["store"] is False
    assert payload["text"]["format"]["type"] == "json_schema"
    assert isinstance(headers, dict) and headers["Authorization"] == "Bearer test-key"


def test_openai_compatible_supports_bearer_key_and_schema(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def fake_post(url: str, **kwargs: object) -> FakeResponse:
        captured["url"] = url
        captured.update(kwargs)
        return FakeResponse(
            {
                "choices": [
                    {
                        "message": {"content": '{"answer": "supported"}'},
                        "finish_reason": "stop",
                    }
                ]
            }
        )

    monkeypatch.setattr("broadcastify_cli.analysis_clients.requests.post", fake_post)
    client = OpenAICompatibleClient(
        base_url="https://example.invalid/v1",
        model="provider-model",
        api_key="provider-key",
        cache_model="compatible:provider-model",
    )
    result = client.chat_json(
        "system",
        "user",
        "answer",
        {
            "type": "object",
            "properties": {"answer": {"type": "string"}},
            "required": ["answer"],
            "additionalProperties": False,
        },
    )

    assert result == {"answer": "supported"}
    assert client.model == "compatible:provider-model"
    assert captured["url"] == "https://example.invalid/v1/chat/completions"
    assert captured["headers"]["Authorization"] == "Bearer provider-key"


def test_codex_cli_is_ephemeral_read_only_and_strips_app_secrets(
    monkeypatch, tmp_path: Path
) -> None:
    executable = tmp_path / "codex.exe"
    executable.write_bytes(b"placeholder")
    captured: dict[str, object] = {}

    def fake_run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        captured["command"] = command
        captured.update(kwargs)
        output_path = Path(command[command.index("--output-last-message") + 1])
        output_path.write_text(json.dumps({"answer": "bounded"}), encoding="utf-8")
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    monkeypatch.setenv("BROADCASTIFY_PASSWORD", "do-not-inherit")
    monkeypatch.setenv("HUGGINGFACE_TOKEN", "do-not-inherit")
    monkeypatch.setenv("CODEX_HOME", str(tmp_path / "codex-home"))
    monkeypatch.setattr("broadcastify_cli.analysis_clients.subprocess.run", fake_run)

    client = CodexCliClient(executable=executable)
    result = client.chat_json(
        "system",
        "untrusted transcript",
        "answer",
        {
            "type": "object",
            "properties": {"answer": {"type": "string"}},
            "required": ["answer"],
            "additionalProperties": False,
        },
    )

    command = captured["command"]
    environment = captured["env"]
    assert result == {"answer": "bounded"}
    assert "--ephemeral" in command
    assert command[command.index("--sandbox") + 1] == "read-only"
    assert "--ignore-user-config" in command
    assert "--ignore-rules" in command
    assert environment["CODEX_HOME"] == str(tmp_path / "codex-home")
    assert "BROADCASTIFY_PASSWORD" not in environment
    assert "HUGGINGFACE_TOKEN" not in environment


def test_openai_diagnostic_never_contacts_model(monkeypatch) -> None:
    monkeypatch.setenv("TEST_OPENAI_KEY", "present")

    def unexpected_request(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("OpenAI readiness must not create API usage")

    monkeypatch.setattr("broadcastify_cli.analysis_providers.requests.get", unexpected_request)
    result = diagnose_analysis_provider(
        AnalysisProviderConfig(
            provider="openai-responses",
            model="fast-model",
            api_key_env="TEST_OPENAI_KEY",
            allow_external=True,
        )
    )

    assert result["ready"] is True
    assert result["verified"] is False
    assert "no billable model request" in str(result["message"])
