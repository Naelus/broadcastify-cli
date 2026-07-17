# Analysis model providers

The default analysis provider is local llama.cpp with quantized Gemma. It is the recommended bulk-processing path because it keeps radio transcript text on the machine, avoids per-token charges, and has already been exercised on the retained corpus.

The current managed default is `ggml-org/gemma-4-12B-it-GGUF:Q4_0`. A model repository can change its available quant filenames, so the launcher first honors an explicit local `.gguf` path, then checks current and older Hugging Face cache snapshots for the requested quant, and only then asks llama.cpp to resolve an uncached repository selector. The former `Q4_K_M` default therefore remains usable when its exact file is already cached; otherwise it migrates to the current `Q4_0` selector. This avoids discarding multi-gigabyte local assets while keeping clean installs functional.

Three opt-in alternatives implement the same `chat_json` / `chat_text` contract and preserve the existing evidence validation, clips, SQLite records, and summaries:

| Provider | Transport | Credential | External-data acknowledgement |
|---|---|---|---|
| `local` | Managed llama.cpp or an existing local `/v1` endpoint | None | No |
| `openai-responses` | OpenAI Responses API with Structured Outputs | Session/env key; optional Windows Credential Locker when explicitly selected | Always required |
| `openai-compatible` | Chat Completions `/v1` endpoint such as llama.cpp, Ollama, LM Studio, or another provider | Optional session/env key or explicitly selected Windows Credential Locker | Required unless the endpoint is loopback |
| `codex-cli` | Ephemeral `codex exec` process using its saved login | Existing Codex CLI login | Always required |

Remote providers receive text, not raw audio. Incident extraction sends bounded transcript windows; questions send retrieved evidence passages and saved incident summaries; weekly and area reports send derived summaries and incident records. Those inputs can still contain sensitive or unverified radio traffic, so remote modes refuse to run until `--allow-external-analysis` (or `ALLOW_EXTERNAL_ANALYSIS=true`) is set.

On Windows, the same choices are available under **Settings → Analysis & AI** and apply to day analysis, automatic post-processing, Q&A, weekly summaries, and area briefs. Non-secret provider settings are saved for the Windows account. An entered API key remains in memory for the session unless **Remember this key in Windows Credential Locker** is explicitly checked; it is never written to the ordinary settings JSON. **Check provider** sends no transcript text. For OpenAI it checks only that a key is present, so it creates no model usage; compatible endpoints receive only `GET /models`; Codex runs only `codex login status`.

## OpenAI Responses API

Store the key in an environment variable, not a command argument or ordinary settings file:

```powershell
$env:OPENAI_API_KEY = "your key"
.\.venv\Scripts\broadcastify-analysis.exe analyze-day `
  --feed-id 90001 `
  --date 2026-07-12 `
  --provider openai-responses `
  --model gpt-5.6-luna `
  --allow-external-analysis
```

The client uses `/v1/responses`, `text.format.type=json_schema`, and `store=false`. It retries bounded 429 and transient 5xx responses, honoring `Retry-After` when present. The model remains configurable; `gpt-5.6-luna` is only the current low-cost/high-volume default, not a permanent compatibility assumption. OpenAI documents the request shape in its [Structured Outputs guide](https://developers.openai.com/api/docs/guides/structured-outputs) and [Responses create reference](https://developers.openai.com/api/reference/resources/responses/methods/create).

## OpenAI-compatible endpoints

A loopback server does not require the external-data switch:

```powershell
.\.venv\Scripts\broadcastify-analysis.exe summarize-week `
  --feed-id 90001 `
  --week-ending 2026-07-12 `
  --provider openai-compatible `
  --server-url http://127.0.0.1:1234/v1 `
  --model local-model
```

For a LAN or internet endpoint, add `--allow-external-analysis`. Use `--api-key-env NAME` to select the environment variable containing its bearer key. Strict JSON Schema is requested first; a provider that returns HTTP 400 for that shape receives one compatibility retry using JSON-object mode.

## Codex subscription/login harness

Codex CLI can be used as a private automation harness when it is installed and signed in:

```powershell
codex login
codex login status
.\.venv\Scripts\broadcastify-analysis.exe ask `
  --feed-id 90001 `
  --start-date 2026-07-11 `
  --end-date 2026-07-12 `
  --question "What serious calls were reported?" `
  --provider codex-cli `
  --allow-external-analysis
```

This path reuses the CLI's saved ChatGPT/Codex authentication by default; it does not turn a ChatGPT login into an OpenAI Platform API key. Plan availability and usage limits still apply. Each call uses `codex exec --ephemeral`, a read-only sandbox, an isolated temporary working directory, ignored user config/rules, and an output schema. Broadcastify, Hugging Face, API-key, token, password, and secret environment variables are removed from the child environment while `CODEX_HOME` remains available for authentication. The transcript prompt explicitly forbids tools and treats ASR text as untrusted data.

OpenAI's current Codex documentation describes saved-login reuse, non-interactive `codex exec`, ephemeral sessions, read-only sandboxing, and `--output-schema` in [Non-interactive mode](https://learn.chatgpt.com/docs/non-interactive-mode).

The contract and command construction are covered by local tests. The native readiness check detected the reference desktop's saved ChatGPT login, but a live Codex model call was deliberately not made. Readiness means authentication was detected, not that a particular plan has remaining usage or that a model response was validated.

## Persistence and switching

Every non-local provider gets a distinct cache identity. OpenAI uses its provider and model name, Codex uses its selected or account-default model, and OpenAI-compatible endpoints include a short hash of the base URL. Switching providers therefore creates a new analysis run instead of silently presenting another provider's cached conclusions. Transcript imports, embeddings, audio, and evidence clips remain reusable.
