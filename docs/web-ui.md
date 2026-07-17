# Cross-platform local Web UI

The `broadcastify-web` entry point is a local browser companion for the same Python worker used by the WinUI app. It is intended to make the retained Library, review, acquisition, and provider experience available on Windows, Linux, and macOS without making a cloud service the owner of radio audio or credentials.

## Start it

From an installed development environment:

```text
broadcastify-web --open
```

The default address is `http://127.0.0.1:8765/`. `--port` may select another local port, `--output-dir` selects the archive root, and `--database` selects its SQLite evidence store. The host option accepts only `127.0.0.1`, `localhost`, or `::1`; there is deliberately no LAN bind switch.

## Implemented surfaces

- Local Library counts, filters, master/detail selection, and exact missing-stage status
- continuous-audio byte-range playback from the retained archive root
- priority-first incident cards with quotes, timestamps, play-at-time, and exact clip preparation
- paged/searchable speaker-labeled transcript viewing
- website-backed feed search and guarded inclusive archive jobs
- automatic analysis queue for newly completed transcripts
- evidence-grounded range Q&A and persisted seven-day briefs with missing dates
- center/radius or ordered-ZIP discovery, explicit area profiles, persisted nearest-first stop/resume queues, and retained regional story-lead briefs with coverage gaps
- automatic/CUDA/CPU/Vulkan/OpenVINO/Apple Metal/Windows ML processing profiles with a collapsed stage-by-stage readiness comparison
- five-step first-run readiness for account, storage, transcription, speaker labels, and analysis
- explicit selected-engine transcription and diarization synthetic-audio self-tests, with user-controlled managed-model download and actual fallback/device reporting
- local Gemma, OpenAI Responses, compatible `/v1`, and Codex-login analysis settings with the same explicit external-text gate as WinUI
- hardware and provider readiness checks that do not send transcript text

## Local security boundary

The service binds to loopback and generates a new random token for every launch. The initial HTML receives that token and a `SameSite=Strict`, `HttpOnly` session cookie. JSON and media routes require the session cookie. Every POST additionally requires the token in `X-Radio-Archive-Token`, an `application/json` body, and a matching local origin. Cross-origin resource sharing is not enabled.

Static transcript, incident, feed, and model text is HTML-escaped before rendering. A restrictive content-security policy allows scripts, styles, media, and connections only from the local origin. Media paths must resolve beneath the configured archive root; absolute paths and traversal are rejected.

Only one worker job may be queued/running at once. Archive job payloads are overwritten to `download_jobs=1` and `keep_originals=true`; diarization also forces daily combination and transcription. Cancel sends an interrupt to the worker process group before using a bounded forced stop, giving model/server contexts an opportunity to clean up.

## Secrets and persistence

Non-secret model/provider choices use browser-local storage. API keys and Hugging Face tokens are never written there; they remain in the active tab and travel only over the same loopback service to the selected child worker. Broadcastify login fields are sent to the website-auth worker and the password field is immediately cleared. For repeat launches, prefer the ignored `.env` variables documented in `.env-example`.

WinUI can additionally use Windows Credential Locker. A portable OS-keychain adapter is still pending and is tracked in `BUGS.md`; the browser UI must not imply that ordinary local storage is a safe replacement.

## Current validation boundary

The complete retained Library/review experience, responsive layout, transcript search, saved weekly summary, saved area digest, provider checks, and the five-step setup view have been exercised on the Windows reference machine. Setup QA passed at desktop and 390×844 with no horizontal overflow or console errors; the hardware action moved the reference machine from two cheap prerequisites to five detected stages. The current local suite is 109 tests. An earlier exact pushed source also ran on an immutable TrueNAS/Linux host: a private non-system dependency install, all then-current 96 tests, loopback startup, session/action-token security, byte-range media, v4 portable profile assets, and the diagnostics worker passed. Linux ASR and quantized LLM components were separately validated on the same AMD host, but a single Web-triggered ASR → pyannote CPU → analysis run is still pending. No real macOS install or Metal model run has occurred.
