# Cross-platform local Web UI

The `broadcastify-web` entry point is a local browser companion for the same Python worker used by the WinUI app. It is intended to make the retained Library, review, acquisition, and provider experience available on Windows, Linux, and macOS without making a cloud service the owner of radio audio or credentials.

## Start it

From an installed development environment:

```text
broadcastify-web --open
```

The default address is `http://127.0.0.1:8765/`. `--port` may select another local port, `--output-dir` selects the archive root, and `--database` selects its SQLite evidence store. The host option accepts only `127.0.0.1`, `localhost`, or `::1`; there is deliberately no LAN bind switch.

Linux users with a systemd user session can install the same entry point as a
failure-restarting user service with `radio-archive-service install`. Its
config, private `.env` reference, health check, lifecycle commands, and
non-systemd fallback are documented in
[linux-service.md](linux-service.md). The unit never contains credential
values and cannot select a non-loopback host.

## Headless Web-job harness

`scripts/web_job_smoke.py` exercises the real HTTP boundary without requiring desktop control. It starts the service on an ephemeral loopback port, receives the same-site cookie and embedded action token, posts one supported `/api/jobs` request with a matching origin, polls the job, and emits or saves the final snapshot.

```text
python scripts/web_job_smoke.py diagnostics --output-dir archives --working-dir .
python scripts/web_job_smoke.py continue-local --payload-file smoke-job.json --output-dir archives --working-dir . --result-file web-job-result.json
```

The payload is never included in the returned job snapshot. Still treat payload files as private when they contain session-only provider keys or first-download model tokens. For cached retained-audio validation, omit credentials entirely.

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
- persistent, deep-linked Setup, Processing, Analysis & AI, and Account settings sections with one visible panel, keyboard navigation, and exact setup-action focus
- one-click sequential profile verification plus explicit transcription, diarization, and analysis tests, with generated input, user-controlled managed-model download, actual fallback/device reporting, and no archive-quota use
- local Gemma, OpenAI Responses, compatible `/v1`, and Codex-login analysis settings with the same explicit external-text gate as WinUI
- hardware and provider readiness checks that do not send transcript text

## Local security boundary

The service binds to loopback and generates a new random token for every launch. The initial HTML receives that token and a `SameSite=Strict`, `HttpOnly` session cookie. JSON and media routes require the session cookie. Every POST additionally requires the token in `X-Radio-Archive-Token`, an `application/json` body, and a matching local origin. Cross-origin resource sharing is not enabled.

Static transcript, incident, feed, and model text is HTML-escaped before rendering. A restrictive content-security policy allows scripts, styles, media, and connections only from the local origin. Media paths must resolve beneath the configured archive root; absolute paths and traversal are rejected.

Only one worker job may be queued/running at once. Archive job payloads are overwritten to `download_jobs=1` and `keep_originals=true`; diarization also forces daily combination and transcription. Cancel sends an interrupt to the worker process group before using a bounded forced stop, giving model/server contexts an opportunity to clean up.

## Secrets and persistence

Non-secret model/provider choices use browser-local storage. API keys and Hugging Face tokens are never written there; they remain in the active tab and travel only over the same loopback service to the selected child worker. A Hugging Face read token is needed for the first gated Community-1 download, but a complete cache can later run offline without one. Broadcastify login fields are sent to the website-auth worker and the password field is immediately cleared. For repeat launches, prefer the ignored `.env` variables documented in `.env-example`.

WinUI can additionally use Windows Credential Locker. A portable OS-keychain adapter is still pending and is tracked in `BUGS.md`; the browser UI must not imply that ordinary local storage is a safe replacement.

## Current validation boundary

The retained Library/review experience, responsive layout, transcript search, saved weekly summary, saved area digest, provider checks, and setup workflow have been exercised on the Windows reference machine. Exact commit `historical-validation` added the task-focused Settings workspace; `historical-validation` added joined profile verification and host-platform choice filtering. Live QA at 1280×720 and a true 390×844 viewport covered every section, direct/hash navigation, reload persistence, Back/Forward synchronization, keyboard selection, setup-card routing/focus, the new verifier, Windows-only Windows ML visibility, hidden Metal, and Vulkan's incompatible-model repair. The phone document and client widths remained exactly 375/375, all four tabs fit without horizontal scrolling, one panel was visible, IDs were unique, every button had an accessible name, and the browser warning/error console was empty. Joined CUDA, OpenVINO, and Windows ML verification completed in 15.4, 13.8, and 13.3 seconds respectively. The current local suite is **171 tests**.

Exact commit `historical-validation` passed the full-model desktop/mobile visual, accessibility, navigation, media, and loopback-security matrix plus all 148 tests on TrueNAS/Linux. Exact commit `historical-validation` was later transferred as a hash-verified archive and passed all **171 tests in 5.90 seconds** inside the immutable AMD/Vulkan image. Exact current commit `historical-validation` repeated that boundary with **172 tests in 5.04 seconds** and a 9.960-second warm-cache joined Vulkan/CPU/Vulkan model proof. Both used networking disabled, a read-only root/source, dropped capabilities, `no-new-privileges`, and an unprivileged numeric user. The container explicitly overrides the whisper.cpp image entrypoint with Python; otherwise the image can accept pytest arguments without running them and produce an empty false success.

The exact `historical-validation` joined verifier completed Vulkan ASR → pyannote CPU → Vulkan analysis on the AMD Radeon 890M in **27.886 seconds** after its child-scoped loader fix prevented llama.cpp GGML libraries from contaminating whisper.cpp. Earlier retained-workflow proofs remain applicable: `historical-validation` completed the protected `/api/jobs` path through embeddings/summary in 15.139 seconds; `historical-validation` completed the matching all-CPU flow in 21.282 seconds; and `historical-validation` passed the managed per-user systemd lifecycle. Platform keychain work, longer portable-diarization/full-day timing, non-developer packaging, Intel GPU/NPU hardware timing, and every real macOS/Metal run remain open.
