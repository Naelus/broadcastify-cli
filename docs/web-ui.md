# Cross-platform local Web UI

The `broadcastify-web` entry point is a local browser companion for the same Python worker used by the WinUI app. It is intended to make the retained Library, review, acquisition, and provider experience available on Windows, Linux, and macOS without making a cloud service the owner of radio audio or credentials.

## Start it

From an installed development environment:

```text
broadcastify-web --open
```

The default address is `http://127.0.0.1:8765/`. `--port` may select another local port, `--output-dir` selects the archive root, and `--database` selects its SQLite evidence store. The host option accepts only `127.0.0.1`, `localhost`, or `::1`; there is deliberately no LAN bind switch.

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
- explicit selected-engine transcription and diarization synthetic-audio self-tests, with user-controlled managed-model download and actual fallback/device reporting
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

The complete retained Library/review experience, responsive layout, transcript search, saved weekly summary, saved area digest, provider checks, and the five-step setup view have been exercised on the Windows reference machine. Setup QA passed at desktop and 390×844 with no horizontal overflow or console errors; the hardware action moved the reference machine from two cheap prerequisites to five detected stages. The current local suite is 113 tests.

Exact commit `historical-validation` also passed all 113 tests inside the immutable TrueNAS/Linux Vulkan image. The headless harness then completed a fresh protected `/api/jobs` request through ASR → pyannote CPU → local Gemma/embedding/summary in 15.139 seconds for a 30-second retained fixture, ending Ready to review. Networking was disabled and no Hugging Face token was supplied; the complete cached speaker model loaded successfully. This closes the joined HTTP-backend gap. A packaged/supervised Linux launcher, Linux full-model visual/accessibility pass, platform keychain work, and every real macOS/Metal run remain open.
