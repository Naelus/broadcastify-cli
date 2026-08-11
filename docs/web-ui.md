# Cross-platform local Web UI

The `broadcastify-web` entry point is a local browser companion for the same Python worker used by the WinUI app. It is intended to make the retained Library, review, acquisition, and provider experience available on Windows, Linux, and macOS without making a cloud service the owner of radio audio or credentials.

## Start it

From an installed development environment:

```text
broadcastify-web --open
```

The default address is `http://127.0.0.1:8765/`. `--port` may select another local port, `--output-dir` selects the archive root, and `--database` selects its SQLite evidence store. `--host` accepts loopback, a numeric private/link-local address, or the wildcard listeners `0.0.0.0` and `::`; public and multicast numeric addresses are rejected.

For an explicitly trusted LAN:

```text
broadcastify-web --host 0.0.0.0 --port 8765
```

Prefer binding or publishing the service on one intended private interface.
TrueNAS SCALE users should use the managed Custom App in
[`deploy/truenas`](../deploy/truenas/README.md), which keeps the container's
wildcard listener inside the App network and constrains the published port to
the selected NAS LAN address.

Linux users with a systemd user session can install the same entry point as a
failure-restarting user service with `radio-archive-service install`. Its
config, private `.env` reference, health check, lifecycle commands, and
non-systemd fallback are documented in
[linux-service.md](linux-service.md). The unit never contains credential
values; loopback remains its default, while an explicit `--host` opt-in can
select the same trusted-LAN addresses as the foreground server.

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
- persistent daily schedules for explicitly selected feeds, with editable
  time, enable state, recent-day lookback, optional self-clearing or recurring
  historical catch-up, cache/stage reuse, service-start recovery, and rolling-quota resume
- visible installation-local 240-of-250 rolling request status with a
  10-request manual reserve and next-safe time
- trusted-LAN source-block pooling before website access, with optional discovery or explicit private peer URLs and one renewable upstream producer lease per quota-scope/feed/day
- automatic analysis queue for newly completed transcripts
- evidence-grounded range/month/whole-feed Q&A, deterministic hotspot aggregates,
  and persisted seven-day briefs with missing dates
- center/radius or ordered-ZIP discovery, explicit area profiles, persisted nearest-first stop/resume queues, and retained regional story-lead briefs with coverage gaps
- automatic/CUDA/CPU/Vulkan/OpenVINO/Apple Metal/Windows ML processing profiles with a collapsed stage-by-stage readiness comparison
- five-step first-run readiness for account, storage, transcription, speaker labels, and analysis
- persistent, deep-linked Setup, Processing, Analysis & AI, and Account settings sections with one visible panel, keyboard navigation, and exact setup-action focus
- one-click sequential profile verification plus explicit transcription, diarization, and analysis tests, with generated input, user-controlled managed-model download, actual fallback/device reporting, and no archive-quota use
- local Gemma, OpenAI Responses, compatible `/v1`, and Codex-login analysis settings with the same explicit external-text gate as WinUI
- hardware and provider readiness checks that do not send transcript text

## Local security boundary

The service binds to loopback unless trusted-LAN mode is explicitly selected and generates a new random token for every launch. The initial HTML receives that token and a `SameSite=Strict`, `HttpOnly` session cookie. JSON and media routes require the session cookie. Every POST additionally requires the token in `X-Radio-Archive-Token`, an `application/json` body, and a matching origin. Cross-origin resource sharing is not enabled.

The session boundary protects routes from blind/cross-origin requests; it is
not a login wall. A client that can reach the listener can load the root page,
receive a valid session, read the configured library, and start supported
jobs. Trusted-LAN mode therefore intentionally exposes retained transcripts
and audio to that LAN. Do not port-forward it, attach it to a public reverse
proxy, or use a public tunnel without a separate authentication/TLS layer.

Static transcript, incident, feed, and model text is HTML-escaped before rendering. A restrictive content-security policy allows scripts, styles, media, and connections only from the local origin. Media paths must resolve beneath the configured archive root; absolute paths and traversal are rejected.

Only one worker job may be queued/running at once. Archive job payloads are
overwritten to `download_jobs=1` and `keep_originals=true`; diarization also
forces daily combination and transcription. Every child worker uses the Web
installation's durable request ledger. The service checks feed schedules in
the background while it is alive, defers behind a user-started job, and
recovers an interrupted schedule on restart. Cancel sends an interrupt to the
worker process group before using a bounded forced stop, giving model/server
contexts an opportunity to clean up.

An independently versioned, read-only LAN archive data surface is available only
when `BROADCASTIFY_LAN_SHARING=true`. Its inventory and block routes do not use
the browser session because native/headless peers do not have one; they expose
only strictly named original MP3 blocks and can require
`BROADCASTIFY_LAN_SYNC_KEY`. Sessionless JSON coordination routes store only
bounded transient feed/day lease state and use the same optional shared key;
they cannot upload an archive or start a remote job. The browser bootstrap
reports only non-secret sharing/discovery/coordination status. See
[Trusted-LAN archive reuse](lan-archive-sync.md) for integrity checks,
explicit peer configuration, and the trusted-network boundary.

## Secrets and persistence

Non-secret model/provider choices use browser-local storage. Saved feed
schedules persist non-secret processing choices in SQLite and never contain
direct passwords, tokens, or API keys.

The browser UI's **Credentials** page can persist the Broadcastify login and a
Hugging Face read token on the app server. It never stores either value in
browser storage or returns a complete secret to the browser:

- Windows servers encrypt the complete credential payload with current-user
  DPAPI.
- Linux and TrueNAS servers use AES-GCM with a randomly generated 256-bit key;
  the key and encrypted payload are restricted to the service account and must
  live in the persistent private data mount.
- Status responses contain only the username and a short password/token prefix.
- Every foreground and scheduled worker receives the decrypted values only in
  its child-process environment.

The default payload is `.credentials.enc`; AES-GCM hosts also create
`.credentials.enc.key` beneath the service working directory. Both patterns are
ignored by Git. Set
`BROADCASTIFY_CREDENTIAL_STORE` to choose another persistent private path.
Filesystem/root access can still recover a local service key, so host and
dataset permissions remain part of the security boundary.

A Hugging Face read token is needed for the first gated Community-1 download;
a complete cache can later run offline without one. Environment variables from
`.env-example` remain a supported headless alternative, and encrypted UI values
take precedence without rewriting that file. Hosted analysis API keys continue
to be session/environment values unless their platform-specific secure-store
path explicitly supports persistence.

## Current validation boundary

The retained Library/review experience, responsive layout, transcript search, saved weekly summary, saved area digest, provider checks, and setup workflow have been exercised on the Windows reference machine. A retained validation revision added the task-focused Settings workspace; a retained validation revision added joined profile verification and host-platform choice filtering. Live QA at 1280×720 and a true 390×844 viewport covered every section, direct/hash navigation, reload persistence, Back/Forward synchronization, keyboard selection, setup-card routing/focus, the new verifier, Windows-only Windows ML visibility, hidden Metal, and Vulkan's incompatible-model repair. The phone document and client widths remained exactly 375/375, all four tabs fit without horizontal scrolling, one panel was visible, IDs were unique, every button had an accessible name, and the browser warning/error console was empty. Joined CUDA, OpenVINO, and Windows ML verification completed in 15.4, 13.8, and 13.3 seconds respectively. The current local and hardened NAS suite is **219 tests**.

A retained validation revision passed the full-model desktop/mobile visual, accessibility, navigation, media, and loopback-security matrix plus all 148 tests on TrueNAS/Linux. A retained validation revision was later transferred as a hash-verified archive and passed all **171 tests in 5.90 seconds** inside the immutable AMD/Vulkan image. A retained validation revision repeated that boundary with **172 tests in 5.04 seconds** and a 9.960-second warm-cache joined Vulkan/CPU/Vulkan model proof. Both used networking disabled, a read-only root/source, dropped capabilities, `no-new-privileges`, and an unprivileged numeric user. The container explicitly overrides the whisper.cpp image entrypoint with Python; otherwise the image can accept pytest arguments without running them and produce an empty false success.

A retained joined verifier completed Vulkan ASR → pyannote CPU → Vulkan analysis on the AMD Radeon 890M in **27.886 seconds** after its child-scoped loader fix prevented llama.cpp GGML libraries from contaminating whisper.cpp. A retained validation revision later completed the faster Vulkan ASR → sherpa CPU preview → Vulkan analysis profile in 9.199 seconds and measured the 23.9-hour speaker stage at 0.0227 RTF; a retained validation revision passed 219 hardened tests plus a real chunk-checkpoint kill/resume/final-cache proof. Earlier retained-workflow proofs remain applicable: a retained validation revision completed the protected `/api/jobs` path through embeddings/summary in 15.139 seconds; a retained validation revision completed the matching all-CPU flow in 21.282 seconds; and a retained validation revision passed the managed per-user systemd lifecycle.

A retained validation revision is also installed as a TrueNAS 25.04 Custom App with a
persistent host-path dataset and a port published on one NAS LAN address. A
separate Windows client reached the root, trusted-LAN health, protected
bootstrap, and protected job API without a tunnel. The live joined verifier
completed Whisper/Vulkan, sherpa CPU, and local Gemma/Vulkan in **4.920
seconds**; a TrueNAS-managed redeploy retained both feed directories, 11
library days, 470 incidents, and 2 weekly summaries. The active packaging work
is tracked in [#1](https://github.com/Naelus/broadcastify-cli/issues/1). Intel
GPU/NPU hardware timing and real macOS/Metal runs remain unvalidated but are not
current priorities.
