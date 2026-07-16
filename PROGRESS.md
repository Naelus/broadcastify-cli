# Progress log

This is the dated verification and delivery log for `GOAL.md`. Keep forward-looking capabilities in `FEATURES.md` and unresolved defects in `BUGS.md`.

## July 16, 2026

### Local Library and UI

- Replaced the long workspace with WinUI navigation for Local Library, New Archive, Review & Ask, Area Watch, and Settings.
- Added a master/detail Local Library viewer at the 1240x900 reference size.
- The selected day now shows a five-stage processing timeline, combined-audio player, bounded timestamped transcript preview, local paths, and explicit continue/review/folder actions.
- Verified visually against real Example City and Example County retained days, including a fully analyzed day and a quota-interrupted partial day.
- Added saved non-secret processing defaults and Windows Credential Locker login persistence.
- Added the explicit `BundleLocalEnv=true` private build; ordinary builds remove stale bundled credentials.

### Pipeline correctness

- Added first-missing-stage discovery and local continuation.
- Existing transcripts can receive speaker labels without rerunning Whisper.
- Tightened diarization status so a request flag alone does not count as completed labeling.
- Python suite before the Windows ML adapter: **69 passed**; after its streaming adapter test: **70 passed**.
- WinUI private Release build: **0 warnings, 0 errors**.

### Hardware parity

- Added per-stage profile diagnostics for CUDA, CPU, Vulkan, OpenVINO, and Windows ML.
- OpenVINO CPU successfully transcribed a real 22.74-second police-radio clip (23 words, 3 segments).
- OpenVINO GPU rejected the current model on this NVIDIA host; automatic retry on CPU succeeded and identifies the fallback in metadata.
- llama.cpp device inspection detected Vulkan on both the RTX 3090 and AMD Radeon integrated graphics.
- Built the official ONNX Runtime GenAI Windows ML helper. A CPU FP32 Whisper Tiny export successfully transcribed the same local test clip.
- Integrated the Windows ML helper with one persistent model process and bounded 28-second chunks. The real clip completed through the Python adapter with 22.74 seconds of audio, one timestamped segment, and 147 output characters; the model self-test reports `decode_ready=true`.
- Verified the native Settings flow: selecting Windows ML plus the validated model path changed the profile from runtime-only to ready after the real decode check; recommended automatic/CUDA defaults were restored afterward.
- DML/WinML exports remain gated after reproducible graph-capture/fused-node errors; tracked as B-001.

### Archive quota reset run

- Completed a guarded feed 90001 resume for July 3–16 after downloads became available again.
- The process made 55 successful new media downloads at five-second pacing: 48/48 July 3 blocks and 7/48 July 4 blocks. The next request received the explicit quota response.
- It made no more media requests, combined the now-complete July 3 audio, reused complete July 11–12 caches, and preserved every incomplete date for the next run.
- The 55-request result contradicts treating the earlier roughly 192-redirect observation as a fixed daily quota; documentation now describes the budget/reset as dynamic or rolling and unknown.

### Portable analysis providers

- Added a common analysis-client contract without changing the validated local llama.cpp default.
- Added OpenAI Responses Structured Outputs with `store=false`, environment-only API keys, bounded transient retries, and an explicit transcript-transmission gate.
- Added generic authenticated OpenAI-compatible Chat Completions for local or remote llama.cpp/Ollama/LM Studio-style endpoints.
- Added an ephemeral, read-only Codex CLI harness that reuses saved CLI authentication, isolates its working directory, requests a JSON schema, and strips unrelated secrets from the child environment.
- Provider/model/endpoint identities are distinct in SQLite caches, preventing conclusions from one provider from masquerading as another provider's run.
- Added a separate native **Analysis & AI** Settings tab instead of adding more controls to the processing page. It exposes local Gemma, OpenAI Responses, compatible `/v1`, and saved-login Codex providers; non-secret choices persist in the user settings file and an API key is session-only unless the user explicitly selects Windows Credential Locker.
- Wired the selected provider into local continuation, post-job analysis, selected-day analysis, Q&A, weekly summaries, and regional story briefs. Explicit blank/false UI values override environment defaults so an old `.env` switch cannot silently re-enable external analysis.
- Added provider-specific readiness checks that never send transcript text: local llama.cpp executable discovery, OpenAI key presence without a billable request, compatible-endpoint `/models`, and local `codex login status`.
- Visually exercised the tab at 1240x900. Local llama.cpp was found, OpenAI correctly reported a missing key, consent-off blocked external use, and the desktop Codex check reported `Logged in using ChatGPT`; no live model request was made. The private local provider and external-sharing toggle were restored before closing.
- Python suite after native provider settings and diagnostics: **77 passed**. WinUI Release build: **0 warnings, 0 errors**.

### Delivery checkpoints

- Git identity: `naelus <9455516+Naelus@users.noreply.github.com>`.
- Remote: `https://Naelus@github.com/Naelus/broadcastify-cli`.
- Staged content is scanned for common token/password patterns before every commit.
- Backend/persistence commit `historical-validation` was pushed to `origin/main` after 69 tests passed and the credential-pattern audit was clean.
- Native Library/UI commit `historical-validation` and Windows ML integration commit `historical-validation` were separately reviewed, audited, and pushed to `origin/main`.

### Cross-platform browser companion

- Added the `broadcastify-web` entry point and a dependency-free Python HTTP service that binds only to loopback. Each launch creates a random same-site session cookie; mutating actions additionally require a token header and matching origin.
- Reused the existing JSON worker for website feed/ZIP search, authentication, guarded range jobs, local continuation, incident clips, day analysis, Q&A, weekly summaries, area profiles/briefs, provider checks, and hardware diagnostics. The service permits one heavy worker at a time, forces archive concurrency to one, preserves source blocks, and keeps the database path explicit across child workers.
- Added safe byte-range streaming limited to the configured archive root, so 24-hour combined audio and generated evidence clips play without exposing arbitrary filesystem paths or contacting Broadcastify.
- Built a responsive Library/New Archive/Review/Area/Settings browser shell. The real retained corpus showed 9 feed-days, 6 ready days, 3 incomplete days, a 24:23:04 combined stream, 40 incidents, 736 diarized transcript segments, the saved 2/7-day brief, and a 10-feed regional profile with explicit 1/10-feed coverage.
- Visually exercised the desktop and 390x844 layouts. Transcript search returned the retained vehicle-fire line, the local llama.cpp readiness check succeeded without loading a model, Codex remained blocked while external sharing was off, mobile navigation opened correctly, and long incident/story surfaces now default to 12/10 highest-ranked records with an explicit show-all action.
- Browser console remained free of warnings/errors. Service/security/media tests added four cases; the complete Python suite is now **81 passed** and the browser JavaScript passes the bundled Node syntax check.

## Earlier validated work

- Feed 90001 completed July 11–12 end to end with 97 retained archive blocks, continuous daily audio, 1,716 transcript segments, 87 incidents, daily summaries, semantic Q&A, and a seven-day brief with explicit missing coverage.
- July 12 full-day pyannote diarization completed in about 25 minutes on the RTX 3090, produced 5,556 turns across five anonymous speaker clusters, and left no transcript words unlabeled.
- Multi-ZIP Example City discovery persisted a six-feed regional profile and created evidence-backed area story leads without treating missing feeds as quiet.
- Example County measurement established the explicit `Download limit exceeded` response, immediate stop policy, and a plausible—but unconfirmed—roughly 200-request account window.
