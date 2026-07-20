# Radio Archive Intelligence

An evidence-first local application for finding Broadcastify feeds, retaining premium archives, combining daily audio, transcribing and diarizing radio traffic, and producing auditable incident, daily, weekly, and regional story summaries. The tested Windows experience uses WinUI 3; the same Python backend has a browser companion for Windows, Linux, and macOS plus the original CLI. It stays on loopback by default and can be explicitly hosted on a trusted LAN, including as a persistent TrueNAS App.

The repository keeps its historical `broadcastify-cli` name for compatibility while the product direction is broader than a downloader.

This fork uses Broadcastify's website login and the same private web endpoints as its archive page. It does **not** use Broadcastify's official API. Those endpoints can change, so all site-specific behavior is isolated in `broadcastify_cli/broadcastify.py`.

## What is implemented

- Native WinUI 3 desktop UI on .NET 10 and current stable Windows App SDK 2.3.1
- A responsive browser UI with the same Library, New Archive, Review & Ask, Area Watch, and Settings workflow for cross-platform use; loopback is the default and trusted-LAN hosting is explicit
- A per-user Linux systemd launcher with install/start/stop/status/log commands, failure restart, private settings, and a configurable loopback or trusted-LAN service boundary
- A TrueNAS Apps deployment with an immutable Vulkan image, persistent host-path data, AMD render-device access, and a LAN-hosted UI managed by TrueNAS
- Feed search by agency, city, county, state, or ZIP, including the county-directory matches returned by the website
- Premium website sign-in with an opt-in Windows Credential Locker login for automatic session refresh
- A navigable WinUI shell for Local Library, New Archive, Review & Ask, Area Watch, and Settings instead of one long scrolling workspace
- A master/detail local-library viewer for every retained feed/day, with durable feed names, a processing timeline, combined-audio playback, timestamped transcript preview, storage paths, and the exact resumable next action; older analysis versions are withheld and fast-preview speaker labels expose an explicit Community-1 upgrade
- Diarization-only continuation for existing transcripts, so adding or improving speaker labels does not rerun Whisper
- Inclusive single-day or date-range archive jobs that acquire the range before starting GPU processing
- Paced archive downloads with timezone-aware cache reuse, shared 429 cooldown, `Retry-After` support, and automatic one-at-a-time fallback
- Trusted-LAN archive reuse before website login/download, with optional one-hop discovery, explicit private peers, read-only Windows/Web/TrueNAS seed nodes, SHA-256 verification, and atomic local publication
- Optional daily MP3 combination with safe source cleanup
- GPU-accelerated `faster-whisper` transcription
- Local pyannote Community-1 accuracy-default diarization plus an opt-in public-model sherpa-onnx fast CPU preview
- Timestamped text plus structured JSON output
- Versioned continuous-audio manifests that trim encoder overlap and measure real media duration across feed outages
- Persistent SQLite evidence store with FTS5 text search
- Local BGE embeddings for semantic retrieval
- Quantized Gemma 4 through llama.cpp for incident extraction, daily briefs, and range Q&A
- Opt-in OpenAI Responses, OpenAI-compatible, and saved-login Codex CLI analysis providers behind the same evidence schema
- A native **Analysis & AI** Settings tab with explicit transcript-sharing consent, no-usage readiness checks, session/env keys, and optional Windows Credential Locker storage
- A five-step first-run setup overview in both UIs covering account, storage, transcription, speaker labels, and analysis, with the complete model-stage verifier and the exact first recovery action directly on Setup; Web Settings uses persistent, deep-linked Setup, Processing, Analysis & AI, and Account sections instead of one long page
- One-click joined hardware-profile verification plus individual transcription, speaker-label, and analysis execution tests; all use generated local fixtures and never consume Broadcastify archive quota
- Cache/resume behavior for downloads, combined audio, transcripts, speaker-label chunks, embeddings, incidents, and summaries; speaker and model-window checkpoints are committed incrementally and retained until their final stage cache is safe
- Automatic transcript import, incident extraction, summary, and semantic indexing after a completed UI job
- Saved-day review with priority filters, all-priority text search, redacted cited-radio quotes, exact evidence clips, optional surrounding radio context, and evidence-grounded date-range questions
- Persisted seven-day activity briefs with explicit coverage gaps, exact category counts, and deterministic notable-record IDs
- Center-plus-radius or ordered-ZIP area watch that follows Broadcastify's ZIP-to-county results, deduplicates feeds, defaults to a live public-safety filter, and saves explicit nearest-first newsroom feed profiles
- Persisted nearest-first multi-feed archive queues that recover interrupted work, skip completed feeds, and stop every lower priority at the first explicit quota response
- Sequential post-transcription analysis plus persisted cross-feed story briefs with deterministic source references, coverage gaps, and an evidence-first ranked-list/selected-evidence master/detail viewer in both UIs; generated assignment prose is collapsed by default
- Newsworthiness ranking separate from dispatch priority, with conservative time/location clustering and routine single-person calls suppressed
- One-click incident playback that cuts and plays a compact clip around the strongest cited transcript segment, plus a clearly labeled five-minutes-before/one-minute-after context option for hearing an earlier dispatch without treating surrounding chatter as evidence
- Native **Export clip…** actions for saved incidents and area-story evidence; exports are local MP3 copies and never trigger an archive download
- Evidence packages on every ranked area lead: obvious-identifier-redacted ASR excerpts, timestamped source clips, confidence, diarization status, and SHA-256 provenance
- Subscription-ready neighborhood and topic tags, with every candidate held in `review_required` status until an editor verifies it
- CLI access to the same Python backend

The saved Broadcastify session is stored in the ignored `cookies.json` file. Use only your own premium account and follow Broadcastify's terms for archive access.

The durable project objective, feature matrix, known blockers, and dated validation log live in [GOAL.md](GOAL.md), [FEATURES.md](FEATURES.md), [BUGS.md](BUGS.md), and [PROGRESS.md](PROGRESS.md). Update those files with material backend or UX work so an installed runtime is never confused with an actually validated workflow.

Broadcastify's published terms restrict commercial use and AI/ML use without a license. Personal experimentation and a commercial newsroom product are not the same authorization; obtain written licensing from Broadcastify before deploying this workflow commercially. See [docs/rate-limits.md](docs/rate-limits.md) for the public guidance and measured archive-quota behavior used by the downloader, and [docs/lan-archive-sync.md](docs/lan-archive-sync.md) for the trusted-LAN reuse boundary.

## Why WinUI 3 on Windows

WinUI 3 remains the right native choice for this application. Microsoft's current [Windows application guidance](https://learn.microsoft.com/en-us/windows/apps/) recommends WinUI with the Windows App SDK for new native Windows apps, while WinForms remains supported for existing or simpler desktop applications. WinUI supplies the current Fluent controls, modern DPI behavior, and Windows App SDK lifecycle without turning the desktop shell into a browser wrapper. Both native projects now use current stable [Windows App SDK 2.3.1](https://github.com/microsoft/WindowsAppSDK/releases/tag/v2.3.1) and BuildTools 10.0.28000.2270.

The Windows UI is an unpackaged x64 desktop app, so development does not require MSIX. A verified `dotnet publish` folder is available, including the namespaced Windows ML helper, but it still uses the repository Python environment and is not yet a standalone installer. The browser companion is the portability surface; WinUI remains the polished Windows reference instead of forcing Windows users into a generic wrapper.

## Local model stack

The default stack is split by responsibility so each stage can be fast and replaced independently:

- `faster-whisper` with the Whisper `turbo` model for transcription
- `pyannote/speaker-diarization-community-1` for speaker boundaries and labels
- BAAI `bge-small-en-v1.5` through FastEmbed for CPU semantic retrieval
- Gemma 4 12B Instruct `Q4_0` through llama.cpp for structured event extraction, summaries, and questions
- CUDA float16 batched ASR; quantized local LLM inference to conserve VRAM

Hardware profiles are stage-specific rather than an all-or-nothing GPU switch. The reference Windows profile uses CUDA for faster-whisper and Community-1 plus llama.cpp's detected GPU backend. Vulkan, OpenVINO, Windows ML, Apple Metal, and Qwen presets combine their selected ASR/analysis path with the much faster sherpa-onnx CPU speaker **preview**; this is portable end-to-end behavior, not a claim that those accelerator APIs run diarization. Every preview transcript remains locally upgradeable to Community-1 without repeating ASR. Exact commit `historical-validation` recorded the earlier Community-1 AMD proof—2.314-second Vulkan ASR, 3.031-second CPU diarization, and 5.548-second Vulkan analysis—while exact `historical-validation` completed the current preview profile in 9.199 seconds. The retained 23.9-hour speaker run then completed in 1,951.971 seconds at 0.0227 RTF, covered 91.05% of Community-1 speech and 93.23% of timestamped word midpoints, and duplicated no ownership across 95 chunk boundaries. Exact `historical-validation` added atomic per-chunk resume retained through the final cache commit and passed a real process-kill/resume proof. Earlier protected retained-audio proofs remain applicable: `historical-validation` for AMD Web, `historical-validation` for all CPU, and `historical-validation` for OpenVINO and Windows ML. Metal still awaits a real Mac run. See [docs/hardware-backends.md](docs/hardware-backends.md), [FEATURES.md](FEATURES.md), and [BUGS.md](BUGS.md) for the setup and honest validation matrix.

ROCm 7.2 now officially includes the NAS's gfx1150/Radeon 890M class, but AMD's Ryzen Linux matrix is limited to Ubuntu 24.04.3 while this appliance runs TrueNAS/Debian. A no-install official-container check did work; on the same Gemma 3 1B Q4 model it measured 1,745/69.7 prompt/generation tokens per second versus 2,934/79.5 for the existing Vulkan build, and its image expanded to 25.1 GB. Because the builds differ this is a local decision measurement, not a universal ROCm ranking. Vulkan remains the default recommendation for this NAS; ROCm is an expert alternative for a supported OS or a model that wins a direct benchmark.

Both UIs provide a primary **Verify model stages** action on first-run Setup and a matching **Verify profile** action beside the advanced hardware controls, plus explicit **Test engine**, **Test speakers**, and **Test analysis** actions. The joined action runs those generated-input proofs sequentially, releases accelerator memory between stages, preserves completed-stage evidence, bounds oversized native diagnostic dumps, and stops at the first actionable setup failure. Detection and failure results also return one exact recovery action: runtime/helper/dependency setup, managed-model preparation, speaker setup, analysis-provider setup, or a focused engine diagnostic. A model-download/build button is not shown until the corresponding runtime can use it. The transcription action decodes generated silence through the exact selected ASR engine/model/path/device and reports the actual backend and fallback; the UI explicitly says that this proves execution, not radio-word accuracy. Windows ML can build a selected managed graph, whisper.cpp can download its selected official GGML file, and Qwen3-ASR can acquire its pinned INT8 graph plus VAD only after the user starts the preparation action. The speaker stage executes either Community-1 on CUDA/CPU or the selected sherpa-onnx preview on CPU; the latter acquires only fixed-hash public assets after the explicit test. The analysis stage requests a tiny structured result without sending archive evidence. Ordinary hardware detection never starts a model or provider download. A profile is **Verified** only after all three execution stages pass with the current settings in the current session.

Whisper remains the validated portable evidence default because the same family has mature CUDA/CPU, Vulkan, Metal, OpenVINO, and ONNX deployment paths. Sparse day-long scanner timelines require speech-aware decoding rather than feeding quiet hours to a small model as one continuous context. The TrueNAS preset therefore uses a fixed-hash Base English Q5_1 model and fixed-hash Silero VAD, bounds detected speech regions to 25 seconds, disables previous-text conditioning, and refuses collapsed or music-only transcript caches before they can reach incident analysis. JSON/TXT publication is atomic and binds the rendered text by SHA-256. Whisper is no longer the only serious local option. The app now includes an opt-in **Fast CPU preview (Qwen3-ASR)** profile. It installs the pinned public [Qwen3-ASR 0.6B](https://github.com/QwenLM/Qwen3-ASR) INT8 sherpa-onnx export and Silero VAD only after **Download & test model**, verifies their fixed byte counts and SHA-256 values, stages extraction safely, and records exact model/runtime identity. The retained-radio evaluation preserved the core event in 13 evidence clips at about 0.08 real-time factor on CPU and recovered consequential phrases including `shots fired`, `Xanax`, and `stolen squad car`. A real joined Windows profile proof completed Qwen CPU, Community-1 CPU, and local Gemma in **15.265 seconds**. Exact pushed source also installed and ran on a 12-core AMD/Linux appliance: four threads processed a five-minute sparse-radio slice in **7.128 seconds** with about **1.58 GiB** peak RSS.

The production caveat is explicit: this sherpa export emits no token timestamps. When Community-1 is enabled, Qwen decodes its exclusive speaker-turn regions and persists those source bounds plus anonymous speaker labels; otherwise Silero speech bounds are used. No synthetic word times are invented. Sub-0.75-second diarization fragments are excluded from this evidence-oriented preview because retained testing showed the generative decoder turning weak radio bursts into plausible filler. The AMD five-minute run demonstrated why: it retained the event concept but changed the critical line to `shops fired` / `nothing seems`; the tighter Community-1 event bound preserved `15 to 20 shots fired, nothing seen` while still generating plausible surrounding words. The validated Turbo transcript is not invalidated, and Qwen still needs longer-corpus false-claim/recall gates before any default change. Qwen's separate official 0.6B forced aligner was measured at 2.199 seconds on RTX 3090 and 19.693 seconds on CPU for five minutes, but it assigned normal timestamps to deliberately wrong words and exposes no mismatch confidence. Its 1.84 GB model can eventually improve seeking; it cannot validate a quote or replace the audio clip.

Growing current-day archives resume conservatively. A completed portable diarization cache records the exact runtime/model identity plus a SHA-256 signature of every retained source block and its manifest timeline. When blocks are appended, the app reuses only chunks whose complete overlap lies inside the verified unchanged prefix; any changed block or uncertain boundary is recomputed. LLM classification similarly retains a prior window only when its model, prompt, index, and input fingerprint all match the current transcript revision. This avoids repeating a completed day without weakening the evidence boundary.

There is now a credible unified local alternative as well. Microsoft's MIT-licensed [VibeVoice-ASR-HF](https://huggingface.co/microsoft/VibeVoice-ASR-HF) is an 8B BF16 model that directly generates structured speaker, timestamp, and content records for up to 60 minutes. An isolated retained-radio run on the RTX 3090 preserved `15 to 20 shots fired, nothing seen`, took **8.751 seconds for 25.44 seconds** of audio (0.344 RTF), and peaked at **15.971 GiB** of allocated GPU memory after a 79-second NAS-backed cold load. A 26.56-second two-speaker fire clip took 8.676 seconds and found two speakers/the fire report, but changed the address and omitted the later disregard. A sparse 120-second stolen-squad-car clip took 8.755 seconds and kept the event, but merged two Community-1 speakers and invented an introductory name/title. Peak allocation reached 16.58 GiB. That makes VibeVoice the leading high-end unified research candidate, not the fast or safest default. Its Transformers 5.3 runtime also conflicts with the app's currently validated Windows ML Transformers 4.x environment, so any product integration must use an isolated sidecar/runtime and pass the same retained evidence gates.

Diarization therefore remains a separate default stage. sherpa-onnx 1.13.4's segmentation/embedding/clustering path is a validated opt-in preview: the production adapter processed the retained five-minute slice in **5.812 seconds** on CPU and found 19 turns, while fresh [Community-1](https://huggingface.co/pyannote/speaker-diarization-community-1) took 68.508 seconds and found 24. A forced five-chunk overlap test returned the same 19 turns in 6.218 seconds. The retained full day returned 3,893 turns in **1,951.971 seconds / 44.05× real time**; comparison to Community-1 showed 91.05% speech coverage and 97.62% candidate precision, while confirming that chunk-local clusters are not stable identities or human-scored DER. The preview safely installs fixed-hash pyannote segmentation 3.0 INT8 (MIT) and NeMo TitaNet-S (Apache-2.0) graphs, bounds long inputs to 15-minute chunks, atomically checkpoints each completed chunk, and scopes anonymous speaker IDs per chunk rather than pretending to identify one person across a day. Community-1 stays the accuracy default, and **Improve speakers** replaces preview labels without repeating ASR. [Parakeet TDT 0.6B v3](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3) was very fast and supplies token timestamps, but blanked on unsegmented sparse-radio input and rendered the key phrase as `drop fire`. NVIDIA's current streaming Sortformer v2.1 plus Multitalker Parakeet is a serious CUDA/NeMo path for overlapping speech, but it consumes external diarization activity and is bounded to a small speaker set rather than solving a day-long many-radio feed portably. WhisperX likewise adds alignment around a separate diarizer. See B-003 and B-009 in [BUGS.md](BUGS.md) for the measured gates.

Embeddings do not replace the generative model. They cheaply retrieve and cluster likely-relevant transcript passages; Gemma turns cited evidence into structured incidents and natural-language answers. SQLite remains the source of truth, including timestamps and transcript evidence, so model output can be audited.

Local Gemma remains the Windows default. API and Codex modes are explicit alternatives for users who prefer a hosted model or an existing Codex subscription login; they never activate merely because a key or login exists. Choose them in **Settings → Analysis & AI**. External providers remain blocked until the transcript-sharing switch is enabled, and **Check provider** verifies configuration without sending transcript text or starting a paid OpenAI model request. See [docs/model-providers.md](docs/model-providers.md) for supported contracts, privacy controls, CLI examples, and validation status.

For a combined daily job, the downloader now concatenates the archive blocks **before** transcription and diarization. This is important: the old order diarized each 30-minute file separately, causing speaker labels and timestamps to restart at every boundary.

## Requirements

- Python 3.12
- FFmpeg executable (the app does not require pyannote/TorchCodec to decode archive audio)
- Broadcastify premium account for archive downloads
- Hugging Face read token only for the first Community-1 download; its complete cache runs offline, while the public sherpa preview needs no token
- llama.cpp for local Gemma analysis

The native Windows shell additionally needs Windows 10 version 1809 or later, x64, and the .NET 10 SDK. NVIDIA CUDA is the tested fast default; CPU, OpenVINO, Windows ML, and Vulkan profiles report their stages separately and fall back only where documented.

Install the Windows prerequisites from PowerShell:

```powershell
winget install --id Python.Python.3.12 -e
winget install --id Gyan.FFmpeg.Shared -e
winget install --id Microsoft.DotNet.SDK.10 -e
winget install --id ggml.llamacpp -e
```

Create the Python environment and install CUDA-enabled dependencies:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install torch==2.11.0+cu128 torchaudio==2.11.0+cu128 --index-url https://download.pytorch.org/whl/cu128
.\.venv\Scripts\python.exe -m pip install -e ".[transcription,analysis,dev]"
```

For the optional fast-CPU Qwen profile, add `.[qwen]`; then select **Fast CPU preview (Qwen3-ASR)** and **Download & test model**. For OpenVINO ASR, add `.[openvino]`. For the Windows ML model builder and helper workflow, add `.[windowsml]`; the native helper's NuGet dependencies are restored by its .NET project.

### Windows ML functional profile

The validated Windows ML path currently uses CPU execution. Windows App SDK 2.3.1 resolves the latest stable NuGet `Microsoft.Windows.AI.MachineLearning` / `Microsoft.WindowsAppSDK.ML` packages at **2.1.74**, and Microsoft's current ONNX-version page now confirms that stable release maps to ORT **1.24.6**; the inference helper retains its separately tested ONNX Runtime GenAI 0.14.1 contract. The newer 2.4.66 preview carries ORT 1.27.1 but is not substituted for the validated stable stack. The helper and official clean-speech fixture proved the persistent C# decoder path. Retained low-SNR radio then showed why execution and quality must be separate: Tiny FP32 returned `[inaudible]`, Tiny INT4 repeated it, multilingual Base captured the core dispatch in 0.212 seconds, and Small took 0.915 seconds without materially improving that phrase. Both UIs start Windows ML at Base and label the generated-silence action as execution-only. Current DML graphs still fail and the available TensorRT RTX provider falls back on unsupported attention nodes, so GPU acceleration remains gated.

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[transcription,windowsml]"
dotnet build .\BroadcastifyCli.WindowsML\BroadcastifyCli.WindowsML.csproj -c Release
```

Choose the Windows ML profile in **Settings → Processing**, then select **Build & test model**. The worker exports the selected public Whisper graph to a staging directory, writes verifiable model/source/provider/precision metadata, atomically installs it under the per-user managed model root, and tests that exact path. A Hugging Face token is inherited only by the child builder when needed and is never placed on its command line; public Whisper export works without one. An advanced explicit path remains supported through **ASR model path**, `WINDOWS_ML_WHISPER_MODEL_PATH`, and `WINDOWS_ML_HELPER_PATH`, but its manifest/architecture must match the selected model.

**Test engine** loads a one-second generated-silence fixture through the exact configured model. A successful result proves only that this model/runtime/provider executed locally; use retained radio to judge accuracy. Microsoft's [Windows ML Model Catalog](https://learn.microsoft.com/en-us/windows/ai/new-windows-ml/model-catalog/overview) is the preferred future distribution surface for a published, checksum-validated graph, but it does not convert an arbitrary Hugging Face model, so preparation remains explicit today.

The helper can inspect Windows ML's provider catalog without downloading anything. Activating an installed provider is separate from explicitly allowing provider acquisition:

```powershell
$helper = ".\BroadcastifyCli.WindowsML\bin\Release\net10.0-windows10.0.26100.0\win-x64\BroadcastifyCli.WindowsML.exe"
& $helper --providers
& $helper --register-winml --providers  # installed packages only
& $helper --ensure-winml --providers    # may download certified provider packages
```

These commands are diagnostic/setup tools; ordinary CPU transcription does not acquire a provider. See [hardware backend evidence](docs/hardware-backends.md#windows-ml-and-onnx-runtime-genai) for the measured DML and TensorRT RTX limitations.

Build and launch the desktop app:

```powershell
dotnet build .\BroadcastifyCli.WinUI\BroadcastifyCli.WinUI.csproj -c Release
& ".\BroadcastifyCli.WinUI\bin\Release\net10.0-windows10.0.26100.0\win-x64\Broadcastify Desktop.exe"
```

Create and verify a normal publish folder:

```powershell
.\scripts\verify_windows_publish.ps1 `
  -OutputDirectory .\BroadcastifyCli.WinUI\bin\Publish\win-x64
```

For a private personal publish that carries the ignored repository `.env` alongside the executable, opt in explicitly:

```powershell
.\scripts\verify_windows_publish.ps1 `
  -OutputDirectory .\BroadcastifyCli.WinUI\bin\Private\publish-win-x64 `
  -BundleLocalEnv
```

This writes `broadcastify-desktop.env` into the dedicated private publish output and verifies it against the ignored source by hash without displaying either file. It is still a plaintext credential file inside that private directory: do not distribute or upload it. A normal verifier run rejects and removes a stale private environment. The verifier also checks the compiled WinUI resources and runs the published Windows ML helper probe. See [docs/windows-publish.md](docs/windows-publish.md) for the verified layout and remaining source-tree/Python boundary.

## Cross-platform local Web UI

The browser companion is served by Python and binds to loopback by default:

```powershell
.\.venv\Scripts\broadcastify-web.exe --open
```

On Linux or macOS, use the equivalent environment entry point:

```bash
./.venv/bin/broadcastify-web --open
```

An explicit private/link-local address or `0.0.0.0` enables trusted-LAN mode:

```bash
./.venv/bin/broadcastify-web --host 0.0.0.0 --port 8765
```

Anyone who can reach a LAN listener can open the page, receive a session
cookie/token, read retained audio/transcripts, and start supported jobs. The
cookie/token, same-origin, CSP, media-containment, and quota controls still
apply, but they are not user authentication. Use LAN mode only on a trusted
network, bind/publish it on the intended private interface, and never
port-forward or publicly proxy it without adding real authentication and TLS.

On TrueNAS SCALE, use the supported **Apps** path instead of the experimental
Containers/LXC feature or a hand-managed systemd process. The reference Custom
App runs the UI and AMD Vulkan compute on the NAS, mounts its archive/model
dataset at `/data`, and publishes only the selected NAS LAN address. See the
[TrueNAS Apps deployment guide](deploy/truenas/README.md).

On a Linux desktop with a systemd user session, the installed package can
supervise the same Web app and reopen it after a process failure:

```bash
./.venv/bin/radio-archive-service install
./.venv/bin/radio-archive-service status
./.venv/bin/radio-archive-service start --open
```

The default user-owned layout is `~/.local/share/radio-archive` for the
working/archive data and `~/.config/radio-archive` for service settings. The
installer creates a comment-only, mode-0600 `.env` template; credentials are
read by the app from that file and are never copied into the systemd unit or
service JSON. Use `--working-dir`, `--output-dir`, `--database`, or
`--env-file` to adopt an existing library. Uninstalling the service preserves
the archive, models, settings, and `.env`. See
[the Linux service guide](docs/linux-service.md) for installation choices,
logs, headless sessions, and the non-systemd fallback.

For headless validation or automation, the repository also includes a real Web-job harness. It starts the loopback service on an ephemeral port, performs the cookie/action-token handshake, submits one supported job, polls it, and emits the final JSON snapshot:

```bash
python scripts/web_job_smoke.py diagnostics \
  --output-dir archives \
  --working-dir . \
  --result-file web-job-result.json
```

Use `continue-local` with `--payload-file` to exercise a retained day without downloading anything. The harness uses the same `/api/jobs` boundary as the browser; it is not a shortcut around its security checks.

It exposes the retained Library and processing timeline, byte-range audio playback, bounded incident and transcript viewers, exact clip preparation, website feed search, guarded archive jobs, local continuation, provider-aware Q&A and weekly summaries, area profiles, retained regional briefs with redacted quote/provenance/clip packages, hardware diagnostics, and session-only sign-in. Settings is divided into keyboard-accessible Setup, Processing, Analysis & AI, and Account sections; direct URLs, Back/Forward, and the last selected section are preserved. Media links are archive-relative and never serialize local filesystem paths. One heavy job can run at a time, every archive job is forced to one download worker, and the explicit Broadcastify quota response still stops the range immediately.

Every launch creates a random session token. API and media routes require the same-site session cookie; actions also require the token header and reject cross-origin submissions. Non-secret settings use browser-local storage. API keys and first-download Hugging Face tokens remain in the active tab or an ignored `.env`; once Community-1 is fully cached, diarization can run offline without supplying the token again. The cross-platform UI does not claim OS-keychain persistence yet. See [docs/web-ui.md](docs/web-ui.md) for the runtime contract and current portability boundary.

## Windows first run

1. Open **Settings → Setup**. The five-step overview checks account, writable storage, the selected transcription path, the selected speaker-label engine, and the selected analysis provider. Detection does not download models or touch Broadcastify archive quota.
2. Select the account item, enter the premium Broadcastify login, and leave **Save this login securely** enabled to keep it encrypted for this Windows account in Windows Credential Locker and automatically refresh an expired session. The session cookie remains in the ignored `cookies.json` file.
3. Keep **Community-1 — accuracy default** for the reference CUDA path. For its first download, accept the access conditions on the [Community-1 model page](https://huggingface.co/pyannote/speaker-diarization-community-1), create a read token, and paste it into **Settings**. The token is not saved by the UI; an environment token can instead stay in the ignored `.env`. The **Fast portable preview — CPU** choice uses public checksum-pinned assets and no token, but can miss short turns.
4. Open **New archive**, search for and select a feed, then choose the inclusive date range. The output folder and model/device defaults are in **Settings**.
5. Enable **Create combined MP3** when you want one continuous timeline. Diarization requires this setting and the UI keeps it enabled so speaker labels cannot restart at archive boundaries.
6. Leave `turbo`, `auto`, and batch size 8 selected for the RTX 3090 starting point. Choose **Verify profile** to prove transcription, speaker labels, and analysis together without using archive audio; use the individual tests when diagnosing one stage.
7. Leave **Extract incidents, summarize, and index after transcription** selected for the complete automatic workflow. Saved days can be reanalyzed manually later without downloading or transcribing again.
8. Open **Local library** at any time to browse retained feed-days. Feed identity is kept in both the persistent catalog and new combined-audio manifests, so human-readable names survive normal catalog rebuilds. Selecting a day shows its five-stage timeline, combined recording, and timestamped transcript preview. **Verify & resume** uses the guarded downloader only when coverage is incomplete; later stages continue entirely locally. A missing speaker pass does not repeat ASR, and a preview-labeled day exposes **Improve speakers** for the same no-ASR Community-1 upgrade.
9. For regional monitoring, open **Area watch**, enter a center ZIP and radius (or choose an exact ordered ZIP list), then explicitly select and save the police/fire agencies for a named profile. Radius discovery caches the small Census ZCTA centroid file and labels its mileage as an approximation; it never downloads archive audio. **Run or resume nearest-first queue** persists every feed/date stop point, reuses the exact cache, and stops the entire profile at the first explicit quota response. Completed transcripts can then be analyzed sequentially. **Find story leads** ranks all retained incidents and prepares playable evidence packages. See [docs/area-coverage.md](docs/area-coverage.md).

Runtime-managed models use their normal local caches the first time they are used. Managed Windows ML, whisper.cpp, Qwen, and portable-speaker assets are instead acquired only by their explicit preparation/test buttons. The first transcription or diarization run will therefore take longer to start. A cached gated Community-1 model remains usable offline; the read token is required for acquisition, not every inference. Qwen and the portable speaker preview use public assets and do not need the Hugging Face token.

Community-1 full-day diarization reports progress for segmentation, speaker counting, and embeddings. Its lossless 16 kHz preparation is atomically cached and retained after an interruption or pipeline error. The app then gives pyannote a memory-mapped waveform dictionary decoded by FFmpeg instead of a filename, avoiding TorchCodec/FFmpeg-DLL compatibility failures while keeping day-long PCM off the Python heap. The raw PCM scratch file is removed after the model call. The app respects a downloaded pipeline's tuned embedding batch size and only raises it when the selected batch is larger.

The portable preview takes a different bounded-memory route: FFmpeg decodes at most a 15-minute 16 kHz float32 chunk plus five seconds of overlap on each edge, sherpa runs segmentation/embedding/clustering, and midpoint ownership keeps overlap turns exactly once. IDs such as `SPEAKER_C003_01` are anonymous clusters scoped to that chunk, not people tracked across the day. Setting minimum and maximum speakers to the same number requests an exact cluster count; a wider range cannot be represented by sherpa's current clustering API and uses the recorded threshold instead.

Archive downloads are resumable and normally start at least five seconds apart. Current URL IDs and server filenames use different identifiers, so cache reuse resolves the URL timestamp in the feed's own published time zone instead of redownloading completed blocks. If Broadcastify rate-limits a job, the app pauses all queued requests for the requested backoff or a conservative 30-second-to-5-minute exponential delay, retries transient failures, and runs the rest of the date-range job one at a time. The progress count includes only successfully saved or cached archives.

Broadcastify can also return HTTP 429 with the explicit body `Download limit exceeded` and no reset time. That is treated as an exhausted account/IP archive quota rather than a transient request-rate response: the app stops immediately, preserves completed files, prevents queued workers and lower-priority area feeds from making more requests, and directs the user to Broadcastify support for quota details. Date-range jobs acquire archives before loading the transcription model, so a long diarization pass cannot delay the next guarded download. If the quota interrupts a range, the app makes no further archive-download requests, reuses any later days already complete in the local cache, processes those complete days, and reports the exact coverage gap. A later rerun resumes from the exact server timestamp and feed-timezone cache key. Progress says **Ready (cached or downloaded)** because a cache hit must not be misreported as new quota use.

Outputs are grouped by feed and date:

```text
archives/
  5318/
    20260713/
      combined_5318_20260713.mp3
      combined_5318_20260713.manifest.json
      evidence-clips/
        5318_2026-07-13_I42.mp3
      transcripts/
        combined_5318_20260713.json
        combined_5318_20260713.txt
```

The text transcript uses entries such as:

```text
[00:03:21.420] SPEAKER_01: Unit 12, copy that.
```

Speaker numbers are anonymous clustering labels, not identified people or radio units.

Persistent analysis defaults to `archives/broadcastify-analysis.sqlite3`. It stores feed days, transcript segments, passages, incidents, daily/weekly briefs, area profiles, area story digests, embeddings, and question history. Every displayed analysis is version-gated: when evidence rules change, retained audio/transcripts remain reusable but older incident, weekly, and area claims are hidden until local reanalysis completes. Raw transcript evidence can contain names or other details spoken on the radio. Public incident fields and displayed quotes apply deterministic obvious-identifier and context-supported private-name redaction, including bounded names immediately following a known incident location. The daily browser boundary assembles only evidence overlapping the exact citation clip, and both daily and area JSON serializers apply redaction again rather than returning raw SQLite text. The original ASR remains internal for audit. Raw clips can still contain spoken identifiers, so they remain review aids rather than publication assets.

Gemma output is not accepted on schema shape alone. Each incident must cite nearby exact transcript segments, share meaningful claim anchors with those citations, and carry any critical claim—such as shots, a weapon, theft, assault, fire, pursuit, collision, overdose, welfare, or trespass—in the cited ASR itself. Clear category/priority contradictions are corrected from evidence, unsupported outcome language is removed or rejected, and daily briefs fall back to deterministic evidence summaries when grounding checks fail. Reports still describe unconfirmed, noisy dispatch traffic rather than findings of fact.

ZIP discovery mirrors the current Broadcastify website flow: a ZIP search first resolves to one or more county-directory links, and the app then parses and deduplicates those county feed tables. The public-safety-only filter is enabled by default. Nearby ZIPs in the same county intentionally produce one feed entry tagged with every matching ZIP rather than duplicate archive jobs.

Area story clustering is conservative and auditable. A matching event category and nearby time are not sufficient by themselves: reports need a compatible reported location or unusually strong descriptive overlap. Cross-feed overlap raises newsworthiness but is never presented as independent confirmation because feeds can rebroadcast the same talkgroup. Exact feed and incident IDs remain attached to each story card. Each source record includes a redacted ASR quote, a cached exact-evidence clip cut from the retained combined audio, model confidence, anonymous speaker-cluster context when available, and source/clip hashes. Gemma only writes the readable assignment brief; both viewers collapse that secondary narrative by default so the publication warning, coverage, and source package stay primary.

Neighborhood/topic tags are generated for audience matching, but scanner-derived items are never automatically publication-ready. A future public subscription surface should accept a neighborhood or radius and desired topics, then deliver only editor-approved stories. Raw clips remain an internal verification aid because radio audio can contain private identifiers or unverified allegations.

## CLI

Search for a feed:

```powershell
.\.venv\Scripts\broadcastify-cli.exe search "Dallas Police"
```

Download, combine, transcribe, and diarize a date range:

```powershell
.\.venv\Scripts\broadcastify-cli.exe download `
  --feed-id 5318 `
  --range 2026-07-12:2026-07-13 `
  --combine `
  --transcribe `
  --diarize `
  --device cuda `
  --model turbo
```

For environment-based sign-in, copy `.env-example` to `.env` and set `BROADCASTIFY_USERNAME` and `BROADCASTIFY_PASSWORD`. The legacy `USERNAME`/`PASSWORD` names remain supported. Set `HUGGINGFACE_TOKEN` for the first pyannote download unless the read token is entered in the UI for that session; a complete cached snapshot can later run offline without it. Optional runtime overrides are `WHISPER_CPP_PATH`, `WHISPER_CPP_MODEL_PATH`, `WHISPER_CPP_VAD_MODEL_PATH`, `OPENVINO_WHISPER_MODEL_PATH`, `QWEN3_ASR_MODEL_PATH`, `QWEN3_ASR_VAD_PATH`, `QWEN3_ASR_THREADS`, `BROADCASTIFY_MODEL_DIR`, `LLAMA_SERVER_PATH`, and `FFMPEG_PATH`; blank values preserve automatic discovery. A hosted Web deployment can publish a validated Automatic preset with the `BROADCASTIFY_DEFAULT_*` processing variables shown in `.env-example`. The server applies that preset only to Automatic (including a pre-upgrade browser that sent the old automatic payload); an explicit custom or named profile is preserved. Immutable Linux hosts can explicitly select a pre-pulled whisper.cpp image with `WHISPER_CPP_CONTAINER_IMAGE`; the app never performs an implicit image pull. Optional analysis-provider variables are documented in [docs/model-providers.md](docs/model-providers.md), and native/container hardware setup is in [docs/hardware-backends.md](docs/hardware-backends.md). Credential files, the Windows-build copy, and the session-cookie file are ignored by Git.

Import, classify, summarize, and embed a completed day:

```powershell
.\.venv\Scripts\broadcastify-analysis.exe analyze-day `
  --feed-id 90001 `
  --date 2026-07-12
```

Ask an evidence-grounded question over a date range:

```powershell
.\.venv\Scripts\broadcastify-analysis.exe ask `
  --feed-id 90001 `
  --start-date 2026-07-11 `
  --end-date 2026-07-12 `
  --question "What were the most serious reported incidents, and which remained unconfirmed?"
```

Generate or reuse a persisted seven-day brief ending on a selected date:

```powershell
.\.venv\Scripts\broadcastify-analysis.exe summarize-week `
  --feed-id 90001 `
  --week-ending 2026-07-12
```

The weekly brief reuses daily incidents and summaries; it does not rerun transcription. If fewer than seven saved days are available, the report names every missing date and never treats missing coverage as inactivity. Use `--force` to refresh the narrative after changing the weekly prompt.

The first analysis run downloads the quantized `ggml-org/gemma-4-12B-it-GGUF:Q4_0` model and the small BGE embedding model into their normal caches. If the former `Q4_K_M` default is still saved in settings, the app reuses that exact file when it is already in the Hugging Face cache; otherwise it migrates to `Q4_0`. An explicit local `.gguf` path is also accepted for fully offline deployments. `analyze-day` resumes from saved incidents after a summary-only failure; use `--force-summary` to refresh only a brief or `--force` to rebuild all derived incidents for the day. Show persistent counts with:

```powershell
.\.venv\Scripts\broadcastify-analysis.exe stats
```

List every saved eventful incident in a day's report without starting either model:

```powershell
.\.venv\Scripts\broadcastify-analysis.exe report-day `
  --feed-id 90001 `
  --date 2026-07-12 `
  --min-priority 3
```

## Validated feed run

Feed `90001` was exercised end-to-end over July 11–12, 2026 through the authenticated website endpoints:

- 97 archive blocks retained locally across two days
- continuous daily audio reached the end of both timelines (the old first-block decoding failure is fixed)
- 1,716 transcript segments and 275 semantic passages persisted
- 87 evidence-backed, deduplicated incident records plus two daily summaries persisted
- cross-day semantic Q&A returned incident citations and distinguished an initial stabbing report from its later radio clarification
- an unchanged full-day download/combine/transcribe rerun dropped from about 232 seconds to about 12 seconds through cache hits
- the July 12 combined 24.4-hour file completed real pyannote diarization in about 25 minutes on the RTX 3090
- that diarization produced 5,556 speaker turns across five anonymous clusters, spanned the complete daily timeline, and left zero transcript words without a speaker label
- after a later TorchCodec/PyTorch mismatch reproduced the file-loader failure, the waveform-dictionary fix passed the generated-audio CUDA self-test in 6.5 seconds and a 250-second retained combined-audio slice in 11.75 seconds, producing 52 turns across three clusters
- cached diarization, transcription, analysis, daily reports, and local Q&A were exercised through the WinUI Release build
- a persisted seven-day brief ending July 12 correctly reported two available days, five missing dates, 87 incidents, 22 priority 4–5 records, and a deterministic list of notable incident IDs
- a live six-ZIP Example City/East Example City/Example City discovery followed Example City and Example County county directories, found the current feed 90001 plus nearby police/fire feeds, and persisted a six-feed `Regional coverage desk` profile
- the first regional brief correctly reported only 2 of 24 feed-days and 1 of 6 feeds with data, ranked 30 newsroom leads from 87 saved incidents, and retained feed/incident references without treating the five missing feeds as quiet
- ranked area leads now retain transcript quotes and generate independently playable, hashed context clips from the matching combined-audio timestamps
- an evidence-v9 refresh acquired all 97 July 15–16 blocks with no 429 and retained 64 supported incidents; the current evidence-v10 finalization adds a narrow exact-phrase recall pass for model-omitted critical calls while preserving the same transcript/audio citation gates, and the quota-free current v8 area rebuild produces 27 ranked leads, 27 references, 27 exact clips, and eight redacted public quotes across 2/2 feed-days while keeping source ASR internal
- retained evidence paths now rebase safely when the same library moves between Windows, Linux, and TrueNAS; the live Example Township I864 card played its citation and exported a hash-verified 25.2-second exact clip after the migration
- the seven-day brief ending July 16 uses those two current days, flags July 11–12 for local analysis updates, names July 10 and July 13–14 as missing, and never treats unavailable coverage as quiet

The five speaker labels are acoustic clusters rather than identified officers or radio unit IDs. Radio compression, overlapping traffic, dispatch consoles, and repeated users of the same equipment can split or merge real speakers, so they should be used as conversation structure rather than identity evidence.

### Joined immutable Linux run

Exact commit `historical-validation` passed all 113 tests inside the recorded whisper.cpp Vulkan image. The headless Web harness then established the real loopback cookie/action-token session and submitted `continue-local` for a fresh 30-second retained-radio fixture. With networking disabled and no Hugging Face token supplied, cached CPU pyannote produced five turns, whisper.cpp retained explicit `Vulkan0 / AMD Radeon 890M Graphics` evidence, local Gemma loaded on the AMD Vulkan device, and SQLite retained one transcript segment, one semantic passage/embedding, and one daily summary. The model correctly returned zero supported incidents for the tiny fixture. Container time was 15.139 seconds and the final local-library state was 100% **Ready to review**.

The container used a read-only root, numeric user, explicit render groups, all capabilities dropped, `no-new-privileges`, bounded tmpfs, PID limit 1024, and no network. The source/model datasets remained unchanged; only user-owned isolated runtime, cache, transcript, database, and log paths were writable. That checkpoint validated the joined HTTP/worker backend; later exact commits `historical-validation` and `historical-validation` respectively validated the managed Linux launcher and full-model visual/accessibility experience. Real macOS behavior remains open.

### Linux full-model visual and accessibility pass

Exact commit `historical-validation4db7f9aa038a0911490f7f366d7af0907` passed all 148 tests on the TrueNAS/Linux AMD host and reopened the retained current-analysis fixture through the loopback Web service. Chromium 148 exercised Local Library, Review & Ask, Area Watch, Settings, retained-media playback, mobile navigation, and keyboard traversal at 1365×900 and 390×844. Every view had exact document/client width, no horizontal overflow, duplicate IDs, unnamed visible buttons, or unlabeled visible controls. The only initial browser error was a missing favicon; the pushed SVG asset and serving test removed it. The final run had zero failures and no warning/error/severe console entries. The service stayed on `127.0.0.1`; the browser ran on the same host in a read-only, capability-dropped container.

### Bounded all-CPU Linux run

Exact commit `historical-validation` passed all 119 tests inside the same immutable image and then completed the protected `continue-local` Web job with no GPU devices exposed. whisper.cpp recorded CPU backend/device and `No devices found`, pyannote produced five CPU speaker turns, llama.cpp listed only the Ryzen CPU and generated around 56-72 tokens/second, and the complete 30-second workflow finished in 21.282 seconds.

The short fixture yielded one evidence-backed weapon-related record from the exact ASR quote, normalized to `Radio traffic reported: a person was chased with a gun.` at confidence 0.90. The local model twice tried to add unsupported daily activity; both responses were rejected and the saved deterministic brief retained only the cited report, explicitly labeled as noisy ASR rather than a confirmed outcome. This proves bounded CPU fallback behavior, not full-day performance.

Run the tests with:

```powershell
.\.venv\Scripts\python.exe -m pytest
```

## Suggested phases

1. **Core workflow:** authenticated feed search/download, overlap-aware continuous combination, fast transcription, persistence, local retrieval, incident extraction, summaries, and Q&A. Implemented and tested on two full days.
2. **Diarization validation:** Community-1 completed a real combined 24.4-hour feed day with continuous labels and cached speaker turns. Implemented and validated.
3. **Analysis UI:** focused WinUI navigation, local pipeline-state library and continuation actions, persisted days, incident timeline, daily/weekly briefs, range questions, model diagnostics, cancellation, automatic post-transcription analysis, multi-ZIP discovery, saved area profiles, and immediately reopened regional lead briefs are implemented and visually exercised in WinUI 3. A responsive loopback browser companion mirrors those surfaces against the same worker contract; its protected full-model job, managed launcher, and desktop/mobile visual-accessibility matrix now pass on real Linux, while real macOS validation remains.
4. **Regional ingestion:** center/radius discovery, explicit nearest-first feed profiles, a persisted shared stop/resume acquisition queue, and sequential analysis are implemented. The next step is a schedule/retention manager so a newsroom can budget the unknown archive quota, storage, and GPU time per profile.
5. **Newsroom product phase:** evidence drill-down, cached story clips, and radius-based coverage markets are implemented. Next add editor approval/redaction/export, profile management, map/jurisdiction review, notifications, a model/cache manager, and installer/MSIX packaging.
6. **Neighborhood subscription phase:** build a separate public web/mobile signup for center-plus-radius and topic preferences; match only editor-approved stories, include unsubscribe/consent controls, and keep raw scanner audio private by default.
7. **Hardware portability phase:** real AMD Vulkan and bounded all-CPU Web jobs are complete, and the fast public-model CPU speaker preview is implemented across Vulkan/OpenVINO/Windows ML/Metal/Qwen profiles with a Community-1 upgrade. Next measure a full day, validate the current preview from exact source on more Intel/macOS hardware, and keep backend claims stage-specific.
8. **Benchmark phase:** compare the saved real-radio corpus against newer ASR/alignment options before changing the `faster-whisper` default; track word quality, missed calls, runtime, and VRAM rather than model-release recency alone.
