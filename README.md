# Broadcastify Desktop

A native Windows app and Python CLI for finding Broadcastify feeds, downloading premium archives, combining daily audio, and producing fast local transcripts with optional speaker diarization.

This fork uses Broadcastify's website login and the same private web endpoints as its archive page. It does **not** use Broadcastify's official API. Those endpoints can change, so all site-specific behavior is isolated in `broadcastify_cli/broadcastify.py`.

## What is implemented

- Native WinUI 3 desktop UI on .NET 10 and Windows App SDK 1.8
- Feed search by agency, city, county, state, or ZIP, including the county-directory matches returned by the website
- Premium website sign-in with an opt-in Windows Credential Locker login for automatic session refresh
- A navigable WinUI shell for Local Library, New Archive, Review & Ask, Area Watch, and Settings instead of one long scrolling workspace
- A master/detail local-library viewer for every retained feed/day, with a processing timeline, combined-audio playback, timestamped transcript preview, storage paths, and the exact resumable next action
- Diarization-only continuation for existing transcripts, so adding speaker labels does not rerun Whisper
- Inclusive single-day or date-range archive jobs that acquire the range before starting GPU processing
- Paced archive downloads with timezone-aware cache reuse, shared 429 cooldown, `Retry-After` support, and automatic one-at-a-time fallback
- Optional daily MP3 combination with safe source cleanup
- GPU-accelerated `faster-whisper` transcription
- Optional local pyannote Community-1 speaker diarization
- Timestamped text plus structured JSON output
- Versioned continuous-audio manifests that trim encoder overlap and measure real media duration across feed outages
- Persistent SQLite evidence store with FTS5 text search
- Local BGE embeddings for semantic retrieval
- Quantized Gemma 4 through llama.cpp for incident extraction, daily briefs, and range Q&A
- Cache/resume behavior for downloads, combined audio, transcripts, embeddings, incidents, and summaries
- Automatic transcript import, incident extraction, summary, and semantic indexing after a completed UI job
- Saved-day review, priority-filtered incident timeline, and evidence-grounded date-range questions in the UI
- Persisted seven-day activity briefs with explicit coverage gaps, exact category counts, and deterministic notable-record IDs
- Multi-ZIP area watch that follows Broadcastify's ZIP-to-county results, deduplicates feeds, and saves explicit newsroom feed profiles
- Sequential multi-feed archive/analyze jobs plus persisted cross-feed story briefs with deterministic source references and coverage gaps
- Newsworthiness ranking separate from dispatch priority, with conservative time/location clustering and routine single-person calls suppressed
- One-click incident playback that cuts and plays a compact clip around the strongest cited transcript segment, avoiding unreliable seeks inside day-long MP3s
- Native **Export clip…** actions for saved incidents and area-story evidence; exports are local MP3 copies and never trigger an archive download
- Evidence packages on every ranked area lead: obvious-identifier-redacted ASR excerpts, timestamped source clips, confidence, diarization status, and SHA-256 provenance
- Subscription-ready neighborhood and topic tags, with every candidate held in `review_required` status until an editor verifies it
- CLI access to the same Python backend

The saved Broadcastify session is stored in the ignored `cookies.json` file. Use only your own premium account and follow Broadcastify's terms for archive access.

The durable project objective, feature matrix, known blockers, and dated validation log live in [GOAL.md](GOAL.md), [FEATURES.md](FEATURES.md), [BUGS.md](BUGS.md), and [PROGRESS.md](PROGRESS.md). Update those files with material backend or UX work so an installed runtime is never confused with an actually validated workflow.

Broadcastify's published terms restrict commercial use and AI/ML use without a license. Personal experimentation and a commercial newsroom product are not the same authorization; obtain written licensing from Broadcastify before deploying this workflow commercially. See [docs/rate-limits.md](docs/rate-limits.md) for the public guidance and measured archive-quota behavior used by the downloader.

## Why WinUI 3

WinUI 3 is Microsoft's current native Windows UI framework and is the right default for this new Windows-only app. It supplies the current Fluent controls, Mica, modern DPI behavior, and Windows App SDK lifecycle while keeping the UI native. WinForms would be simpler for a disposable utility, but it is an older UI stack and was not the best long-term choice here.

The UI is an unpackaged x64 desktop app, so development does not require MSIX. Packaging and an installer can be added after the archive and model workflows are fully validated.

## Local model stack

The default stack is split by responsibility so each stage can be fast and replaced independently:

- `faster-whisper` with the Whisper `turbo` model for transcription
- `pyannote/speaker-diarization-community-1` for speaker boundaries and labels
- BAAI `bge-small-en-v1.5` through FastEmbed for CPU semantic retrieval
- Gemma 4 12B Instruct `Q4_K_M` through llama.cpp for structured event extraction, summaries, and questions
- CUDA float16 batched ASR; quantized local LLM inference to conserve VRAM

Hardware profiles are stage-specific rather than an all-or-nothing GPU switch. The reference Windows profile uses CUDA for faster-whisper and pyannote plus llama.cpp's detected GPU backend. Vulkan uses whisper.cpp and llama.cpp on Vulkan while diarization falls back to CPU. OpenVINO uses the devices exposed by its runtime and retries a rejected accelerator/model pairing on CPU. Windows ML uses the optional ONNX Runtime GenAI helper and does not report ready until its configured model completes a real decode self-test. See [FEATURES.md](FEATURES.md) and [BUGS.md](BUGS.md) for the current validation matrix.

Whisper remains the practical fast ASR default for this radio workflow. Diarization is deliberately separate: current all-in-one audio models do not yet offer a clearly better combination of speed, mature speaker labeling, Windows support, and local deployment. WhisperX can remain an optional future word-alignment mode rather than adding its extra pass to every job.

Embeddings do not replace the generative model. They cheaply retrieve and cluster likely-relevant transcript passages; Gemma turns cited evidence into structured incidents and natural-language answers. SQLite remains the source of truth, including timestamps and transcript evidence, so model output can be audited.

For a combined daily job, the downloader now concatenates the archive blocks **before** transcription and diarization. This is important: the old order diarized each 30-minute file separately, causing speaker labels and timestamps to restart at every boundary.

## Requirements

- Windows 10 version 1809 or later, x64
- Python 3.12
- .NET 10 SDK
- FFmpeg with shared libraries (`Gyan.FFmpeg.Shared`)
- NVIDIA GPU recommended; CPU mode is supported but slower
- Broadcastify premium account for archive downloads
- Hugging Face read token for the first pyannote model download
- llama.cpp for local Gemma analysis

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

For OpenVINO ASR, add `.[openvino]`. For the Windows ML model builder and helper workflow, add `.[windowsml]`; the native helper's NuGet dependencies are restored by its .NET project.

### Windows ML functional profile

The validated Windows ML path currently uses CPU execution. It proves functional parity and keeps DML acceleration gated while the current generated DML Whisper graph is incompatible with ONNX Runtime GenAI 0.14.1. Build a small validation model and the helper with:

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[transcription,windowsml]"
$env:HF_TOKEN = $env:HUGGINGFACE_TOKEN
.\.venv\Scripts\python.exe -m onnxruntime_genai.models.builder `
  -m openai/whisper-tiny -e cpu -p fp32 `
  -o "$env:LOCALAPPDATA\Broadcastify Desktop\models\windowsml\tiny"
dotnet build .\BroadcastifyCli.WindowsML\BroadcastifyCli.WindowsML.csproj -c Release
```

Set **Settings → Processing → Advanced → ASR model path** to that model directory, choose the Windows ML profile, and select **Refresh check**. The helper loads a one-second local silence fixture through the model; only a successful decode makes the profile ready. `WINDOWS_ML_WHISPER_MODEL_PATH` and `WINDOWS_ML_HELPER_PATH` provide the equivalent `.env` overrides. The Tiny model is for validation; use a larger compatible Whisper export only after measuring radio accuracy and runtime.

Build and launch the desktop app:

```powershell
dotnet build .\BroadcastifyCli.WinUI\BroadcastifyCli.WinUI.csproj -c Release
& ".\BroadcastifyCli.WinUI\bin\Release\net10.0-windows10.0.26100.0\win-x64\Broadcastify Desktop.exe"
```

For a private personal build that carries the ignored repository `.env` alongside the executable, opt in explicitly:

```powershell
dotnet build .\BroadcastifyCli.WinUI\BroadcastifyCli.WinUI.csproj `
  -c Release `
  -p:BundleLocalEnv=true
```

This writes `broadcastify-desktop.env` into that build output. It is ignored by Git and the worker loads it before starting a job, but it is still a plaintext credential file inside the private build directory. Do not distribute or upload that build. Omit `BundleLocalEnv` for a normal shareable build; a normal build also removes any stale private env copy from its output.

## First run

1. Open **Settings**, select **Sign in**, and enter the premium Broadcastify login. Leave **Save this login securely** enabled to keep it encrypted for this Windows account in Windows Credential Locker and automatically refresh an expired session. The session cookie remains in the ignored `cookies.json` file.
2. Open **New archive**, search for and select a feed.
3. Select the date range. The output folder and model/device defaults are in **Settings**.
4. Enable **Create combined MP3** when you want one continuous timeline. Diarization requires this setting and the UI keeps it enabled so speaker labels cannot restart at archive boundaries.
5. Leave `turbo`, `auto`, and batch size 8 selected for the RTX 3090 starting point.
6. For diarization, accept the access conditions on the [pyannote Community-1 model page](https://huggingface.co/pyannote/speaker-diarization-community-1), create a read token, and paste it into **Settings**. The token is not saved by the UI. An environment token can instead be kept in the ignored `.env` file.
7. Leave **Extract incidents, summarize, and index after transcription** selected for the complete automatic workflow. Saved days can be reanalyzed manually later without downloading or transcribing again.
8. Open **Local library** at any time to browse retained feed-days. Selecting a day shows its five-stage timeline, combined recording, and timestamped transcript preview. **Verify & resume** uses the guarded downloader only when coverage is incomplete; later stages continue entirely locally, and an existing non-diarized transcript gains speakers without repeating Whisper.
9. For regional monitoring, open **Area watch**, enter several ZIP codes, discover feeds, and explicitly select the police/fire agencies to retain in a named profile. **Archive + analyze selected** processes one feed at a time using the shared Settings defaults; **Find story leads** ranks all already-saved incidents across the selected feeds and prepares a playable evidence package for every retained lead. The latest saved brief reopens immediately without starting the LLM.

Models are downloaded into their normal local caches the first time they are used. The first transcription or diarization run will therefore take longer to start.

Full-day diarization reports progress for segmentation, speaker counting, and embeddings. Its lossless 16 kHz preparation is atomically cached while a run is active and retained after an interruption or pipeline error, so a retry does not repeat a multi-gigabyte conversion. The app respects a downloaded pyannote pipeline's tuned embedding batch size and only raises it when the selected batch is larger.

Archive downloads are resumable and normally start at least five seconds apart. Current URL IDs and server filenames use different identifiers, so cache reuse resolves the URL timestamp in the feed's own published time zone instead of redownloading completed blocks. If Broadcastify rate-limits a job, the app pauses all queued requests for the requested backoff or a conservative 30-second-to-5-minute exponential delay, retries transient failures, and runs the rest of the date-range job one at a time. The progress count includes only successfully saved or cached archives.

Broadcastify can also return HTTP 429 with the explicit body `Download limit exceeded` and no reset time. That is treated as an exhausted account/IP archive quota rather than a transient request-rate response: the app stops immediately, preserves completed files, prevents queued workers from making more requests, and directs the user to Broadcastify support for quota details. Date-range jobs acquire archives before loading the transcription model, so a long diarization pass cannot delay the next guarded download. If the quota interrupts a range, the app makes no further archive-download requests, reuses any later days already complete in the local cache, processes those complete days, and reports the exact coverage gap. A later rerun resumes from the exact server timestamp and feed-timezone cache key.

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

Persistent analysis defaults to `archives/broadcastify-analysis.sqlite3`. It stores feed days, transcript segments, passages, incidents, daily/weekly briefs, area profiles, area story digests, embeddings, and question history. Raw transcript evidence can contain names or other details spoken on the radio; derived prompts omit private identifiers, and reports must be treated as unconfirmed dispatch traffic rather than findings of fact.

ZIP discovery mirrors the current Broadcastify website flow: a ZIP search first resolves to one or more county-directory links, and the app then parses and deduplicates those county feed tables. The public-safety-only filter is enabled by default. Nearby ZIPs in the same county intentionally produce one feed entry tagged with every matching ZIP rather than duplicate archive jobs.

Area story clustering is conservative and auditable. A matching event category and nearby time are not sufficient by themselves: reports need a compatible reported location or unusually strong descriptive overlap. Cross-feed overlap raises newsworthiness but is never presented as independent confirmation because feeds can rebroadcast the same talkgroup. Exact feed and incident IDs remain attached to each story card. Each source record includes an ASR quote, a cached context clip cut from the retained combined audio, model confidence, anonymous speaker-cluster context when available, and source/clip hashes. Gemma only writes the readable assignment brief.

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

For environment-based sign-in, copy `.env-example` to `.env` and set `BROADCASTIFY_USERNAME` and `BROADCASTIFY_PASSWORD`. The legacy `USERNAME`/`PASSWORD` names remain supported. Set `HUGGINGFACE_TOKEN` for pyannote, unless the read token is entered in the UI for that session. Optional overrides are `WHISPER_CPP_PATH`, `WHISPER_CPP_MODEL_PATH`, `OPENVINO_WHISPER_MODEL_PATH`, `BROADCASTIFY_MODEL_DIR`, `LLAMA_SERVER_PATH`, and `FFMPEG_PATH`; blank values preserve automatic discovery. Credential files, the Windows-build copy, and the session-cookie file are ignored by Git.

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

The first analysis run downloads the quantized `ggml-org/gemma-4-12B-it-GGUF:Q4_K_M` model and the small BGE embedding model into their normal caches. `analyze-day` resumes from saved incidents after a summary-only failure; use `--force-summary` to refresh only a brief or `--force` to rebuild all derived incidents for the day. Show persistent counts with:

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
- cached diarization, transcription, analysis, daily reports, and local Q&A were exercised through the WinUI Release build
- a persisted seven-day brief ending July 12 correctly reported two available days, five missing dates, 87 incidents, 22 priority 4–5 records, and a deterministic list of notable incident IDs
- a live six-ZIP Example City/East Example City/Example City discovery followed Example City and Example County county directories, found the current feed 90001 plus nearby police/fire feeds, and persisted a six-feed `Regional coverage desk` profile
- the first regional brief correctly reported only 2 of 24 feed-days and 1 of 6 feeds with data, ranked 30 newsroom leads from 87 saved incidents, and retained feed/incident references without treating the five missing feeds as quiet
- ranked area leads now retain transcript quotes and generate independently playable, hashed context clips from the matching combined-audio timestamps

The five speaker labels are acoustic clusters rather than identified officers or radio unit IDs. Radio compression, overlapping traffic, dispatch consoles, and repeated users of the same equipment can split or merge real speakers, so they should be used as conversation structure rather than identity evidence.

Run the tests with:

```powershell
.\.venv\Scripts\python.exe -m pytest
```

## Suggested phases

1. **Core workflow:** authenticated feed search/download, overlap-aware continuous combination, fast transcription, persistence, local retrieval, incident extraction, summaries, and Q&A. Implemented and tested on two full days.
2. **Diarization validation:** Community-1 completed a real combined 24.4-hour feed day with continuous labels and cached speaker turns. Implemented and validated.
3. **Analysis UI:** focused WinUI navigation, local pipeline-state library and continuation actions, persisted days, incident timeline, daily/weekly briefs, range questions, model diagnostics, cancellation, automatic post-transcription analysis, multi-ZIP discovery, saved area profiles, and immediately reopened regional lead briefs are implemented and visually exercised in WinUI 3.
4. **Regional ingestion:** explicit sequential multi-feed archive/analyze is implemented. The next step is a schedule/retention manager so a newsroom can budget storage and GPU time per profile instead of manually starting each date range.
5. **Newsroom product phase:** evidence drill-down and cached story clips are implemented. Next add editor approval/redaction/export, radius-based coverage markets, profile management, map/geocoding review, notifications, a model/cache manager, and installer/MSIX packaging.
6. **Neighborhood subscription phase:** build a separate public web/mobile signup for center-plus-radius and topic preferences; match only editor-approved stories, include unsubscribe/consent controls, and keep raw scanner audio private by default.
7. **Hardware portability phase:** validate the complete workflow in CPU-only mode with smaller quantized LLM profiles; add backend-aware diagnostics and settings; support suitable `llama.cpp` Vulkan/HIP/SYCL builds for AMD and Intel GPUs; and investigate practical non-CUDA acceleration for ASR and diarization while retaining a reliable CPU fallback.
8. **Benchmark phase:** compare the saved real-radio corpus against newer ASR/alignment options before changing the `faster-whisper` default; track word quality, missed calls, runtime, and VRAM rather than model-release recency alone.
