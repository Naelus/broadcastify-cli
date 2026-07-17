# Feature inventory

Last updated: July 17, 2026

Status labels: **Validated**, **Implemented**, **Foundation**, or **Planned**.

## Acquisition and feed discovery

| Capability | Status | Notes |
|---|---|---|
| Website-login feed search | Validated | Uses Broadcastify's website endpoints, not the official API. |
| City/county/state/ZIP discovery | Validated | Includes county-directory results and feed catalog persistence. |
| Radius and ordered-ZIP area profiles | Validated | Census ZCTA centroid expansion, live opt-out public-safety filter, explicit feed review, and persisted approximate distance/priority; discovery makes no archive requests. |
| Persisted nearest-first acquisition cascade | Validated | Shared native/Web queue skips completed feeds, recovers interrupted items, and stops all lower priorities on explicit quota exhaustion. |
| Sequential cache-aware archive acquisition | Validated | Five-second minimum pacing, retries, shared cooldown, exact timezone cache keys. |
| Explicit quota exhaustion stop | Validated | Stops new media requests and preserves all completed work. |

## Processing and persistence

| Capability | Status | Notes |
|---|---|---|
| Continuous daily MP3 with overlap trimming | Validated | Combination occurs before ASR/diarization. |
| faster-whisper CUDA/CPU | Validated | CUDA is the Windows reference default; CPU is slower fallback. |
| OpenVINO Whisper | Validated | Real AUTO/CPU decode passed; initialization and generation failures retry on CPU and retain the actual backend/fallback stage. |
| whisper.cpp Vulkan | Validated | Native and opt-in locked-down container adapters exist; exact app path completed a real AMD Radeon 890M decode and retains backend evidence. |
| Windows ML Whisper | Validated | Batched streaming C# helper and real CPU decode/self-test pass; provider discovery/acquisition is explicit, while DML/TensorRT acceleration stays gated after measured incompatibilities. |
| pyannote diarization CUDA/CPU | Validated | Existing transcripts can gain labels without rerunning ASR; FFmpeg-to-memory-mapped waveform input bypasses broken TorchCodec file loaders. |
| Non-CUDA diarization acceleration | Planned | Current portable behavior is CPU fallback. |
| SQLite transcript/incident/summary store | Validated | Includes FTS, embeddings, Q&A history, area profiles/queues, and cached briefs. |
| Evidence/identity safety gates | Validated | Exact-citation claim/concept checks, bounded evidence gaps, deterministic category/priority correction, contained-evidence dedupe, outcome-language rejection, and public identifier/name redaction; original ASR remains available for internal audit. |
| Quantized llama.cpp analysis | Validated | Gemma GGUF selector or explicit local path; retained removed quants are reused from older Hub snapshots, clean installs use the current `Q4_0` default, and real AMD Vulkan generation offloaded all tested model layers. |
| Provider API/Codex harness | Validated | Native Settings and CLI support local Gemma, OpenAI Responses, compatible `/v1` endpoints, and ephemeral saved-login Codex with explicit transcript-sharing consent and readiness checks. |

## User experience

| Capability | Status | Notes |
|---|---|---|
| WinUI 3 navigation shell | Validated | Local Library, New Archive, Review & Ask, Area Watch, Settings. |
| Local Library master/detail viewer | Validated | Processing timeline, audio playback, transcript preview, resume/open actions. |
| Saved secure Broadcastify login | Implemented | Windows Credential Locker plus session refresh. |
| Private `.env` build option | Implemented | Explicit opt-in only; normal builds remove stale copies. |
| Daily incident/evidence review | Validated | Exact local clips and export actions; a real Example City long-span card was verified to cut/play the cited event rather than seek to the broad incident-envelope start. |
| Seven-day summaries | Validated | Explicitly records available and missing days. |
| Regional story leads | Validated | Evidence-backed assignment leads, not confirmations. |
| Cross-platform browser UI | Validated | Windows visual QA plus an immutable Linux host passed loopback security, byte-range media, worker lifecycle, an exact protected ASR/diarization/analysis job, and the managed service lifecycle; Linux full-model visual QA and macOS remain open. |
| Managed Linux Web launcher | Validated | Exact `historical-validation` wheel/unit passed systemd verification, start/health/session/bootstrap, failure restart, 0600 owner logs, graceful stop, and data-preserving uninstall on TrueNAS/Linux. |
| Explicit ASR runtime self-test | Validated | Native and Web Settings decode local synthetic audio using the selected model/device, with managed downloads only after a user starts the test. |
| Explicit diarization runtime self-test | Validated | Native and Web Settings load Community-1 and execute generated waveform audio on CUDA/CPU; a complete cache now works offline without retaining a token. |
| First-run readiness overview | Validated | Five concise account/storage/transcription/speaker/analysis steps share stage diagnostics and direct setup/test actions in native and Web UIs. |
| Verified Windows publish | Validated | Runnable WinUI resources plus namespaced Windows ML runtime; normal/private credential isolation is verified. |
| Installer/model manager/onboarding | Foundation | In-app readiness, verified Windows publish, and the managed Linux service are implemented; dependency/model acquisition and non-developer packaging remain. |

## Hardware profile matrix

| Profile | ASR | Diarization | Analysis | Current state |
|---|---|---|---|---|
| Automatic/NVIDIA | faster-whisper CUDA | pyannote CUDA | llama.cpp Vulkan/auto-offload | Validated on RTX 3090 |
| CPU only | faster-whisper INT8 CPU or whisper.cpp CPU | pyannote CPU | llama.cpp CPU | Exact `historical-validation` protected Web job completed all stages on CPU in 21.282 seconds for 30 seconds of retained audio; full-day timing remains |
| Vulkan | whisper.cpp Vulkan | pyannote CPU | llama.cpp Vulkan | Exact `historical-validation` protected Web job completed the 30-second offline AMD workflow in 15.139 seconds; full-day CPU diarization timing remains |
| OpenVINO | OpenVINO Whisper AUTO/CPU | pyannote CPU | llama.cpp SYCL/auto/CPU | Exact `historical-validation` protected 60-second job completed every stage to Ready to review in 31 seconds; accelerator-to-CPU fallback also validated |
| Windows ML | ONNX Runtime GenAI CPU | pyannote CPU | llama.cpp auto/CPU | Exact `historical-validation` protected 60-second job completed every stage in 32 seconds and resumed without repeating work; DML fails and the current TensorRT RTX provider is slower than CPU for Whisper |
| Apple Metal | whisper.cpp Metal | pyannote CPU | llama.cpp Metal | Implemented with native-backend detection and explicit self-test; real Mac validation pending |

Detailed outcomes and commands belong in `PROGRESS.md`; defects and blockers belong in `BUGS.md`.
