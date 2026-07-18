# Feature inventory

Last updated: July 18, 2026

Status labels: **Validated**, **Implemented**, **Foundation**, or **Planned**.

## Acquisition and feed discovery

| Capability | Status | Notes |
|---|---|---|
| Website-login feed search | Validated | Uses Broadcastify's website endpoints, not the official API. |
| City/county/state/ZIP discovery | Validated | Includes county-directory results and durable feed catalog persistence. |
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
| Windows ML Whisper | Validated | Batched streaming C# helper, exact model identity, managed Base/Tiny/Small export, and real CPU decode pass; the UI labels generated silence as an execution check rather than an accuracy test. Base is the measured radio starter, while DML/TensorRT acceleration stays gated. |
| Qwen3-ASR 0.6B INT8 through sherpa-onnx | Implemented | Opt-in CPU profile with fixed-hash managed model/VAD acquisition, safe staged extraction, exact identity diagnostics/self-test, and Community-1 or Silero source-region provenance without invented word timestamps. It preserved 13 retained events near 0.08 RTF, completed a real three-stage Windows proof in 15.265 seconds, and exact pushed source processed five minutes in 7.128 seconds on AMD/Linux; quote-level/default gates remain. |
| Parakeet TDT 0.6B v3 INT8 through sherpa-onnx | Evaluated | Very fast CPU inference and token timestamps, but it returned blank text on unsegmented sparse-radio clips and rendered the critical event as `drop fire` after segmentation. It is not the default. |
| VibeVoice-ASR 8B BF16 unified transcription/diarization | Evaluated | Official local model directly emitted who/when/what and preserved three retained core events on RTX 3090. It also changed an address, omitted a disregard, merged speakers, and hallucinated noise; the 15.53 GiB snapshot/16.58 GiB peak allocation and Transformers 5.x isolation requirement prevent promotion. |
| pyannote diarization CUDA/CPU | Validated | Existing transcripts can gain labels without rerunning ASR; FFmpeg-to-memory-mapped waveform input bypasses broken TorchCodec file loaders. |
| Non-CUDA diarization acceleration | Foundation | sherpa-onnx 1.13.4 processed a retained five-minute radio slice in 5.722 seconds versus 68.508 seconds for fresh Community-1 CPU. It found 19 versus 24 turns, covered 89.2% of Community-1 speech with 99.5% candidate precision, and produced one extra acoustic cluster at the tuned 0.95 threshold. Community-1 remains the accuracy default. |
| SQLite transcript/incident/summary store | Validated | Includes FTS, embeddings, Q&A history, area profiles/queues, cached briefs, per-model-window incident-analysis checkpoints, and prompt-version invalidation so retained media survives an interruption or evidence-rule upgrade while older claims are withheld. |
| Evidence/identity safety gates | Validated | Exact-citation claim/concept checks, bounded evidence gaps, deterministic category/priority correction, contained-evidence dedupe, outcome-language rejection, and public identifier/name redaction including location-adjacent dispatch forms; original ASR remains available for internal audit. |
| Quantized llama.cpp analysis | Validated | Gemma GGUF selector or explicit local path; retained removed quants are reused from older Hub snapshots, clean installs use the current `Q4_0` default, real AMD Vulkan generation offloaded all tested model layers, and release-bundled loader libraries are scoped to the llama-server child so they cannot contaminate whisper.cpp. |
| Provider API/Codex harness | Validated | Native Settings and CLI support local Gemma, OpenAI Responses, compatible `/v1` endpoints, and ephemeral saved-login Codex with explicit transcript-sharing consent and readiness checks. |

## User experience

| Capability | Status | Notes |
|---|---|---|
| WinUI 3 navigation shell | Validated | Local Library, New Archive, Review & Ask, Area Watch, Settings; rebuilt and interactively verified on current stable Windows App SDK 2.3.1. |
| Local Library master/detail viewer | Validated | Processing timeline, audio playback, transcript preview, resume/open actions, durable human-readable names from catalog/manifest identity, and a distinct local-only reanalysis state for summaries created under older evidence rules. |
| Saved secure Broadcastify login | Implemented | Windows Credential Locker plus session refresh. |
| Private `.env` build option | Implemented | Explicit opt-in only; normal builds remove stale copies. |
| Daily incident/evidence review | Validated | Priority-filtered browsing plus search across every priority, redacted cited-radio quotes, exact playable/exportable clips, and an explicitly non-evidentiary surrounding-context clip for hearing an earlier dispatch before a cited disposition. |
| Seven-day summaries | Validated | Explicitly separates available, missing, and retained-but-needing-reanalysis days. |
| Regional story leads | Validated | Evidence-backed assignment leads, not confirmations; explicit feed selection survives rediscovery/filtering, older briefs are version-gated, and both UIs use a bounded ranked-list/selected-evidence master/detail viewer with redacted quotes plus exact playable/exportable clips. |
| Cross-platform browser UI | Validated | Windows QA plus exact Linux commit `historical-validation` passed 148 host tests and full-model desktop/mobile visual, accessibility, navigation, media, and loopback-security checks; safe area-evidence packages expose no local paths. Real macOS validation remains open. |
| Managed Linux Web launcher | Validated | Exact `historical-validation` wheel/unit passed systemd verification, start/health/session/bootstrap, failure restart, 0600 owner logs, graceful stop, and data-preserving uninstall on TrueNAS/Linux. |
| Explicit ASR runtime self-test | Validated | Native and Web Settings decode generated silence using the exact selected model/device and clearly call this an execution check, not radio accuracy. Windows ML models can be built and whisper.cpp GGML models downloaded only after the user starts the action. |
| Explicit diarization runtime self-test | Validated | Native and Web Settings load Community-1 and execute generated waveform audio on CUDA/CPU; a complete cache now works offline without retaining a token. |
| Explicit analysis runtime self-test | Validated | Native and Web Settings request a tiny structured JSON result through the selected local, API, or consented Codex provider without sending archive evidence; local execution reports the actual model/backend. |
| Joined hardware-profile verifier | Validated | One native/Web action—also prominent in first-run Setup—runs generated-input transcription, diarization, and analysis sequentially, releases accelerator memory between stages, preserves successful earlier-stage evidence, bounds oversized native dumps, and stops with actionable recovery at the first failure without using archive quota. Exact `historical-validation` reran all three stages on AMD Vulkan/CPU/Vulkan in 9.960 seconds with warm caches. |
| First-run readiness overview | Validated | Five concise account/storage/transcription/speaker/analysis steps share stage diagnostics and direct setup/test actions in native and Web UIs; detected, configured, and execution-verified are separate states. The joined verifier retains running and final state on Setup. Web Settings uses persistent/deep-linked Setup, Processing, Analysis & AI, and Account sections with keyboard navigation and exact action focus. |
| Native crash/state recovery | Validated | Stable per-user atomic settings with legacy migration and debounced autosave, restored review/profile selection, append-only activity/crash logs, and a real startup probe of the formerly crashing XAML event path. |
| Verified Windows publish | Validated | Windows App SDK 2.3.1 WinUI resources plus a namespaced Windows ML 2.1.74 runtime; normal/private credential isolation and the visible three-stage model proof are verified. |
| Installer/model manager/onboarding | Foundation | In-app readiness, verified Windows publish, and the managed Linux service are implemented; dependency/model acquisition and non-developer packaging remain. |

## Hardware profile matrix

| Profile | ASR | Diarization | Analysis | Current state |
|---|---|---|---|---|
| Automatic/NVIDIA | faster-whisper CUDA | pyannote CUDA | llama.cpp Vulkan/auto-offload | Validated on RTX 3090; the joined native verifier completed all three stages in 14.9 seconds and the joined Web verifier completed in 15.4 seconds |
| CPU only | faster-whisper INT8 CPU or whisper.cpp CPU | pyannote CPU | llama.cpp CPU | Exact `historical-validation` protected Web job completed all stages on CPU in 21.282 seconds for 30 seconds of retained audio; full-day timing remains |
| Vulkan | whisper.cpp Vulkan | pyannote CPU | llama.cpp Vulkan | Exact `historical-validation` first joined verification completed in 27.886 seconds on AMD Radeon 890M; exact current `historical-validation` completed a warm-cache rerun in 9.960 seconds (3.677/2.783/2.559) and passed 172 hardened tests; exact `historical-validation` retains the protected offline Web-job proof; full-day CPU diarization timing remains |
| OpenVINO | OpenVINO Whisper AUTO/CPU | pyannote CPU | llama.cpp SYCL/auto/CPU | The joined Web verifier completed in 13.8 seconds with a 1.0-second OpenVINO AUTO decode; exact `historical-validation` retains the protected 60-second Ready-to-review proof |
| Windows ML | ONNX Runtime GenAI CPU, Base starter | pyannote CPU | llama.cpp auto/CPU | Exact model/provider identity and managed preparation are implemented. Base captured the retained low-SNR event's core dispatch in 0.212 seconds; Tiny did not. Exact `historical-validation` retains the protected 60-second/resume proof, while catalog-only WebGPU/TensorRT and current DML paths remain gated |
| Fast CPU/Qwen | Qwen3-ASR 0.6B INT8 / sherpa-onnx CPU | pyannote CPU, Silero bounds fallback | llama.cpp auto/CPU | Managed model and source-region timestamp contract implemented; joined Windows verification completed in 15.265 seconds; exact AMD/Linux inference completed a five-minute slice in 7.128 seconds at four threads, while critical-word drift kept the quality gate open |
| Apple Metal | whisper.cpp Metal | pyannote CPU | llama.cpp Metal | Implemented with native-backend detection and explicit self-test; real Mac validation pending |

Detailed outcomes and commands belong in `PROGRESS.md`; defects and blockers belong in `BUGS.md`.
