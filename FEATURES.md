# Feature inventory

Last updated: July 16, 2026

Status labels: **Validated**, **Implemented**, **Foundation**, or **Planned**.

## Acquisition and feed discovery

| Capability | Status | Notes |
|---|---|---|
| Website-login feed search | Validated | Uses Broadcastify's website endpoints, not the official API. |
| City/county/state/ZIP discovery | Validated | Includes county-directory results and feed catalog persistence. |
| Multi-ZIP area profiles | Validated | Downloads only explicitly selected feeds. |
| Nearest-feed/radius cascade | Planned | Needed for low-quota regional coverage without indiscriminate acquisition. |
| Sequential cache-aware archive acquisition | Validated | Five-second minimum pacing, retries, shared cooldown, exact timezone cache keys. |
| Explicit quota exhaustion stop | Validated | Stops new media requests and preserves all completed work. |

## Processing and persistence

| Capability | Status | Notes |
|---|---|---|
| Continuous daily MP3 with overlap trimming | Validated | Combination occurs before ASR/diarization. |
| faster-whisper CUDA/CPU | Validated | CUDA is the Windows reference default; CPU is slower fallback. |
| OpenVINO Whisper | Validated | Real CPU decode passed; exposed GPU failure retries on CPU. |
| whisper.cpp Vulkan | Foundation | Adapter and diagnostics exist; a Vulkan whisper-cli/model install and real AMD test remain. |
| Windows ML Whisper | Implemented | Integrated streaming C# helper and real CPU decode/self-test pass; DML model-builder output still fails and stays gated. |
| pyannote diarization CUDA/CPU | Validated | Existing transcripts can gain labels without rerunning ASR. |
| Non-CUDA diarization acceleration | Planned | Current portable behavior is CPU fallback. |
| SQLite transcript/incident/summary store | Validated | Includes FTS, embeddings, Q&A history, area profiles, and cached briefs. |
| Quantized llama.cpp analysis | Validated | Gemma GGUF local path; Vulkan device detection is implemented. |
| Provider API/Codex harness | Planned | Must be opt-in, auditable, secret-safe, and preserve evidence schemas. |

## User experience

| Capability | Status | Notes |
|---|---|---|
| WinUI 3 navigation shell | Validated | Local Library, New Archive, Review & Ask, Area Watch, Settings. |
| Local Library master/detail viewer | Validated | Processing timeline, audio playback, transcript preview, resume/open actions. |
| Saved secure Broadcastify login | Implemented | Windows Credential Locker plus session refresh. |
| Private `.env` build option | Implemented | Explicit opt-in only; normal builds remove stale copies. |
| Daily incident/evidence review | Validated | Exact local clips and export actions. |
| Seven-day summaries | Validated | Explicitly records available and missing days. |
| Regional story leads | Validated | Evidence-backed assignment leads, not confirmations. |
| Cross-platform browser UI | Planned | Should consume a local service and mirror the native viewer experience. |
| Installer/model manager/onboarding | Planned | Required before non-developer distribution. |

## Hardware profile matrix

| Profile | ASR | Diarization | Analysis | Current state |
|---|---|---|---|---|
| Automatic/NVIDIA | faster-whisper CUDA | pyannote CUDA | llama.cpp Vulkan/auto-offload | Validated on RTX 3090 |
| CPU only | faster-whisper INT8 CPU | pyannote CPU | llama.cpp CPU | Implemented; full end-to-end timing still needed |
| Vulkan | whisper.cpp Vulkan | pyannote CPU | llama.cpp Vulkan | ASR installation/real AMD validation pending |
| OpenVINO | OpenVINO Whisper AUTO/CPU | pyannote CPU | llama.cpp SYCL/auto/CPU | Real CPU decode and accelerator-to-CPU fallback validated |
| Windows ML | ONNX Runtime GenAI | pyannote CPU | llama.cpp auto/CPU | CPU model decode integrated and self-tested; DML acceleration remains blocked |

Detailed outcomes and commands belong in `PROGRESS.md`; defects and blockers belong in `BUGS.md`.
