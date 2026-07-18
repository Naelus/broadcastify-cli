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
| Windows ML Whisper | Validated | Batched streaming C# helper and real CPU decode/self-test pass; provider discovery/acquisition is explicit, while DML/TensorRT acceleration stays gated after measured incompatibilities. |
| pyannote diarization CUDA/CPU | Validated | Existing transcripts can gain labels without rerunning ASR; FFmpeg-to-memory-mapped waveform input bypasses broken TorchCodec file loaders. |
| Non-CUDA diarization acceleration | Planned | Current portable behavior is CPU fallback. sherpa-onnx's ONNX diarization is the strongest current portable-CPU benchmark candidate, but it must match Community-1 on retained noisy radio before integration. |
| SQLite transcript/incident/summary store | Validated | Includes FTS, embeddings, Q&A history, area profiles/queues, cached briefs, per-model-window incident-analysis checkpoints, and prompt-version invalidation so retained media survives an interruption or evidence-rule upgrade while older claims are withheld. |
| Evidence/identity safety gates | Validated | Exact-citation claim/concept checks, bounded evidence gaps, deterministic category/priority correction, contained-evidence dedupe, outcome-language rejection, and public identifier/name redaction including location-adjacent dispatch forms; original ASR remains available for internal audit. |
| Quantized llama.cpp analysis | Validated | Gemma GGUF selector or explicit local path; retained removed quants are reused from older Hub snapshots, clean installs use the current `Q4_0` default, and real AMD Vulkan generation offloaded all tested model layers. |
| Provider API/Codex harness | Validated | Native Settings and CLI support local Gemma, OpenAI Responses, compatible `/v1` endpoints, and ephemeral saved-login Codex with explicit transcript-sharing consent and readiness checks. |

## User experience

| Capability | Status | Notes |
|---|---|---|
| WinUI 3 navigation shell | Validated | Local Library, New Archive, Review & Ask, Area Watch, Settings. |
| Local Library master/detail viewer | Validated | Processing timeline, audio playback, transcript preview, resume/open actions, durable human-readable names from catalog/manifest identity, and a distinct local-only reanalysis state for summaries created under older evidence rules. |
| Saved secure Broadcastify login | Implemented | Windows Credential Locker plus session refresh. |
| Private `.env` build option | Implemented | Explicit opt-in only; normal builds remove stale copies. |
| Daily incident/evidence review | Validated | Priority-filtered browsing plus search across every priority, redacted cited-radio quotes, exact playable/exportable clips, and an explicitly non-evidentiary surrounding-context clip for hearing an earlier dispatch before a cited disposition. |
| Seven-day summaries | Validated | Explicitly separates available, missing, and retained-but-needing-reanalysis days. |
| Regional story leads | Validated | Evidence-backed assignment leads, not confirmations; explicit feed selection survives rediscovery/filtering, older briefs are version-gated, and both UIs use a bounded ranked-list/selected-evidence master/detail viewer with redacted quotes plus exact playable/exportable clips. |
| Cross-platform browser UI | Validated | Windows QA plus exact Linux commit `historical-validation` passed 148 host tests and full-model desktop/mobile visual, accessibility, navigation, media, and loopback-security checks; safe area-evidence packages expose no local paths. Real macOS validation remains open. |
| Managed Linux Web launcher | Validated | Exact `historical-validation` wheel/unit passed systemd verification, start/health/session/bootstrap, failure restart, 0600 owner logs, graceful stop, and data-preserving uninstall on TrueNAS/Linux. |
| Explicit ASR runtime self-test | Validated | Native and Web Settings decode local synthetic audio using the selected model/device, with managed downloads only after a user starts the test. |
| Explicit diarization runtime self-test | Validated | Native and Web Settings load Community-1 and execute generated waveform audio on CUDA/CPU; a complete cache now works offline without retaining a token. |
| Explicit analysis runtime self-test | Validated | Native and Web Settings request a tiny structured JSON result through the selected local, API, or consented Codex provider without sending archive evidence; local execution reports the actual model/backend. |
| First-run readiness overview | Validated | Five concise account/storage/transcription/speaker/analysis steps share stage diagnostics and direct setup/test actions in native and Web UIs; detected, configured, and execution-verified are separate states. |
| Native crash/state recovery | Validated | Stable per-user atomic settings with legacy migration and debounced autosave, restored review/profile selection, append-only activity/crash logs, and a real startup probe of the formerly crashing XAML event path. |
| Verified Windows publish | Validated | Runnable WinUI resources plus namespaced Windows ML runtime; normal/private credential isolation is verified. |
| Installer/model manager/onboarding | Foundation | In-app readiness, verified Windows publish, and the managed Linux service are implemented; dependency/model acquisition and non-developer packaging remain. |

## Hardware profile matrix

| Profile | ASR | Diarization | Analysis | Current state |
|---|---|---|---|---|
| Automatic/NVIDIA | faster-whisper CUDA | pyannote CUDA | llama.cpp Vulkan/auto-offload | Validated on RTX 3090; exact `historical-validation` native tests completed the three selected stages in 3.5, 5.3, and 7.80 seconds |
| CPU only | faster-whisper INT8 CPU or whisper.cpp CPU | pyannote CPU | llama.cpp CPU | Exact `historical-validation` protected Web job completed all stages on CPU in 21.282 seconds for 30 seconds of retained audio; full-day timing remains |
| Vulkan | whisper.cpp Vulkan | pyannote CPU | llama.cpp Vulkan | Exact `historical-validation` protected Web job completed the 30-second offline AMD workflow in 15.139 seconds; exact `historical-validation` separately verified ASR, diarization, and analysis in 4.413/4.221/5.622 seconds; full-day CPU diarization timing remains |
| OpenVINO | OpenVINO Whisper AUTO/CPU | pyannote CPU | llama.cpp SYCL/auto/CPU | Exact `historical-validation` protected 60-second job completed every stage to Ready to review in 31 seconds; a current Tiny CPU self-test completed in 1.188 seconds with no fallback |
| Windows ML | ONNX Runtime GenAI CPU | pyannote CPU | llama.cpp auto/CPU | Exact `historical-validation` protected 60-second job completed every stage in 32 seconds and resumed without repeating work; provider-catalog presence stays unverified until a real decode, and current DML/TensorRT Whisper paths remain gated |
| Apple Metal | whisper.cpp Metal | pyannote CPU | llama.cpp Metal | Implemented with native-backend detection and explicit self-test; real Mac validation pending |

Detailed outcomes and commands belong in `PROGRESS.md`; defects and blockers belong in `BUGS.md`.
