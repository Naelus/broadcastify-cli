# Known bugs and gaps

Last updated: July 16, 2026

Use this file for reproducible defects and concrete blockers, not the general roadmap. Remove an entry only after its fix and verification are recorded in `PROGRESS.md`.

## Active

### B-001 — Windows ML DML Whisper model fails during generation

- **Severity:** High for Windows ML parity; no impact on the default CUDA path.
- **Observed:** ONNX Runtime GenAI 0.14.1 with official builder output for `openai/whisper-tiny` fails on DML/WinML. With `past_present_share_buffer=false`, DML graph capture rejects the generator; changing it to true reaches a `DmlFusedNode` invalid-key error.
- **Control:** The C# helper, Python streaming adapter, and a CPU FP32 model successfully transcribed a real 23-second radio clip. The profile becomes ready only when its configured model passes an actual decode self-test; the broken DML models remain unavailable.
- **Next:** Test a compatible published DML model or upstream fix; add self-test/model discovery before enabling the profile.

### B-003 — Portable diarization is CPU-only outside CUDA

- **Severity:** Medium; functionally correct but potentially slow.
- **Observed:** pyannote's supported app path is PyTorch CUDA or CPU. Vulkan/OpenVINO/Windows ML profiles deliberately fall back to CPU.
- **Next:** Benchmark CPU on full-day audio and investigate supported Intel XPU/other backends without weakening the reliable fallback.

### B-004 — Full CPU-only end-to-end timing is not recorded

- **Severity:** Medium.
- **Observed:** CPU components and profile exist, but the complete multi-hour pipeline has not been benchmarked on the retained corpus.
- **Next:** Use a bounded representative slice first, then a full day if practical; record ASR, diarization, and analysis time separately.

### B-005 — Complete cross-platform workflow is not yet validated or packaged on Linux/macOS

- **Severity:** Medium for product portability; no impact on native Windows use.
- **Observed:** Linux AMD Vulkan ASR and quantized LLM components now pass real runs through the app adapter, including an immutable-host container path. The complete Web UI install, media stream, pyannote CPU continuation, analysis server lifecycle, and packaging have not run together on Linux; no real Mac run has occurred. Cross-platform credential persistence is `.env`/session-only rather than an OS keychain.
- **Next:** Exercise a clean ordinary-user Linux install and full short workflow, then a macOS install/CPU-or-Metal workflow; add a supervised launcher/package and platform keychain adapter without weakening the loopback/session-token boundary.

### B-006 — Feed display names are missing for some retained legacy days

- **Severity:** Low.
- **Observed:** Local Library shows `Feed 90003` when that feed was downloaded before its search result entered the persistent catalog.
- **Next:** Backfill feed metadata from saved manifests or a quota-free directory lookup.

### B-007 — Archive quota size/reset schedule remains inferred

- **Severity:** Operational.
- **Observed:** Broadcastify publishes no numeric archive-download quota or reset timestamp. One measured window allowed roughly 192 successful archive redirects, while the July 16 availability follow-up allowed only 55 new media downloads before the same explicit limit response.
- **Control:** Sequential pacing, exact cache reuse, and immediate stop on explicit exhaustion.
- **Next:** Ask Broadcastify support for authoritative details and prioritize future regional acquisition by user distance/importance rather than a guessed quota size.

### B-008 — Nearby-feed acquisition is ZIP/profile based, not a true distance cascade

- **Severity:** Medium for quota-efficient newsroom coverage.
- **Observed:** Multi-ZIP discovery and explicit area profiles work, but the app does not yet rank a center/radius market by geographic distance and spend the unknown archive budget on the nearest missing feed-day first.
- **Next:** Add geocoded feed coverage metadata, center-plus-radius settings, and a persisted nearest-first acquisition queue. Keep feed selection reviewable and stop the whole queue on the first explicit quota response.

## Recently fixed

### F-001 — Diarization restarted for every downloaded block

Combined audio is now created before transcription/diarization, preserving one continuous timeline.

### F-002 — Failed download futures appeared as completed progress

Progress counts only successful cached or saved archives; explicit quota exhaustion cancels queued work.

### F-003 — Existing transcript forced a second Whisper pass to add speakers

The library now runs diarization-only continuation and atomically updates the transcript/text output.

### F-004 — A `diarization_requested` flag could look like completed labeling

Library detection now requires completion evidence and new transcripts write `diarization_completed=true` only after labels are produced.

### F-005 — Native analysis actions always used local Gemma

The separate **Analysis & AI** Settings tab now selects local Gemma, OpenAI Responses, an OpenAI-compatible endpoint, or a saved-login Codex CLI harness. Every native analysis action receives the selected provider settings, external transcript sharing is an explicit opt-in, optional API-key persistence uses Windows Credential Locker, and readiness checks do not send transcript text.

### F-006 — Vulkan readiness was inferred without a real AMD run

The exact whisper.cpp adapter now completed a real AMD Radeon 890M decode after a direct container CLI control run, and llama.cpp offloaded all layers of a public Q4_K Gemma test model to the same Vulkan device. Diagnostics now require an actual Vulkan llama.cpp device instead of treating any `llama-server` executable as Vulkan-ready. Linux/macOS binary discovery, explicit pre-pulled container support, MP3-to-WAV preparation, rootless cache fallbacks, clamped progress, and retained backend evidence are covered by tests.
