# Known bugs and gaps

Last updated: July 16, 2026

Use this file for reproducible defects and concrete blockers, not the general roadmap. Remove an entry only after its fix and verification are recorded in `PROGRESS.md`.

## Active

### B-001 — Windows ML GPU providers do not yet accelerate Whisper reliably

- **Severity:** High for Windows ML parity; no impact on the default CUDA path.
- **Observed:** Stable 0.13.1 and installed 0.14.1 builder/runtime combinations fail with current DML Whisper Tiny exports. A non-shared cache is rejected by automatic graph capture; forcing the shared cache reaches `DmlFusedNode_0_0` with `invalid unordered_map<K, T> key`.
- **Second provider:** Windows ML successfully acquired and registered certified `NvTensorRTRTXExecutionProvider` 1.8.24.0, but TensorRT RTX reported 36 unsupported Whisper attention nodes. The retained 22.7-second clip decoded correctly through partition/fallback in 15.895 seconds, versus 0.622 seconds on the CPU model after warm caches, so it is deliberately not selected as an accelerated profile.
- **Control:** The corrected C# helper, Python streaming adapter, and CPU FP32 model perform a real decode and identify the backend as CPU. Broken or partially-falling-back models must pass the selected-engine self-test and are not advertised as acceleration.
- **Next:** Retest a published compatible graph or upstream runtime/provider fix. Prefer the validated CUDA, OpenVINO, or whisper.cpp Vulkan paths for Windows GPU ASR in the meantime.

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
- **Observed:** Linux AMD Vulkan ASR and quantized LLM components pass real runs through the app adapter, including an immutable-host container path. The exact pushed Web UI commit now also passes on that Linux host: private dependency bootstrap, 96 tests, loopback startup, action-token enforcement, byte-range media, and a diagnostics child worker. Those component validations have not yet been joined into one pyannote-plus-analysis short workflow. Apple Metal is an explicit native whisper.cpp/llama.cpp profile with CPU diarization and macOS auto-selection when both Metal backends are detected, but no real Mac run has occurred. Cross-platform credential persistence is `.env`/session-only rather than an OS keychain.
- **Next:** Run a short retained Linux day through ASR, pyannote CPU continuation, and local analysis from the Web UI, then a macOS install/CPU-or-Metal workflow; add a supervised launcher/package and platform keychain adapter without weakening the loopback/session-token boundary.

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

### F-007 — OpenVINO fallback looked like the requested accelerator succeeded

OpenVINO now retries both pipeline initialization and first-generation failures on CPU, removes an obsolete NPU-only constructor flag, and writes the actual fallback backend and failure stage into transcripts. Native and Web Settings expose an explicit selected-engine synthetic-audio test; the ordinary readiness check cannot silently trigger a multi-gigabyte model download.

### F-008 — Windows ML Whisper used the wrong C# processor overload

The helper passed one prompt through the scalar multimodal overload, which caused `DivideByZeroException` in both CPU and DML models even though Microsoft’s Python sample worked. It now uses the batched-prompt overload for the one-audio batch, constructs the model through `Config`, reports the configured provider, and passes both synthetic and retained-radio CPU decodes.
