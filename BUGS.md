# Known bugs and gaps

Last updated: July 17, 2026

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
- **Observed:** pyannote's supported app path is PyTorch CUDA or CPU. Vulkan/OpenVINO/Windows ML profiles deliberately fall back to CPU. A 60-second retained combined-audio slice completed on the Windows CPU in 19.203 seconds; the exact Linux AMD joined run also completed CPU diarization before Vulkan ASR/analysis.
- **Next:** Benchmark CPU on full-day audio and investigate supported Intel XPU/other backends without weakening the reliable fallback.

### B-004 — Full CPU-only end-to-end timing is not recorded

- **Severity:** Medium.
- **Observed:** Exact commit `historical-validation` completed a protected 30-second all-CPU Web job in 21.282 seconds: whisper.cpp CPU, pyannote CPU, llama.cpp CPU, embeddings, persistence, and grounded daily summary. This proves the fallback contract but does not predict multi-hour throughput.
- **Next:** Run a longer representative slice and then a full day if practical; record CPU ASR, diarization, and CPU LLM time separately rather than extrapolating the short result.

### B-005 — Cross-platform visual QA, keychain, and real macOS validation remain incomplete

- **Severity:** Medium for product portability; no impact on native Windows use.
- **Observed:** Exact commit `historical-validation` completed a fresh network-disabled retained-radio workflow through the real loopback HTTP job boundary: same-site session cookie, action token, `continue-local` worker, whisper.cpp on AMD Vulkan, pyannote on CPU, local Gemma on AMD Vulkan, embeddings, summary persistence, and final Ready-to-review state. Exact commit `historical-validation` then closed the managed Linux launch gap: its verified 0.4.0 wheel and systemd user unit passed real start/health/session/bootstrap, loopback bind, failure restart, 0600 config/env/unit/owner-log, useful log viewing without journal access, graceful stop, and data-preserving uninstall on the same TrueNAS/Linux host. There is still no Linux browser visual/accessibility pass against a full model job. Apple Metal is implemented with CPU diarization and macOS auto-selection when both Metal backends are detected, but no real Mac run has occurred. Cross-platform credential persistence is `.env`/session-only rather than an OS keychain.
- **Next:** Visually exercise the complete browser workflow during a Linux full-model job, then run a macOS install/CPU-or-Metal workflow; add a platform keychain adapter without weakening the loopback/session-token boundary.

### B-006 — Feed display names are missing for some retained legacy days

- **Severity:** Low.
- **Observed:** Local Library shows `Feed 90003` when that feed was downloaded before its search result entered the persistent catalog.
- **Next:** Backfill feed metadata from saved manifests or a quota-free directory lookup.

### B-007 — Archive quota size/reset schedule remains inferred

- **Severity:** Operational.
- **Observed:** Broadcastify publishes no numeric archive-download quota or reset timestamp. One measured window allowed roughly 192 successful archive redirects, the July 16 availability follow-up allowed only 55 before the explicit limit response, and a later Example City window allowed 97 consecutive new media responses with no 429.
- **Control:** Sequential pacing, exact cache reuse, and immediate stop on explicit exhaustion.
- **Next:** Ask Broadcastify support for authoritative details and prioritize future regional acquisition by user distance/importance rather than a guessed quota size.

## Recently fixed

### F-015 — Model incident cards could cite unrelated audio and leak radio identifiers

A live Example City run produced a “stolen squad car in Example Township” card whose citations said only that a subject was under arrest/being transported and that a person from an unrelated domestic call was waiting in a black car. Incident retention now requires meaningful lexical support plus exact-evidence coverage for critical event concepts, rejects citations scattered more than ten minutes apart, deterministically normalizes clear category/priority contradictions, and deduplicates citation subsets. Public text and quotes redact obvious identifiers and context-supported private names while retaining source ASR internally. Daily briefs reject unsupported outcome language, and local JSON extraction is deterministic. The final two-day Example City v9 audit retained 64 cards with zero post-persistence support failures or detected public-field name leaks; 12 exact clips were generated and all 142 tests pass.

### F-014 — A removed Gemma quant selector broke resumed local analysis

The upstream Gemma 4 12B GGUF repository removed the former `Q4_K_M` file while existing settings still selected it, so llama.cpp exited before incident analysis even when that exact 7.4 GB file remained in an older local Hub snapshot. The managed launcher now detects explicit local GGUF paths and selected GGUFs in current or older Hugging Face snapshots, launches cached files with a stable API alias, and preserves the true model identity used for SQLite caching. The default is the repository's available `Q4_0` quant; an old saved selector reuses its exact cached file or migrates to `Q4_0` when no cache exists. The complete Python suite passes **133 tests**, the WinUI Release build has **0 warnings and 0 errors**, and the reference cache resolves without a network request.

### F-013 — Local daily briefs could invent incident IDs and overstate ASR certainty

An exact CPU job retained only incident I1, but the free-form daily brief invented I2-I4 and claimed four priority events. Daily summaries now reject unknown incident IDs and counts above the supplied record set, retry once, and fall back to a deterministic evidence-only brief when still ungrounded. Incident normalization caps noisy-ASR extraction confidence below certainty, caps one coarse 20-second-or-longer segment at 0.90, and marks unhedged claims as radio-reported. Exact commit `historical-validation` rejected both unsupported live summary attempts and persisted only the cited I1 evidence.

### F-012 — llama.cpp rejected the daily JSON schema on the CPU path

llama.cpp b9637 expanded the daily summary's `maxLength: 2500` into a grammar repetition above its sane parser limit and returned HTTP 500. The application now enforces the 250-word bound after generation, omits the parser-hostile schema repetition, and retries only recognized schema/parser failures with llama.cpp's simpler JSON-object response format. Unrelated server 500s still surface.

### F-011 — A cached diarization model still required a Hugging Face token offline

The normal transcriber and explicit speaker self-test rejected an empty token before calling pyannote, even when the gated Community-1 snapshot was complete in the local Hugging Face cache. Both paths now attempt cached loading with `token=None`; a missing token is requested only when no usable cache can be loaded. Exact commit `historical-validation` completed the protected Linux Web job with networking disabled and no token supplied, while regression tests preserve the clear first-download error.

### F-010 — pyannote file decoding failed when TorchCodec could not load compatible FFmpeg DLLs

Current pyannote/Torchaudio delegated filename decoding to TorchCodec, which was installed but incompatible with the reference PyTorch/FFmpeg combination. Both the new speaker self-test and the same combined-file path failed before model inference. Diarization now uses FFmpeg to create a bounded-lifetime float32 PCM scratch file, memory-maps it as a PyTorch waveform dictionary, and passes that directly to pyannote. This avoids TorchCodec while keeping a day-long waveform off the Python heap. The generated-audio CUDA test passed in 6.5 seconds; a 250-second slice from retained combined feed 90001 audio passed in 11.75 seconds with 52 turns and three acoustic clusters.

### F-009 — `dotnet publish` collided on Windows App SDK assets and omitted XAML resources

The build-only Windows ML `ProjectReference` was replaced with an explicit helper build and namespaced `windowsml/` copy. This prevents the two self-contained Windows App SDK graphs from merging. The publish target also copies the unpackaged WinUI `App.xbf`, `MainWindow.xbf`, and application PRI that the SDK omitted. A normal/private/normal credential cycle passed, the published helper completed a real CPU Whisper decode, and the actual published desktop stayed open, loaded the retained library, and logged `Windows ML helper: bundled runtime`. The remaining installer/Python/model-manager work stays under B-005 and the roadmap.

### F-008 — Area jobs could continue after a feed exhausted the archive quota

Native and Web clients now call one shared SQLite-backed nearest-first queue. Its processing fingerprint excludes credentials, completed feeds are skipped, interrupted `running` items return to `pending`, the first incomplete feed resumes against the exact cache, and `download_limited` stops every lower-priority feed before another archive request. A real cached July 11 queue completed once and the second run skipped the feed without authenticating or touching archive metadata.

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
