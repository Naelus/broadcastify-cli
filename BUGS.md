# Known bugs and gaps

Last updated: July 18, 2026

Use this file for reproducible defects and concrete blockers, not the general roadmap. Remove an entry only after its fix and verification are recorded in `PROGRESS.md`.

## Active

### B-008 — Reference Windows host intermittently loses its NVMe storage path

- **Severity:** High for long unattended validation; this is not currently attributable to application code.
- **Observed:** The July 17 unattended run ended in Windows bugcheck `0x154 UNEXPECTED_STORE_EXCEPTION`; `volmgr` then failed to create the dump, so no driver stack survived. The user also observed that the SSD was absent until a full power cycle. A separate earlier `0x193 VIDEO_DXGKRNL_LIVEDUMP` was nonfatal and does not identify the storage failure.
- **Current evidence:** The Solidigm P44 Pro reports healthy SMART state and current firmware `001C`. The Gigabyte B850 AORUS ELITE WIFI7 remains on launch BIOS F1; later stable BIOS releases include PCIe-compatibility work. This makes motherboard firmware, chipset/PCIe power management, slot/contact/power, and the drive/controller path more plausible than an application memory failure, but the missing dump prevents a definitive cause.
- **Next:** Keep current backups, install the current stable (not beta) board BIOS and AMD chipset package when the user is ready, then retest. If the drive again disappears from BIOS until a power cycle, treat it as a hardware/firmware-path fault and pursue slot/drive diagnostics or RMA. The app now checkpoints model windows and writes state/logs durably, but software cannot prevent a controller disappearing from firmware.

### B-001 — Windows ML GPU providers do not yet accelerate Whisper reliably

- **Severity:** High for Windows ML parity; no impact on the default CUDA path.
- **Observed:** Stable 0.13.1 and installed 0.14.1 builder/runtime combinations fail with current DML Whisper Tiny exports. A non-shared cache is rejected by automatic graph capture; forcing the shared cache reaches `DmlFusedNode_0_0` with `invalid unordered_map<K, T> key`.
- **Second provider:** Windows ML successfully acquired and registered certified `NvTensorRTRTXExecutionProvider` 1.8.24.0, but TensorRT RTX reported 36 unsupported Whisper attention nodes. The retained 22.7-second clip decoded correctly through partition/fallback in 15.895 seconds, versus 0.622 seconds on the CPU model after warm caches, so it is deliberately not selected as an accelerated profile.
- **Current platform boundary:** Modern Windows ML exposes a certified execution-provider catalog rather than only legacy DirectML. The July 2026 Windows ML 1.8 table lists AMD MIGraphX/VitisAI, Intel OpenVINO, Qualcomm QNN, and NVIDIA TensorRT RTX alongside CPU/DirectML for compatible Windows 11 24H2+ systems; it also explicitly says the current MIGraphX provider is not supported for GenAI scenarios. Provider presence, download, and registration still do not prove that a particular Whisper graph is supported or accelerated. After updating both native projects to Windows App SDK 1.8.10, the reference-machine catalog still reports TensorRT RTX `NotReady`; no provider acquisition was performed.
- **Control:** The corrected C# helper, Python streaming adapter, and CPU FP32 model perform a real decode and identify the backend as CPU. Exact source `historical-validation` completes a protected 60-second CPU-pyannote → Windows ML CPU → local-Gemma Web job in 32 seconds and resumes idempotently; the current joined Web verifier completed the same three stage classes in 13.3 seconds with a 0.4-second Windows ML CPU decode. Diagnostics distinguish catalog detection/configuration from session-specific execution. Broken, untested, or partially-falling-back models are not advertised as acceleration.
- **Next:** Retest a published graph explicitly supported by one of the certified vendor providers or an upstream runtime/provider fix. Keep acquisition behind an explicit user action, and require an actual timed decode before enabling any provider. Prefer the validated CUDA, OpenVINO, or whisper.cpp Vulkan paths for Windows GPU ASR in the meantime.

### B-003 — Portable diarization is CPU-only outside CUDA

- **Severity:** Medium; functionally correct but potentially slow.
- **Observed:** Community-1's documented local path is PyTorch CUDA or CPU. Vulkan/OpenVINO/Windows ML profiles deliberately fall back to CPU. A 60-second retained combined-audio slice completed on the Windows CPU in 19.203 seconds; exact joined Vulkan, OpenVINO, and Windows ML jobs all completed CPU diarization before their selected ASR/analysis stages. Exact `historical-validation` also loaded the tokenless cached model and completed a generated-audio CPU proof in 4.221 seconds on the AMD host.
- **Modern candidates:** sherpa-onnx now packages offline ASR plus ONNX speaker diarization for many desktop/mobile bindings and is the best current portable-CPU benchmark candidate, but its diarizer is a separate pyannote-segmentation/embedding/clustering pipeline rather than Community-1 and its documented packaged GPU route is CUDA. NVIDIA Sortformer is promising for bounded NVIDIA workloads, but the published offline model is limited to four speakers, warns about noisy/out-of-domain audio, and cites about 12 minutes maximum on a 48 GB A6000; its noncommercial license also conflicts with the newsroom direction.
- **Joined proof:** Exact commit `historical-validation` completed the one-action AMD profile with CPU Community-1 in 10.595 seconds between Vulkan ASR and Vulkan analysis. This proves correct fallback and bounded synthetic execution, not a full-day throughput claim.
- **Next:** Benchmark sherpa-onnx against Community-1 on retained day-long police-radio slices for diarization error, unmatched transcript words, time, RAM, and interruption behavior. Also benchmark current Community-1 CPU on a full day. Do not replace the reliable fallback or add another model manager until the retained corpus demonstrates a material win.

### B-004 — Full CPU-only end-to-end timing is not recorded

- **Severity:** Medium.
- **Observed:** Exact commit `historical-validation` completed a protected 30-second all-CPU Web job in 21.282 seconds: whisper.cpp CPU, pyannote CPU, llama.cpp CPU, embeddings, persistence, and grounded daily summary. This proves the fallback contract but does not predict multi-hour throughput.
- **Next:** Run a longer representative slice and then a full day if practical; record CPU ASR, diarization, and CPU LLM time separately rather than extrapolating the short result.

### B-005 — Cross-platform keychain and real macOS validation remain incomplete

- **Severity:** Medium for product portability; no impact on native Windows use.
- **Observed:** Exact commit `historical-validation` completed a fresh network-disabled retained-radio workflow through the real loopback HTTP job boundary: same-site session cookie, action token, `continue-local` worker, whisper.cpp on AMD Vulkan, pyannote on CPU, local Gemma on AMD Vulkan, embeddings, summary persistence, and final Ready-to-review state. Exact commit `historical-validation` closed the managed Linux launch gap with a verified wheel and per-user systemd lifecycle. Exact commit `historical-validation` now closes the Linux full-model visual/accessibility gap: 148/148 host tests plus Local Library, Review, Area Watch, Settings, media, navigation, and keyboard QA at 1365×900 and 390×844, with exact document/client widths and no browser errors. Apple Metal is implemented with CPU diarization and macOS auto-selection when both Metal backends are detected, but no real Mac run has occurred. Cross-platform credential persistence is `.env`/session-only rather than an OS keychain.
- **Next:** Run a real macOS install and CPU-or-Metal workflow; add a platform keychain adapter without weakening the loopback/session-token boundary. Non-developer packaging/model acquisition also remains roadmap work.

### B-007 — Archive quota size/reset schedule remains inferred

- **Severity:** Operational.
- **Observed:** Broadcastify publishes no numeric archive-download quota or reset timestamp. One measured window allowed roughly 192 successful archive redirects, the July 16 availability follow-up allowed only 55 before the explicit limit response, and a later Example City window allowed 97 consecutive new media responses with no 429.
- **Control:** Sequential pacing, exact cache reuse, and immediate stop on explicit exhaustion.
- **Next:** Ask Broadcastify support for authoritative details and prioritize future regional acquisition by user distance/importance rather than a guessed quota size.

## Recently fixed

### F-026 — Profile verification was fragmented and portable GPU libraries could collide

Users had to run three separate tests before a selected profile became Verified, the Web selector exposed impossible Apple Metal choices on Windows, and switching a managed `distil-large-v3` OpenVINO setup to whisper.cpp could retain a model with no portable GGML mapping. Missing whisper.cpp guidance also suggested the Linux container route for Metal, even though containers cannot expose Apple Metal.

Commit `historical-validation` adds one sequential generated-input **Verify profile** action to WinUI and Web Settings, releases model/cache memory between stages, preserves completed-stage results, and stops at the first actionable failure without turning a setup problem into a worker crash. Platform-inapplicable profile choices are hidden, Vulkan/Metal reset the unsupported managed model to `turbo` with an explanation, and native setup errors are device-specific. Windows native CUDA completed the joined proof in 14.9 seconds; Web CUDA, OpenVINO, and Windows ML completed in 15.4, 13.8, and 13.3 seconds respectively.

The first exact AMD joined run then found a deeper process-boundary defect: a global llama.cpp `LD_LIBRARY_PATH` made whisper.cpp load llama.cpp's incompatible GGML library and abort before model inference. Commit `historical-validation` scopes release-bundled loader libraries to the llama-server child process. Exact source `historical-validation` passed all 171 tests in the hardened NAS container, then completed Vulkan ASR, CPU diarization, and Vulkan Gemma analysis in 27.886 seconds on the Radeon 890M. The llama log records `Vulkan0 : AMD Radeon 890M Graphics`; no validation container remained.

Commit `historical-validation` places the same joined verifier directly in both first-run Setup overviews and keeps native running/success/cancel/failure state visible beside the readiness cards. Oversized native runtime dumps are reduced to one actionable signal plus the individual diagnostic action instead of flooding the UI. Exact hash-verified source passed all 172 tests in the hardened AMD container and reran Vulkan ASR, CPU Community-1, and Vulkan Gemma analysis in 3.677/2.783/2.559 seconds (9.960 seconds total, warm caches); the llama log again identifies the Radeon 890M and no container remained.

### F-025 — Web Settings remained a long scrolling page

The cross-platform Settings view exposed the right controls but rendered setup, processing, analysis, and account configuration as one long page, unlike the native task-oriented navigation. Setup cards also scrolled approximately rather than taking the user to and focusing the requested control. Exact commit `historical-validation` introduces an accessible four-tab Settings workspace with one visible panel, roving keyboard focus, URL deep links, Back/Forward synchronization, and remembered selection. Desktop and true 390×844 browser QA covered all four sections, action routing, reload persistence, and hash-only navigation; all tabs fit without horizontal overflow, duplicate IDs, unnamed buttons, or console warnings/errors. The first live pass exposed a hash-navigation synchronization defect, which was fixed before the final pass.

### F-024 — Hardware detection could masquerade as execution readiness

Cheap diagnostics previously allowed installed runtimes, cached models, or listed accelerators to make a profile look ready even when no selected model had executed. The native worker also wrote a UTF-8 BOM to JSON stdin, which Python rejected before native self-tests while the Web path remained healthy. Exact commit `historical-validation` separates detected/configured capability from session-specific verification, recognizes a complete cached Community-1 snapshot without pretending it has run, adds a real structured analysis generation test, and marks a profile **Verified** only after ASR, diarization, and analysis all pass. The worker now writes BOM-free UTF-8. Native Windows CUDA completed the three stages in 3.5/5.3/7.80 seconds; Web completed them in 3.4/5.2/7.77 seconds; the locked-down AMD host completed Vulkan ASR, CPU diarization, and Vulkan analysis in 4.413/4.221/5.622 seconds. Both UIs hide stale setup guidance after verification.

### F-023 — A supported single-feed stolen-vehicle report was hidden from the area brief

The retained July 16 feed contained an exact Example Township stolen-squad-car report, a possible wreck follow-up, and recovery traffic, but the saved extraction cited only the recovery and remained P2/score 39. Daily Review also defaulted to P3+, hiding it. Review search now searches every priority, the cited public quote is visible, and an optional six-minute surrounding-context clip reaches the earlier dispatch while remaining clearly separate from exact evidence. Vehicle theft is deterministically at least P3 for future extraction and area ranking; the v7 area contract gives it an editorial impact bonus without requiring a second feed. The rebuilt two-day Example City brief contains the Example Township item at score 55 with one exact clip and the separately generated context clip.

### F-022 — Native startup/state and full-day model progress were lost on interruption

Three native crash reports traced to a XAML selection event calling `SelectedComboValue` before all controls existed. Settings were also saved only on a clean close and could follow a parent application's redirected LocalAppData path. Startup now begins in loading mode, combo lookup is null-safe, settings migrate to a stable user path and autosave atomically, the last review/profile reopens, and activity/crash logs append on disk. Incident analysis stores each validated model window in SQLite and resumes unfinished days from the next window. A simulated window-2 failure reused window 1, and the rebuilt WinUI executable remained open through the formerly crashing startup path.

### F-021 — Retained feed days could lose their human-readable name

Archive jobs now carry the selected feed name through native, Web, and area-queue requests, save it to the persistent feed catalog before acquisition, and record it in the combined-audio manifest. A cached current manifest can be enriched atomically without re-encoding its MP3, while Library discovery falls back to the manifest if the catalog is rebuilt. A directory-only lookup repaired legacy feed `90003` as `Example Regional Public Safety.` with no archive-media request; all five retained days show that name after a native Library refresh. The complete suite passes **150 tests**, and the isolated WinUI Release build has zero warnings/errors.

### F-020 — An inline location-adjacent alarm quote could retain a private name

The v5 sanitizer covered strong radio-name contexts and a name ending a quote after the known incident location, but a retained alarm line placed the same two-token form between the location and `for an intrusion alarm`. The sanitizer now recognizes the bounded `location, First Last, for/regarding…` dispatch form, the area contract advances to `police-radio-area-stories-v6-evidence-v9`, and older digests stay hidden until regenerated. The local-only July 15–16 rebuild retains 25 leads/references/clips, increases public redactions from five to six, and leaves the original transcript internal. All **149 tests** pass.

### F-019 — The packaged Web UI emitted a missing-favicon console error

The full-model Linux browser pass found a single SEVERE console entry: Chromium's automatic `/favicon.ico` request returned 404. The Web package now declares and serves a code-native SVG favicon with the correct `image/svg+xml` content type, and the HTTP regression test covers both the document link and static response. Exact commit `historical-validation` passes all **148 tests** locally and on TrueNAS/Linux; the repeated desktop/mobile matrix has zero failures and no warning/error/severe browser entries.

### F-018 — Regional story review rendered every evidence package in one long page

The retained Example City brief contains 25 ranked leads, and the earlier Web viewer expanded them as full cards while the native view had no persistent selected-story audit pane. Both UIs now use the same master/detail interaction: a bounded ranked index on the left and one independently scrollable evidence package on the right. Selection preserves the source quote, provenance, exact-clip playback/export, audience guidance, and score context. The Web layout becomes one column at 390×844, caps the lead index at 300 pixels, scrolls a chosen story into view, and provides a **Ranked leads** return action. Live July 15–16 Example City checks passed at 1365×900, 390×844, and the native 1228×894 window with 25 leads, one active selection, open evidence, no horizontal overflow, and no browser console errors. The complete suite remains **148 passed** and the WinUI Release build has zero warnings/errors.

### F-017 — Area evidence quotes could retain a private name immediately after a location

The first current Example City area-brief audit found a two-token private-person name in a displayed quote immediately after a known incident location. The incident sanitizer already handled stronger radio-name contexts, but this location-adjacent form was not covered. Area quote redaction now receives the incident location, replaces a matching following name with `[private person]`, and leaves the original transcript internal for audit. The area prompt advanced to `police-radio-area-stories-v5-evidence-v9`, which hides the older brief. The regenerated July 15–16 brief retains 25 source-backed leads and 25 exact clips, contains five redacted quotes, and omits the known name. Regression coverage includes this location-adjacent form.

### F-016 — Older incident and regional summaries looked current after evidence rules changed

Daily summaries, incidents, weekly briefs, and area briefs were durable, but their viewers did not distinguish the earlier extraction prompt from the current evidence-gated v9 rules. A saved Example City area profile could therefore reopen older unsupported claims even though newer days used stricter citation validation. Storage queries now expose/filter prompt versions; Local Library marks affected days **Analysis update available**, disables review, and offers local reanalysis that reuses retained audio, transcript, and speaker labels. Daily, weekly, range-Q&A, and area aggregation read only current-version incidents, and both native and Web viewers hide stale saved briefs. The explicit area-feed selection also survives rediscovery instead of silently selecting every result. Live Example City native and 390×844 Web checks passed, the WinUI Release build has zero warnings/errors, and all **148 tests** pass.

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
