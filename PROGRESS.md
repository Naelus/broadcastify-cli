# Progress log

This is the dated verification and delivery log for `GOAL.md`. Keep forward-looking capabilities in `FEATURES.md` and unresolved defects in `BUGS.md`.

## July 16, 2026

### Local Library and UI

- Replaced the long workspace with WinUI navigation for Local Library, New Archive, Review & Ask, Area Watch, and Settings.
- Added a master/detail Local Library viewer at the 1240x900 reference size.
- The selected day now shows a five-stage processing timeline, combined-audio player, bounded timestamped transcript preview, local paths, and explicit continue/review/folder actions.
- Verified visually against real Example City and Example County retained days, including a fully analyzed day and a quota-interrupted partial day.
- Added saved non-secret processing defaults and Windows Credential Locker login persistence.
- Added the explicit `BundleLocalEnv=true` private build; ordinary builds remove stale bundled credentials.

### Pipeline correctness

- Added first-missing-stage discovery and local continuation.
- Existing transcripts can receive speaker labels without rerunning Whisper.
- Tightened diarization status so a request flag alone does not count as completed labeling.
- Python suite before the Windows ML adapter: **69 passed**; after its streaming adapter test: **70 passed**.
- WinUI private Release build: **0 warnings, 0 errors**.

### Hardware parity

- Added per-stage profile diagnostics for CUDA, CPU, Vulkan, OpenVINO, and Windows ML.
- OpenVINO CPU successfully transcribed a real 22.74-second police-radio clip (23 words, 3 segments).
- OpenVINO GPU rejected the current model on this NVIDIA host; automatic retry on CPU succeeded and identifies the fallback in metadata.
- llama.cpp device inspection detected Vulkan on both the RTX 3090 and AMD Radeon integrated graphics.
- Built the official ONNX Runtime GenAI Windows ML helper. A CPU FP32 Whisper Tiny export successfully transcribed the same local test clip.
- Integrated the Windows ML helper with one persistent model process and bounded 28-second chunks. The real clip completed through the Python adapter with 22.74 seconds of audio, one timestamped segment, and 147 output characters; the model self-test reports `decode_ready=true`.
- Verified the native Settings flow: selecting Windows ML plus the validated model path changed the profile from runtime-only to ready after the real decode check; recommended automatic/CUDA defaults were restored afterward.
- DML/WinML exports remain gated after reproducible graph-capture/fused-node errors; tracked as B-001.

#### AMD Vulkan / immutable Linux host

- Validated on an isolated TrueNAS/Debian 12 host with an AMD Radeon 890M (`RADV GFX1150`). No host package, service, storage, or group configuration was changed; artifacts remain in one user-owned test directory.
- Pulled the official `ghcr.io/ggml-org/whisper.cpp:main-vulkan` image. Recorded image ID `sha256:2f8d2507ee587a8b94c514d27545089234810e6da3e6dd0f2e1327f2f96de861` and repository digest `sha256:3ac0269a3752513c64c31ee8000b1a3354ed68cab790f51008a832a56b3e461a`.
- Verified the 77,704,715-byte `ggml-tiny.en.bin` model at SHA-256 `921e4cf8686fdd993dcd081a5da5b6c365bfde1162e72b08d75ac75289920b1f`.
- The first direct CLI run decoded the 22.7-second retained fixture in 1.023 seconds and emitted normalized JSON. The exact Python app adapter then passed with `--network none`, read-only root, dropped capabilities, no-new-privileges, bounded tmpfs, exact read-only audio/model mounts, and one writable output mount.
- The final app-adapter run completed in 1.022 seconds, produced one 159-character segment, clamped upstream's short-input progress to 100%, and retained initialization evidence naming `AMD Radeon 890M Graphics`, `Vulkan0`, and `using Vulkan0 backend`.
- Native Linux/macOS `whisper-cli` discovery now covers PATH and common local build trees. Non-WAV inputs are atomically converted to a 16 kHz mono PCM WAV so a build without optional FFmpeg decoding does not fail on combined MP3s.
- Downloaded the official llama.cpp b9637 Ubuntu Vulkan release (38,391,553 bytes, SHA-256 `6ca268d758aae9e8518afa43042678e8b60b47f0d34df7d6efff4ca622c74313`) and ran it inside the already-validated Vulkan container because the immutable host intentionally lacks a Vulkan loader on its no-exec user dataset.
- Public `ggml-org/gemma-3-1b-it-GGUF:Q4_K_M` (806,058,240 bytes) offloaded all 27/27 layers. Vulkan model memory was 762.49 MiB; prompt evaluation measured 320.05 tokens/second and cached generation measured 105.39 tokens/second.
- A numeric-user container initially failed before model load because it lacked a writable home. The managed llama.cpp launcher now supplies private `HOME`, `LLAMA_CACHE`, and `HF_HOME` fallbacks only when the inherited POSIX home is absent/unwritable.
- Fixed an honesty bug: Vulkan profile readiness now requires a detected llama.cpp Vulkan device rather than any `llama-server` executable. The container image is never pulled implicitly; diagnostics only inspect an explicitly configured image already present locally.
- Complete Python suite after the native/container runtime work: **87 passed**.

#### OpenVINO selected-engine validation

- Updated the OpenVINO path for the installed 2026.2.1 runtime: removed the obsolete NPU static-pipeline override, added the official Distil Large V3 INT8 mapping, and normalized Web UI `.en` model aliases.
- Added CPU recovery when GPU/NPU/AUTO model construction itself fails, complementing the existing first-generation fallback. Transcripts now replace the requested backend label with the actual fallback backend and retain whether initialization or generation failed.
- Re-ran the retained 22.7-second radio fixture with the already-cached official Tiny INT8 model. AUTO completed in 1.093 seconds with 23 words and 3 timestamped segments. Explicit GPU failed during generation, retried on CPU, and returned the identical transcript in 1.828 seconds.
- Added an explicit selected-engine self-test worker to both WinUI and the loopback Web UI. It generates one second of local silence, loads the exact selected engine/model/device, may download a missing managed model only after the user clicks **Test engine**, and returns backend/timing/fallback metadata without returning generated transcript text.
- The full worker path passed with OpenVINO AUTO in 0.766 seconds. Explicit GPU correctly surfaced `OpenVINO CPU (fallback from GPU)` in 1.735 seconds instead of the old misleading GPU label.
- Visually and interactively verified WinUI at 1228×894 and the browser UI at desktop and 390×844. Both showed a clear successful OpenVINO AUTO result in 0.8 seconds; automatic/turbo settings were restored after testing. WinUI Release build: **0 warnings, 0 errors**.
- Complete Python suite after the OpenVINO/self-test work: **92 passed**; browser JavaScript syntax check passed with the bundled Node runtime.

#### Windows ML runtime validation and provider boundary

- Rebuilt Whisper Tiny with the stable ONNX Runtime GenAI 0.13.1 builder and compared it with the installed 0.14.1 builder/runtime. Fresh CPU and DML models initially reproduced the helper's `DivideByZeroException`.
- The tag-matched Microsoft Python Whisper sample decoded the same CPU model, isolating the app defect to the C# call. Whisper requires the batched-prompt multimodal overload even for one audio input; switching from the scalar overload fixed the helper. The final CPU helper decoded the retained 22.7-second radio clip in 0.622 seconds after warm caches and returned the same 147-character text as the Python control.
- Added read-only Windows ML provider discovery, activation of already-installed providers, and explicit provider acquisition. Ordinary transcription never downloads a provider. The helper uses ONNX Runtime GenAI's native provider-registration entry point and reports configured provider/backend instead of calling every success GPU-accelerated.
- The explicit acquisition path installed and registered Microsoft's certified `NvTensorRTRTXExecutionProvider` 1.8.24.0. Provider acquisition took 45.124 seconds on this host; later activation was local and quick.
- Current 0.13.1/0.14.1 DML Whisper exports remain blocked: graph capture rejects the builder's non-shared cache, while a forced shared cache fails `DmlFusedNode_0_0` with an invalid-key error.
- A corrected TensorRT RTX export completed both synthetic and real decodes, but the provider rejected 36 Whisper `Attention`/`MultiHeadAttention` nodes and partitioned/fell back. The 22.7-second real clip took 15.895 seconds, versus 0.622 seconds on the CPU model, so this is diagnostic evidence rather than an enabled acceleration path.
- The final Python streaming adapter run retained 22.74 seconds, one segment, 147 characters, and `Windows ML / ONNX Runtime GenAI CPU`, proving the honest backend label crosses the helper boundary.
- Final WinML, DirectML-flavor, and complete WinUI builds all completed with **0 warnings, 0 errors**. The complete Python suite remains **92 passed**; provider activation and its download boundary are documented in `docs/hardware-backends.md`.

#### Apple Metal profile and portable settings UX

- Added `metal` to the persisted job contract, CLI, Web settings, automatic engine normalization, and whisper.cpp adapter validation. Metal is native-only; the app rejects a container request because Docker/Podman cannot expose Apple Metal through this adapter.
- Native whisper.cpp backend inspection recognizes adjacent ggml-metal libraries and, on macOS, verifies `Metal.framework`/`ggml-metal` linkage with `otool`. llama.cpp device output already normalizes `Metal0` to the same profile contract.
- macOS automatic selection chooses whisper.cpp only when a native Metal build is actually detected; otherwise it retains the CPU fallback. The Apple profile requires both Metal ASR and Metal analysis while keeping pyannote on CPU.
- Added a compact Web hardware-profile selector and a collapsed stage-by-stage comparison after **Check hardware**. Each detected profile can apply compatible transcription/device/diarization defaults without exposing the advanced controls first.
- Browser QA passed at 1365×900 and 390×844: six Windows profiles rendered with 4/6 ready, the Vulkan preset applied and restored correctly, the comparison stayed collapsed by default, there was no horizontal overflow, and the console remained clean.
- Python coverage increased to **96 passed**; browser JavaScript syntax validation passed. Real-machine macOS/Metal timing remains intentionally open in B-005.

### Archive quota reset run

- Completed a guarded feed 90001 resume for July 3–16 after downloads became available again.
- The process made 55 successful new media downloads at five-second pacing: 48/48 July 3 blocks and 7/48 July 4 blocks. The next request received the explicit quota response.
- It made no more media requests, combined the now-complete July 3 audio, reused complete July 11–12 caches, and preserved every incomplete date for the next run.
- The 55-request result contradicts treating the earlier roughly 192-redirect observation as a fixed daily quota; documentation now describes the budget/reset as dynamic or rolling and unknown.

### Portable analysis providers

- Added a common analysis-client contract without changing the validated local llama.cpp default.
- Added OpenAI Responses Structured Outputs with `store=false`, environment-only API keys, bounded transient retries, and an explicit transcript-transmission gate.
- Added generic authenticated OpenAI-compatible Chat Completions for local or remote llama.cpp/Ollama/LM Studio-style endpoints.
- Added an ephemeral, read-only Codex CLI harness that reuses saved CLI authentication, isolates its working directory, requests a JSON schema, and strips unrelated secrets from the child environment.
- Provider/model/endpoint identities are distinct in SQLite caches, preventing conclusions from one provider from masquerading as another provider's run.
- Added a separate native **Analysis & AI** Settings tab instead of adding more controls to the processing page. It exposes local Gemma, OpenAI Responses, compatible `/v1`, and saved-login Codex providers; non-secret choices persist in the user settings file and an API key is session-only unless the user explicitly selects Windows Credential Locker.
- Wired the selected provider into local continuation, post-job analysis, selected-day analysis, Q&A, weekly summaries, and regional story briefs. Explicit blank/false UI values override environment defaults so an old `.env` switch cannot silently re-enable external analysis.
- Added provider-specific readiness checks that never send transcript text: local llama.cpp executable discovery, OpenAI key presence without a billable request, compatible-endpoint `/models`, and local `codex login status`.
- Visually exercised the tab at 1240x900. Local llama.cpp was found, OpenAI correctly reported a missing key, consent-off blocked external use, and the desktop Codex check reported `Logged in using ChatGPT`; no live model request was made. The private local provider and external-sharing toggle were restored before closing.
- Python suite after native provider settings and diagnostics: **77 passed**. WinUI Release build: **0 warnings, 0 errors**.

### Delivery checkpoints

- Git identity: `naelus <9455516+Naelus@users.noreply.github.com>`.
- Remote: `https://Naelus@github.com/Naelus/broadcastify-cli`.
- Staged content is scanned for common token/password patterns before every commit.
- Backend/persistence commit `historical-validation` was pushed to `origin/main` after 69 tests passed and the credential-pattern audit was clean.
- Native Library/UI commit `historical-validation` and Windows ML integration commit `historical-validation` were separately reviewed, audited, and pushed to `origin/main`.

### Cross-platform browser companion

- Added the `broadcastify-web` entry point and a dependency-free Python HTTP service that binds only to loopback. Each launch creates a random same-site session cookie; mutating actions additionally require a token header and matching origin.
- Reused the existing JSON worker for website feed/ZIP search, authentication, guarded range jobs, local continuation, incident clips, day analysis, Q&A, weekly summaries, area profiles/briefs, provider checks, and hardware diagnostics. The service permits one heavy worker at a time, forces archive concurrency to one, preserves source blocks, and keeps the database path explicit across child workers.
- Added safe byte-range streaming limited to the configured archive root, so 24-hour combined audio and generated evidence clips play without exposing arbitrary filesystem paths or contacting Broadcastify.
- Built a responsive Library/New Archive/Review/Area/Settings browser shell. The real retained corpus showed 9 feed-days, 6 ready days, 3 incomplete days, a 24:23:04 combined stream, 40 incidents, 736 diarized transcript segments, the saved 2/7-day brief, and a 10-feed regional profile with explicit 1/10-feed coverage.
- Visually exercised the desktop and 390x844 layouts. Transcript search returned the retained vehicle-fire line, the local llama.cpp readiness check succeeded without loading a model, Codex remained blocked while external sharing was off, mobile navigation opened correctly, and long incident/story surfaces now default to 12/10 highest-ranked records with an explicit show-all action.
- Browser console remained free of warnings/errors. Service/security/media tests added four cases; the complete Python suite is now **81 passed** and the browser JavaScript passes the bundled Node syntax check.

## Earlier validated work

- Feed 90001 completed July 11–12 end to end with 97 retained archive blocks, continuous daily audio, 1,716 transcript segments, 87 incidents, daily summaries, semantic Q&A, and a seven-day brief with explicit missing coverage.
- July 12 full-day pyannote diarization completed in about 25 minutes on the RTX 3090, produced 5,556 turns across five anonymous speaker clusters, and left no transcript words unlabeled.
- Multi-ZIP Example City discovery persisted a six-feed regional profile and created evidence-backed area story leads without treating missing feeds as quiet.
- Example County measurement established the explicit `Download limit exceeded` response, immediate stop policy, and a plausible—but unconfirmed—roughly 200-request account window.
