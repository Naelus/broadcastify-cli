# Progress log

This is the dated verification and delivery log for `GOAL.md`. Keep forward-looking capabilities in `FEATURES.md` and unresolved defects in `BUGS.md`.

## July 16, 2026

### Managed Linux Web launcher

- Added the wheel-packaged `radio-archive-service` entry point with install, start, restart, status, stop, journal, unit-preview, and conservative uninstall commands for a per-user systemd service.
- The service persists only absolute non-secret paths and a fixed `127.0.0.1` endpoint in owner-only JSON. It references a separate mode-0600, comment-only `.env` template; credential values never enter the unit or service JSON.
- The generated unit supervises failure restart, uses SIGINT plus a bounded stop window, an owner-only umask, `NoNewPrivileges`, private temporary storage, and read-only system paths without blocking outbound website/model traffic. The shared Web cookie/action-token/origin and archive quota boundaries remain unchanged.
- The default working/archive layout follows XDG user directories. Custom absolute paths can adopt an existing library, uninstall preserves all data/private settings, and the recorded venv Python path is not dereferenced to a dependency-free system interpreter.
- Refactored `broadcastify-web` into a reusable runner and added explicit `--working-dir`, keeping service cookies, `.env` discovery, and worker children on the selected private path.
- Eight new service/parser tests, including a real selected-venv import check, bring the complete local suite to **127 passed in 6.09 seconds**. The final `broadcastify_cli-0.4.0-py3-none-any.whl` is 182,202 bytes with SHA-256 `154c8fcd3a17cfcdd4882f05abefb75a58b5e548565c371a16388c32e3fb0f55`; archive inspection confirms the service module, all Web static assets, and the `radio-archive-service` console entry. Exact real-Linux unit/wheel validation remains open under B-005 and will be recorded separately.

### Local Library and UI

- Replaced the long workspace with WinUI navigation for Local Library, New Archive, Review & Ask, Area Watch, and Settings.
- Added a master/detail Local Library viewer at the 1240x900 reference size.
- The selected day now shows a five-stage processing timeline, combined-audio player, bounded timestamped transcript preview, local paths, and explicit continue/review/folder actions.
- Verified visually against real Example City and Example County retained days, including a fully analyzed day and a quota-interrupted partial day.
- Added saved non-secret processing defaults and Windows Credential Locker login persistence.
- Added the explicit `BundleLocalEnv=true` private build; ordinary builds remove stale bundled credentials.

### First-run readiness and diarization regression

- Added a compact five-step **Setup** tab to WinUI and a matching full-width Web overview. Account, writable storage, transcription, speaker labels, and analysis now show detected/configured/verified states with direct actions instead of requiring the user to infer readiness across three settings tabs.
- Hardware profiles now expose independent `transcription_ready`, `diarization_ready`, and `analysis_ready` evidence. The ordinary check remains download-free; explicit transcription, speaker-label, and provider actions distinguish detection from model execution.
- Added `diarization-self-test` across the worker, WinUI, and Web UI. It loads Community-1 on the selected CUDA/CPU device and runs generated local audio without making a Broadcastify request.
- The first real UI test exposed the remaining combined-file failure: current pyannote/Torchaudio tried to route filenames through an incompatible TorchCodec/FFmpeg-DLL combination. The same problem reproduced outside the UI.
- Replaced filename input with an FFmpeg-decoded float32 scratch file, memory-mapped as a PyTorch `waveform`/`sample_rate` dictionary. This bypasses TorchCodec, keeps day-long PCM off the Python heap, removes the raw scratch file after inference, and retains the compact lossless retry input.
- The corrected native self-test passed on CUDA in **6.5 seconds** with one synthetic turn. A **250-second** slice from retained combined feed 90001 audio passed in **11.75 seconds**, returning **52 turns across three acoustic clusters**.
- Native visual/accessibility QA passed at **1240×900**. Web QA passed at desktop and **390×844**, with no horizontal overflow and an empty console; the detection action moved the reference machine from 2/5 cheap prerequisites to 5/5 available stages.
- Full Python suite: **109 passed**. Browser JavaScript syntax and the Release WinUI build passed with **0 warnings, 0 errors**. The public publish verifier confirmed no private environment and a ready bundled Windows ML runtime; the final owner-only publish independently verified the explicitly bundled ignored `.env`.

### Bounded CPU and joined Linux parity

- A 60-second retained combined-audio slice completed real pyannote CPU diarization on the Windows reference machine in **19.203 seconds**, producing **23 turns across two clusters**. This is useful bounded evidence, not a linear full-day prediction; B-004 remains open.
- Created a fresh isolated TrueNAS clone at exact pushed commit `historical-validation`. Its existing pure-Python target passed **109 tests in 0.87 seconds** without host changes.
- TrueNAS home and `/tmp` are intentionally `noexec`. CPU Torch therefore could not map from the source dataset even though its wheels installed. A Python 3.12 private runtime was built inside the already-recorded whisper.cpp Vulkan image and placed in user-writable executable `/var/tmp`; no NAS package, service, group, or storage setting changed.
- The exact commit plus that runtime passed **109 tests in 10.35 seconds** inside image `sha256:2f8d2507ee587a8b94c514d27545089234810e6da3e6dd0f2e1327f2f96de861` with networking disabled and the source mounted read-only.
- A staged failure deliberately proved resumability: CPU diarization completed and cached before a library-path error; the next run reused the diarization and whisper preparation, completed Vulkan ASR, persisted the transcript, and a later run resumed at analysis rather than repeating either audio stage.
- The final fresh warm-cache run used a new output directory, **network disabled**, read-only root, all capabilities dropped, `no-new-privileges`, numeric user 950, explicit AMD render groups, and bounded tmpfs. It completed the full 30-second retained-radio workflow in **15.729 seconds**.
- Retained metadata reports `whisper.cpp`, device/backend `vulkan`, explicit `Vulkan0 / AMD Radeon 890M Graphics (RADV GFX1150)` runtime evidence, one transcript segment, CPU diarization complete with **5 turns**, and no fallback.
- Local `ggml-org/gemma-3-1b-it-GGUF:Q4_K_M` loaded on the same AMD Vulkan device. The app correctly extracted **0** supported incidents from the short fixture, persisted **1 passage, 1 embedding, and 1 daily summary**, and finished at pipeline 100% / **Ready to review** instead of inventing an event.
- The completed test container was removed. Exact source, model caches, transcript/database, and backend logs remain only under the user-owned isolated test paths for audit and future Web-launcher work.

### Linux Web job boundary and offline cache repair

- Added `scripts/web_job_smoke.py`, a dependency-free headless harness that starts the real loopback service on an ephemeral port, establishes its same-site cookie and action token, submits one supported `/api/jobs` request, polls it, and writes the final JSON snapshot. Its diagnostics integration test raised the local suite to **110 passed**.
- Exact pushed commit `historical-validation` passed all **110 tests in 2.28 seconds** inside the same network-disabled, read-only Vulkan image. The first full HTTP job then exposed a real offline bug: normal diarization rejected a missing Hugging Face token before asking pyannote to reuse the complete local cache.
- Normal processing and the explicit speaker self-test now try the cached Community-1 pipeline with `token=None`; only a cache miss asks for a read token for the first download. Regression coverage includes successful tokenless cache reuse and the first-download error. Exact pushed commit `historical-validation` passes **113 local tests** and all **113 tests in 2.31 seconds** inside the immutable image.
- A fresh headless Web job from that exact commit completed through the protected HTTP boundary without a Hugging Face token. The harness established the real cookie/token session, posted `continue-local`, and the child worker completed 30 seconds of retained radio in **15.139 seconds** of container time.
- Transcript evidence records whisper.cpp on `vulkan`, explicit `Vulkan0 / AMD Radeon 890M Graphics (RADV GFX1150)`, CPU diarization with **5 turns**, and no ASR fallback. Local Gemma loaded on the same AMD Vulkan device.
- SQLite retained **1 feed-day, 1 transcript segment, 1 passage, 1 embedding, and 1 daily summary**. The short fixture correctly produced **0 incidents** and finished at pipeline 100% / **Ready to review**.
- The completed container used image `sha256:2f8d2507ee587a8b94c514d27545089234810e6da3e6dd0f2e1327f2f96de861`, numeric user 950, read-only root, network `none`, `cap-drop=ALL`, `no-new-privileges`, render groups 44/107, and PID limit 1024. It was removed after inspection; exact source and result artifacts remain in the isolated user-owned `app-historical-validation` and `webjob-historical-validation-1` paths.

### All-CPU Web workflow and analysis credibility

- Removed every GPU device/group from a fresh Web-job container and selected whisper.cpp CPU, pyannote CPU, and the CPU device of the same local llama.cpp build. The first exact-current run proved CPU ASR and diarization, then exposed llama.cpp b9637 rejecting the daily schema's `maxLength: 2500` grammar with HTTP 500 even though the 1B Q4 model was generating at roughly 70 tokens/second.
- Removed the parser-hostile grammar bound, retained a deterministic 250-word application clamp, and made the local client retry only recognized schema/parser errors through llama.cpp's simpler JSON-object mode. Exact commit `historical-validation` passed **116 tests** and completed the all-CPU job.
- That run exposed a more important credibility defect: with only incident I1 retained, the generated daily brief invented incidents I2-I4 and an unsupported count of four priority events. Daily summaries now reject unknown incident IDs and activity counts above the supplied record set, retry once with the exact allowed IDs, then use the existing deterministic evidence summary if the model remains ungrounded.
- Incident normalization now treats confidence as extraction confidence from noisy ASR rather than event certainty: it caps every incident below 1.0, caps a single coarse 20-second-or-longer segment at **0.90**, and prefixes unhedged claims with `Radio traffic reported:`.
- Exact pushed commit `historical-validation` passes **119 local tests** and all **119 tests in 2.28 seconds** inside the immutable image. A fresh protected all-CPU Web job completed in **21.282 seconds** with networking disabled, no Hugging Face token, and no GPU devices or supplemental render groups.
- Transcript evidence records whisper.cpp backend/device `cpu`, `ggml_vulkan: No devices found`, one segment, and CPU diarization complete with **5 turns**. llama.cpp listed only the AMD Ryzen CPU, loaded the 1B Q4 model in 0.645 seconds, and generated at about 56-72 tokens/second.
- The short fixture retained one evidence-backed `person_with_weapon` record from the exact quote `I was chased with someone with a gun`, normalized to `Radio traffic reported: a person was chased with a gun.` at confidence **0.90**. Both free-form summary attempts were rejected as unsupported; the persisted brief contains only that one reported category and explicitly says noisy ASR is not a confirmed outcome.
- SQLite retained **1 feed-day, 1 transcript segment, 1 incident, 1 passage, 1 embedding, and 1 daily summary**, finishing at pipeline 100% / **Ready to review**. This is bounded fallback evidence, not a full-day CPU benchmark; B-004 remains open.

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

#### Linux Web UI real-host validation

- Cloned exact public commit `historical-validation` into a new child of the existing isolated TrueNAS test directory; the prior Vulkan artifacts were not overwritten.
- TrueNAS system Python 3.11.9 intentionally lacked `venv`, `ensurepip`, and pip. No host package was installed. PyPA's standalone `pip.pyz` (1,756,180 bytes, SHA-256 `6ddc3444b803a48d83ccf1c4ad846717b42c8ffc9d74713a53ae829a97201365`) installed the project/dev dependencies into a private 17 MiB `.python` directory.
- The complete suite passed on the Linux host: **96 passed in 0.80 seconds**.
- Started the exact commit on `127.0.0.1:18765` only, then exercised it from the same host. Health/bootstrap reported Linux and `loopback_only=true`; the v4 static UI included the portable profile controls; a POST without the action token returned 403; a retained-file range request returned 206 with the exact requested bytes; and the diagnostics child worker completed with six Linux profiles.
- Stopped only the verified test PID after the smoke run, confirmed the port closed and no matching process remained, and left all artifacts under the user-owned isolated test directory. No NAS package, service, group, or storage setting changed.

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

### Radius-based regional discovery

- Added a bounded, cached reader for the official 2025 US Census ZCTA Gazetteer and great-circle ZIP-centroid distance calculation. Radius searches accept 1–100 miles, inspect at most 20 nearby ZIP areas, and retain the exact distance basis; ordered-ZIP mode remains available for Census-missing USPS ZIPs.
- Updated reverse-engineered Broadcastify website discovery to retain the nearest matched ZIP, approximate mileage, and stable priority rank while still deduplicating county-directory results by feed ID. Discovery makes no archive-media requests.
- Persisted center ZIP, radius, ZIP cap, searched ZIP metadata, explicit feed selection, and nearest-first order in SQLite with an additive migration for existing databases.
- Added matching native WinUI and loopback Web controls. A real archive-free 12345/10-mile/4-ZIP run downloaded and validated the 930 KB Census cache, then returned eight Example County feeds led by feed 90001; all shared the honest 0-mile county-directory approximation because the queried ZIPs mapped to the same county.
- Focused geography/search/storage/Web/worker suite: **30 passed**. WinUI isolated Debug build: **0 warnings, 0 errors**. Browser JavaScript syntax check passed.

### Persisted nearest-first acquisition queue

- Moved multi-feed execution from the native loop into a shared backend runner used by WinUI and the loopback Web UI. SQLite retains the profile/date/processing fingerprint, feed priority, approximate distance, status, attempts, completed and missing days, explicit quota state, and compact job result.
- Credentials are excluded from the persisted processing JSON and fingerprint. Every area job is forced to one download worker and preserved source blocks at the Web service boundary.
- Completed feeds are skipped without authentication or archive metadata, interrupted `running` items recover to `pending`, partial feeds rerun against exact cached blocks, and the first `download_limited` result stops every lower-priority feed.
- Native Area Watch now requires a saved reviewed profile, runs/resumes the shared queue, performs sequential analysis only for completed transcripts, and displays the latest retained queue. Web Area Watch mirrors the processing switches, queue action, and per-feed status.
- Real browser smoke: a one-feed July 11 queue reused all 48 cached blocks and completed with model work disabled; the exact rerun logged only `already complete`. No media download could have occurred because all 48 cache resolutions finished inside five seconds despite the five-second network pacing guard. Mobile 390×844 had 375/375 px document width, and browser console logs were empty.
- Renamed ambiguous archive progress from `Downloaded` to `Ready (cached or downloaded)` so cache reuse is not mistaken for fresh quota consumption.

### Area Watch UX polish

- Visually inspected the current native build at its real 1240×900 window. Local Library remains a non-scrolling master/detail workspace; Area Watch keeps discovery/queue controls in a two-column card and story evidence in its own tab, with the retained queue visible beside the saved profile.
- Made the native public-safety checkbox a live filter rather than a one-time search option. The Web client now has the same checked-by-default filter, retains all discovered results in memory, and can reveal optional weather/rail categories immediately without another website request.
- The real 12345 radius result showed **6 of 8** public-safety feeds by default; opting out showed all 8 including NOAA weather and Example City-area rail, then the UI was restored to the recommended filter. Browser console logs remained empty.
- A saved Web area profile now reselects and reopens after saving instead of dropping the user back to an unselected profile state.
- Produced the current private runnable Windows build at `BroadcastifyCli.WinUI/bin/Private/win-x64` with the ignored `.env` verified byte-for-byte by SHA-256 comparison without displaying it. A normal Release rebuild then verified its output contains no bundled environment file. Both builds completed with **0 warnings, 0 errors**.
- Attempted `dotnet publish` and documented the real `NETSDK1152` duplicate Windows App SDK asset collision as B-009 rather than treating an unpackaged build as a successful publish.

### Runnable Windows publish and bundled Windows ML runtime

- Removed the build-only Windows ML `ProjectReference` from the WinUI dependency graph. A custom MSBuild target now builds the sibling executable independently and copies its complete self-contained runtime to `windowsml/`, eliminating the `NETSDK1152` duplicate `MsixContent` collision.
- Found and fixed a second publish-only defect: the SDK omitted `App.xbf`, `MainWindow.xbf`, and `Broadcastify Desktop.pri`, causing an immediate `Microsoft.UI.Xaml.dll`/`0xc000027b` crash. The publish target now verifies and copies those compiled resources.
- Added `scripts/verify_windows_publish.ps1`. It validates required desktop/helper/runtime files, private/normal environment isolation, stale-private cleanup, and a live helper probe without printing credentials.
- A private → normal publish cycle passed. The private `.env` matched by hash without being displayed or leaking into normal build output; the subsequent normal publish removed the stale private copy.
- The private published helper completed a real FP32 CPU Whisper decode in **0.589 seconds**, reporting `decode_ready=true`, provider `CPU`, and backend `Windows ML / ONNX Runtime GenAI CPU`.
- Launched the corrected normal publish through Windows, confirmed it remained open and loaded all nine retained library days, and inspected its Activity Log: `.venv Python`, the expected repository, and `Windows ML helper: bundled runtime`.
- The verified private publish is `BroadcastifyCli.WinUI/bin/Private/publish-win-x64`. It is runnable from the source tree; supervised Python/dependency/model packaging remains explicit future work.

## Earlier validated work

- Feed 90001 completed July 11–12 end to end with 97 retained archive blocks, continuous daily audio, 1,716 transcript segments, 87 incidents, daily summaries, semantic Q&A, and a seven-day brief with explicit missing coverage.
- July 12 full-day pyannote diarization completed in about 25 minutes on the RTX 3090, produced 5,556 turns across five anonymous speaker clusters, and left no transcript words unlabeled.
- Multi-ZIP Example City discovery persisted a six-feed regional profile and created evidence-backed area story leads without treating missing feeds as quiet.
- Example County measurement established the explicit `Download limit exceeded` response, immediate stop policy, and a plausible—but unconfirmed—roughly 200-request account window.
