# Progress log

This is the dated verification and delivery log for `GOAL.md`. Keep forward-looking capabilities in `FEATURES.md` and unresolved defects in `BUGS.md`.

## July 18, 2026

### Execution-verifiable hardware readiness and current runtime review

- Replaced profile readiness inferred from executable/device/model discovery with three explicit states: **detected**, **configured**, and **verified in this session**. Cheap refresh remains network- and model-execution-free; only successful selected-engine ASR, diarization, and analysis tests make the active profile Verified. A complete cached Community-1 snapshot is now recognized without retaining a token, but cache presence alone is never execution proof.
- Added a real analysis self-test to the shared worker contract. It asks the selected local, API, or consented Codex provider for a tiny structured JSON object using synthetic text only, sends no archive transcript/evidence, reports the actual local model/runtime, and keeps hosted usage behind the explicit action. The native bridge test exposed and fixed a pre-existing UTF-8 BOM on every JSON worker stdin payload.
- Rebuilt and exercised the Release WinUI app on the RTX 3090 reference host. The selected Automatic profile completed faster-whisper CUDA float16 in **3.5 seconds**, Community-1 CUDA in **5.3 seconds**, and local Gemma 4 12B `Q4_K_M` in **7.80 seconds**. The five-step setup summary showed all five configured stages and all three execution checks; the profile banner and selected comparison card changed to Verified and removed stale next steps.
- Exercised the same loopback Web controls with real models: CUDA ASR in **3.4 seconds**, CUDA diarization in **5.2 seconds**, and local 12B Gemma analysis in **7.77 seconds**. Cache version 16 removes stale instructions after verification. Desktop and **390×844** phone checks retained exact **375/375** document/client width, no horizontal overflow, and an empty warning/error console.
- Ran a separate current OpenVINO Tiny English CPU self-test in **1.188 seconds** with no fallback. Read-only Windows ML catalog inspection returned `NvTensorRTRTXExecutionProvider:NotReady`; no provider was acquired or registered. This preserves the explicit system-change boundary and confirms that provider presence is not being mislabeled as model execution.
- Reviewed current primary runtime documentation. Whisper remains the portable default because its model family spans faster-whisper CUDA/CPU, whisper.cpp Vulkan/Metal/CUDA/ROCm, OpenVINO CPU/GPU/NPU, and ONNX/Windows ML deployment. OpenVINO 2026 adds Whisper word timestamps across CPU/GPU/NPU. Windows ML now provides a cross-vendor certified provider catalog, but every provider/model pair still needs a timed decode. NVIDIA Parakeet 0.6B v3 is a credible optional high-throughput NVIDIA ASR benchmark, not a diarization or cross-vendor replacement.
- Community-1 remains the validated open local diarizer and officially documents CUDA or CPU. sherpa-onnx now offers one portable ONNX toolkit with offline ASR and a separate segmentation/embedding/clustering diarizer; it is the leading CPU benchmark candidate, not yet a quality-equivalent replacement. NVIDIA Sortformer is unsuitable as the default for this corpus because its published offline checkpoint is capped at four speakers, warns about noisy/out-of-domain audio, has a roughly 12-minute limit even on a 48 GB A6000, and uses a noncommercial license.
- Archived exact pushed source `historical-validation` at SHA-256 `d6d770a150a322bdd9435b67aff3e61412660456ceb4233c424570a10c7d676e`, transferred and verified it in a fresh TrueNAS path, and ran only completed `--rm` containers with networking disabled, read-only source/root, dropped capabilities, `no-new-privileges`, numeric user 950, and the existing AMD render groups. No NAS package, service, storage, group, or host setting changed.
- On the AMD Radeon 890M (`RADV GFX1150`), cheap diagnostics correctly reported the Automatic/Vulkan profiles configured but not verified. Exact stage proofs then completed whisper.cpp Vulkan ASR in **4.413 seconds**, tokenless cached Community-1 CPU diarization in **4.221 seconds**, and Gemma 3 1B `Q4_K_M` llama.cpp Vulkan analysis in **5.622 seconds**. A protected loopback `analysis-self-test` HTTP job independently completed in **1.413 seconds** and crossed the same cookie/action-token boundary as the UI.
- The exact isolated Linux source passed **163 tests in 3.82 seconds** after exposing it through the same installed-runtime import boundary. Windows also passed all **163 tests**, Python compilation, bundled-Node syntax, Git whitespace, and the Release WinUI build with **0 warnings and 0 errors**. A changed-file credential scan found no secret-like values, and `.env` remained untracked. Audit artifacts remain under `/home/truenas_admin/radio-archive-vulkan-test/app-historical-validation` and `proof-historical-validation`.

### Crash diagnosis and durable native recovery

- Recovered the actual July 15–16 processing metadata: Whisper `turbo` through faster-whisper on `cuda:0`/float16, pyannote diarization on CUDA, and completed continuous-file speaker labeling. July 15 retained 968 transcript segments and 5,932 speaker turns; July 16 retained 854 segments and 5,215 turns.
- Isolated three native crashes to `HardwareProfile_SelectionChanged → UpdateSetupSummary → SelectedComboValue` firing from XAML construction before every control existed. The window now starts with settings loading enabled and combo access is null-safe.
- Migrated native settings from the parent-process-redirected package cache to `AppData\Local\Broadcastify Desktop\settings.json`. Settings are atomically write-through/autosaved, flushed before long operations, and retain the last area profile plus review feed/date. Activity and exception logs append under the same stable user directory instead of overwriting one temporary startup file.
- A real rebuilt Release startup probe stayed alive for ten seconds on the formerly failing path, produced no stable crash log, and restored `Example City 12345`, feed `90001`, and July 16. WinUI Release built with zero warnings/errors.
- Distinguished the machine failure from the app crash. Windows recorded bugcheck `0x154 UNEXPECTED_STORE_EXCEPTION` at the unexpected shutdown and failed dump creation; the user observed the NVMe device required a full power cycle to return. Current P44 Pro SMART/firmware are healthy/current, while the board remains on launch BIOS F1. The evidence supports a storage/PCIe path investigation but cannot name a driver without a dump; B-008 records the safe next steps.

### Example Township evidence recovery and single-feed ranking

- Audited retained segment evidence and the continuous-audio manifest. Feed 90001 reported that Example Township Police had a squad car stolen at approximately **18:22:29**, said it sounded as though it had wrecked around **18:23:26**, and reported it located unoccupied at Bruch/Garfield at **18:26:29** on July 16. A later drone search contains an uncertain ASR place name and is not asserted as the same event.
- Existing incident I717 correctly cites only the recovery line but was hidden by the P3 default and omitted from the area brief at score 39. Daily Review now searches title/summary/type/location/time across all priorities whenever text is entered and shows a redacted cited-radio quote.
- Generated the exact retained recovery clip and a separate **385.2-second surrounding-context clip** from 18:21:21 through 18:27:46. The latter reaches the initial theft transmission but is labeled context—not cited evidence—and neither operation contacted Broadcastify. The context artifact SHA-256 is `cd4dd07455eaaaffd166be02ecb1c3794022102bb92c97f3cc35800c88715437`.
- Future vehicle-theft extraction and existing area scoring now enforce a P3 editorial floor and impact bonus; cross-feed corroboration remains helpful but is never required. The evidence-v7 July 15–16 `Example City 12345` rebuild used only retained SQLite/audio/model caches and now ranks **27 leads from 64 incidents**, including Example Township at score 55/P3 with its exact quote and clip. Coverage is deterministically rendered as 2/2 feed-days and 1/1 selected feeds so a model cannot alter it or treat ZIP 12345 as an incident geofence.
- Expanded public quote redaction for direct address, `check for <name>`, and a single name immediately following a known location. Synthetic regressions cover each form; original ASR remains internal.

### Model-window analysis checkpoints

- Added an additive SQLite `analysis_window_checkpoints` table keyed by day, transcript hash, provider/model identity, prompt version, window index, and content fingerprint. Each validated window—including a supported empty result—is committed before the next model call.
- A changed transcript invalidates its checkpoints; explicit force-analysis clears matching checkpoints; a normal retry reuses completed windows, deduplicates their retained incidents with newly completed windows, and only then atomically replaces the day result.
- A two-window regression deliberately raises during window 2. The retry reuses the first saved window and makes only the missing model call, then persists both supported incidents.
- Complete suite: **159 tests passed**. Python compilation, Git whitespace checks, and the WinUI Release build pass.

## July 17, 2026

### Durable feed identity and legacy Library repair

- Added a bounded `feed_name` to the shared archive-job contract and carried it through native WinUI, the cross-platform Web UI, and each per-feed regional queue item. The worker persists selected identity before acquisition, and completed combination records it in the versioned audio manifest.
- Current cached manifests can now gain a missing or corrected feed name through the atomic manifest writer without invoking FFmpeg or touching source audio. Local Library resolves identity from the persistent catalog first, then the retained manifest, and only then falls back to `Feed <id>`.
- Used the website's feed-directory search—not its archive listing or media endpoint—to refresh legacy feed `90003` as `Example Regional Public Safety.`. A direct offline Library scan then resolved all five retained July 3–7 days by name.
- Refreshed the already-running native Library and verified the repaired name on the July 7 quota-interrupted day and the four retained combined/transcribed/diarized days. The viewer still distinguishes **Resume archive download** from local-only **Re-run evidence analysis**, and the rendered 1840×1335 window remains readable without changing the selected Example City/Example County data.
- Focused persistence/UI suite: **34 passed**. Complete suite: **150 passed in 8.37 seconds**. Python compilation, bundled-Node JavaScript syntax, Git whitespace, and an isolated WinUI Release build passed with **0 warnings and 0 errors**; the alternate build output kept the inspected native executable open.

### Example City inline quote-safety follow-up

- Reopened the current July 15–16 `Example City 12345` digest read-only and audited all 25 public evidence packages before reporting them. The audit found one additional two-token private-person name between a known numeric location and a continuing dispatch clause; the original source ASR remains internal.
- Extended location-adjacent quote redaction to cover both sentence-final names and the inline `location, First Last, for/regarding…` form. Added a focused regression with synthetic names and advanced the aggregate contract to `police-radio-area-stories-v6-evidence-v9`, so older saved briefs are hidden until rebuilt.
- Regenerated the brief from the 64 retained current-version incidents with cached local Gemma in **20.3 seconds**. No feed lookup, authentication, archive listing, media download, combination, ASR, diarization, or incident reanalysis ran.
- The persisted result remains **2/2 feed-days, 25 ranked leads, 25 source references, and 25 playable exact clips**. Public quote redaction increased from five to six; the newly matched package retains its location, alarm context, timestamp, diarization provenance, and clip while replacing only the private name.
- Refreshed the already-running native Area Watch through its named accessibility control, selected the affected alarm lead, and verified the v6 `[private person]` quote, current coverage line, exact source record, and responsive window in the rendered UI. The app remains open on that corrected evidence package.
- Focused area-watch suite: **7 passed**. Complete suite: **149 passed in 6.39 seconds**. Python compilation, Git whitespace, and a changed-file credential-pattern scan passed.

### Linux full-model visual and accessibility validation

- Cloned exact pushed commit `historical-validation4db7f9aa038a0911490f7f366d7af0907` into a fresh user-owned TrueNAS/Linux source tree and kept the earlier full-model fixture intact. The clean checkout passed the complete suite: **148 passed in 2.84 seconds**.
- Refreshed the retained 30-second full-model fixture under the current evidence-v9 contract without an archive request. The resumable Web job reused its combined audio, transcript, and CPU speaker labels; cached Gemma 3 1B `Q4_K_M` ran on the AMD Radeon 890M through Vulkan at roughly 55 generated tokens/second, FastEmbed reused the persisted passage index, and the day returned to 100% / **Ready to review** with the correct zero-incident summary.
- The NAS user dataset is intentionally `noexec`, so ONNX Runtime could not map from a private user-site install. Copying only the isolated QA runtime to executable `/var/tmp` restored FastEmbed without installing a host package or changing a system service, storage setting, group, or archive.
- Served the exact source on `127.0.0.1:18767` and ran Chromium 148/Selenium on the same host in a read-only, capability-dropped, `no-new-privileges`, numeric-user container. The archive service never bound to the LAN; SSH forwarding was unavailable and was not worked around by widening the listener.
- Exercised Local Library, Review & Ask, Area Watch, Settings, the mobile navigation drawer, retained MP3 byte-range playback, and keyboard traversal at 1365×900 and 390×844. Desktop document/client width was **1350/1350** and mobile was **375/375**; every view had no horizontal overflow, duplicate IDs, unnamed visible buttons, or unlabeled visible controls.
- Direct screenshot review found one harness artifact where the first phone capture landed during the 200 ms drawer transition. Waiting for the computed sidebar position produced a clean steady-state Review view; no product CSS change was needed.
- The first run exposed one real browser defect: Chromium requested an absent `/favicon.ico`, producing the only SEVERE console entry. Commit `historical-validation` packages an SVG favicon and a serving/content-type regression test. The final matrix reported **zero failures and no warning/error/severe browser entries**. Four DEBUG-only Chromium recommendations remain for the intentionally session-only Hugging Face/API-key password fields, which are not website login forms.
- Stopped the verified loopback QA PID and removed only the exact isolated Gemma container/image after validation. Health closed as expected; the exact source checkout, browser report/screenshots, reanalysis result/logs, private QA runtime, retained media, transcript, and SQLite fixture remain preserved for audit.

### Regional story master/detail review

- Replaced the Web Area Watch stack of full story cards with a bounded two-column review workspace: all 25 ranked leads remain searchable by scrolling the index while only the selected story renders its complete source package. The generated assignment brief stays separately collapsible.
- Added the matching native WinUI workspace with a 330-pixel ranked list, independent selected-story scroller, preserved selection across refreshes, and explicit empty/stale states. Changing stories stops old playback before rebinding the quote, provenance, exact-clip actions, and media player.
- The Web phone layout changes to one column at 560 pixels, caps the ranked index at 300 pixels, scrolls the selected detail into view, and offers a keyboard-focus-preserving **Ranked leads** return action.
- Live July 15–16 `Example City 12345` QA verified **25 leads, one active selection, open evidence, and exact media URLs** at 1365×900 and 390×844. Selecting the West Parker report updated its headline and redacted transcript evidence; the phone return action restored focus to lead 2. Both sizes had exact document/client width and the browser console had no errors.
- The rebuilt native Release app opened the same retained brief at 1228×894, selected the first lead automatically, then switched to the West Parker report while retaining its redacted quote and playable/exportable exact clip. The app remains open on that evidence pane for inspection.
- Added static-asset regression checks for the selected-story state, ranked-item data contract, master/detail CSS, and cache version 13. Full Python suite: **148 passed in 8.75 seconds**; Python compilation, bundled-Node JavaScript syntax, Git whitespace, and the WinUI Release build passed with **0 warnings and 0 errors**.

### Viewer integrity and area-selection UX

- Added prompt-version integrity across the persistent viewer contract. Days analyzed before `police-radio-events-v9` now remain 80% complete with **Analysis update available**, retain their combined recording/transcript/diarization, and offer local-only reanalysis instead of review. Current July 15–16 Example City days remain 100% reviewable.
- Daily report, date-range Q&A, weekly summary, and area-story queries now filter incident evidence to the current prompt. Weekly and area coverage explicitly list retained dates/feed-days that need reanalysis. Saved weekly/area summaries also require the current aggregate prompt and incident prompt before either UI will display their claims.
- Native and Web Area Watch preserve the explicitly saved feed set through discovery and public-safety filtering. **Nearest 3** and **Clear** provide intentional bulk actions; save labels/status report the actual selected count rather than silently selecting every discovered feed.
- The responsive Web navigation now includes an outside scrim with an accessible close action. Static cache versions were advanced so the updated behavior reaches an already-running local service after refresh.
- Live QA against the real library verified 31 current incidents for July 16, 33 for July 15, explicit reanalysis states for July 11–12, and hidden pre-v9 Example City area claims in both the native 1240×900 app and 390×844 Web UI. The loopback service was restarted to load the new Python prompt gates; no archive request was made.
- Regenerated the one-feed `Example City 12345` brief for July 15–16 under `police-radio-area-stories-v5-evidence-v9`: **2/2 feed-days, 64 current incidents, 25 ranked leads, 25 source references, and 25 exact clips**. Native Story Leads exposes each quote, provenance record, player, and export action. The Web viewer exposes the same packages through archive-relative media URLs, never serializes a local path, and has no horizontal overflow at 390×844.
- A privacy audit caught a two-token private-person name immediately following a known incident location in an earlier area quote. The current quote sanitizer redacts that context, invalidates the older area prompt, and regenerated the brief with five redacted quotes and the known name absent.
- Regenerated the seven-day brief ending July 16 from current evidence only: July 15–16 are available with **64 incidents / 14 priority 4–5 records**; July 11–12 are explicitly labeled analysis updates, while July 10 and July 13–14 are missing. No unavailable day is described as quiet.
- Full Python suite: **148 passed in 12.63 seconds** using an isolated selected-runtime import check. JavaScript syntax, Python compilation, live native/Web checks, and the WinUI Release build passed with **0 warnings and 0 errors**.

### retained two-day refresh and evidence hardening

- Refreshed quota-free Example City discovery by city and ZIP/radius. The relevant live public-safety catalog included Example City Public Safety (`90001`), Example City City Fire Dispatch (`90004`), Example County Fire Digital (`11466`), Example County Sheriff (`20301`), Logan-Trivoli Fire/Rescue (`18278`), and the much broader Example State State Police Troops 1–5 feed (`21049`). The closest police feed was processed first rather than spending the unknown archive allowance across every match.
- Acquired the two latest complete feed `90001` days: **48/48 July 16 blocks** and **49/49 July 15 blocks**, or **97 successful new archive media responses with no 429**. July 16 combined to 85,984.914 seconds and July 15 to 87,784.934 seconds. The extra July 15 listing block reflects the feed archive boundary/overlap rather than a failed duplicate.
- The CUDA reference workflow retained **854 transcript segments / 5,685 words / 5,215 diarized turns / 3 acoustic clusters** for July 16 and **968 segments / 6,036 words / 5,932 turns / 4 clusters** for July 15. Both days have zero unlabeled turns.
- A manual evidence audit caught a serious model error before reporting: an invented “stolen squad car in Example Township” card cited only an arrest/transport line and an unrelated domestic call. Structured extraction now requires at least two meaningful claim anchors in the exact cited ASR, rejects evidence bundles separated by more than ten minutes, and requires critical concepts such as shots, weapons, theft, assault, threats, fire, pursuit, collision, overdose, welfare, and trespass to occur in those exact citations.
- Deterministic evidence normalization now corrects clear category contradictions, caps routine priorities, downgrades P5 medical cards without imminent-life evidence such as not-breathing/CPR/unconscious language, strips unsupported outcome sentences and legal-intent labels, removes ASR-fused location suffixes and plate-like pseudo-locations, and deduplicates contained citation sets.
- Public incident fields and displayed quotes now redact obvious phone/email/long identifiers, DOB forms, and private-person names found in strong radio contexts. Exact source ASR stays internal for audit. Daily briefs reject unknown incident IDs, impossible counts, and unsupported “confirmed/identified/resolved” outcome wording.
- Local llama.cpp JSON extraction now uses temperature zero and a fixed seed. The frozen v9 run retained **31 supported July 16 incidents (6 P4–5)** and **33 supported July 15 incidents (8 P4–5)**. A post-persistence audit found **zero unsupported cards and zero detected names in public fields**.
- Prepared 12 independently playable, SHA-256-hashed evidence clips for the strongest or most reviewable reports. Each clip is 21.64–30.22 seconds and is cut from the selected evidence segment, including a long-span group-assault card whose correct clip begins at the later cited traffic rather than the incident envelope start.
- Full Python suite: **142 passed in 5.10 seconds**; focused analysis/area-watch suite: **28 passed**; Python compile and Git whitespace checks passed. No Example City reanalysis or clip export made an archive request.

### Joined OpenVINO and Windows ML parity

- Resumed the prepared OpenVINO fixture that had already persisted CPU diarization and OpenVINO CPU ASR before the removed Gemma quant stopped analysis. Current code resolved the exact 7.4 GB legacy quant from the older local Hub snapshot, reused every completed audio stage, and reached analysis, one passage/embedding, one daily summary, and `Ready to review` in a ten-second protected loopback Web job.
- Repeated OpenVINO from a fresh fixture containing only the retained 60-second MP3, exact clean source commit `historical-validation`, cached models, and Hugging Face/Transformers offline flags. The single protected Web job completed all **18** lifecycle events in **31 seconds**: CPU pyannote, OpenVINO CPU ASR, local Gemma, embedding persistence, deterministic zero-incident summary, and `local_complete`.
- The OpenVINO transcript records **14 ASR segments, 278 word records, 21 speaker turns, two anonymous clusters, one unmatched ASR segment**, `OpenVINO CPU`, requested device `CPU`, and no fallback. SQLite records one feed-day, 14 transcript segments, one passage/embedding, and one daily summary.
- A fresh Windows ML fixture initially stopped before processing because the headless harness did not inherit the packaged desktop's helper location. Supplying the documented `WINDOWS_ML_HELPER_PATH` to the exact Release helper completed the protected job without changing the runtime/model.
- The fresh Windows ML job completed all **22** lifecycle events in **32 seconds**: CPU pyannote, three persistent-helper ASR chunks, cached Gemma, embedding persistence, deterministic zero-incident summary, and `local_complete`. Metadata records `Windows ML / ONNX Runtime GenAI CPU`, **3 ASR segments, 21 turns, two clusters, zero unlabeled segments**, 28-second chunks, and no fallback.
- An immediate Windows ML resume completed in about one second of job time. It reported `operation=reused`, skipped ASR/diarization, reused the saved zero-incident analysis, indexed zero new passages, and remained 100% / `Ready to review`.
- Both Windows runs used offline model flags and retained local inputs; neither made a Broadcastify archive request. This closes functional joined-stage parity for OpenVINO and Windows ML. B-001 remains open for genuine Windows ML GPU acceleration, B-003 for portable diarization acceleration, and B-004 for full-day CPU timing.

## July 16, 2026

### Local Gemma model continuity

- A fresh joined OpenVINO job reached local analysis after completing and persisting OpenVINO CPU transcription plus CPU diarization, then exposed an upstream compatibility break: the Gemma 4 12B GGUF repository had removed the configured `Q4_K_M` file and now offers `Q4_0`, `Q8_0`, and BF16.
- Changed the clean-install default to `ggml-org/gemma-4-12B-it-GGUF:Q4_0` in Python, WinUI, Web UI, and documentation.
- Added deterministic Hugging Face cache discovery across configured/default Hub roots and older snapshots. An existing `Q4_K_M` setting retains that exact cached model; without it, the setting normalizes to `Q4_0`. Explicit local `.gguf` paths launch directly.
- Managed llama.cpp now uses `--model` for resolved local files, `--hf-repo` for uncached repository selectors, and a stable `--alias` that the analysis client also uses. Provider cache identity follows the effective model instead of attributing new results to a removed selector.
- The reference machine resolved its existing 7,381,382,048-byte `Q4_K_M` file from the older cached snapshot without network access. Installed llama-server help confirms the model-path, Hub-repository, and API-alias flags used by the launcher.
- Regression coverage brings the complete Python suite to **133 passed in 5.43 seconds**. Python compile validation passed; WinUI Release build: **0 warnings, 0 errors**.

### Managed Linux Web launcher

- Added the wheel-packaged `radio-archive-service` entry point with install, start, restart, status, stop, owner-log, unit-preview, and conservative uninstall commands for a per-user systemd service.
- The service persists only absolute non-secret paths and a fixed `127.0.0.1` endpoint in owner-only JSON. It references a separate mode-0600, comment-only `.env` template; credential values never enter the unit or service JSON.
- The generated unit supervises failure restart, uses SIGINT plus a bounded stop window, an owner-only umask, `NoNewPrivileges`, private temporary storage, and read-only system paths without blocking outbound website/model traffic. The shared Web cookie/action-token/origin and archive quota boundaries remain unchanged.
- The default working/archive layout follows XDG user directories. Custom absolute paths can adopt an existing library, uninstall preserves all data/private settings, and the recorded venv Python path is not dereferenced to a dependency-free system interpreter.
- Refactored `broadcastify-web` into a reusable runner and added explicit `--working-dir`, keeping service cookies, `.env` discovery, and worker children on the selected private path.
- Ten service/parser tests, including a real selected-venv import check and bounded owner-log viewing, bring the complete local suite to **129 passed in 5.89 seconds**. The current `broadcastify_cli-0.4.0-py3-none-any.whl` is 182,581 bytes with SHA-256 `5234c6bd92321772902b93bc974ab3778f0055928a12401d45376efcc66f1cda`; archive inspection confirms the service module, all Web static assets, and the `radio-archive-service` console entry. Its exact real-Linux lifecycle validation is recorded below.

### Real Linux managed-service validation

- Transferred the exact 182,581-byte 0.4.0 wheel from pushed commit `historical-validation` to a fresh user-owned TrueNAS/Linux test directory and verified SHA-256 `5234c6bd92321772902b93bc974ab3778f0055928a12401d45376efcc66f1cda` before installing it with `--no-index --no-deps`. The host remained package- and archive-request-free.
- The first pushed unit exposed real systemd syntax rejecting ExecStart-style quotes around `WorkingDirectory`; commit `historical-validation` added directive-specific path escaping. That unit passed `systemd-analyze --user verify`, but the appliance account could not read the system journal, so commit `historical-validation` moved service output to a portable owner log and made `logs` tail it directly.
- The final generated unit SHA-256 was `a8d48c9b42dfbf0a0d1052d5779fba24b24e9cef40f280b64efa05ab675d44e7`. systemd 252 accepted its absolute working directory, exact wheel runner, failure restart, control-group SIGINT shutdown, 0077 umask, `NoNewPrivileges`, private temporary directory, read-only system paths, and append-only owner log.
- The installed service reported active/ready and listened only on `127.0.0.1:18766`. `/health` returned the exact loopback-only response; `/` set `HttpOnly`/`SameSite=Strict`, carried a fresh action token, and authenticated `/api/bootstrap`, which reported Linux and `loopback_only=true`.
- `service.json`, the comment-only private `.env`, installed unit, and service log were all mode **0600**. `radio-archive-service logs --lines 8` returned the startup and local HTTP lines even though this account cannot read the NAS journal.
- A deliberate SIGKILL changed the main PID and returned the health endpoint through `Restart=on-failure`; the owner log retained the restart. In the same user-manager session, `stop` reached inactive with no listener. `uninstall` removed/disabled the unit to `LoadState=not-found` while preserving config, `.env`, log, and archive directory.
- Exact wheel/config/unit/log evidence remains under `/home/truenas_admin/radio-archive-vulkan-test/service-historical-validation`; its no-exec-home test wrapper remains under `/var/tmp`. No Broadcastify request, model download, host package, storage setting, group, or system service was changed.

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
