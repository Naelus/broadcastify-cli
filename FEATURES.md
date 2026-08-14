# Feature inventory

Last updated: August 14, 2026

Status labels: **Validated**, **Implemented**, **Evaluated**, **Foundation**, or
**Planned**. Detailed behavior and retained validation evidence belong in
[`docs/`](docs/README.md). Actionable work is tracked in
[GitHub Issues](https://github.com/Naelus/broadcastify-cli/issues).

## Acquisition and discovery

| Capability | Status | Notes |
|---|---|---|
| Website-login feed search | Validated | Agency/place/ZIP search through Broadcastify website endpoints, not the official API. |
| Radius and ordered-ZIP profiles | Validated | Approximate ZCTA radius expansion, county-directory parsing, public-safety filter, explicit feed review. |
| Cache-aware archive acquisition | Validated | One-to-one provider-ID/timeline index with legacy collapse detection and targeted repair, duplicate-listing suppression, sequential pacing, newest/previous priority, current-day refresh, process-safe 240-of-250 rolling ledger, and immediate 429 stop. |
| Source-specific progress | Validated | Each ready block reports local cache versus Broadcastify download; LAN copies name the exact peer-sourced block. |
| Nearest-first area queue | Validated | Persists stop points, skips complete feeds, resumes partial work, stops lower priorities on quota exhaustion. |
| Per-feed daily schedules | Validated | Explicitly selected feeds retain their processing profile, recent-day lookback, optional historical catch-up boundary in either self-clearing or recurring full-range mode, and a non-secret account policy. Written-authorization mode uses acquisition-only turns to rotate immediately and sequentially across isolated account credentials, cookies, and 240+10 rolling ledgers before long local model stages; otherwise schedules remain single-account. Visible default-on Windows startup keeps the desktop scheduler available, interrupted/incomplete runs resume retained work, quota deferrals resume at the next known rolling slot, and every scheduler binds jobs to its currently selected Library so a stale path cannot re-fetch the same archive IDs into another root. |
| Trusted-LAN work pool | Validated | One globally sequential upstream archive stream across feeds, days, machines, and authorized accounts; per-account quota results allow safe pool rotation. Followed-feed reconciliation hash-verifies raw blocks plus model-fingerprint-matched combined audio, timeline manifests, transcript JSON, and rendered text. Renewable model/day claims prevent duplicate inference while different days may run on Windows and TrueNAS in parallel. Credentials, cookies, models, analysis databases, incidents, and summaries are never shared. |

See [archive acquisition](docs/features/archive-acquisition.md),
[rate limits](docs/rate-limits.md), and [LAN sharing](docs/lan-archive-sync.md).

## Processing and persistence

| Capability | Status | Notes |
|---|---|---|
| Continuous daily audio | Validated | Combination precedes ASR/diarization; overlap-aware timeline manifest and atomic refresh. |
| faster-whisper CUDA/CPU | Validated | CUDA is the Windows reference; CPU is available as a slower fallback. |
| whisper.cpp Vulkan/Metal/CPU | Validated/Implemented | Vulkan is real-machine validated; Metal is implemented but awaits a real Mac run. |
| OpenVINO Whisper | Validated | Real decode with honest recorded CPU fallback when an accelerator rejects the model. |
| Windows ML Whisper | Validated | Packaged helper and real CPU Base decode; GPU validation is scoped to the future packaged app in [#4](https://github.com/Naelus/broadcastify-cli/issues/4). |
| Qwen3-ASR CPU preview | Implemented | Fast bounded-region preview with no fabricated word timing; retained full-day quality evaluation is tracked in [#2](https://github.com/Naelus/broadcastify-cli/issues/2). |
| Community-1 diarization | Validated | Accuracy default on CUDA/CPU; can improve an existing transcript without repeating ASR. |
| Portable CPU speaker preview | Validated | Checksum-managed sherpa-onnx models, bounded chunks, atomic resume, and chunk-scoped anonymous labels; human scoring and supported accelerator research are tracked in [#3](https://github.com/Naelus/broadcastify-cli/issues/3). |
| Persistent evidence store | Validated | SQLite/FTS, embeddings, incidents, summaries, Q&A, profiles/queues, prompt/source versioning and checkpoints; every review/query/brief/clip consumer enforces the retained revision. |

See [audio processing](docs/features/audio-processing.md),
[storage and resume](docs/reference/storage-and-resume.md), and
[hardware backends](docs/hardware-backends.md).

## Analysis and evidence

| Capability | Status | Notes |
|---|---|---|
| Quantized local analysis | Validated | Gemma 4 12B `Q4_0` through llama.cpp; Vulkan/CPU/auto-offload, payload-bounded windows, and per-window resume. |
| Semantic retrieval | Validated | BGE Small/FastEmbed passage index persisted in SQLite. |
| Evidence-gated incidents | Validated | Exact citations, critical-concept support, bounded gaps, outcome rejection, category/priority correction. |
| Cited-name private review | Validated | Spoken names preserved; inferred/normalized identities forbidden; high-risk identifiers masked. |
| Daily and weekly briefs | Validated | Current-version incidents only, explicit coverage gaps and grounding checks. |
| Range, monthly, and whole-feed Q&A | Validated | Friendly-name feed picker, visibly labeled responsive question-scope controls, one-click calendar-month or entire-downloaded-span scope, local downloaded/question-ready coverage preview, compact gap enforcement, date-diverse retrieved evidence, deterministic category/location/weekday/time-block pattern records with required E/I/P citations, backend-owned cited event dates/times (or an explicit archive offset when clock time is unavailable), bounded multi-turn follow-ups, visible month/hotspot starters, and local Q&A audit persistence; New chat returns to scope selection instead of scrolling it away, and quantized Gemma 4 12B remains the default provider. |
| Evidence clips and export | Validated | Exact hashed citation clip plus separately labeled surrounding context. |
| External provider contracts | Validated | OpenAI Responses, compatible `/v1`, and saved-login Codex behind explicit transcript-sharing consent. |

See [evidence analysis](docs/features/evidence-analysis.md),
[privacy decisions](docs/decisions/evidence-and-privacy.md), and
[model providers](docs/model-providers.md).

## User experience and deployment

| Capability | Status | Notes |
|---|---|---|
| Native WinUI 3 shell | Validated | Task navigation for Library, New Archive, Review & Ask, Area Watch, Settings, and About & Support; the support surface shows the installed version/runtime and opens local data or copies credential-free diagnostic context. Acquisition/transcription runs as a background pipeline while non-conflicting review/chat/library actions remain available, same-feed file handles are guarded, model analysis is serialized, and bursty worker output remains off the UI thread and in the rotating diagnostic log. A fixed 48-pixel app-owned chrome row supplies its own drag surface, pin, minimize, maximize/restore, close, and docked-sidebar controls instead of depending on framework title-bar visibility. Pinning reserves either Windows desktop edge through an AppBar, retains the target monitor plus independent floating/maximized placement, uses a captured inner-edge resize grip, removes the floating non-client border and rounded corners while pinned, keeps the chrome row visible, and switches every major surface to a stacked compact layout. The build-owned rendered E2E gate validates nine window sizes around every responsive boundary; all named page, list, transcript, activity-log, dialog, and flyout scroll surfaces; active/inactive light/dark dock borders; startup, left/right, resize/reclaim, shell restart, available-monitor, DPI, visible and hittable chrome controls after normal/maximized unpin, custom minimize/maximize/restore/close actions, and abrupt-process cleanup without desktop-control fallbacks or live-provider use. |
| Local Library | Validated | Five-stage timeline, playback, transcript preview, stale-result detection, retained-versus-temporary storage, schedule-aware feed backlog/last-known source coverage, always-available existing-feed catch-up entry even while another sequential pipeline is active, durable catch-up from one prior date through current that queues only absent or unfinished days, survives restart/quota/cancellation, uses the authorized account pool one profile at a time, can be promoted to a recurring daily schedule, and rejoins global Resume, guarded retrying whole-feed deletion with media players detached through the directory rename and nonmodal completion feedback, shell-based folder/transcript launching, selector-driven quota-safe Resume/prioritize, safe orphan cleanup, and exact per-day resume/upgrade/review actions. |
| Area Watch | Validated | Profile discovery, queue, coverage-aware ranked leads, selected evidence, clips and exports. |
| Neighborhood subscription delivery | Foundation | Story eligibility and neighborhood/topic tags exist; opt-in delivery is explicitly low priority in [#5](https://github.com/Naelus/broadcastify-cli/issues/5). |
| Cross-platform browser UI | Validated | Windows/Linux desktop and mobile validation; loopback default and explicit trusted-LAN mode; visible About/build identity, existing-feed missing-day catch-up, recurring schedules, month/whole-feed questions, and automatic account-pool status and selection. |
| Encrypted local credentials | Validated | One-click Windows Credential Locker UI; browser/server DPAPI or AES-GCM store; named authorized account profiles keep credentials and sessions isolated, and only short non-secret previews return to either UI. |
| First-run/profile verifier | Validated | Real generated-input execution across ASR, speakers, and analysis without archive quota. |
| Linux user service | Validated | Install/start/stop/status/log/restart and data-preserving uninstall. |
| TrueNAS App | Validated | Persistent host-path data, trusted-LAN UI, AMD Vulkan, unprivileged/read-only container boundary, and the same named-account/catch-up/schedule controls as the desktop workflow. |
| Verified Windows publish | Validated | WinUI resources, namespaced Windows ML helper, normal/private environment isolation. |
| Standalone Windows installer | Validated | Per-user Inno Setup package with self-contained .NET app/helper, embedded Python, FFmpeg, Windows ML and portable inference runtimes; every native/package build requires committed source and the full offline product regression gate before compilation, embeds that commit in ProductVersion and the manifest, public stages reject PDBs/private state, install/upgrade/uninstall preserve application data, interactive setup discloses default-on login startup, and silent setup requires explicit startup/launch flags. |
| Packaged CUDA runtime manager | Validated | Explicit isolated Python/CUDA install from a fully hashed Windows lock; source/license/size/path disclosure, progress, cancellation, retained cache, safe resume, atomic promotion, app-update reuse, and clean staged-package ASR/Community-1 execution are proven. |
| Additional accelerator add-ons | Foundation | OpenVINO/Vulkan packaged dependency lifecycles and release signing remain future work; their source/runtime paths continue to work when explicitly configured. |

See [Library and Review](docs/features/library-and-review.md),
[Area Watch](docs/features/area-watch.md),
[UI/platform decisions](docs/decisions/ui-and-platforms.md), and the
[documentation index](docs/README.md).

## Evaluated alternatives

| Candidate | State | Decision |
|---|---|---|
| Qwen3 ForcedAligner | Evaluated | Useful possible seek aid; wrong supplied words still receive timestamps, so it cannot validate quotes. |
| Parakeet TDT 0.6B | Evaluated | Fast with token timing but weak on sparse-radio segmentation/critical wording. |
| VibeVoice-ASR 8B | Evaluated | Promising unified speaker/time/text output, but high VRAM and retained hallucination/address/speaker errors prevent promotion. |

The rationale and current measurements are in
[model-stack decisions](docs/decisions/model-stack.md) and
[retained-corpus validation](docs/validation/retained-corpus.md).
