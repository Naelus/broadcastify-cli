# Feature inventory

Last updated: July 22, 2026

Status labels: **Validated**, **Implemented**, **Evaluated**, **Foundation**, or
**Planned**. Detailed behavior and retained validation evidence belong in
[`docs/`](docs/README.md). Actionable work is tracked in
[GitHub Issues](https://github.com/Naelus/broadcastify-cli/issues).

## Acquisition and discovery

| Capability | Status | Notes |
|---|---|---|
| Website-login feed search | Validated | Agency/place/ZIP search through Broadcastify website endpoints, not the official API. |
| Radius and ordered-ZIP profiles | Validated | Approximate ZCTA radius expansion, county-directory parsing, public-safety filter, explicit feed review. |
| Cache-aware archive acquisition | Validated | Sequential pacing, timezone identity, newest/previous priority, current-day refresh, retry/backoff, explicit quota stop. |
| Source-specific progress | Validated | Each ready block reports local cache versus Broadcastify download; LAN copies name the exact peer-sourced block. |
| Nearest-first area queue | Validated | Persists stop points, skips complete feeds, resumes partial work, stops lower priorities on quota exhaustion. |
| Trusted-LAN archive pool | Validated | One upstream producer lease; followers hash-verify raw blocks from any peer; no credentials/transcripts/analysis shared. |

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
| Persistent evidence store | Validated | SQLite/FTS, embeddings, incidents, summaries, Q&A, profiles/queues, prompt/source versioning and checkpoints. |

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
| Range Q&A | Validated | Structured plus retrieved evidence with required E/I citations. |
| Evidence clips and export | Validated | Exact hashed citation clip plus separately labeled surrounding context. |
| External provider contracts | Validated | OpenAI Responses, compatible `/v1`, and saved-login Codex behind explicit transcript-sharing consent. |

See [evidence analysis](docs/features/evidence-analysis.md),
[privacy decisions](docs/decisions/evidence-and-privacy.md), and
[model providers](docs/model-providers.md).

## User experience and deployment

| Capability | Status | Notes |
|---|---|---|
| Native WinUI 3 shell | Validated | Task navigation for Library, New Archive, Review & Ask, Area Watch, and Settings. |
| Local Library | Validated | Five-stage timeline, playback, transcript preview, exact resume/upgrade/review action. |
| Area Watch | Validated | Profile discovery, queue, coverage-aware ranked leads, selected evidence, clips and exports. |
| Neighborhood subscription delivery | Foundation | Story eligibility and neighborhood/topic tags exist; opt-in delivery is explicitly low priority in [#5](https://github.com/Naelus/broadcastify-cli/issues/5). |
| Cross-platform browser UI | Validated | Windows/Linux desktop and mobile validation; loopback default and explicit trusted-LAN mode. |
| Saved Windows login | Implemented | Windows Credential Locker plus ignored session cookie. |
| First-run/profile verifier | Validated | Real generated-input execution across ASR, speakers, and analysis without archive quota. |
| Linux user service | Validated | Install/start/stop/status/log/restart and data-preserving uninstall. |
| TrueNAS App | Validated | Persistent host-path data, trusted-LAN UI, AMD Vulkan, unprivileged/read-only container boundary. |
| Verified Windows publish | Validated | WinUI resources, namespaced Windows ML helper, normal/private environment isolation. |
| Standalone installer/model manager | Foundation | Verified publish exists; the remaining non-developer packaging and managed-model work is tracked in [#1](https://github.com/Naelus/broadcastify-cli/issues/1). |

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
