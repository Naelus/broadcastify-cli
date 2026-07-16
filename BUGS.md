# Known bugs and gaps

Last updated: July 16, 2026

Use this file for reproducible defects and concrete blockers, not the general roadmap. Remove an entry only after its fix and verification are recorded in `PROGRESS.md`.

## Active

### B-001 — Windows ML DML Whisper model fails during generation

- **Severity:** High for Windows ML parity; no impact on the default CUDA path.
- **Observed:** ONNX Runtime GenAI 0.14.1 with official builder output for `openai/whisper-tiny` fails on DML/WinML. With `past_present_share_buffer=false`, DML graph capture rejects the generator; changing it to true reaches a `DmlFusedNode` invalid-key error.
- **Control:** The UI profile remains not ready. The C# helper and a CPU FP32 model successfully transcribed a real 23-second radio clip, proving the adapter itself works.
- **Next:** Test a compatible published DML model or upstream fix; add self-test/model discovery before enabling the profile.

### B-002 — Vulkan ASR has not completed a real decode on AMD hardware

- **Severity:** High for Vulkan parity.
- **Observed:** The whisper.cpp adapter, backend inspection, and model lookup are implemented, but no local `whisper-cli` Vulkan build/model is installed on the Windows reference machine.
- **Next:** Inspect the permitted TrueNAS AMD host, build/install a Vulkan-enabled whisper.cpp non-destructively, and run the retained short evidence clip or a synthetic fixture.

### B-003 — Portable diarization is CPU-only outside CUDA

- **Severity:** Medium; functionally correct but potentially slow.
- **Observed:** pyannote's supported app path is PyTorch CUDA or CPU. Vulkan/OpenVINO/Windows ML profiles deliberately fall back to CPU.
- **Next:** Benchmark CPU on full-day audio and investigate supported Intel XPU/other backends without weakening the reliable fallback.

### B-004 — Full CPU-only end-to-end timing is not recorded

- **Severity:** Medium.
- **Observed:** CPU components and profile exist, but the complete multi-hour pipeline has not been benchmarked on the retained corpus.
- **Next:** Use a bounded representative slice first, then a full day if practical; record ASR, diarization, and analysis time separately.

### B-005 — Cross-platform UI/service is not implemented

- **Severity:** Medium for product portability; no impact on native Windows use.
- **Observed:** The backend is Python and largely portable, but the current polished shell is WinUI-only and worker launch/settings assume a repository checkout.
- **Next:** Extract a localhost service contract and build a browser UI that mirrors Library/Review/Area/Settings.

### B-006 — Feed display names are missing for some retained legacy days

- **Severity:** Low.
- **Observed:** Local Library shows `Feed 90003` when that feed was downloaded before its search result entered the persistent catalog.
- **Next:** Backfill feed metadata from saved manifests or a quota-free directory lookup.

### B-007 — Archive quota size/reset schedule remains inferred

- **Severity:** Operational.
- **Observed:** Broadcastify publishes no numeric archive-download quota or reset timestamp. Measured runs suggest a budget near 200 successful requests and a reset no later than the next day, but this is not proven.
- **Control:** Sequential pacing, exact cache reuse, and immediate stop on explicit exhaustion.
- **Next:** Record the July 16 reset run and ask Broadcastify support for authoritative details.

## Recently fixed

### F-001 — Diarization restarted for every downloaded block

Combined audio is now created before transcription/diarization, preserving one continuous timeline.

### F-002 — Failed download futures appeared as completed progress

Progress counts only successful cached or saved archives; explicit quota exhaustion cancels queued work.

### F-003 — Existing transcript forced a second Whisper pass to add speakers

The library now runs diarization-only continuation and atomically updates the transcript/text output.

### F-004 — A `diarization_requested` flag could look like completed labeling

Library detection now requires completion evidence and new transcripts write `diarization_completed=true` only after labels are produced.
