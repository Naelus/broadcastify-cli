# Project goal

Last updated: July 16, 2026

Build a dependable, evidence-first radio archive intelligence application that is pleasant for an ordinary user on Windows and has a credible path to the same core workflow on Linux and macOS.

The tested Windows experience remains the reference behavior. A user should be able to discover relevant feeds, acquire a quota-safe date range, see retained work in a local library, resume only the missing processing stages, inspect transcripts and speaker labels, review evidence-backed incidents and clips, and ask for daily, weekly, or regional summaries without understanding the pipeline internals.

## Current acceptance targets

1. **Pipeline parity:** CUDA, CPU, Vulkan, OpenVINO, and Windows ML profiles must each report stage-by-stage readiness honestly. A profile is called ready only after real transcription, diarization/fallback, and local-analysis checks appropriate to that profile pass.
2. **Resumability:** downloads, combination, transcription, diarization, embeddings, incident extraction, and summaries persist independently. A retry starts at the first missing or invalid stage.
3. **Evidence:** every surfaced event can be traced to timestamped transcript text and a playable/exportable local clip. Missing coverage is never described as inactivity.
4. **Quota safety:** archive acquisition is sequential by default, paced, cache-aware, responsive to `Retry-After`, and stops all further media requests after the explicit Broadcastify download-limit response.
5. **Usable UI:** primary workflows use clear navigation and master/detail views rather than one long page. Common actions fit at the reference 1240x900 window size; advanced hardware controls stay optional.
6. **Portable providers:** local engines remain the default, with explicit provider contracts for supported OpenAI-compatible APIs or an authenticated Codex CLI harness. Secrets must not be persisted in ordinary settings or committed.
7. **Cross-platform direction:** the Python service and a future browser UI should expose the same library/review experience on Windows, Linux, and macOS, while the native WinUI app remains the polished Windows shell.
8. **Reviewable delivery:** substantial sections are committed and pushed separately, with tests, real-machine evidence, limitations, and security checks recorded in `PROGRESS.md` and `BUGS.md`.

## Definition of done for the active parity phase

- CUDA reference flow remains green on the RTX 3090.
- CPU transcription, CPU diarization, and CPU LLM fallback are exercised.
- OpenVINO performs a real Whisper decode and safely falls back to CPU when an exposed accelerator rejects the model.
- Vulkan performs a real whisper.cpp decode and llama.cpp generation on suitable AMD or Intel hardware; diarization uses the documented CPU fallback.
- Windows ML performs a real ONNX Whisper decode through the packaged helper and is integrated only after the model/runtime self-test passes.
- Settings explain the selected stage backends and prevent unsupported combinations from looking ready.
- The Local Library viewer makes unfinished diarization or analysis obvious and offers the exact next action.
- The application receives another visual QA and accessibility pass after every material navigation/viewer change.

This goal is intentionally not marked complete yet.
