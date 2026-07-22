# Retained-corpus validation

This page summarizes representative real-radio validations. Exact implementation
history remains available through Git commits, while actionable follow-up is
tracked in [GitHub Issues](https://github.com/Naelus/broadcastify-cli/issues).

## reference feed workflow

Feed `90001` has been exercised through authenticated website acquisition,
continuous combination, transcription, speaker labeling, embeddings, incident
analysis, daily/weekly review, questions, and regional story leads.

Representative retained results include:

- complete July 11–12 acquisition with 97 blocks;
- continuous day audio and timestamped transcript persistence;
- a real July 12 Community-1 full-day run in roughly 25 minutes on RTX 3090;
- exact evidence clips and cross-day Q&A grounded in retained transcript IDs;
- July 15–20 full-day processing and review;
- versioned reanalysis from retained transcripts without another download,
  transcription, or diarization pass; and
- 12 current analyzed days rebuilt under the cited-name policy, producing 389
  incidents, two weekly briefs, and three regional digests with 90 ranked leads.

These counts are validation evidence, not a fixed product fixture or an estimate
of future feed activity.

## Example County dense-traffic recovery

Feed `90002` July 20 retained 1,829 transcript segments. Its original two-hour
analysis window tokenized to 66,558 tokens and was rejected by the 32,768-token
llama.cpp context. Payload-bounded windowing covered the same corpus in four
requests whose largest measured prompt was 24,113 tokens. Recovery reused the
existing audio, transcript, and Community-1 diarization, completed in 117.5
seconds, retained 13 supported incidents, and indexed 41 new passages without a
new Broadcastify request.

## CUDA reference

The Windows RTX 3090 profile runs faster-whisper and Community-1 on CUDA plus
quantized llama.cpp analysis. The joined profile verifier has completed all
three generated-input stages, and retained full-day processing has been
exercised separately.

## AMD Vulkan and TrueNAS

The exact application path has completed whisper.cpp Vulkan ASR, sherpa-onnx CPU
speaker preview, and llama.cpp Vulkan analysis on Radeon 890M hardware. A joined
preview profile completed in about nine seconds on the retained short fixture.

The portable speaker preview processed a 23.9-hour feed day in 1,951.971 seconds
(44.05× real time), returned 3,893 turns, covered 91.05% of Community-1 reference
speech and 93.23% of timestamped word midpoints, and duplicated no boundary
ownership. Community-1 is model output rather than human ground truth, so those
figures are not DER.

The same Web workflow and Vulkan/CPU/Vulkan stages run as a persistent TrueNAS
App on the trusted LAN with host-path persistence and the AMD render device.

## CPU, OpenVINO, and Windows ML

- A bounded all-CPU Linux Web job completed the full retained workflow in
  21.282 seconds for a 30-second fixture. Full-day all-CPU ASR/LLM timing remains
  open.
- OpenVINO completed real decodes and a protected Web workflow while recording
  honest CPU fallback when an exposed accelerator rejected the model.
- Windows ML completes real CPU Whisper decode through the packaged helper.
  Multilingual Base captured the retained low-SNR core dispatch where Tiny did
  not. Current DML/TensorRT acceleration remains gated.

## Qwen and unified-model evaluation

Qwen3-ASR preserved the core event across the retained clip set near 0.08 RTF
and processed five minutes in 7.128 seconds on AMD/Linux, but critical-word drift
and missing token timing keep it opt-in.

VibeVoice-ASR directly emitted speaker/timestamp/content records and preserved
several retained events, but used roughly 16 GiB GPU memory and also changed an
address, omitted a disposition, merged speakers, and hallucinated short noise.
It remains a research candidate.

## UI and service validation

The native Windows shell has been exercised at its 1240×900 reference size.
The full browser experience has passed desktop and 390×844 mobile navigation,
media, accessibility, and no-overflow checks on Linux. The per-user Linux service
and TrueNAS App have passed start, health, restart, graceful-stop, and
data-preserving update/uninstall boundaries.

Real macOS/Metal workflow validation remains open.
