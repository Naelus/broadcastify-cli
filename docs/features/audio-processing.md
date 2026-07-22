# Audio combination, transcription, and diarization

## Pipeline order

For each feed-day the application performs:

```text
source archive blocks → continuous combined audio → transcription → diarization
```

Combination must occur before ASR and speaker labeling. Diarizing each
30-minute block separately restarts timestamps and acoustic cluster identities
at every boundary.

## Continuous combination

The manifest records every source block, archive start, real media duration,
timeline offset, and feed identity. Encoder overlap is trimmed and actual media
duration—not nominal wall-clock spacing—controls the continuous timeline across
feed outages.

Refreshing a growing day writes and validates a sibling partial MP3, releases
native playback handles, atomically replaces the destination, and only then
updates the manifest. A persistent external file lock preserves the older
recording and reports a recoverable error. The Library treats a combined file as
current only when its manifest exactly matches the retained source set.

## Transcription profiles

Whisper remains the evidence default because it has mature CUDA/CPU,
whisper.cpp Vulkan/Metal, OpenVINO, and ONNX/Windows ML deployment paths.
Sparse scanner audio is decoded in bounded speech-aware regions, with cache
quality gates that reject collapsed repeated output.

The optional Qwen3-ASR 0.6B INT8 profile is fast on CPU but has no token-level
timestamps in the tested sherpa export. It inherits Community-1 turns or Silero
speech-region bounds and never fabricates word timing. It remains a preview
because retained radio showed critical-word drift despite strong throughput.

See [the model decision](../decisions/model-stack.md) and
[hardware backends](../hardware-backends.md).

## Speaker labeling

Community-1 is the accuracy default. It accepts a memory-mapped waveform decoded
by FFmpeg, avoiding the optional TorchCodec file-loader compatibility path.
Existing transcripts can gain or improve labels without repeating ASR.

The portable sherpa-onnx preview uses checksum-pinned public segmentation and
embedding graphs. Long recordings are processed in 15-minute chunks with
five-second overlap; midpoint ownership prevents duplicate boundary turns.
Labels such as `SPEAKER_C003_01` are anonymous, chunk-scoped acoustic clusters,
not identified people or radio units.

## Growing-day reuse

Transcription and diarization caches bind the exact combined-source signature.
When new blocks append to an unchanged prefix, only safely reusable completed
regions/chunks are retained. Changed blocks, uncertain overlap, model changes,
or corrupt metadata invalidate only the affected stage.

Portable diarization checkpoints every completed chunk atomically and keeps the
checkpoint until the final cache is committed. Model-window incident analysis
uses the same durable principle at a later stage.

## What execution tests prove

The profile verifier runs transcription, speaker labeling, and analysis on
generated local fixtures without archive quota. A green result proves the
selected runtime/model/backend executed; it does not establish radio-word
accuracy or human-scored diarization quality.
