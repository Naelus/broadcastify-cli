# Model-stack decisions

## Why the pipeline is split

No single tested local model currently offers the best combination of scanner
accuracy, speaker labeling, timestamps, speed, VRAM use, and portability.
Independent stages let the application choose the strongest practical engine
for each responsibility and rerun only the stage whose model or policy changed.

The Windows reference stack is:

| Stage | Default | Reason |
|---|---|---|
| Transcription | faster-whisper Turbo | Mature CUDA path, timestamps, strong retained-radio baseline |
| Diarization | Community-1 | Best validated speaker-boundary quality in this project |
| Retrieval | BGE Small/FastEmbed | Cheap CPU semantic search with persistent vectors |
| Analysis | Gemma 4 12B `Q4_0`/llama.cpp | Local structured output and summaries with reduced VRAM |
| Persistence | SQLite/FTS5 | Auditable local truth independent of model memory |

## Why Whisper remains the default

Whisper is not retained merely because it is familiar. It has the strongest
validated deployment matrix here: faster-whisper on CUDA/CPU, whisper.cpp on
Vulkan/Metal/CPU, OpenVINO, and ONNX/Windows ML. It also provides the timestamp
contract required for exact evidence clips.

Sparse day-long radio must be speech-aware. The portable path uses VAD, bounded
regions, disabled previous-text conditioning, atomic output, and quality gates
that reject collapsed repetition or music-only hallucination.

## Why Qwen is a preview

Qwen3-ASR 0.6B INT8 is very fast on CPU and preserved the core meaning of the
retained event set. The tested sherpa export has no token timing, however, and
longer radio windows showed critical-word drift. It therefore uses honest source
region bounds and never fabricates word timestamps. It is useful for fast
preview, not yet the quote-level default.

## Why diarization remains separate

Community-1 remains the accuracy default and can run on CUDA or CPU. The public
sherpa-onnx segmentation/embedding/clustering path supplies a much faster CPU
preview with bounded memory, but its labels are chunk-scoped acoustic clusters
and it is not human-scored accuracy-equivalent.

A preview transcript can be upgraded to Community-1 without repeating ASR.

## Why embeddings and an LLM are both used

Embeddings cheaply find relevant passages over many days. They do not classify
complex incidents or write grounded summaries. Gemma consumes a time- and
payload-bounded cited set and produces schema-constrained records; deterministic
validators then decide whether those records are supportable. Each extraction
chunk has a durable fingerprint/checkpoint, so dense traffic cannot overflow the
managed context or force successful earlier chunks to repeat. SQLite retains the
evidence and cache identity for both stages.

## Why Gemma is quantized

The 12B instruction model provides better structured reasoning than the smaller
tested local choices while a GGUF `Q4_0` quant reduces VRAM and works through
llama.cpp on CUDA-offload, Vulkan, Metal, and CPU. Provider abstraction permits
OpenAI Responses, compatible endpoints, or a saved-login Codex harness, but
local processing remains the default and external transcript sharing is opt-in.

## Unified models evaluated

VibeVoice-ASR 8B BF16 directly emitted speaker/timestamp/text records and
preserved several core retained events. It also required roughly 16 GiB GPU
memory, altered an address, omitted a disregard, merged speakers, and
hallucinated short noise. It is the leading unified research candidate, not the
fast or evidence-safe default.

Parakeet and forced alignment were also measured. Parakeet was fast but weak on
unsegmented sparse radio; the tested forced aligner timestamps supplied wrong
words without a mismatch confidence. Neither replaces source audio as evidence.

Detailed versions, commands, and measurements belong in
[hardware-backends.md](../hardware-backends.md) and [`PROGRESS.md`](../../PROGRESS.md).
