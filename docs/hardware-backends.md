# Hardware backends and portable runtimes

The application chooses a backend independently for transcription, diarization, and analysis. A GPU-capable ASR engine does not imply that pyannote supports the same API, and a generic `llama-server` executable does not prove that its Vulkan or SYCL backend loaded. **Settings → Refresh check** reports cheap stage diagnostics; **Verify profile** then executes all three stages sequentially with generated input.

Readiness uses three deliberately different terms:

- **Detected** means a runtime, device, model cache, or provider catalog entry exists.
- **Configured** means the selected stages have enough local configuration to attempt execution.
- **Verified** means ASR, diarization, and analysis each completed a real selected-runtime self-test with the current settings in the current UI session.

Refreshing hardware does not download a model or execution provider and never promotes detection to verification. **Verify profile** may download a missing managed model after the user starts it, releases each model before loading the next stage, stops at the first actionable failure, and never consumes Broadcastify archive quota.

## Current validation boundary

| Profile | Transcription | Diarization | Analysis | Evidence |
|---|---|---|---|---|
| Windows CUDA | faster-whisper CUDA | pyannote CUDA | llama.cpp auto-offload | Full retained-day reference workflow; current Windows App SDK 2.3.1 private publish completed joined native verification in 14.9 seconds on RTX 3090 |
| CPU | faster-whisper INT8 or whisper.cpp CPU | pyannote CPU | llama.cpp CPU | Protected 30-second all-CPU Web job validated; full-day benchmark remains open |
| AMD Vulkan/Linux | whisper.cpp Vulkan | pyannote CPU fallback | llama.cpp Vulkan | Exact `historical-validation` first joined verification completed in 27.886 seconds on Radeon 890M; exact current `historical-validation` warm-cache rerun completed in 9.960 seconds and passed 172 hardened tests; protected retained-audio Web job also validated |
| OpenVINO | OpenVINO GenAI AUTO/CPU | pyannote CPU fallback | llama.cpp SYCL/auto/CPU | Joined Web verification completed in 13.8 seconds; exact `historical-validation` protected 60-second job completed all stages in 31 seconds |
| Windows ML | ONNX Runtime GenAI CPU | pyannote CPU fallback | llama.cpp auto/CPU | Joined Web verification completed in 13.3 seconds; exact `historical-validation` protected 60-second job completed all stages in 32 seconds and resumed idempotently; GPU providers remain gated |
| Fast CPU/Qwen | Qwen3-ASR 0.6B INT8 through sherpa-onnx CPU | pyannote CPU, with Silero fallback bounds | llama.cpp auto/CPU | Managed model/runtime and retained-region provenance are implemented; the real joined Windows proof completed in 15.265 seconds; exact AMD/Linux source processed five minutes in 7.128 seconds at four threads, with critical-word drift keeping the quality gate open |
| macOS | whisper.cpp Metal or CPU | pyannote CPU | llama.cpp Metal or CPU | Explicit profile and detection exist; a real Mac install and model run remain required |

The official [whisper.cpp project](https://github.com/ggml-org/whisper.cpp) documents Windows, Linux, macOS, Docker, quantized models, Metal, OpenVINO, and `GGML_VULKAN=1`. The official [llama.cpp project](https://github.com/ggml-org/llama.cpp) documents native packages/releases, Vulkan and SYCL backends, quantized GGUF models, and its OpenAI-compatible server.

## Current model and runtime decision

Whisper remains the default ASR family, not because it is the newest model name but because it currently gives this app the strongest portable deployment matrix: faster-whisper for the measured CUDA/CPU reference, whisper.cpp for Vulkan/Metal/ROCm and quantized CPU use, OpenVINO GenAI for Intel-oriented CPU/GPU/NPU execution, and ONNX Runtime for the Windows ML boundary. Radio accuracy still has to be measured on the retained corpus; backend availability is not a quality claim.

[Qwen3-ASR 0.6B](https://github.com/QwenLM/Qwen3-ASR) INT8 through sherpa-onnx is now an implemented, separately selectable fast-CPU preview. With English forced per stream, it preserved the core meaning of 13 retained evidence clips at about 0.08 real-time factor and completed a five-minute segmented ASR pass in 6.027 seconds. The complete experimental sherpa diarization plus Qwen path took 12.361 seconds. The production profile retains Community-1 as its speaker-label default: Qwen decodes Community-1's exclusive speaker turns when available, otherwise it uses Silero VAD speech regions.

The model manager pins the official 878,702,423-byte sherpa archive and 643,854-byte Silero graph, verifies their recorded SHA-256 values, rejects unsafe archive members, extracts into a staging directory, and atomically installs only a complete model. Managed discovery requires a matching engine/model/source/runtime/provider/precision/hash manifest. The current ONNX export still has no token timestamps, so the transcript truthfully contains region-level segments and zero fabricated word-timestamp records. Metadata identifies whether timing came from Community-1 or Silero. Sub-0.75-second speaker fragments are omitted because retained testing showed weak noise bursts producing plausible filler. A real selected-profile proof completed Qwen CPU in 2.140 seconds, Community-1 CPU in 4.078 seconds, and local Gemma in 7.703 seconds—15.265 seconds total.

Exact pushed source `historical-validation` subsequently passed all 191 tests in a hardened AMD/Linux container, ran the same production preparation path against hash-verified local assets, and loaded sherpa-onnx 1.13.4 from an isolated executable runtime. On the Ryzen AI 9 HX PRO 370, a retained 25.44-second event-bound fixture took 1.236 seconds at four threads and 1.172 seconds at twelve, with about 1.42 GiB peak RSS. A five-minute Silero-segmented slice took 7.128 seconds at four threads and 6.587 seconds at eight, with about 1.58 GiB peak RSS. The event-bound clip preserved `15 to 20 shots fired, nothing seen` but generated surrounding words; the five-minute run changed the critical phrase to `shops fired` / `nothing seems`. It therefore remains opt-in because throughput is excellent while quote-level fidelity, names/locations/codes, and longer-corpus false-claim gates remain open.

The official Qwen stack itself supports timestamp output only when its separate [Qwen3-ForcedAligner 0.6B](https://huggingface.co/Qwen/Qwen3-ForcedAligner-0.6B) is loaded. The public ASR and aligner repositories are about 1.75 and 1.71 GiB respectively before runtime/cache overhead, and streaming timestamps are not available in the same mode. That GPU-oriented PyTorch/vLLM route remains a possible evidence-alignment upgrade; it is not silently conflated with the smaller sherpa CPU deployment.

[NVIDIA Parakeet TDT 0.6B v3](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3) remains a credible high-throughput model on clean speech. The tested official INT8 ONNX export decoded a 3.845-second clean fixture in 0.107 seconds with token timestamps, but returned blank output for unsegmented sparse-radio clips. After sherpa speech segmentation it processed five minutes in 1.657 seconds of ASR time but rendered the key phrase as `drop fire`. It does not perform diarization and is not the current radio default.

[Microsoft VibeVoice-ASR-HF](https://huggingface.co/microsoft/VibeVoice-ASR-HF) is the strongest current answer to “can one open local model produce who, when, and what?” Its MIT-licensed 8B BF16 checkpoint uses Transformers 5.3, accepts up to 60 minutes, and directly emits speaker/timestamp/content records. The exact 15.53 GiB snapshot was retained on the NAS and tested locally on the RTX 3090 without changing the app environment. It preserved `15 to 20 shots fired, nothing seen` in both unprompted and radio-context runs; 25.44 seconds took 8.751/9.017 seconds (0.344/0.354 RTF), allocated 15.971 GiB at peak, and cold-loaded over the NAS in 79.158 seconds.

Two more retained fixtures exposed the boundary. A 26.56-second fire exchange took 8.676 seconds (0.327 RTF), identified two speakers, and kept the fire report, but changed `447 Fallen Oak` to `3447 Solano` and omitted the later disregard. A sparse 120-second stolen-squad-car context clip took 8.755 seconds (0.073 RTF) and preserved the central report, but invented an introductory name/title and merged two Community-1 speaker clusters into one. Peak allocation rose to 16.58 GiB. VibeVoice is therefore the leading high-end unified research candidate, but it is slower/larger than Qwen on short audio, CUDA-oriented in this measured setup, and not safer evidence by construction. It needs a broader long-window false-claim/diarization score plus an isolated sidecar because Transformers 5.x conflicts with the validated Windows ML 4.x environment.

[pyannote Community-1](https://huggingface.co/pyannote/speaker-diarization-community-1) remains the validated open local diarizer. Its current model card documents better speaker assignment/counting than 3.1, exclusive diarization for easier ASR reconciliation, offline use, CPU by default, and CUDA through PyTorch. It does not document Vulkan, OpenVINO, Windows ML, or Metal inference.

[sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx) **1.13.4** is the strongest measured lighter portable CPU runtime. It packages offline ASR and [offline ONNX speaker diarization](https://k2-fsa.github.io/sherpa/onnx/speaker-diarization/index.html) across Windows, Linux, macOS, mobile, and many language bindings. On the retained five-minute slice, its segmentation/TitaNet/clustering pipeline took 5.722 seconds versus 68.508 seconds for fresh Community-1, but found 19 rather than 24 turns and needed a 0.95 clustering threshold to avoid severe over-splitting. It covered 89.2% of Community-1 speech with 99.5% candidate precision and produced one extra cluster. Its documented providers are CPU, CUDA, and Core ML rather than Vulkan/DirectML; Community-1 remains the accuracy default.

NVIDIA's current [streaming Sortformer](https://huggingface.co/nvidia/diar_streaming_sortformer_4spk-v2) and [Multitalker Parakeet](https://huggingface.co/nvidia/multitalker-parakeet-streaming-0.6b-v1) materially improve the NVIDIA-specific answer. Sortformer provides online speaker activity for up to four active speakers; Multitalker Parakeet consumes those external speaker tracks and runs a target-speaker recognizer per speaker, which helps overlapping speech. It is still a coupled NeMo/CUDA pipeline whose cost scales with speakers and whose four-speaker streaming assumption does not match anonymous identities accumulated across a full scanner day. The older offline Sortformer checkpoint additionally had a short-recording/noncommercial boundary. This path deserves a bounded NVIDIA benchmark, not a cross-vendor default claim. WhisperX also continues to wrap a separate pyannote diarization stage.

Mistral's current Voxtral line splits the answer differently. [Voxtral Mini 4B Realtime](https://huggingface.co/mistralai/Voxtral-Mini-4B-Realtime-2602) is open-weight and designed for very low-latency local/vLLM streaming on a 16 GiB-class GPU, while the documented Voxtral Mini Transcribe V2 service adds diarization and word timestamps. The downloadable realtime model is not evidence that the local checkpoint supplies the complete offline diarization contract, so this app does not label it a unified local replacement.

## Managed Qwen3-ASR CPU preview

Install the optional runtime without changing the default profile:

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[qwen]"
```

Choose **Fast CPU preview (Qwen3-ASR)** in native or Web Settings, then select **Download & test model**. The explicit action downloads the pinned public assets, verifies and installs them under the per-user model root, stores the exact selected path, and executes the decoder on generated silence. The generated-silence result proves runtime execution only. Use retained radio and the Library's evidence clips to judge quality.

Advanced installations can set `QWEN3_ASR_MODEL_PATH` and `QWEN3_ASR_VAD_PATH` to complete local assets. `QWEN3_ASR_THREADS` overrides the conservative four-thread default. On the Windows reference Ryzen, 4 and 8 threads decoded the retained clip in 1.284 and 1.200 seconds after model load, while 16 regressed to 1.859 seconds. On the 12-core AMD/Linux appliance, 1/2/4/8/12/16/24 threads took 2.169/1.422/1.236/1.203/1.172/1.338/1.377 seconds. Twelve minimized one-clip latency but improved only about 5% over four; four remains the portable sustained-work default. The managed public path needs no Hugging Face token.

## Diarization decoding and proof

Both UIs expose **Test speakers** separately from hardware detection. It loads `pyannote/speaker-diarization-community-1`, selects the requested CUDA or CPU device, and executes generated local audio. The action is explicit because its first run may download the gated model; it never uses Broadcastify archive quota. A complete local snapshot is enough to configure an offline run without retaining a Hugging Face token, but the profile remains unverified until inference succeeds.

The app does not give archive filenames to pyannote's optional TorchCodec loader. FFmpeg decodes the already-prepared 16 kHz mono input to a temporary float32 PCM file, PyTorch memory-maps it, and pyannote receives `waveform` plus `sample_rate`. This avoids TorchCodec/PyTorch/FFmpeg-DLL compatibility failures and prevents a day-long waveform from being copied onto the Python heap. The raw scratch file is removed after inference; the compact lossless preparation remains reusable after an interruption.

On the Windows CUDA reference machine, the original generated-audio proof completed in 6.5 seconds and the current `historical-validation` proof completed in 5.3 seconds. A separate 250-second slice from retained combined feed 90001 audio completed in 11.75 seconds and returned 52 turns across three anonymous acoustic clusters. On the AMD host, the tokenless cached CPU proof completed in 4.221 seconds. These tests prove execution and continuous-file decoding; the clusters are not officer identities.

## Native whisper.cpp

Build a native Vulkan binary with the upstream flags:

```bash
git clone https://github.com/ggml-org/whisper.cpp.git
cd whisper.cpp
cmake -B build -DGGML_VULKAN=1
cmake --build build -j --config Release
```

Set `WHISPER_CPP_PATH` to `build/bin/whisper-cli` and `WHISPER_CPP_MODEL_PATH` to the selected GGML model, or put the binary/model in one of the paths listed by `.env-example`. Linux and macOS binaries without an `.exe` suffix are discovered from `PATH`, `tools/whisper.cpp`, and a local `whisper.cpp/build/bin` tree.

Combined archive audio is normally MP3. Before invoking whisper.cpp, the adapter atomically prepares a 16 kHz mono PCM WAV with FFmpeg, reuses a complete interrupted preparation, and deletes it after a successful transcript. This avoids assuming that a particular upstream build includes optional FFmpeg decoding.

## Immutable Linux / containerized whisper.cpp

The container path is opt-in. It is useful for an appliance or immutable host that has a working Vulkan driver and Docker/Podman but should not receive compilers or host libraries. Pull the official image yourself and record what was received:

```bash
docker pull ghcr.io/ggml-org/whisper.cpp:main-vulkan
docker image inspect --format '{{json .RepoDigests}} {{.Id}}' \
  ghcr.io/ggml-org/whisper.cpp:main-vulkan
```

Then configure a local model and the already-present tag:

```dotenv
WHISPER_CPP_CONTAINER_IMAGE="ghcr.io/ggml-org/whisper.cpp:main-vulkan"
WHISPER_CPP_MODEL_PATH="/absolute/path/to/ggml-large-v3-turbo-q5_0.bin"
BROADCASTIFY_CONTAINER_RUNTIME="docker"
WHISPER_CPP_CONTAINER_DEVICE="/dev/dri"
```

The app checks `docker image inspect`; it never pulls an image in the background. A transcription run uses `--network none`, a read-only root filesystem, `no-new-privileges`, no Linux capabilities, a bounded temporary filesystem, the calling UID/GID, the detected render-device group, and exactly three bind mounts: prepared audio read-only, model read-only, and the temporary JSON output directory read/write. Docker/Podman access and `/dev/dri` permissions must already be granted to the account running the app. The app never stores or supplies a sudo password. `WHISPER_CPP_CONTAINER_GROUP_ID` can override render-group detection on a carefully administered host.

The adapter retains sanitized initialization lines such as the selected Vulkan adapter and loaded backend in transcript metadata. Progress is clamped to 0–100 because upstream can report a value above 100 on very short inputs.

## llama.cpp analysis

The managed local provider discovers `llama-server` from `LLAMA_SERVER_PATH` or `PATH` on every platform. On Linux/macOS it creates a private writable fallback `HOME`, `LLAMA_CACHE`, and `HF_HOME` only when the inherited home is absent or unwritable, which matters for numeric-user containers. `BROADCASTIFY_RUNTIME_DIR` can select that fallback root.

Official release archives often place `llama-server` beside backend and GGML shared libraries. The launcher prepends that sibling directory to `LD_LIBRARY_PATH` on Linux or `DYLD_LIBRARY_PATH` on macOS **only for the llama-server child**. Do not export a llama release directory globally: exact joined AMD testing showed that whisper.cpp can otherwise load llama.cpp's incompatible GGML library first. Windows resolves the release DLLs through the executable directory and receives no Unix loader override.

An existing llama.cpp/Ollama/LM Studio-style loopback server can be used instead: choose the local provider and enter its `/v1` endpoint. This is the recommended container boundary for analysis because the UI can health-check the server and the model process can be supervised independently. The upstream [llama-server documentation](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md) describes native and container launches. Expose it only on loopback for this app; do not publish an unauthenticated model endpoint to the LAN.

## Intel/OpenVINO

Install the optional packages with `pip install -e ".[openvino]"`, choose OpenVINO plus AUTO/GPU/NPU, and select **Verify profile** or **Test engine**. That explicit action may download the selected official OpenVINO Whisper model, then decodes one second of generated silence through the same long-recording adapter used by archive jobs. It proves execution, not radio accuracy. The general hardware check remains download-free.

The adapter uses INT8 model mappings for Tiny through Large V3 Turbo, including `OpenVINO/distil-whisper-large-v3-int8-ov`. English-suffixed Web UI choices such as `tiny.en` normalize to the same managed model. The current [OpenVINO GenAI documentation](https://docs.openvino.ai/2026/openvino-workflow-generative/inference-with-genai.html) lists WhisperPipeline on CPU, GPU, and NPU, and the [2026 release notes](https://docs.openvino.ai/nightly/about-openvino/release-notes-openvino.html) add word-level Whisper timestamps across those devices. OpenVINO 2026.2.1's [NPU Whisper guidance](https://docs.openvino.ai/2026/openvino-workflow-generative/inference-with-genai/inference-with-genai-on-npu.html) says the ordinary Whisper GenAI pipeline works on NPU without NPU-specific pipeline flags, so the app no longer injects the obsolete static-pipeline override.

If the requested accelerator rejects model compilation, initialization retries on CPU. If it compiles but rejects generation, that first chunk retries on CPU and subsequent chunks stay there. The transcript and self-test result record the actual backend, requested device, fallback stage, and bounded first-line reason. The Windows reference machine exposes an AMD CPU and NVIDIA GPU through OpenVINO rather than Intel hardware: AUTO decoded the retained 22.7-second clip in 1.093 seconds; explicit GPU failed at generation and the CPU fallback returned the identical 23 words/3 segments in 1.828 seconds. This validates the contract and fallback, not Intel GPU/NPU performance.

Exact clean source commit `historical-validation` also completed a fresh protected loopback Web job from only a retained 60-second MP3, with Hugging Face/Transformers offline flags. CPU pyannote produced 21 turns across two anonymous clusters; OpenVINO CPU produced 14 segments/278 word records; local Gemma retained zero unsupported incidents; one passage, embedding, and daily summary persisted; and the library reached 100% `Ready to review` in 31 seconds. One ASR segment did not overlap a diarization turn, so portable diarization is functional rather than perfect.

The later `historical-validation` selected-engine test loaded current Tiny English on OpenVINO CPU, completed the real synthetic decode in 1.188 seconds, and reported no fallback.

## Windows ML and ONNX Runtime GenAI

The validated Windows ML functional path is a CPU FP32 Whisper model. The helper uses `Config`, passes a one-item prompt batch to the multimodal processor, keeps one model process alive across archive chunks, and reports the provider parsed from `genai_config.json`. A scalar prompt call is not equivalent for Whisper and caused the formerly misleading `DivideByZeroException`.

Verified Windows publish folders carry the helper and its complete runtime under `windowsml/`. The desktop sets `WINDOWS_ML_HELPER_PATH` for Python children only when the user has not explicitly configured another helper. Managed preparation exports to a staging directory, atomically installs only a complete graph, writes `broadcastify-model.json`, and revalidates selected model/source/provider/precision before execution. An explicit missing selected path does not fall back to an unrelated cache or environment path.

The official ONNX Runtime GenAI fixture transcribed correctly through the persistent helper, proving the decoder contract. The retained low-SNR evidence clip then separated execution from quality: Tiny FP32 returned `[inaudible]`, Tiny INT4 repeated it, multilingual Base captured the core dispatch in **0.212 seconds**, and Small took **0.915 seconds** without materially improving the phrase. Base is the Windows ML starter. **Test engine** uses generated silence and proves only that the selected graph executed; retained radio remains the quality gate.

Both native projects use the current stable Windows App SDK **2.3.1** and BuildTools **10.0.28000.2270**. Restore resolves `Microsoft.Windows.AI.MachineLearning` and `Microsoft.WindowsAppSDK.ML` **2.1.74** while the inference helper retains its explicitly tested ONNX Runtime GenAI 0.14.1 contract; see [windows-publish.md](windows-publish.md). Microsoft's [Windows ML ONNX-version guidance](https://learn.microsoft.com/en-us/windows/ai/new-windows-ml/onnx-versions) identifies 2.1.74 as the current supported Windows ML release and maps it to ONNX Runtime **1.24.6**. Package/runtime availability is still not model execution proof.

Provider management is explicit and follows Microsoft's [Windows ML execution-provider catalog](https://learn.microsoft.com/en-us/windows/ai/new-windows-ml/initialize-execution-providers). As of July 18, 2026, the current [supported-provider table](https://learn.microsoft.com/en-us/windows/ai/new-windows-ml/supported-execution-providers) lists dynamically delivered AMD MIGraphX/VitisAI, Intel OpenVINO, Qualcomm QNN, and NVIDIA TensorRT RTX routes for compatible Windows 11 24H2+ devices, alongside included CPU and legacy DirectML. For the 2.x line it currently records MIGraphX 1.8.57.0, TensorRT RTX 0.0.40.0, OpenVINO 1.8.80.0, QNN 2.2450.47.0, and VitisAI 1.8.63.0 packages. Microsoft explicitly says the current MIGraphX provider is not supported for GenAI scenarios. Catalog status is only detection, provider versions update independently, and each selected provider/model pair still needs a real timed decode. The newer [Windows ML Model Catalog](https://learn.microsoft.com/en-us/windows/ai/new-windows-ml/model-catalog/overview) is a good future distribution surface for a published provider-compatible graph, but it does not convert an arbitrary Hugging Face model. These commands respectively inspect, activate only an already-installed provider, or allow Windows to download and register compatible certified providers:

```powershell
$helper = ".\BroadcastifyCli.WindowsML\bin\Release\net10.0-windows10.0.26100.0\win-x64\BroadcastifyCli.WindowsML.exe"
& $helper --providers
& $helper --register-winml --providers
& $helper --ensure-winml --providers
```

`--ensure-winml` is never invoked by ordinary transcription. It can take minutes and changes system-wide provider package state, so it belongs behind an explicit setup/test action. Registration uses ONNX Runtime GenAI's native provider environment; registering only the general C# `OrtEnv` does not make a provider visible to GenAI.

GPU acceleration remains gated. ONNX Runtime GenAI 0.13.1 and 0.14.1 DML Whisper Tiny exports fail at graph capture or a fused DML node. The certified Windows ML TensorRT RTX 1.8.24.0 provider registered and decoded, but rejected 36 attention nodes and took 15.895 seconds for the retained 22.7-second clip while the CPU model took 0.622 seconds after warm caches. A successful decode with provider partitioning is therefore not reported as a validated speed path.

During the exact `historical-validation` review, read-only provider inspection returned `NvTensorRTRTXExecutionProvider:NotReady`. After the Windows App SDK 2.3.1 migration, current read-only inspection returned uncertified `WebGpuExecutionProvider:NotPresent` and certified installed `NvTensorRTRTXExecutionProvider:NotReady`. No acquisition or system registration was performed in either review.

Exact clean source commit `historical-validation` completed the matching fresh protected 60-second Web job after the headless harness received the documented `WINDOWS_ML_HELPER_PATH`. CPU pyannote produced 21 turns/two clusters with zero unlabeled ASR segments; the persistent helper decoded three 28-second-bounded chunks into three segments; local Gemma, one passage/embedding, and a daily summary persisted; and the library reached 100% `Ready to review` in 32 seconds. An immediate resume reported `operation=reused`, repeated no audio/model stage, indexed zero new passages, and completed in about one second of job time.

## macOS and Apple Metal

Metal is a native profile, not a container profile. Build current upstream whisper.cpp and llama.cpp on the Mac that will run them:

```bash
xcode-select --install
git clone https://github.com/ggml-org/whisper.cpp.git
cmake -S whisper.cpp -B whisper.cpp/build -DGGML_METAL=ON
cmake --build whisper.cpp/build --config Release -j

git clone https://github.com/ggml-org/llama.cpp.git
cmake -S llama.cpp -B llama.cpp/build -DGGML_METAL=ON
cmake --build llama.cpp/build --config Release -j --target llama-server
```

Set `WHISPER_CPP_PATH`, `WHISPER_CPP_MODEL_PATH`, and `LLAMA_SERVER_PATH`, choose **Apple Metal**, then run **Verify profile** or **Test engine**. The app accepts Metal only when the native whisper.cpp directory/linkage exposes ggml-metal, and the hardware comparison requires llama.cpp to list a `Metal` device as well. Metal is never offered as a container recovery path because Docker and Podman cannot expose Apple Metal through this adapter. The upstream [whisper.cpp repository](https://github.com/ggml-org/whisper.cpp) also documents optional Core ML encoder acceleration on Apple Silicon; its compiled encoder directory must accompany the matching GGML model. The upstream [llama.cpp repository](https://github.com/ggml-org/llama.cpp) describes Apple Silicon as a first-class Metal target.

If either native backend is absent, Automatic stays on the portable CPU route. pyannote remains a CPU stage because it does not expose a supported Metal backend through this app. No macOS performance claim is made until the exact self-test and a retained radio clip run on real Apple hardware.

## Measured AMD/Vulkan reference

The isolated Linux validation used an AMD Radeon 890M (RADV GFX1150), the official `main-vulkan` whisper.cpp image, and the 77,704,715-byte Tiny English model. The exact app adapter transcribed a 22.7-second retained radio clip in 1.022 seconds, returned one 159-character segment, reported the AMD adapter, and loaded its model on `Vulkan0`.

The separate llama.cpp b9637 Vulkan run used public `ggml-org/gemma-3-1b-it-GGUF:Q4_K_M`: all 27 layers offloaded, the Q4_K model occupied about 762 MiB of Vulkan memory, prompt evaluation reached about 320 tokens/second, and generation reached about 105 tokens/second. These are compatibility measurements, not promises for a 12B production model or other hardware.

No packages, services, or storage configuration were changed on the appliance host. Image/model hashes and the remaining limitations are recorded in `PROGRESS.md` and `BUGS.md`.

Exact pushed source `historical-validation` was later archived at SHA-256 `d6d770a150a322bdd9435b67aff3e61412660456ceb4233c424570a10c7d676e` and independently extracted in a fresh user-owned path. Its cheap diagnostics saw the Vulkan runtime, cached Community-1 model, and llama.cpp Vulkan device but correctly left the profile configured/unverified. The three explicit proofs then completed whisper.cpp Vulkan ASR in 4.413 seconds, CPU Community-1 diarization in 4.221 seconds, and Gemma 3 1B `Q4_K_M` Vulkan generation in 5.622 seconds. A protected loopback analysis job completed in 1.413 seconds, and the exact source passed all 163 tests in 3.82 seconds inside the immutable runtime boundary.

Exact pushed source `historical-validation` was then hash-verified and extracted in another fresh path. Its one-action profile verifier completed whisper.cpp Vulkan in **4.578 seconds**, tokenless cached Community-1 CPU in **10.595 seconds**, and Gemma 3 1B `Q4_K_M` llama.cpp Vulkan in **10.645 seconds**, **27.886 seconds total**. The llama log identifies `Vulkan0 : AMD Radeon 890M Graphics (RADV GFX1150)`. The exact source passed all **171 tests in 5.90 seconds** with networking disabled, read-only source/root, dropped capabilities, `no-new-privileges`, numeric user 950, and only the existing render device/group. No validation container remained.

Exact pushed source `historical-validation` was archived at SHA-256 `56b1ad6f1d92ed3b79cce24a623cecb38c1caf683b242a9872f4f2a108a0b976`, matched after transfer, and extracted into a fresh path. The hardened container passed all **172 tests in 5.04 seconds**. Its warm-cache joined rerun completed whisper.cpp Vulkan in **3.677 seconds**, Community-1 CPU in **2.783 seconds**, and Gemma 3 1B `Q4_K_M` llama.cpp Vulkan in **2.559 seconds**, **9.960 seconds total**. The source/root stayed read-only, networking was disabled, the llama log again identifies `Vulkan0 : AMD Radeon 890M Graphics (RADV GFX1150)`, and no container remained. These generated-input timings prove execution and integration; they are not day-long throughput measurements.

### Joined current-worker measurement

The exact current commit (`historical-validation`) was placed in a fresh isolated clone. Because the TrueNAS home and `/tmp` mounts are deliberately `noexec`, Python 3.12 CPU wheels were built inside the already-recorded Vulkan image and stored in user-writable executable `/var/tmp`; no host package or mount setting changed. The resulting image/runtime combination passed 113 tests with networking disabled.

The headless Web harness established the real loopback cookie/action-token session and posted a fresh `continue-local` job for 30 seconds of retained radio with `--network none`. No Hugging Face token was supplied: the complete cached Community-1 model loaded, CPU pyannote completed five turns, whisper.cpp retained `cpu` plus `vulkan` availability and explicit `using Vulkan0 backend` evidence for AMD Radeon 890M, and the 1B Q4 Gemma server detected the same Vulkan device. One passage, one embedding, and one daily summary persisted; the final state was Ready to review with zero invented incidents. Container time was 15.139 seconds. The root filesystem was read-only, all capabilities were dropped, `no-new-privileges` and PID limit 1024 were set, and the completed container was removed.

### Bounded all-CPU measurement

Exact commit `historical-validation` passed 119 tests in the immutable image, then ran the same protected Web job with no `/dev/dri` device and no supplemental render groups. whisper.cpp was explicitly selected as CPU and retained `ggml_vulkan: No devices found`; pyannote ran on CPU; llama.cpp listed only the Ryzen CPU, loaded the cached 1B Q4 model in about 0.645 seconds, and generated around 56-72 tokens/second.

The fresh 30-second workflow completed in 21.282 seconds, persisted one segment, five speaker turns, one evidence-backed incident, one passage/embedding, and one daily summary, and ended Ready to review. The local model's two unsupported daily briefs were rejected by the incident-ID/count grounding gate; the stored deterministic brief contains only the cited report. The container used a read-only root, network `none`, numeric user 950, dropped all capabilities, `no-new-privileges`, PID limit 1024, no token, and an empty device list. Full-day CPU throughput remains unmeasured.
