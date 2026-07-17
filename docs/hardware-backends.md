# Hardware backends and portable runtimes

The application chooses a backend independently for transcription, diarization, and analysis. A GPU-capable ASR engine does not imply that pyannote supports the same API, and a generic `llama-server` executable does not prove that its Vulkan or SYCL backend loaded. **Settings → Refresh check** therefore reports all three stages separately.

## Current validation boundary

| Profile | Transcription | Diarization | Analysis | Evidence |
|---|---|---|---|---|
| Windows CUDA | faster-whisper CUDA | pyannote CUDA | llama.cpp auto-offload | Full retained-day reference workflow on RTX 3090 |
| CPU | faster-whisper INT8 | pyannote CPU | llama.cpp CPU | Components work; full-day wall-clock benchmark remains open |
| AMD Vulkan/Linux | whisper.cpp Vulkan | pyannote CPU fallback | llama.cpp Vulkan | Protected end-to-end Web job validated on Radeon 890M; full-day CPU diarization timing remains open |
| OpenVINO | OpenVINO GenAI AUTO/CPU | pyannote CPU fallback | llama.cpp SYCL/auto/CPU | Real CPU ASR and rejected-accelerator-to-CPU fallback |
| Windows ML | ONNX Runtime GenAI CPU | pyannote CPU fallback | llama.cpp auto/CPU | Real ASR decode; DML fails and TensorRT RTX currently partitions/falls back slower than CPU |
| macOS | whisper.cpp Metal or CPU | pyannote CPU | llama.cpp Metal or CPU | Explicit profile and detection exist; a real Mac install and model run remain required |

The official [whisper.cpp project](https://github.com/ggml-org/whisper.cpp) documents Windows, Linux, macOS, Docker, quantized models, Metal, OpenVINO, and `GGML_VULKAN=1`. The official [llama.cpp project](https://github.com/ggml-org/llama.cpp) documents native packages/releases, Vulkan and SYCL backends, quantized GGUF models, and its OpenAI-compatible server.

## Diarization decoding and proof

Both UIs expose **Test speakers** separately from hardware detection. It loads `pyannote/speaker-diarization-community-1`, selects the requested CUDA or CPU device, and executes generated local audio. The action is explicit because its first run may download the gated model; it never uses Broadcastify archive quota.

The app does not give archive filenames to pyannote's optional TorchCodec loader. FFmpeg decodes the already-prepared 16 kHz mono input to a temporary float32 PCM file, PyTorch memory-maps it, and pyannote receives `waveform` plus `sample_rate`. This avoids TorchCodec/PyTorch/FFmpeg-DLL compatibility failures and prevents a day-long waveform from being copied onto the Python heap. The raw scratch file is removed after inference; the compact lossless preparation remains reusable after an interruption.

On the Windows CUDA reference machine, the generated-audio proof completed in 6.5 seconds. A separate 250-second slice from retained combined feed 90001 audio completed in 11.75 seconds and returned 52 turns across three anonymous acoustic clusters. These tests prove execution and continuous-file decoding; the clusters are not officer identities.

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

An existing llama.cpp/Ollama/LM Studio-style loopback server can be used instead: choose the local provider and enter its `/v1` endpoint. This is the recommended container boundary for analysis because the UI can health-check the server and the model process can be supervised independently. The upstream [llama-server documentation](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md) describes native and container launches. Expose it only on loopback for this app; do not publish an unauthenticated model endpoint to the LAN.

## Intel/OpenVINO

Install the optional packages with `pip install -e ".[openvino]"`, choose OpenVINO plus AUTO/GPU/NPU, and select **Test engine**. That explicit test may download the selected official OpenVINO Whisper model, then decodes one second of local synthetic audio through the same long-recording adapter used by archive jobs. The general hardware check remains download-free.

The adapter uses INT8 model mappings for Tiny through Large V3 Turbo, including `OpenVINO/distil-whisper-large-v3-int8-ov`. English-suffixed Web UI choices such as `tiny.en` normalize to the same managed model. OpenVINO 2026.2.1's [NPU Whisper guidance](https://docs.openvino.ai/2026/openvino-workflow-generative/inference-with-genai/inference-with-genai-on-npu.html) says the ordinary Whisper GenAI pipeline works on NPU without NPU-specific pipeline flags, so the app no longer injects the obsolete static-pipeline override.

If the requested accelerator rejects model compilation, initialization retries on CPU. If it compiles but rejects generation, that first chunk retries on CPU and subsequent chunks stay there. The transcript and self-test result record the actual backend, requested device, fallback stage, and bounded first-line reason. The Windows reference machine exposes an AMD CPU and NVIDIA GPU through OpenVINO rather than Intel hardware: AUTO decoded the retained 22.7-second clip in 1.093 seconds; explicit GPU failed at generation and the CPU fallback returned the identical 23 words/3 segments in 1.828 seconds. This validates the contract and fallback, not Intel GPU/NPU performance.

## Windows ML and ONNX Runtime GenAI

The validated Windows ML functional path is a CPU FP32 Whisper model. The helper uses `Config`, passes a one-item prompt batch to the multimodal processor, keeps one model process alive across archive chunks, and reports the provider parsed from `genai_config.json`. A scalar prompt call is not equivalent for Whisper and caused the formerly misleading `DivideByZeroException`.

Verified Windows publish folders carry the helper and its complete runtime under `windowsml/`. The desktop sets `WINDOWS_ML_HELPER_PATH` for Python children only when the user has not explicitly configured another helper. The published helper's FP32 CPU model self-test completed in 0.589 seconds and reported `Windows ML / ONNX Runtime GenAI CPU`; see [windows-publish.md](windows-publish.md).

Provider management is explicit and follows Microsoft's [Windows ML execution-provider catalog](https://learn.microsoft.com/en-us/windows/ai/new-windows-ml/initialize-execution-providers). These commands respectively inspect, activate only an already-installed provider, or allow Windows to download and register compatible certified providers:

```powershell
$helper = ".\BroadcastifyCli.WindowsML\bin\Release\net10.0-windows10.0.26100.0\win-x64\BroadcastifyCli.WindowsML.exe"
& $helper --providers
& $helper --register-winml --providers
& $helper --ensure-winml --providers
```

`--ensure-winml` is never invoked by ordinary transcription. It can take minutes and changes system-wide provider package state, so it belongs behind an explicit setup/test action. Registration uses ONNX Runtime GenAI's native provider environment; registering only the general C# `OrtEnv` does not make a provider visible to GenAI.

GPU acceleration remains gated. ONNX Runtime GenAI 0.13.1 and 0.14.1 DML Whisper Tiny exports fail at graph capture or a fused DML node. The certified Windows ML TensorRT RTX 1.8.24.0 provider registered and decoded, but rejected 36 attention nodes and took 15.895 seconds for the retained 22.7-second clip while the CPU model took 0.622 seconds after warm caches. A successful decode with provider partitioning is therefore not reported as a validated speed path.

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

Set `WHISPER_CPP_PATH`, `WHISPER_CPP_MODEL_PATH`, and `LLAMA_SERVER_PATH`, choose **Apple Metal**, then run **Test engine**. The app accepts Metal only when the native whisper.cpp directory/linkage exposes ggml-metal, and the hardware comparison requires llama.cpp to list a `Metal` device as well. The upstream [whisper.cpp repository](https://github.com/ggml-org/whisper.cpp) also documents optional Core ML encoder acceleration on Apple Silicon; its compiled encoder directory must accompany the matching GGML model. The upstream [llama.cpp repository](https://github.com/ggml-org/llama.cpp) describes Apple Silicon as a first-class Metal target.

If either native backend is absent, Automatic stays on the portable CPU route. pyannote remains a CPU stage because it does not expose a supported Metal backend through this app. No macOS performance claim is made until the exact self-test and a retained radio clip run on real Apple hardware.

## Measured AMD/Vulkan reference

The isolated Linux validation used an AMD Radeon 890M (RADV GFX1150), the official `main-vulkan` whisper.cpp image, and the 77,704,715-byte Tiny English model. The exact app adapter transcribed a 22.7-second retained radio clip in 1.022 seconds, returned one 159-character segment, reported the AMD adapter, and loaded its model on `Vulkan0`.

The separate llama.cpp b9637 Vulkan run used public `ggml-org/gemma-3-1b-it-GGUF:Q4_K_M`: all 27 layers offloaded, the Q4_K model occupied about 762 MiB of Vulkan memory, prompt evaluation reached about 320 tokens/second, and generation reached about 105 tokens/second. These are compatibility measurements, not promises for a 12B production model or other hardware.

No packages, services, or storage configuration were changed on the appliance host. Image/model hashes and the remaining limitations are recorded in `PROGRESS.md` and `BUGS.md`.

### Joined current-worker measurement

The exact current commit (`historical-validation`) was placed in a fresh isolated clone. Because the TrueNAS home and `/tmp` mounts are deliberately `noexec`, Python 3.12 CPU wheels were built inside the already-recorded Vulkan image and stored in user-writable executable `/var/tmp`; no host package or mount setting changed. The resulting image/runtime combination passed 113 tests with networking disabled.

The headless Web harness established the real loopback cookie/action-token session and posted a fresh `continue-local` job for 30 seconds of retained radio with `--network none`. No Hugging Face token was supplied: the complete cached Community-1 model loaded, CPU pyannote completed five turns, whisper.cpp retained `cpu` plus `vulkan` availability and explicit `using Vulkan0 backend` evidence for AMD Radeon 890M, and the 1B Q4 Gemma server detected the same Vulkan device. One passage, one embedding, and one daily summary persisted; the final state was Ready to review with zero invented incidents. Container time was 15.139 seconds. The root filesystem was read-only, all capabilities were dropped, `no-new-privileges` and PID limit 1024 were set, and the completed container was removed.
