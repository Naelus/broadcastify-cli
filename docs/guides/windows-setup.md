# Windows setup and first run

The native Windows application is the reference experience. It is an unpackaged
x64 WinUI 3 application built on .NET 10 and Windows App SDK 2.3.1. It still
uses the repository Python environment; the verified publish is not yet a
standalone installer.

## Prerequisites

- Windows 10 version 1809 or later, x64
- Python 3.12
- FFmpeg
- .NET 10 SDK
- llama.cpp for the default local analysis provider
- A Broadcastify premium account
- A Hugging Face read token for the first Community-1 download only

Install the basic tools:

```powershell
winget install --id Python.Python.3.12 -e
winget install --id Gyan.FFmpeg.Shared -e
winget install --id Microsoft.DotNet.SDK.10 -e
winget install --id ggml.llamacpp -e
```

## Python environment

The tested NVIDIA setup uses CUDA 12.8 PyTorch wheels:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install torch==2.11.0+cu128 torchaudio==2.11.0+cu128 --index-url https://download.pytorch.org/whl/cu128
.\.venv\Scripts\python.exe -m pip install -e ".[transcription,analysis,dev]"
```

Optional groups:

- `.[qwen]` for the fast CPU Qwen3-ASR preview
- `.[openvino]` for OpenVINO ASR
- `.[windowsml]` for Windows ML model export/build support

## Build and run

```powershell
dotnet build .\BroadcastifyCli.WinUI\BroadcastifyCli.WinUI.csproj -c Release
& ".\BroadcastifyCli.WinUI\bin\Release\net10.0-windows10.0.26100.0\win-x64\Broadcastify Desktop.exe"
```

## First-run workflow

1. Open **Settings → Setup**. The overview distinguishes detection,
   configuration, and verified execution for the account, storage,
   transcription, speaker, and analysis stages.
2. Enter the premium Broadcastify login. **Save this login securely** stores it
   in Windows Credential Locker for the current Windows account. Session cookies
   remain in the ignored `cookies.json` file.
3. Keep **Community-1 — accuracy default** on the CUDA reference path. Accept
   the Community-1 model terms and supply a read token for its first download.
   The token is not required after the complete model cache exists.
4. Use **Verify profile**. It runs generated-input transcription, speaker
   labeling, and analysis sequentially, releases model memory between stages,
   and does not touch Broadcastify archive quota.
5. Open **New archive**, search for a feed, select the inclusive date range, and
   keep **Create combined MP3** enabled for transcription/diarization.
6. To revisit a feed automatically, select that search result and choose
   **Schedule this feed**. Pick a local daily time and recent-day lookback.
   Schedules run while the desktop app is open; use the managed Web/TrueNAS
   service for continuous unattended scheduling.
7. Open **Local library** to continue an interrupted or partially processed
   day. **Improve speakers** replaces preview labels with Community-1 without
   repeating transcription.
8. Use **Review & Ask** for incidents, evidence clips, daily/weekly briefs, and
   range questions. Use **Area watch** for radius discovery and regional leads.

Model self-tests prove that the selected runtime actually executes; generated
silence is not a radio-accuracy benchmark. See [hardware backends](../hardware-backends.md).

## Environment file

Copy `.env-example` to the ignored `.env` when environment-based configuration
is preferred. The important account variables are:

```dotenv
BROADCASTIFY_USERNAME=""
BROADCASTIFY_PASSWORD=""
HUGGINGFACE_TOKEN=""
```

Runtime/model overrides and hosted processing defaults are documented inline in
`.env-example`. Never commit `.env`, `cookies.json`, or a private publish.

## Verified publish

Normal publish:

```powershell
.\scripts\verify_windows_publish.ps1 `
  -OutputDirectory .\BroadcastifyCli.WinUI\bin\Publish\win-x64
```

Private personal publish with the ignored `.env` bundled as plaintext inside
the dedicated output directory:

```powershell
.\scripts\verify_windows_publish.ps1 `
  -OutputDirectory .\BroadcastifyCli.WinUI\bin\Private\publish-win-x64 `
  -BundleLocalEnv
```

Do not distribute a private publish. The verifier checks environment isolation,
compiled WinUI resources, and the bundled Windows ML helper. See
[Windows publish layout](../windows-publish.md).
