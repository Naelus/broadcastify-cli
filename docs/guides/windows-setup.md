# Windows setup and first run

The reference Windows experience is a native x64 WinUI 3 application on .NET
10 and Windows App SDK 2.3.1.

## Install a release

Download `BroadcastifyDesktop-<version>-win-x64-setup.exe` from
[GitHub Releases](https://github.com/Naelus/broadcastify-cli/releases) and run
it. The installer is per-user by default and creates a Start-menu shortcut; a
desktop shortcut is optional.

The package includes:

- the self-contained WinUI application and Windows ML helper;
- embedded Python 3.12 and the application worker;
- FFmpeg and FFprobe;
- Windows ML model-build dependencies;
- portable Qwen3-ASR and Sherpa ONNX speaker runtimes.

It deliberately does not bundle multi-gigabyte models, llama.cpp, PyTorch, or
CUDA. Model acquisition remains an explicit action in Setup. The portable CPU
profile works without a system Python installation; CUDA, Community-1,
OpenVINO, Vulkan, and local Gemma have the boundaries documented in
[hardware backends](../hardware-backends.md).

For CUDA faster-whisper or Community-1, **Settings → Processing → Optional
Python runtime** can select `python.exe` from an environment where this project
and its `transcription` optional dependencies are installed. The saved path is
validated on the next app start; an invalid environment falls back to the
bundled portable runtime with a visible warning. `BROADCASTIFY_PYTHON` remains
the headless equivalent. Archive jobs load the selected local audio stack
before contacting Broadcastify, so a missing dependency or model cannot spend
archive requests before failing.

The current installer is unsigned. Windows can show an unknown-publisher or
SmartScreen warning until code signing is configured. Starting trusted-LAN
sharing from a newly installed path can also cause a one-time Windows Firewall
prompt for the bundled Python executable.

## Credentials

Choose **Credentials** in the bottom-left navigation:

- Broadcastify username/password are stored in Windows Credential Locker for
  the current Windows account.
- A Hugging Face read token is stored in its own Credential Locker entry.
- Saved secrets are never pre-filled. The UI shows only the username and a
  short password/token prefix so the user can identify the selected credential.
- Saved values are supplied to foreground and scheduled worker processes
  without being written to ordinary settings or job JSON.

The app performs the same login-page preflight and same-site redirect sequence
as the website. Broadcastify uses HTTP 302 redirects for both successful and
rejected form submissions, so the app verifies the premium session cookie
instead of treating the status code itself as the result.

Use a Hugging Face **read** token. The page links directly to the
[token settings](https://huggingface.co/settings/tokens),
[token documentation](https://huggingface.co/docs/hub/en/security-tokens), and
[Community-1 terms](https://huggingface.co/pyannote/speaker-diarization-community-1).
A token is needed only for the first gated Community-1 download; a complete
model cache can later run offline.

## First-run workflow

1. Open **Credentials** and sign in through the Broadcastify website session.
2. Open **Settings → Setup**. The overview distinguishes detected,
   configured, and execution-verified stages.
3. Choose the hardware profile. The packaged portable paths are available on
   a clean machine; install optional accelerator dependencies only when needed.
4. Use **Verify profile**. It runs generated-input transcription, speaker
   labeling, and analysis without consuming Broadcastify archive quota.
5. Open **New archive**, find a feed, select an inclusive date range, and keep
   combination enabled for transcription or speaker labels.
6. Use **Local library** to continue interrupted or partially processed days.
   **Improve speakers** can replace preview labels without repeating ASR.
7. Use **Review & Ask** for evidence clips, daily/weekly briefs, and range
   questions. Use **Area watch** for radius discovery and regional leads.

Desktop schedules run while the desktop application is open. Use the managed
Web/TrueNAS service when a continuously supervised schedule is required.

## Data, upgrades, and uninstall

The installed application uses:

- program files: `%LOCALAPPDATA%\Programs\Broadcastify Desktop`
- settings, quota ledger, logs, and managed models:
  `%LOCALAPPDATA%\Broadcastify Desktop`
- default installed library:
  `%LOCALAPPDATA%\Broadcastify Desktop\archives`

Upgrades use a stable installer identity. Uninstall removes the application
runtime but intentionally never removes the data directory. A source build now
persists its library as an absolute path. When upgrading an older source build
whose setting was the relative value `archives`, the installed app can
reconnect to a valid previous library recorded in its own activity history and
then persists that absolute location.

See [storage and resume](../reference/storage-and-resume.md) before moving a
library.

## Build from source

Source development still uses Python 3.12, FFmpeg, the .NET 10 SDK, and the
desired optional inference backends. The tested NVIDIA environment is:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install torch==2.11.0+cu128 torchaudio==2.11.0+cu128 --index-url https://download.pytorch.org/whl/cu128
.\.venv\Scripts\python.exe -m pip install -e ".[transcription,analysis,dev]"
dotnet build .\BroadcastifyCli.WinUI\BroadcastifyCli.WinUI.csproj -c Release
& ".\BroadcastifyCli.WinUI\bin\Release\net10.0-windows10.0.26100.0\win-x64\Broadcastify Desktop.exe"
```

Optional dependency groups include `qwen`, `portable-diarization`, `openvino`,
and `windowsml`.

## Build an installer

Install the pinned Inno Setup compiler once, then build:

```powershell
.\scripts\install_inno_setup.ps1
.\scripts\build_windows_installer.ps1
```

The public artifact is written to
`dist\windows\BroadcastifyDesktop-<version>-win-x64-setup.exe`. Verified Python
and FFmpeg archives plus an exact Python dependency constraints file make the
portable runtime repeatable. Tagged `v*` pushes run the Windows release
workflow and attach this asset to the existing GitHub release.

For a private local build only:

```powershell
.\scripts\build_windows_installer.ps1 -BundleLocalEnv
```

This copies the ignored `.env` as plaintext into that private application
runtime. Never distribute a private installer. Public builds reject a stale or
unexpected `broadcastify-desktop.env`.
