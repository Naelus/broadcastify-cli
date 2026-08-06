# Radio Archive Intelligence

Radio Archive Intelligence is a local, evidence-first application for finding
Broadcastify feeds, retaining premium archives, transcribing and diarizing radio
traffic, and reviewing incident, daily, weekly, and regional summaries against
the exact source audio.

Windows uses a native WinUI 3 desktop app. The same Python backend also provides
a browser UI for Windows, Linux, macOS, and TrueNAS, plus command-line tools.

> This project signs in through Broadcastify's website and uses the same private
> web endpoints as its archive page. It does not use the official Broadcastify
> API. The current [Broadcastify Terms and
> Conditions](https://www.broadcastify.com/terms/) require a separate advance
> license for programmatic/automated access and AI/ML processing, regardless of
> scale or personal motive; Premium service alone is not that license. Use this
> software only with authorization that covers the intended workflow.

## What it does

- Searches feeds by agency, place, or ZIP and builds explicit radius-based area
  profiles.
- Downloads archive blocks conservatively with cache reuse, sequential pacing,
  resumable jobs, a durable 240-of-250 rolling-window guard, and optional
  trusted-LAN sharing.
- Saves daily schedules for explicitly selected feeds; each run revisits a
  short recent window, can retain an older catch-up boundary until every gap
  is complete, and continues at rolling quota release times instead of
  sleeping for a fixed day.
- Combines each feed-day into one continuous timeline before transcription and
  speaker labeling.
- Runs local Whisper-family ASR, Community-1 or a fast CPU speaker preview, BGE
  retrieval, and quantized Gemma analysis by default.
- Stores audio, transcripts, speaker turns, incidents, summaries, evidence
  clips, and processing state locally and independently.
- Provides a local library with per-feed backlog/last-known source coverage,
  selector-driven quota-safe resume, exact-clip review, named-feed evidence
  chat, seven-day briefs, and evidence-backed regional story leads.
- Keeps acquisition, transcription, and diarization in a background pipeline so
  non-conflicting Library, review, schedule, and chat actions remain usable.

See [FEATURES.md](FEATURES.md) for the current capability matrix.

## Choose an interface

| Interface | Best for | Start here |
|---|---|---|
| WinUI 3 desktop | Tested Windows reference experience | [Windows setup](docs/guides/windows-setup.md) |
| Browser UI | Windows, Linux, macOS, TrueNAS, or trusted-LAN access | [Web UI guide](docs/web-ui.md) |
| CLI | Automation, diagnostics, and headless workflows | [CLI guide](docs/guides/cli.md) |

## Quick start on Windows

Download `BroadcastifyDesktop-<version>-win-x64-setup.exe` from
[Releases](https://github.com/Naelus/broadcastify-cli/releases). The per-user
installer includes the native WinUI app, Python worker, FFmpeg, Windows ML
helper, the .NET desktop runtime, portable CPU/Qwen/Sherpa runtimes, and a
checksum-verified installer for the optional NVIDIA CUDA audio runtime. It
preserves application data across upgrades and uninstall. Large dependencies
and models remain explicit first-use downloads.
Public releases contain no `.env`, account password, API token, or saved
session. Credentials entered in the app are encrypted for the current Windows
account and remain outside the installation directory.
Interactive setup visibly offers **Start with Windows** and enables it by
default so saved schedules keep running; the same toggle remains available
under **Settings → Setup**.

The installer is currently unsigned, so Windows may show an unknown-publisher
warning until release signing is configured. LAN sharing may also request a
one-time Windows Firewall allowance for the bundled Python path.

For source development instead:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install torch==2.11.0+cu128 torchaudio==2.11.0+cu128 --index-url https://download.pytorch.org/whl/cu128
.\.venv\Scripts\python.exe -m pip install -e ".[transcription,analysis,dev]"
dotnet build .\BroadcastifyCli.WinUI\BroadcastifyCli.WinUI.csproj -c Release
& ".\BroadcastifyCli.WinUI\bin\Release\net10.0-windows10.0.26100.0\win-x64\Broadcastify Desktop.exe"
```

In the app:

1. Open **Credentials** to enter the Broadcastify website login and optional
   Hugging Face read token. Windows encrypts them for the current account and
   shows only a short prefix afterward.
2. Open **Settings → Setup**, review **Start with Windows**, and configure
   storage, processing, and analysis. For the NVIDIA profile, explicitly choose
   **Install runtime** in the **Packaged CUDA runtime** card; allow about 5.9 GB.
   Cancellation retains a verified cache and **Resume install** continues it.
3. Use **Verify profile** to execute all three model stages on generated local
   input without consuming archive quota.
4. Open **New archive**, search for a feed, and select an inclusive date range.
5. Optionally choose **Schedule this feed** to save a daily time and recent-day
   lookback for that specific result.
6. Leave combination enabled when transcribing or adding speaker labels.
7. Review completed or interrupted days in **Local library**. Expand feed
   coverage to see scheduled gaps and last-known provider blocks, or use
   **Resume / prioritize…** to select feeds, local work, guarded downloads, and
   ordering; a confirmed **Delete feed** action removes a feed's local evidence.
8. Use **Review & Ask** to choose a feed by name and chat with cited evidence,
   or use **Area watch** for clips, weekly briefs, and regional leads.

The full prerequisite, first-run, model, and private-build instructions are in
[docs/guides/windows-setup.md](docs/guides/windows-setup.md).

## Local data and recovery

Installed Windows archive data defaults to
`%LOCALAPPDATA%\Broadcastify Desktop\archives`; source and server runs default
to `archives/`. Analysis lives in the selected library as
`broadcastify-analysis.sqlite3`. Downloads, combined audio,
transcription, diarization, embeddings, incidents, and summaries have separate
cache identities. A retry starts at the first missing or invalid stage rather
than repeating successful work.

Changing an evidence-analysis policy does not require another download,
transcription, or diarization pass. Current private-use analysis preserves names
explicitly spoken in cited evidence while masking phone numbers, dates of birth,
emails, and long numeric identifiers in derived/display text.

See [storage and resume](docs/reference/storage-and-resume.md) before moving or
backing up a library.

## Documentation

The [documentation index](docs/README.md) organizes setup guides, feature
behavior, design decisions, deployment, operations, storage, and validation.
Useful starting points:

- [Pipeline and local model choices](docs/decisions/model-stack.md)
- [Why WinUI 3 plus a browser companion](docs/decisions/ui-and-platforms.md)
- [Evidence and privacy policy](docs/decisions/evidence-and-privacy.md)
- [Archive acquisition and quota behavior](docs/features/archive-acquisition.md)
- [Transcription and diarization](docs/features/audio-processing.md)
- [Incident analysis and evidence](docs/features/evidence-analysis.md)
- [Area Watch and regional leads](docs/features/area-watch.md)
- [Hardware backends and measured limits](docs/hardware-backends.md)
- [Credential storage and precedence](docs/reference/credentials.md)
- [TrueNAS Apps deployment](deploy/truenas/README.md)

## Development tracking

- [FEATURES.md](FEATURES.md) is the durable capability/status matrix.
- [GitHub Issues](https://github.com/Naelus/broadcastify-cli/issues) tracks
  reproducible defects and actionable enhancements.
- Local `GOAL.md` and `PROGRESS.md` files may be used as temporary working notes
  during active development; they are intentionally ignored by Git.

Run the complete test suite with:

```powershell
.\.venv\Scripts\python.exe -m pytest
```

The historical repository name `broadcastify-cli` is retained for compatibility;
the product is now broader than a downloader.
