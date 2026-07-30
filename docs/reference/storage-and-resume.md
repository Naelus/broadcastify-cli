# Storage and resumability

## Default locations

Repository/development defaults:

- archive library: `archives/`
- analysis database: `archives/broadcastify-analysis.sqlite3`
- session cookies: ignored `cookies.json`
- repository environment: ignored `.env`

Native Windows state:

- settings: `%LOCALAPPDATA%\Broadcastify Desktop\settings.json`
- activity log: `%LOCALAPPDATA%\Broadcastify Desktop\activity.log`
- exception log: `%LOCALAPPDATA%\Broadcastify Desktop\crash.log`
- archive request ledger: `%LOCALAPPDATA%\Broadcastify Desktop\archive-quota.sqlite3`
- installed default library: `%LOCALAPPDATA%\Broadcastify Desktop\archives`
- managed model root: `%LOCALAPPDATA%\Broadcastify Desktop\models`
- program runtime: `%LOCALAPPDATA%\Programs\Broadcastify Desktop`

The program runtime is replaceable and is removed by uninstall. The data
directory is explicitly retained. New settings persist the selected library as
an absolute path. The installed app can reconnect an older source build's
relative `archives` setting to a valid prior library recorded in the local
activity history; it does not copy or delete that library.

Native secrets are separate from these files. Windows Credential Locker stores
the Broadcastify password, Hugging Face token, and any explicitly remembered
analysis key for the current account.

Linux service defaults:

- data/working directory: `~/.local/share/radio-archive`
- configuration: `~/.config/radio-archive`
- archive request ledger: `.broadcastify-archive-quota.sqlite3` in the service
  working directory unless `BROADCASTIFY_QUOTA_LEDGER` selects another durable
  installation-local path

The Linux and TrueNAS layouts are configurable so an existing library can be
adopted without moving it.

## Feed-day layout

```text
archives/
  5318/
    20260713/
      202607130000-…-5318.mp3
      combined_5318_20260713.mp3
      combined_5318_20260713.manifest.json
      evidence-clips/
        5318_2026-07-13_I42.mp3
      transcripts/
        combined_5318_20260713.json
        combined_5318_20260713.txt
```

Text transcript entries look like:

```text
[00:03:21.420] SPEAKER_01: Unit 12, copy that.
```

Speaker values are anonymous acoustic clusters, not identified people or radio
units.

## SQLite contents

The persistent database stores:

- feed catalog and feed-days;
- timestamped transcript segments and FTS index;
- passages and embeddings;
- incidents and exact evidence references;
- daily and weekly summaries;
- question history;
- area profiles, acquisition queues, story digests, and per-feed schedules;
  and
- model-window checkpoints and prompt/source identities.

SQLite is derived from retained transcripts/audio but is worth backing up with
the archive tree because it also contains review and queue state.

The archive request ledger is separate from the evidence database. Back it up
with the installation state and never delete or clone it to obtain more request
capacity. Each installation mints its own stable ledger identity; ledgers are
not synchronized over LAN.

## Stage identities

Each expensive stage has an independent cache contract:

- archive blocks: feed/timezone/source identity and non-empty local file;
- combined audio: exact ordered source list, modification times, and timeline
  manifest;
- ASR: audio/model/engine/backend plus rendered-text integrity;
- diarization: audio signature, model/runtime/quality/chunk policy;
- embeddings: model and passage identity;
- incidents: transcript SHA, model, prompt version, and window fingerprint;
- summaries/area digests: current incident versions and source fingerprint.

A later stage cannot make an earlier stale stage look complete.

## Interruption behavior

- Downloads use partial files and publish only completed MP3s.
- Combined audio is written to a sibling partial and atomically replaces the
  prior recording only after validation.
- Transcript JSON/TXT and manifests use atomic replacement.
- Portable diarization checkpoints every completed chunk and retains the
  checkpoint until final-cache commit.
- Incident analysis checkpoints every completed model window.
- Feed schedules store last-run and next-safe quota state; a startup recovery
  defers any schedule left running for one collision-avoidance minute and
  resumes from retained work. Windows login startup runs this recovery before
  claiming scheduled work.
- Native settings are atomically written; activity/crash logs append.

An application or machine interruption may leave a partial/checkpoint, but it
must not overwrite the last known-good final artifact.

## Moving and backing up a library

Back up the complete archive root and SQLite database together. Stop active
processing or copy from a filesystem snapshot so the pair is consistent.
Evidence paths are rebased against the configured archive root when a library
moves between Windows, Linux, and TrueNAS; paths outside that root are not
served by the Web UI.

Do not delete raw archive blocks after combination unless you intentionally give
up exact cache reconstruction and LAN sharing. The normal application preserves
them.
