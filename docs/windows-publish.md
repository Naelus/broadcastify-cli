# Windows publish, installer, and release layout

Last verified: August 1, 2026.

## Raw native publish

The development verifier still produces and probes the native WinUI publish:

```powershell
.\scripts\verify_windows_publish.ps1 `
  -OutputDirectory .\BroadcastifyCli.WinUI\bin\Publish\win-x64
```

It checks compiled XAML/PRI resources, the namespaced Windows ML helper, and
normal/private environment isolation. This raw folder intentionally remains a
developer artifact that uses the repository Python environment.

## Standalone installer

The consumer build layers a portable runtime over the verified native publish:

```powershell
.\scripts\install_inno_setup.ps1
.\scripts\build_windows_installer.ps1
```

Output:

```text
dist/windows/BroadcastifyDesktop-0.4.13-win-x64-setup.exe
```

The application stage contains:

```text
Broadcastify Desktop.exe
windowsml/
runtime/
  python/
    python.exe
    Lib/site-packages/
  tools/
    ffmpeg.exe
    ffprobe.exe
build-manifest.json
LICENSE
THIRD-PARTY-NOTICES.txt
```

The main application and namespaced Windows ML helper each carry the .NET 10
runtime files they require. Public builds disable debug symbols and reject any
staged PDB so local build paths cannot enter a release installer.

`WorkerClient` detects `runtime/python/python.exe`, stops searching for
`pyproject.toml`, and runs workers from the writable per-user data directory.
Relative output paths therefore resolve under
`%LOCALAPPDATA%\Broadcastify Desktop`, never under the program directory.
Bundled workers disable user-site packages and bytecode writes. FFmpeg receives
an explicit path. The build removes pip's generated console-launcher directory:
those stubs contain an absolute path to the build interpreter, while every
installed worker is launched portably as a module through the bundled
`python.exe`.

Python 3.12.10 and the FFmpeg 8.1.2 essentials archive are SHA-256 pinned.
Python wheel versions are exact in
`installer/windows-runtime-constraints.txt`. The package includes Windows ML,
Qwen3-ASR, portable Sherpa diarization, and IANA timezone data used to map
archive listings to feed-local cache identities, but excludes the multi-
gigabyte CUDA/PyTorch stack. Models are not bundled.

## Installer behavior

The Inno Setup package:

- installs per-user under
  `%LOCALAPPDATA%\Programs\Broadcastify Desktop`;
- uses a stable `AppId` for in-place upgrades;
- creates a Start-menu shortcut and offers an optional desktop shortcut;
- visibly offers current-user Windows startup during interactive setup and
  selects it by default;
- leaves startup and post-install launch off for a fresh silent install unless
  `/ENABLESTARTUP` and `/LAUNCHAFTERINSTALL` are explicitly supplied;
- preserves an existing startup choice during a flagless silent upgrade;
- registers one normal uninstall entry;
- removes generated application-runtime residue on uninstall;
- marks `%LOCALAPPDATA%\Broadcastify Desktop` as never uninstall;
- replaces the bundled Python and Windows ML runtime trees on upgrade so
  removed dependencies and old package metadata cannot survive;
- never copies `.env` in a public build;
- contains no PDB/debug-symbol files or developer build paths;
- removes root debug symbols left by an earlier developer/private build during
  an in-place public upgrade;
- deletes a stale `broadcastify-desktop.env` when a public build upgrades a
machine that previously ran an owner-only private build.

The recommended silent deployment is:

```powershell
.\BroadcastifyDesktop-0.4.13-win-x64-setup.exe `
  /VERYSILENT /SUPPRESSMSGBOXES /NORESTART `
  /ENABLESTARTUP /LAUNCHAFTERINSTALL
```

The launched app receives `--prompt-setup`: if the archive account or selected
gated speaker-label path is not configured, it remains visible and directs the
user to Credentials while scheduled jobs wait. Normal login launches use
`--startup --prompt-setup` and minimize only after readiness and
interrupted-schedule recovery checks.

The retained lifecycle test completed install, native launch with the bundled
Python child, same-version upgrade, uninstall, and clean reinstall. Settings and
all data files were unchanged; the corrected uninstall removed the complete
program directory; and the clean launch generated no Python cache directories.

The current local installer is unsigned. A release certificate can be added
without changing the layout; signing should cover the application executable
and final setup executable. Until then, Windows reputation warnings are
expected. Trusted-LAN sharing can cause a separate one-time firewall consent
for the bundled Python path.

## Private build

`-BundleLocalEnv` is an explicit owner-only escape hatch:

```powershell
.\scripts\build_windows_installer.ps1 -BundleLocalEnv
```

The build verifies the copied ignored `.env` by hash without printing it.
It defaults to `dist/windows-private` and is forbidden from writing into the
public `dist/windows` release tree. Never distribute that artifact. The default
public and GitHub Actions paths reject any bundled private environment.

## GitHub release workflow

`.github/workflows/windows-release.yml` runs on `v*` tags or manual dispatch.
It installs the checksum- and signature-verified Inno Setup 7.0.2 compiler,
builds the installer, uploads a workflow artifact, and creates a tagged GitHub
release with generated notes or replaces its existing installer asset.

The workflow never creates a tag and requires the tag/build version to match
`pyproject.toml`. It contains no credentials; repository `GITHUB_TOKEN` supplies
only release permission.
