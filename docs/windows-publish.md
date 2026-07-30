# Windows publish, installer, and release layout

Last verified: July 29, 2026.

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
dist/windows/BroadcastifyDesktop-0.4.1-win-x64-setup.exe
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

`WorkerClient` detects `runtime/python/python.exe`, stops searching for
`pyproject.toml`, and runs workers from the writable per-user data directory.
Relative output paths therefore resolve under
`%LOCALAPPDATA%\Broadcastify Desktop`, never under the program directory.
Bundled workers disable user-site packages and bytecode writes. FFmpeg receives
an explicit path.

Python 3.12.10 and the FFmpeg 8.1.2 essentials archive are SHA-256 pinned.
Python wheel versions are exact in
`installer/windows-runtime-constraints.txt`. The package includes Windows ML,
Qwen3-ASR, and portable Sherpa diarization dependencies but excludes the
multi-gigabyte CUDA/PyTorch stack. Models are not bundled.

## Installer behavior

The Inno Setup package:

- installs per-user under
  `%LOCALAPPDATA%\Programs\Broadcastify Desktop`;
- uses a stable `AppId` for in-place upgrades;
- creates a Start-menu shortcut and offers an optional desktop shortcut;
- registers one normal uninstall entry;
- removes generated application-runtime residue on uninstall;
- marks `%LOCALAPPDATA%\Broadcastify Desktop` as never uninstall;
- never copies `.env` in a public build.

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
Never distribute that artifact. The default and GitHub Actions paths are always
public builds and reject any bundled private environment.

## GitHub release workflow

`.github/workflows/windows-release.yml` runs on `v*` tags or manual dispatch.
It installs the checksum- and signature-verified Inno Setup 7.0.2 compiler,
builds the installer, uploads a workflow artifact, and creates a tagged GitHub
release with generated notes or replaces its existing installer asset.

The workflow never creates a tag and requires the tag/build version to match
`pyproject.toml`. It contains no credentials; repository `GITHUB_TOKEN` supplies
only release permission.
