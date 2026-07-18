# Windows publish and runtime layout

Last verified: July 18, 2026.

The Windows reference shell can now be produced with `dotnet publish` without merging the Windows ML helper into the WinUI dependency graph:

```powershell
.\scripts\verify_windows_publish.ps1 `
  -OutputDirectory .\BroadcastifyCli.WinUI\bin\Publish\win-x64
```

The verifier performs a real Release publish, checks the unpackaged WinUI compiled XAML and PRI resources, confirms the private environment is absent, and executes `windowsml\BroadcastifyCli.WindowsML.exe --probe`. The helper is kept in a namespaced subdirectory with its complete runtime so duplicate Windows App SDK filenames cannot collide with the desktop shell. The desktop passes that exact helper path to every Python child unless `WINDOWS_ML_HELPER_PATH` was explicitly configured by the user.

Both native projects currently target the maintained Windows App SDK 1.8 servicing line at **1.8.10 / 1.8.260710003** (released July 14, 2026). This updates the validated 1.8 runtime without silently crossing to a newer major Windows ML/runtime contract. The current isolated Release build succeeds with zero warnings or errors, and the bundled helper still performs a real CPU Whisper decode; GPU execution remains gated by the model/provider tests in [hardware-backends.md](hardware-backends.md).

For the owner's private build only:

```powershell
.\scripts\verify_windows_publish.ps1 `
  -OutputDirectory .\BroadcastifyCli.WinUI\bin\Private\publish-win-x64 `
  -BundleLocalEnv
```

That opt-in verifies the published `broadcastify-desktop.env` against the ignored repository `.env` by hash without printing either file. Re-running the verifier without `-BundleLocalEnv` against the same output removes and rejects a stale private environment.

## Current boundary

This is a verified runnable publish, but not yet a standalone installer. `WorkerClient` still locates the source tree (`pyproject.toml`) and uses its configured Python/`.venv`; model caches also remain external. A distributable installer must supervise or bundle Python and the selected optional dependencies, choose per-hardware components, and provide an explicit model/cache manager. Do not describe the current folder as a standalone package.
