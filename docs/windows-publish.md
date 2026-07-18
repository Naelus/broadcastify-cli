# Windows publish and runtime layout

Last verified: July 18, 2026.

The Windows reference shell can now be produced with `dotnet publish` without merging the Windows ML helper into the WinUI dependency graph:

```powershell
.\scripts\verify_windows_publish.ps1 `
  -OutputDirectory .\BroadcastifyCli.WinUI\bin\Publish\win-x64
```

The verifier performs a real Release publish, checks the unpackaged WinUI compiled XAML and PRI resources, confirms the private environment is absent, and executes `windowsml\BroadcastifyCli.WindowsML.exe --probe`. The helper is kept in a namespaced subdirectory with its complete runtime so duplicate Windows App SDK filenames cannot collide with the desktop shell. The desktop passes that exact helper path to every Python child unless `WINDOWS_ML_HELPER_PATH` was explicitly configured by the user.

Both native projects now target the current stable [Windows App SDK **2.3.1** release](https://github.com/microsoft/WindowsAppSDK/releases/tag/v2.3.1) and `Microsoft.Windows.SDK.BuildTools` **10.0.28000.2270**. The supported metapackage graph resolves `Microsoft.Windows.AI.MachineLearning` and `Microsoft.WindowsAppSDK.ML` **2.1.74**. The earlier 1.8.10 build remains historical validation evidence, not the current runtime.

Both projects build with zero warnings or errors. The upgraded native app completed the real CUDA/Community-1/Gemma setup proof in **14.9 seconds**, and the helper completed a real CPU FP32 Whisper decode. A current joined Windows ML profile completed in **13.546 seconds**, including its **0.437-second** CPU ASR decode. GPU execution remains gated by the model/provider tests in [hardware-backends.md](hardware-backends.md).

`dotnet list package --outdated --include-transitive` reports a few independently versioned transitive components newer than the minimum versions selected by Windows App SDK 2.3.1. They are intentionally not overridden one-by-one: the Windows App SDK metapackage owns the tested WinUI, Windows ML, WebView2, and MSIX component graph.

For the owner's private build only:

```powershell
.\scripts\verify_windows_publish.ps1 `
  -OutputDirectory .\BroadcastifyCli.WinUI\bin\Private\publish-win-x64 `
  -BundleLocalEnv
```

That opt-in verifies the published `broadcastify-desktop.env` against the ignored repository `.env` by hash without printing either file. Re-running the verifier without `-BundleLocalEnv` against the same output removes and rejects a stale private environment.

The current migration repeated a normal → private → normal → private cycle. The final private publish retained the matching ignored environment, the normal pass proved stale-private removal, and the published desktop visibly completed all three setup model stages. The bundled helper's read-only provider inspection reported `WebGpuExecutionProvider:NotPresent` and certified installed `NvTensorRTRTXExecutionProvider:NotReady`; no provider was downloaded, acquired, or registered.

## Current boundary

This is a verified runnable publish, but not yet a standalone installer. `WorkerClient` still locates the source tree (`pyproject.toml`) and uses its configured Python/`.venv`; model caches also remain external. A distributable installer must supervise or bundle Python and the selected optional dependencies, choose per-hardware components, and provide an explicit model/cache manager. Do not describe the current folder as a standalone package.
