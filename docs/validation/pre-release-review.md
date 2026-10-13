# Inspect a Windows candidate before release

Build a committed candidate locally, then open a separate preview window:

```powershell
dotnet build .\BroadcastifyCli.WinUI\BroadcastifyCli.WinUI.csproj -c Release --no-restore
.\scripts\launch_windows_ui_smoke.ps1 -WorkflowPreview -Executable '.\BroadcastifyCli.WinUI\bin\Release\net10.0-windows10.0.26100.0\win-x64\Broadcastify Desktop.exe'
```

The build runs the full offline suite and the native UI verification before it
can pass. It does not publish a release or replace the installed app. The preview
uses a fresh temporary settings directory, SQLite database and sample library.
It disables archive/model worker commands and does not start the scheduler or
LAN service. Closing the preview leaves the installed app running independently.
FFmpeg is needed to generate two seconds of silent, playable sample audio.

The launch command prints the preview's process ID and data directory and saves
`smoke-session.json` there. Use that directory to identify the preview's files;
the banner also identifies the window as a pre-release preview.

## Review these outcomes

1. **Library:** two sample feeds, three unfinished feed-days, and one completed
   day. One date has saved analysis but needs its search index, another needs
   transcription, and one date has no retained audio. Today's empty listing is
   shown separately as awaiting publication. Future source audio is not counted
   as completed processing.
2. **Review:** the full queue remains visible after switching pages. A day with
   saved analysis remains readable while its search index is unfinished.
3. **What changed:** choose the first sample feed and refresh the last seven
   days. Two explicitly synthetic incidents appear with quoted evidence. Mark
   the batch caught up, refresh, and reopen the same preview data: the marked
   content stays read. The sample audio is silence, not a real radio recording.
4. **Isolation:** attempting a model job shows a preview-disabled message.
   Production settings, quota history, credentials and archive files are outside
   this preview. No inference or Broadcastify requests are needed for this review.

To reopen the same preview, launch its recorded executable with the same
`BROADCASTIFY_DESKTOP_TEST_DATA_ROOT`, `BROADCASTIFY_LIBRARY_ROOT`,
`BROADCASTIFY_SECURE_ANALYSIS_DB`, `BROADCASTIFY_QUOTA_LEDGER`,
`BROADCASTIFY_DESKTOP_E2E_ISOLATED=1`, and `BROADCASTIFY_DESKTOP_PREVIEW=1` process
environment values from the launch script. A fresh launch command creates fresh
samples, so its acknowledgements intentionally start over.

## Evidence required before publishing

The offline regression verifies newest-first acquisition with oldest-first
bounded processing, including newly growing recent audio; durable index-only
resume; metadata-only import preservation; actual worker restart/acknowledgement
behavior; and the authenticated follower's view of the master's queue. Native
verification checks Review refresh without replacing playback selection,
current-step progress without overwriting the full queue, preview model-command
rejection, and the existing compact/docked layout matrix.

Record the source commit, suite result, native report, and manual preview outcome
with the candidate. A green build is not evidence of real-world model speed or
an overall ETA: end-to-end timing calibration remains outstanding. Do not resume
paused production model processing merely to obtain that measurement.
