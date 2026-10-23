# Review the installed app and a committed candidate

The installed Windows app provides **About & Support → Open isolated preview**.
It opens a separate Broadcastify Desktop window seeded with synthetic sample
data, using the bundled Python runtime and FFmpeg. The preview has its own fresh
temporary Library and settings; model/archive commands, credentials, scheduler,
and LAN services are unavailable. It does not replace or control the production
window. The temporary sample directory is retained for inspection.

The browser companion has the same **About → Open isolated preview** action. It
opens a saved-data-only preview under a separate temporary preview URL. The
preview allows browsing and acknowledging synthetic incidents; download/model,
credential, schedule, and LAN routes are disabled. The browser preview runs
inside the existing web listener and does not make provider calls.

Use the following developer workflow only when validating a new committed
candidate; ordinary users can use the installed About action above.

Build a committed candidate locally, then open a separate preview window:

```powershell
dotnet build .\BroadcastifyCli.WinUI\BroadcastifyCli.WinUI.csproj -c Release --no-restore
.\scripts\launch_windows_ui_smoke.ps1 -WorkflowPreview -Executable '.\BroadcastifyCli.WinUI\bin\Release\net10.0-windows10.0.26100.0\win-x64\Broadcastify Desktop.exe'
```

The build does not publish a release or replace the installed app. The script
preview uses a fresh temporary settings directory, SQLite database and sample
library. It disables archive/model worker commands and does not start the
scheduler or LAN service. Closing the preview leaves the installed app running
independently. FFmpeg is needed to generate two seconds of silent, playable
sample audio.

The script launch prints the preview's process ID and data directory and saves
`smoke-session.json` there. Use that directory to identify the preview's files;
the banner identifies the window as an isolated synthetic-data preview.

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

To reopen the same preview, launch its recorded executable with `--workflow-preview` and the same
`BROADCASTIFY_DESKTOP_TEST_DATA_ROOT`, `BROADCASTIFY_LIBRARY_ROOT`,
`BROADCASTIFY_SECURE_ANALYSIS_DB`, `BROADCASTIFY_QUOTA_LEDGER`, `BROADCASTIFY_CREDENTIAL_STORE`,
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

Resume verification also invokes the real worker planner while repeating Resume
and Catch up clicks. Only one options dialog may open; canceling it must start no
work and restore the controls. A worker result arriving after window closure must
not open a dialog or start processing. These checks protect the observed WinUI
crashes from overlapping dialogs and late resume results.

Record the source commit, suite result, native report, and manual preview outcome
with the candidate. The full-workflow estimate is based on local successful
timing samples for the current machine and processing profile. New profiles
begin in a learning state; provisional input sizes are labeled, and source or
quota waits have no confident finish time. A green build or synthetic preview
does not establish real-world processing speed or validate an installed
release. Do not use production archive or model work solely to obtain timing
samples.

In the browser, also inspect a measured ETA and expand its remaining stage
budgets, plus a learning/waiting estimate with some calibrated stages. A fresh
profile with no timing samples does not exercise duration rendering. Confirm
that a formatting error cannot masquerade as a master disconnect.
