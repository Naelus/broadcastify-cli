# Product regression gate

## Purpose

Every Broadcastify Desktop native or installer build is a claim about the whole
retained product, not only the files changed most recently. The MSBuild entry
point therefore runs the full offline Python suite before compilation. A failed
test prevents the executable and installer from being produced.

The committed-source guard runs first. This preserves the one-build/one-commit
relationship: tests, source, documentation, version metadata, and release
metadata must already be committed before the build gate can run.

## End-to-end retained workflows

`tests/e2e/test_product_workflows.py` crosses the same JSON worker, SQLite,
filesystem, and loopback HTTP boundaries used by the shipped application. Its
fixtures are temporary and synthetic; they do not authenticate, contact
Broadcastify, download models, or consume archive quota.

The retained catch-up workflow proves that:

- an existing feed can save a prior start date through the current day;
- global Resume queues only absent or unfinished days and skips completed days;
- promotion to a recurring schedule preserves sequential download and original
  archive retention settings;
- a claimed schedule spans the historical boundary through the current day;
- an interrupted run is recovered from persisted state;
- successful recurring runs retain their historical boundary; and
- deleting a feed while a live local front end remains open removes its files,
  evidence, catch-up, and schedule, and the open front end immediately observes
  the empty Library.

The feed-question workflow proves that:

- entire-feed scope resolves the earliest through latest retained day;
- calendar-month scope counts all requested dates and distinguishes
  question-ready, local-processing, and missing-audio days;
- a question uses only current retained transcripts and incidents;
- deterministic category, repeated-location, weekday, and time-block pattern
  records are supplied as citeable evidence;
- follow-up context is bounded and is not treated as evidence;
- missing coverage is appended as a backend-owned limitation;
- cited monthly and whole-feed events retain backend-owned archive dates and
  times, with offsets explicitly distinguished from unavailable clock time; and
- the answer and its evidence records are written to the local Q&A audit table.

Native UI and release contracts remain in the full suite. They cover the About
version surface, media-player detachment before deletion, nonmodal completion,
startup recovery, version consistency, immutable commit metadata, private-file
exclusion, and installer data preservation.

After the offline suite passes and the native executable is compiled, every
desktop build launches that just-built executable in an isolated test-data
root. The app-owned E2E probe does not use desktop-control fallbacks,
UIAutomation, saved credentials, a model provider, or Broadcastify. It renders
and navigates every primary page at nine logical sizes: 520×640, 600×900,
680×840, 720×720, 960×720, 1099×760, 1101×760, 1240×900, and 1600×1000. The
matrix brackets the compact-layout threshold and exercises every named page
scroll host, every bounded list, the Library transcript, the activity log, all
Settings tabs, Credentials, About, and the expanded hardware/profile content in
both directions. It verifies keyboard focus, persistent background status,
calendar-month scope, required event day/time wording, and rejects horizontally
clipped critical controls.

The native docking phase starts from a saved right-edge preference, then proves
real left/right Windows AppBar reservation, a single theme-aware inner border,
square docked corners, suppressed outer frame, active/inactive light/dark
states, reachable dialogs and flyouts, pointer-captured resizing with temporary
reservation release and reclaim, logical-width behavior at 96/120/144/192 DPI,
saved monitor identity, an alternate display when present, shell restart
re-registration, floating and maximized restoration, and work-area cleanup
after both ordinary unpin and an intentionally abrupt isolated process exit. A
JSON report is written atomically, checked by the launcher, and cleaned only
after success. A failed or timed-out probe fails the MSBuild target and retains
its isolated artifacts for diagnosis.

The v0.4.32 pre-package validation passed all 418 offline tests, compiled the
native application with no warnings or errors, and established the original
four-size rendered matrix plus left/right/resize/unpin AppBar baseline. The
expanded matrix and native-frame lifecycle checks above are now mandatory for
every later native or installer build.

## Running the gate

Use the focused set while iterating:

```powershell
.\scripts\run_product_regression_gate.ps1 -Focused
```

Run the authoritative gate directly with:

```powershell
.\scripts\run_product_regression_gate.ps1
```

The authoritative offline command is equivalent to `python -m pytest -q`. A
normal desktop `dotnet build` or `dotnet publish` invokes it automatically after
the committed-source check and before compilation, then runs
`scripts/run_windows_ui_e2e.ps1` against the compiled target. The Windows
release workflow installs the ordinary project plus `dev` test dependencies
before invoking the same build path.

The gate clears inherited account, token, library, database, quota-ledger, and
credential-store environment variables for its child test process. Tests use
only temporary paths and fakes. Live-provider validation and retained private
corpus scoring remain explicit validation activities rather than build steps.
