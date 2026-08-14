# Broadcastify AI Assistant engineering guide

The Codex project is **Broadcastify AI Assistant**. The shipped Windows product
name remains **Broadcastify Desktop**. Keep the existing repository and release
identity unchanged.

## Change discipline

- Preserve retained archives, transcripts, checkpoints, analysis databases,
  quota ledgers, settings, and encrypted credentials across upgrades and fixes.
- Never stage `.env`, credential/session stores, local archives, model caches,
  diagnostics containing secrets, or machine-specific working notes.
- Test archive planning, resume, and quota behavior with temporary files,
  fakes, or cached fixtures. Do not spend live Broadcastify requests in tests.
- Keep archive requests sequential and behind the persistent 240-request rolling
  guard, leaving the documented manual-use reserve intact. Cached/LAN reuse and
  local processing must happen before a website request.
- Destructive Library actions require a clear non-default confirmation, strict
  path containment, and database/filesystem recovery behavior.
- Keep worker output off the WinUI thread and coalesce bursty UI updates.
- Maintain the native Windows reference experience and shared Python backend;
  do not rename the Windows executable or installer product.
- For a minor change, start with the observed failure and make the smallest
  direct correction that fits the existing design. Do not add a harness,
  service, abstraction, or replacement subsystem unless the current path
  cannot implement or verify the requirement; record that evidence when a
  broader change is necessary.

## Productive catch-up checks

- During long coding, test, build, packaging, deployment, transcription, or
  analysis stages, keep resumable catch-up making useful progress whenever a
  backlog and an authorized archive allowance remain. Do not equate a saved
  or enabled schedule with an active catch-up.
- Check the actual worker/job state, next schedule time, global acquisition
  lease, retained-work progress, and remaining allowance for every authorized
  account profile. If local-only processing is holding the workflow while the
  website stream is idle, checkpoint it and let another node/profile continue
  the next sequential acquisition turn; model work may proceed on different
  feed-days in parallel under the processing-claim rules.
- Use every authorized account's allowance over sequential acquisition turns;
  never issue concurrent archive requests merely to consume it. A blocked or
  exhausted profile must hand off promptly to the next eligible profile, while
  LAN reuse and retained local processing continue independently.

## Efficient verification

Every native or installer build is commit-gated. Commit all tracked source,
tests, documentation, version, and release metadata first; never build from a
dirty tracked worktree. If source changes after a build, create a new commit
before rebuilding. A versioned installer is immutable for its source commit.

Run focused tests while iterating, then the full offline suite and native build:

```powershell
.\.venv\Scripts\python.exe -m pytest -q tests\test_library.py tests\test_ui_contracts.py
.\.venv\Scripts\python.exe -m pytest -q
dotnet build .\BroadcastifyCli.WinUI\BroadcastifyCli.WinUI.csproj -c Release --no-restore
```

Use `scripts/build_windows_installer.ps1` only after the source commit so the
packaged product version embeds that commit. Public packages must pass a secret
and forbidden-file scan before release.

## Documentation map

- `README.md`: short product overview and entry points.
- `FEATURES.md`: current capability matrix.
- `docs/features/`: feature behavior and resume/failure rules.
- `docs/decisions/`: durable architectural and privacy decisions.
- `docs/reference/storage-and-resume.md`: data locations and recovery.
- GitHub Issues: outstanding bugs and lower-priority work.
