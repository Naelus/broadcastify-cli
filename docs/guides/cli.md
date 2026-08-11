# CLI workflows

The CLI and both UIs use the same backend. CLI commands are useful for
automation, diagnosis, and reproducible local processing.

## Feed search

```powershell
.\.venv\Scripts\broadcastify-cli.exe search "Dallas Police"
```

Search uses the Broadcastify website flow and can include county-directory
matches. It is not the official API.

## Download and process a range

```powershell
.\.venv\Scripts\broadcastify-cli.exe download `
  --feed-id 5318 `
  --range 2026-07-12:2026-07-13 `
  --combine `
  --transcribe `
  --diarize `
  --device cuda `
  --model turbo
```

The range is inclusive. Archive acquisition finishes before GPU-heavy model
work begins. Cached blocks count as ready but do not consume another media
request.

## Analyze retained data

```powershell
.\.venv\Scripts\broadcastify-analysis.exe analyze-day `
  --feed-id 90001 `
  --date 2026-07-12
```

`analyze-day` imports the existing transcript, extracts incidents, writes the
daily brief, and updates embeddings. It does not download or transcribe. Use
`--force-summary` for only the brief or `--force` to rebuild all derived
incidents for the day.

## Ask and summarize

```powershell
.\.venv\Scripts\broadcastify-analysis.exe ask `
  --feed-id 90001 `
  --entire-feed `
  --question "Where and when do supported reports cluster across everything retained?"

.\.venv\Scripts\broadcastify-analysis.exe summarize-week `
  --feed-id 90001 `
  --week-ending 2026-07-12
```

Use explicit `--start-date` and `--end-date` instead of `--entire-feed` for a
bounded range. Questions print local downloaded/question-ready coverage and
retrieve only from current imported transcript dates. Whole-feed pattern
questions also receive deterministic category/location/weekday/time-block
aggregates. Weekly briefs name missing dates. Neither surface treats missing
coverage as inactivity.

## Inspect saved results

```powershell
.\.venv\Scripts\broadcastify-analysis.exe stats
.\.venv\Scripts\broadcastify-analysis.exe report-day `
  --feed-id 90001 `
  --date 2026-07-12 `
  --min-priority 3
```

## Exercise the Web job boundary

The smoke harness performs the real session-cookie/action-token handshake and
submits a normal Web job:

```powershell
.\.venv\Scripts\python.exe scripts\web_job_smoke.py diagnostics `
  --output-dir archives `
  --working-dir . `
  --result-file web-job-result.json
```

Use `continue-local` plus `--payload-file` for a retained day without an archive
request.

## Tests

```powershell
.\.venv\Scripts\python.exe -m pytest
```

Analysis-provider configuration is documented in
[model-providers.md](../model-providers.md); hardware-specific runtime variables
are in [hardware-backends.md](../hardware-backends.md).
