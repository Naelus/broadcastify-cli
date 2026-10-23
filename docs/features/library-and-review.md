# Local Library and Review

## What changed

The Review & Ask surface includes a **What changed** view for the selected
retained feed and date range. It reads saved current-prompt incidents and their
radio quotes without starting transcription, embeddings, or analysis. The
default range is the last seven days; ranges are bounded to 366 days.

Each record includes its archive date/time (or an explicit archive offset),
category, priority, summary, location, and cited radio quote. **Review cited
day and exact clips** opens the existing evidence review. Dates without saved
analysis remain visible as coverage gaps; older indexed evidence remains
readable while newer source files await processing. Neither an empty view nor
a missing date proves that no activity occurred.

Reading or refreshing does not mark records read. **Mark this batch caught
up** acknowledges only the displayed batch, at most 100 records. Bookmarks
use the incident fingerprint and displayed-content digest rather than row IDs
or analysis timestamps. Identical reanalysis and confidence-only changes do
not create new catch-up items. A changed title, summary, category, location,
priority, timing, or cited quote can reappear as updated. Records changed after
the view was loaded remain unread on a late acknowledgement; additional
batches stay available. Removing an incident does not currently create a
separate retraction notification.

Bookmarks persist in the analysis database and are shared across the desktop
and its trusted browser companion. TrueNAS followers read and acknowledge
through the authenticated Windows-master relay. An unavailable master produces
an explicit error; the follower does not start an independent model pipeline.
Exact-clip availability still follows the existing retained-audio guards.

This view shipped in Broadcastify Desktop 0.4.93. Offline execution checks cover
durable acknowledgements and the authenticated relay. The installed preview
provides synthetic incidents for exploring this view without production data;
see [pre-release review](../validation/pre-release-review.md).

## Local Library

The Library is a master/detail view over every retained feed-day. It shows:

- durable feed name and date;
- source-block count plus retained and temporary-working storage;
- five-stage processing state;
- combined-audio playback;
- timestamped transcript preview;
- local paths; and
- the exact next action.

The expandable **Feed coverage and backlog** section groups those days by the
friendly feed name. For an enabled schedule it compares the schedule's recent
lookback plus any retained catch-up boundary with locally retained days through
today. It reports missing days, local processing work, guarded source/network
days, newest retained date, and pipeline percentage. It also compares retained
raw blocks with the last authenticated provider-list snapshot. This view is
read-only: refreshing the Library never signs in, loads an archive listing, or
spends an archive-media request. Today's snapshot becomes eligible for a
refresh after 30 minutes; it is refreshed only by an explicit/scheduled guarded
resume.

The Library refreshes its counts and coverage every ten seconds while open,
including during long background jobs. Refreshes do not overlap or interrupt
recording playback. **Backlog** counts unfinished feed-days, not processing
operations: combining audio can advance a day without removing it from the
count while transcription, speaker labels, or analysis remain. The summary
separates ready-to-review days, local work, source checks, confirmed unavailable
archives, and days waiting for publication, with the last refresh time.

An authenticated empty listing checked after a historical day ended is shown
as **No archive available**. It remains visible as a coverage gap, contributes
no unfinished work, and is not offered by Resume or counted as analyzed.
The manual source-check action remains available. An empty folder, invalid
completion record, or missing downloaded file is never sufficient evidence to
dismiss a day. An empty listing captured before a day ended requires a final
check. Empty listings for today and the previous two days are **Waiting for source
audio** and become eligible for a new source check after 30 minutes. These recent
listings keep refreshing even after midnight, so delayed final blocks can be
acquired. Older days need a final listing checked at least one full calendar day
after they ended before their snapshot can be reused indefinitely. Existing retained evidence is
never hidden just because a later listing is empty.

When every listed raw block is retained but the combined recording is missing
or outdated, the day is **Ready to combine**. Finish locally combines those
verified files before transcription, preserving the originals and spending no
archive allowance.

The five stages are archive audio, combination, transcription, speaker labels,
and analysis. Each stage is detected from completion evidence, not merely from a
requested flag or filename.

## Resume actions

- **Catch up missing feed days…** selects one retained, scheduled, or previously
  saved feed and one prior start date. The local planner expands every calendar
  day from that date through today, including gaps older than a schedule's
  normal lookback and unscheduled feeds. Only absent days and retained days with
  unfinished processing are queued; a locally complete day is skipped even if
  its normal current-day source refresh would otherwise be due. The exact
  retained-local and source/network counts are shown before work starts. By
  default, starting the plan saves the start date in the Library database before
  the first day runs. While any work remains, its effective end follows the
  current day whenever Library or **Resume / prioritize…** reloads it. The saved
  catch-up survives app restart, cancellation, quota pause, machine restart, and
  upgrades; it is not limited to reopening the catch-up dialog. It self-clears
  only after every calendar day through current has local completion evidence
  or a confirmed historical empty listing; waiting for live audio keeps it saved.
  The dialog can explicitly replace or clear it, or promote the selected feed
  and start date into a daily recurring catch-up. Recurring mode keeps that
  boundary after a complete run and checks it through the then-current day at
  each scheduled time. Starting and resuming use the
  same sequential quota checks and checkpoints, so completed days and retained
  stages are never repeated. The entry remains available while another
  background pipeline is active: the chosen resumable range is saved behind
  that pipeline instead of disappearing or starting a concurrent download.
  With locally recorded written authorization, source-needed days rotate across
  eligible account profiles one at a time; each profile keeps independent
  credentials, cookies, 240+10 ledger state, and 429 release time.
- **Resume / prioritize…** builds a fresh read-only plan, synthesizes missing
  days from enabled feed schedules, and skips work already current. Before
  execution, the user chooses feeds, local-only work, whether to check/download
  missing source audio now, and one of local-first, selected-feed-first,
  newest-first, or oldest-first ordering. Network-needed days remain sequential.
  The app checks the persistent rolling ledger before each network day and
  stops on the first unavailable slot or explicit provider limit. Planning and
  local stages never contact Broadcastify, and completed checkpoints are reused.
  A current day can appear in both categories: if its refresh is quota-blocked,
  selected retained local stages still finish and its guarded source refresh
  remains queued for the next pass.

- **Verify & resume** contacts the guarded archive path only when source
  completeness needs verification.
- **Check for new source audio** is available on every retained day, including
  days already ready for review. It rechecks that exact feed/date, reuses
  retained blocks, downloads only missing source files, and obeys the rolling
  archive-request ledger.
- **Transcribe locally** uses retained combined audio.
- **Add speaker labels** does not repeat ASR.
- **Improve speakers** replaces portable preview labels with Community-1 and
  then refreshes dependent analysis.
- **Re-run evidence analysis** uses retained transcripts and makes no archive
  request.
- **Open review** appears only for the current evidence-policy version.

## Deleting a feed

The native Library exposes **Delete feed** in the selected feed header. A
non-default destructive confirmation lists the number of days, approximate
storage, and data types that will be removed. If the feed has a daily schedule,
the dialog offers to remove that schedule too and selects that option by
default so the feed is not unexpectedly downloaded again.

Deletion first detaches the numeric feed directory within the selected Library
root, then transactionally removes its imported days, transcript segments,
passages and embeddings, incidents, daily/weekly summaries, saved questions,
cached area digests that cite the feed, daily schedule, and saved catch-up
range. If the database transaction fails, the detached directory is restored.
A transient filesystem cleanup failure leaves an ignored tombstone that a
later Library refresh safely retries. The result is shown in a nonmodal Library
banner after the destructive confirmation closes, avoiding overlapping Windows
dialogs. Saved
Area Watch profiles and acquisition history remain configuration/audit records;
running one of those profiles can intentionally acquire the feed again.

On Windows, playback is released before detaching the directory and transient
sharing violations are retried with bounded backoff. Media players remain
detached through the directory rename, and folder/transcript buttons use the
Windows shell rather than retaining WinRT directory handles. If another worker, File
Explorer window, media handle, or outside process still owns the folder, the
operation reports that the feed is in use and removes neither files nor database
records. A feed currently used by the background pipeline or archive chat is
blocked before confirmation; a different feed can still be deleted.

An older combined MP3 is not considered current when its manifest differs from
the retained raw blocks. That day is labeled **New audio pending combine**; the
older recording is preserved, while its obsolete transcript/review state is
withheld until refresh.

A transcript created before a later successful combine is also withheld, even
when its filename still matches. The next action returns to local
transcription/diarization for the current recording instead of presenting stale
quotes or incidents.

Community-1 may temporarily use substantially more disk than the compressed
archive audio while it prepares and memory-maps a full-day waveform. The
Library labels that separately as **temporary**, keeps a completed lossless
preparation after interruption so retry work is reusable, and releases it once
an exact completed diarization cache exists. Refreshing the Library also removes
PID-owned partial combines and raw speaker scratch files whose worker no longer
exists; active or ownership-free worker files and retained archive evidence are
never removed.

The same current-revision gate applies to native and browser details, Review &
Ask, weekly and area briefs, and clip playback/export. When a combined recording
or transcript changes, older database segments, incidents, summaries, and
offsets are preserved for recovery but withheld until the current transcript
revision has been imported and analyzed. The command-line review, question, and
weekly-summary surfaces enforce the same gate.

## Review and Ask

Review supports priority filters plus all-priority search, daily briefs,
timestamped source quotes, exact and surrounding-context playback, local clip
export, and range questions. The feed picker displays retained/scheduled feed
names rather than requiring a numeric ID. **Ask the archive** is a multi-turn
chat over one named feed and date range. A month selector sets the full past
calendar month or the current month through today. **Use entire downloaded
feed** selects the earliest-to-latest locally retained span, including internal
calendar gaps, then previews how many days
have downloaded audio and how many have a current imported transcript. Missing
audio and locally unfinished dates remain explicit coverage gaps; they are
excluded from retrieval and cannot be interpreted as days with no activity.
Starter questions cover shots reports, unusual events, weekly importance, a
coverage-aware monthly summary, and whole-feed hotspots. The backend supplies
deterministic P records for exact category, repeated-location, weekday, and
six-hour archive-time counts over current analyzed days. These support cluster
descriptions but cannot be presented as population-normalized crime rates or
trends. Recent turns help resolve follow-ups, but every new material claim must
cite fresh E/I/P evidence for that turn. The
configured provider is used—quantized local Gemma 4 12B through llama.cpp by
default—and each answer remains in the existing local Q&A audit history.
Incident playback seeks within the generated clip, not an unrelated offset in
the full day.

Archive acquisition, combination, ASR, diarization, and incident extraction run
as a background pipeline. Navigation, Library browsing, reviewing completed
feeds, feed search, schedule management, and archive chat remain available.
Actions that would mutate or hold files for a feed currently being updated are
disabled for that feed only. Model-backed archive chat and pipeline analysis
share one analysis slot, preventing two competing local Gemma servers; chat can
run during download/transcription, and whichever model request reaches the slot
second waits without blocking the window.

Speaker labels are anonymous acoustic clusters. They provide conversation
structure but do not identify officers, dispatchers, callers, or radio units.

## Durable UI state

### Full catch-up visibility and measured ETA

The persistent status area distinguishes the complete retained work queue from
the current worker step. It counts unfinished feed-days across all feeds and
dates, with remaining source checks/downloads, combination, transcription,
speaker labeling, import, incident analysis, and semantic indexing. Stage
counts overlap when the same day needs several steps. Routine source checks
of completed days, unavailable historical audio, and audio awaiting publication
are separate; fresh source audio can add downstream work.

The snapshot uses the existing resume planner. Imported days with missing
embeddings remain incomplete, because saving a summary precedes search indexing.
Resume reuses the saved summary and transcript and finishes missing embeddings;
native and shared scheduled analysis no longer skip that remainder.
Native status refreshes while Review is open without replacing the selected
recording; the browser reads the Windows master's queue when configured as a
follower and does not substitute its partial local archive if the master fails.

The persistent Library status includes a **Full workflow (all retained work)**
estimate across retained and scheduled backlog, distinct from the current-stage
counter. It uses recent successful local timing samples for the active machine
and processing configuration, then combines estimates for unfinished stages.
The estimate is a range, not a promise; the browser follower displays the
Windows master's timestamped estimate. Per-job rows show a separate
**Full workflow (requested scope)** estimate, including known queue delay.

New machine/configuration profiles initially show a learning message until
local measurements exist. Provisional input sizes are labeled. The estimate
does not predict archive publication, website access, or future quota
availability: source and quota waits appear as waiting states without a
confident finish time. Paused/failed work and stale or disconnected master data
also hide finish times; estimates older than two minutes are treated as stale.
An estimate only describes remaining planned work and never determines whether
a feed-day or saved evidence is complete.
Completing one day's analysis indexes only that feed-day. Asking a question
indexes only the question-ready dates in the selected feed/range. Cached index
checks do not load an embedding model. An interrupted index resumes from its
committed batches without rerunning saved analysis or indexing unrelated feeds.
Bounded model passes process the oldest unfinished retained date, independently
of newest-first acquisition. Native scheduled model work selects one pending
date rather than repeatedly traversing the whole historical range; remaining
dates keep the saved schedule resumable. Download-only schedules skip that model
pass. Offline scheduling and recovery checks pass. The installed Windows app
and browser About surfaces open an isolated synthetic-data preview; it has no
production credentials, archive acquisition, model workers, scheduler, or LAN
service. Candidate-specific build and release validation remain separate; see
the [pre-release review](../validation/pre-release-review.md).

Completed **recurring catch-up** schedules recheck after 30 minutes, matching the
recent-source listing freshness window. This also applies with model processing
paused: acquisition can continue using the existing cache/LAN-first path and
rolling quota guard. Ordinary daily schedules still wait for the next scheduled
day, and an explicit Cancel suppresses further checks for that day. The next
source-check time survives restart; a failure during that later pass retains the
normal retry backoff rather than being hidden by the earlier completed pass.

Native settings and last selections live under
`AppData\Local\Broadcastify Desktop`. Settings are atomic and operation/crash
logs append rather than replacing the previous diagnostic. The browser UI keeps
non-secret view settings in browser-local storage. See
[storage and resume](../reference/storage-and-resume.md).
