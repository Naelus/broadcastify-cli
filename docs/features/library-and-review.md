# Local Library and Review

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
  only after every calendar day through current has local completion evidence.
  The dialog can explicitly replace or clear it. Starting and resuming use the
  same sequential quota checks and checkpoints, so completed days and retained
  stages are never repeated.
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
chat over one named feed and date range, with starter questions for shots
reports, unusual events, and the most important events in a week. Recent turns
help resolve follow-ups, but every new material claim must cite fresh E/I
evidence for that turn. The configured provider is used—quantized local Gemma 4
12B through llama.cpp by default—and each answer remains in the existing local
Q&A audit history. Incident playback seeks within the generated clip, not an
unrelated offset in the full day.

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

Native settings and last selections live under
`AppData\Local\Broadcastify Desktop`. Settings are atomic and operation/crash
logs append rather than replacing the previous diagnostic. The browser UI keeps
non-secret view settings in browser-local storage. See
[storage and resume](../reference/storage-and-resume.md).
