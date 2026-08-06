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

The five stages are archive audio, combination, transcription, speaker labels,
and analysis. Each stage is detected from completion evidence, not merely from a
requested flag or filename.

## Resume actions

- **Resume all** builds a fresh read-only plan, skips ready days, finishes every
  local-only day first, and then handles network-needed days sequentially. It
  checks the persistent rolling ledger before each network day and stops on the
  first unavailable slot or explicit provider limit. Planning and local stages
  never contact Broadcastify, and completed checkpoints are reused.

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
and cached area digests that cite the feed. If the database transaction fails,
the detached directory is restored. A transient filesystem cleanup failure
leaves an ignored tombstone that a later Library refresh safely retries. Saved
Area Watch profiles and acquisition history remain configuration/audit records;
running one of those profiles can intentionally acquire the feed again.

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
export, and range questions. Incident playback seeks within the generated clip,
not an unrelated offset in the full day.

Speaker labels are anonymous acoustic clusters. They provide conversation
structure but do not identify officers, dispatchers, callers, or radio units.

## Durable UI state

Native settings and last selections live under
`AppData\Local\Broadcastify Desktop`. Settings are atomic and operation/crash
logs append rather than replacing the previous diagnostic. The browser UI keeps
non-secret view settings in browser-local storage. See
[storage and resume](../reference/storage-and-resume.md).
