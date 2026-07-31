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
old partial combines and raw speaker scratch files whose worker no longer
exists; active worker files and retained archive evidence are never removed.

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
