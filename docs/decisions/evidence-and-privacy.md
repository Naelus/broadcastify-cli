# Evidence, identity, and privacy

## Evidence first

Scanner ASR is noisy and radio calls are reports, not findings of fact. A model
output is surfaced only when it can be traced to timestamped transcript segments
and the corresponding local audio. Coverage gaps are explicit and never treated
as quiet days.

Incident titles, summaries, locations, categories, priorities, and outcomes are
validated against the cited text. Cross-feed overlap can increase interest but
is not independent confirmation because feeds may rebroadcast the same traffic.

## Spoken names

The private-use application preserves a person's name when it was explicitly
spoken in cited evidence. This is deliberate: hiding names from an investigator's
local review can remove important context while leaving the same information in
the source audio.

The analysis model is still forbidden from:

- guessing an identity;
- correcting an uncertain ASR name;
- normalizing a partial name into a full identity; or
- treating a named person as guilty or as a confirmed suspect without evidence.

Phone numbers, dates of birth, emails, and long numeric identifiers remain
masked in derived/display text. Raw audio and original transcript evidence are
unchanged and may contain those details.

## Derived data versus source evidence

Downloads, combined audio, transcripts, word/segment timing, and anonymous
speaker labels are source/processing artifacts. Incidents, daily/weekly briefs,
questions, and regional leads are versioned derived analysis.

Changing this policy invalidates only the derived prompt versions. Existing
audio, ASR, and diarization are reused. Historical analysis rows can remain in
SQLite for audit while the UI exposes only the current policy.

## Clips and publication

Exact evidence clips are local review aids. They may contain unverified
allegations and identifiers even when the displayed quote is sanitized. Regional
leads therefore remain `review_required` and must not be auto-published.

The earlier name-redaction policy was designed around a future public newsroom
surface and was too restrictive for private use. A future public product should
apply an explicit publication/export policy after editorial review rather than
silently weakening the investigator's local evidence view.

## External analysis providers

Local Gemma sends nothing away. External providers remain disabled until the
user explicitly allows transcript excerpts to leave the computer. Readiness
checks do not send transcript text or create a paid model request. See
[model-providers.md](../model-providers.md).
