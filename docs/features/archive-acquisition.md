# Archive acquisition

## Scope

Acquisition covers website feed discovery, premium sign-in, archive listing,
cache reuse, media download, current-day refresh, and trusted-LAN sharing. Site
behavior is isolated in `broadcastify_cli/broadcastify.py` because these are
reverse-engineered website endpoints rather than the official API.

## Discovery

- Text search accepts agency, city, county, state, or ZIP terms.
- ZIP discovery follows the website's ZIP-to-county flow and parses the county
  feed tables.
- Radius discovery expands nearby Census ZCTA centroids, deduplicates feeds,
  labels distance as approximate, and makes no archive-media request.
- Users explicitly select feeds before an area profile is saved.

## Cache and ordering

Archive cache identity uses the feed's published timezone and source timestamp,
not the downloaded filename alone. A range job:

1. resolves the requested listings;
2. reuses exact local or trusted-LAN blocks;
3. acquires the newest completed and immediately previous block first;
4. continues older backlog sequentially; and
5. refreshes a current-day listing once at the end to catch a block finalized
   while the run was active.

The live block cannot be downloaded until Broadcastify publishes it as an
archive. Each ready message says **cached locally** or **downloaded from
Broadcastify** and includes the retained filename, so quota-consuming work is
visible without mistaking cache reuse for a new request.

## Quota behavior

Operational account guidance describes a 250-request rolling 24-hour allowance
that counts archive play and download requests. Each installation mints a stable
local ledger identity and admits at most 240 automated archive requests in its
own rolling 24-hour window, leaving 10 unspent for manual review.

The ledger is reserved immediately before every archive-media request. Network
and 5xx retries therefore consume another entry, just as they do upstream.
Cached local or LAN blocks and local processing consume none. Any HTTP 429
stops without retry. The ledger then waits until its oldest known active
request ages out of the rolling window (plus a small clock-skew grace) and
admits at most the next guarded request. Another 429 advances the stop to the
next known release; it does not produce a tight retry loop.

The ledger cannot observe manual website activity, requests made before it was
created, or another installation. Do not run separate desktop/NAS instances
against the same provider allowance concurrently. See
[rate-limits.md](../rate-limits.md) for the exact boundary and current terms.

## Per-feed schedules

A schedule belongs to one explicitly selected feed, not a search term or a
hardcoded locality. It stores:

- the feed ID and display name;
- one local wall-clock time;
- a one-to-fourteen-day lookback, including the current day;
- the selected combination, ASR, speaker, analysis, and LAN settings; and
- enabled, last-run, retry, and recovery state.

At the scheduled time the job revisits that recent range. Exact source blocks
and valid processing stages are reused, so overlap is intentional and does not
repeat completed work. If the rolling archive guard is closed, cached days can
still finish locally and the missing acquisition is deferred until the
ledger's next-safe time. The desktop checks schedules while it is open. The
Web/TrueNAS service owns a background coordinator and can run them continuously
under its normal service supervisor.

Schedules live in the evidence database and survive restart. An interrupted
running schedule is returned to a deferred state on startup and resumes from
retained files/checkpoints. Only one local worker job runs at a time. Stored
schedule JSON removes direct Hugging Face and analysis API-key values; those
secrets must remain in the platform credential store, active session, or
private environment.

## Trusted-LAN pool

The optional LAN node shares original archive MP3 blocks, not credentials,
transcripts, analysis, or combined audio. One renewable producer lease owns a
feed/day upstream acquisition; followers assemble the exact completion manifest
from any peer and verify size plus SHA-256.

Broadcastify source labels can drift slightly across midnight even when the
track belongs to the prior website archive page. LAN manifests accept that
bounded next-day rollover consistently; the requested feed/day directory and a
30-hour timestamp ceiling still prevent an unrelated block from being admitted.

Today/yesterday successful manifests are short-lived rolling snapshots so a new
track can be discovered. Explicit quota state follows the producer ledger's
next known rolling-window release. LAN failure never prevents local cache use
or the paced website fallback. See
[lan-archive-sync.md](../lan-archive-sync.md).

## Failure and resume rules

- Failed futures do not count as completed progress.
- A quota response cancels queued media requests.
- Existing blocks are never deleted during an interrupted range.
- Archive acquisition precedes GPU processing so a long model pass cannot delay
  the next guarded download.
- Re-running the same request starts from the exact first missing block or later
  invalid processing stage.
