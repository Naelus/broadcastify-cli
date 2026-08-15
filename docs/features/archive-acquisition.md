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

Archive cache identity uses the exact provider archive ID returned by the
website listing. The downloaded `Content-Disposition` filename can carry a
different timestamp/token and is not treated as that identity. Each feed-day
keeps the relationship in an atomic hidden
`.broadcastify-archive-index.json` file. Older retained blocks without an index
are matched once using the feed timezone and bounded source-timestamp fallback,
then migrated to the exact index. A complete legacy day can also be reconciled
chronologically when listing/file counts, timestamp ordering, clock drift, and
any already-known mappings all agree; partial or ambiguous days are rejected
rather than guessed.

After an authenticated listing has been satisfied in full, the day also gets
an atomic `.broadcastify-archive-complete.json` snapshot of every listed
provider ID. This is deliberately separate from the identity index because an
index may describe only a partially downloaded day. The snapshot is accepted
without networking only when every identity still resolves to its retained
file at the recorded size. A hash-verified trusted-LAN completion manifest can
establish the same local proof. A range job:

1. resolves the requested listings;
2. reuses exact local or trusted-LAN blocks;
3. acquires the newest completed and immediately previous block first;
4. continues older backlog sequentially; and
5. refreshes a current-day listing once at the end to catch a block finalized
   while the run was active.

Repeated archive IDs in one listing are collapsed before progress accounting
or acquisition. Distinct IDs, however, remain distinct timeline positions and
must each have a local filename whose recorded byte size still matches. Even
if two authenticated responses use the same `Content-Disposition` filename or
contain equal bytes, the second position is materialized under a deterministic
alternate raw-archive filename. Equal content is not evidence that one point
in time can replace another.

Before workers check a partial legacy day, exact timestamp matches are claimed
in one pass. This prevents a neighboring one-hour fallback from racing ahead
and assigning a source file to the wrong archive ID. Older indexes that already
map several IDs to one filename are treated as incomplete: the Library hides
their stale combined/review output, identifies the day as needing archive
timeline repair, and a normal network-enabled retry downloads only the missing
positions before rebuilding downstream stages. Existing source files are
preserved.

The live block cannot be downloaded until Broadcastify publishes it as an
archive. Each ready message says **cached locally** or **downloaded from
Broadcastify** and includes the retained filename, so quota-consuming work is
visible without mistaking cache reuse for a new request.

## Quota behavior

Operational account guidance describes a 250-request rolling 24-hour allowance
that counts archive play and download requests. Each configured account profile
mints a stable local ledger identity and admits at most 240 automated archive
requests in its own rolling 24-hour window, leaving 10 unspent for manual
review. A locally authorization-gated development/testing deployment can use N
approved accounts for N × standard capacity; other deployments remain on one
profile.

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
- an optional fixed historical catch-up start date and one-time or recurring mode;
- one non-secret account profile or the locally authorized automatic pool;
- the selected combination, ASR, speaker, analysis, and LAN settings; and
- enabled, last-run, retry, and recovery state.

At the scheduled time the job revisits that recent range. Exact source blocks
and valid processing stages are reused, so overlap is intentional and does not
repeat completed work. When a historical catch-up date is set, it remains the
range start across rolling-quota retries and day changes. One-time mode clears
the date only after the complete requested range has no missing days. Recurring
mode retains it after success and checks that start through the new current day
at the next daily run. A LAN-deferred or otherwise incomplete result retries
shortly rather than being recorded as complete. Range acquisition, matching
LAN reconciliation, and scheduled processing all start with the newest day and
work backward toward the saved boundary; already-valid days are reused rather
than repeated. A scheduled processing pass runs at most one model/day before it
yields the worker back to acquisition. Remaining retained model days stay
explicitly queued, so newly released allowance on any authorized profile is
checked before another long inference pass. Explicit user-started jobs are not
subject to this scheduler fairness bound. If the rolling archive guard is
closed, cached days can still finish locally and missing acquisition remains
deferred until the ledger's next-safe time. The schedule itself becomes eligible
every five minutes so newly retained LAN blocks or transcript results continue
moving between nodes instead of waiting for that distant provider release. The
acquisition runner reads that local ledger before authentication; a
closed guard permits trusted-LAN reuse, local processing, and cache reuse only
for days with a valid local completion snapshot. It does not authenticate,
load archive listings, or request archive media. A locally proven range
therefore completes normally, while a merely partial or unproven day retains
the quota deferral. The desktop checks schedules while it is open; its
visible, default-on Windows startup option keeps it available after user
sign-in. The Web/TrueNAS service owns a background coordinator and can run them
continuously under its normal service supervisor.

Schedules live in the evidence database and survive restart. An interrupted
running schedule is returned to a deferred state on startup, waits one minute
to avoid colliding with an orphaned worker, and resumes from retained
files/checkpoints. A failed worker attempt remains incomplete and retries the
same due date after a fifteen-minute backoff instead of being treated as that
day's successful run. A quota-paused schedule is rechecked after startup and then
at five-minute intervals, so a far-future website retry cannot strand
combination, transcription, speaker labeling, analysis, or LAN reconciliation
that needs no provider request. The persistent guard still prevents
authentication, listing, or archive-media requests until the exact profile is
safe. The Windows activity log reports startup recovery. Only one local worker
job runs at a time. Stored schedule JSON removes direct Hugging
Face and analysis API-key values; those secrets must remain in the platform
credential store, active session, or private environment.

Automatic account mode is still one worker and one spaced archive request at a
time. Credentials, cookies, request attempts, 429 blocks, and next-safe times
are isolated per profile. When one account becomes unavailable, the same saved
job's acquisition-only pass is replayed immediately with the next eligible
profile; exact cache identities and completion snapshots prevent already
retained blocks from being requested again. Combination, transcription,
speaker labeling, and analysis begin only after the available acquisition turns
are checkpointed, so loading or running a local model cannot strand unused
allowance between account profiles. A coordinated peer may take the next global
website turn while this node processes a different claimed feed/day. If every
profile is closed, cached days still process locally, the schedule persists the
earliest next-safe time, and missing acquisition resumes there.

The Windows and Web/TrueNAS account surfaces use the same profile IDs and
automatic policy. The Web service may load named profiles from its encrypted
AES-GCM credential store or its private, persistent environment file. It shows
only profile labels, usernames, session availability, and per-profile quota
status; passwords never enter bootstrap/status responses. A real sign-in writes
one cookie file per profile, and aggregate capacity is reported as the sum of
the independent 240-request automated budgets plus each ten-request reserve.

On Windows, **Manage schedules** can edit the daily time, lookback, historical
catch-up date, one-time/recurring mode, account policy, enabled state, local processing stages, and incident analysis
for an existing feed. The Web/TrueNAS schedule list exposes the same edit and
enable/disable controls rather than requiring removal and recreation.
Changing those basics preserves the schedule's saved model, accelerator,
speaker-tuning, and LAN choices. The schedule always writes into the Library
currently selected in Settings. The desktop and Web/NAS scheduler pass that
canonical root at claim time, so it also replaces a stale absolute path left by
a previous Library selection. Direct/legacy callers without an active Library
selection resolve a relative `archives` value against the evidence database
instead of the installed app's working directory. Both rules prevent a silent
split that could otherwise fetch the same archive IDs into two roots.
Select the refresh option in the
editor to intentionally replace those advanced choices with the current
Settings page values. Removing a schedule never removes retained evidence.

Area acquisition uses explicit rank first, then measured distance. Older or
hand-curated profiles that contain neither retain their saved feed order, so a
name sort cannot redirect scarce archive requests away from the intended
nearest/highest-priority feed.

## Trusted-LAN pool

The optional LAN node shares retained archive MP3 blocks and complete
model-fingerprint-matched transcript sets, never credentials or analysis
databases. One renewable producer lease owns the single global upstream stream;
followers assemble the exact feed/day completion manifest from any peer and
verify size plus SHA-256. Inventories and completion manifests
carry the one optional exact provider ID and listing prefix for each block, so
a copied block retains its no-request cache identity on the receiving node. A
legacy peer manifest that collapses multiple positions into one block is
rejected and must be repaired by an updated producer. Once the exact completed manifest is
assembled and hash-verified, the receiving node writes its own local completion
snapshot; the snapshot itself does not need to be shared.

At every LAN-enabled job start, feed-wide reconciliation discovers all dates a
peer retained for the selected feed, not only the current job's recent window.
The long-running Windows node and browser/TrueNAS host also repeat this
LAN-only convergence independently of archive/model jobs, every five minutes
by default. Thus a transcript completed by one host appears on the other while
a long, unrelated model stage is still active; the pass cannot contact the
provider or consume quota. Equivalent model output travels as combined audio
(when present), its exact timeline manifest, transcript JSON, and rendered
text. A per-fingerprint/day processing lease prevents two machines from running
the same work; different days remain eligible for parallel model processing.
Scheduled jobs defer an active peer-owned acquisition or model/day immediately
and retry it from retained state on a later pass.

Broadcastify source labels can drift slightly across midnight even when the
track belongs to the prior website archive page. LAN manifests accept that
bounded next-day rollover consistently; the requested feed/day directory and a
30-hour timestamp ceiling still prevent an unrelated block from being admitted.

Today/yesterday successful manifests are short-lived rolling snapshots so a new
track can be discovered. Explicit quota state follows the producer ledger's
next known rolling-window release for the selected account profile. A different
authorized profile may take the next sequential turn. With a configured
authoritative coordinator, Windows and TrueNAS use the same per-account ledger;
coordinator failure pauses new website requests. LAN failure never prevents
local cache use. See
[lan-archive-sync.md](../lan-archive-sync.md).

## Failure and resume rules

- Failed futures do not count as completed progress.
- A quota response cancels queued media requests.
- Existing blocks are never deleted during an interrupted range.
- Archive acquisition precedes GPU processing, and scheduled processing yields
  after one model/day so the next released guarded download is checked before
  another long model pass.
- Re-running the same request starts from the exact first missing block or later
  invalid processing stage.
