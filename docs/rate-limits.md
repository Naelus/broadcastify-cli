# Broadcastify archive request limits

Last reviewed: August 13, 2026.

## Authorization comes first

The current [Broadcastify Terms and
Conditions](https://www.broadcastify.com/terms/) say that programmatic or
automated access and AI/ML processing require a separate commercial license in
advance, regardless of scale or personal/hobby intent. Premium access alone is
not authorization for this workflow. The controls below reduce accidental
overuse for an appropriately licensed or otherwise authorized installation;
they do not grant permission.

## Operational ceiling

Broadcastify support has described the standard archive allowance as **250
requests in a rolling 24-hour window**, counting both archive plays and archive
downloads. Older requests restore capacity individually as they age past 24
hours. Requests should be made one file at a time and spaced apart.

That support guidance is recorded here without personal correspondence,
account details, or identifying deployment information. It is an operational
boundary, not a guarantee that every request will succeed or that the service
will never change its policy.

## Authorized account capacity

The provider has supplied written, deployment-specific authorization for this
development/testing installation to create and operate multiple accounts. Each
account retains its standard allowance, so the effective aggregate allowance
is **N × the standard account limit** for N authorized accounts. The same
authorization requires limited concurrency, spaced requests, and avoidance of
unnecessary load; it does not waive any other term or apply automatically to
other users or deployments.

Accordingly, multi-account pooling is off unless the private local setting
`BROADCASTIFY_AUTHORIZED_ACCOUNT_POOL=1` is present. Account IDs may be listed
in `BROADCASTIFY_ACCOUNT_PROFILES` and contain no secrets. This repository does
not retain the private correspondence, personal details, or credentials that
establish authorization. Anyone without equivalent written permission must
leave pooling off and use one account.

## Application policy: 240 plus a reserve of 10 per account

Every account profile creates a durable random `instance_id` in the app's
SQLite request ledger. Before each `/archives/download/...` attempt it
atomically reserves one of that profile's **240 automated requests** in the
previous 24 hours. The remaining **10 of 250 on that account** are intentionally
unavailable to automation so a person can still play or download a small
number of archives manually. With N authorized profiles the UI therefore shows
N × 240 automated capacity and N × 10 total reserve, while every underlying
decision remains account-specific.

Credentials, premium cookies, 429 blocks, request attempts, and next-safe times
are isolated by the same non-secret account profile ID. The runner never uses a
default account cookie for a named profile. An automatic catch-up checks local
cache/LAN state first, selects an eligible profile, and continues retained
missing work on the next eligible profile only after the current one becomes
unavailable. It never runs two archive downloads concurrently: `download_jobs`
remains forced to one and the normal inter-request spacing still applies across
the whole coding session.

The ledger counts attempts, not just successful files:

- a 200, redirect chain, 4xx, 5xx, timeout, or connection failure consumes one
  local entry once the archive request starts;
- a retry after a network error or transient 5xx consumes another entry;
- an exact local cache hit or verified LAN block consumes no entry;
- archive metadata listings, transcription, diarization, analysis, playback of
  retained local audio, and clip export do not use this archive-media ledger;
- any HTTP 429 stops immediately without retry and blocks new archive requests
  until the oldest request known to that installation leaves the rolling
  window, plus a small clock-skew grace.

At that next-safe time the installation may admit one guarded request. If the
provider still returns 429 because of unseen browser/other-client activity, the
new attempt is counted, stops immediately, and advances the block to the next
known local release. Scheduled jobs preserve this deferral and resume then;
they never poll the archive-media endpoint while the ledger is closed.

A successfully verified feed-day also gets a local
`.broadcastify-archive-complete.json` snapshot containing the exact provider
IDs that were satisfied. While the rolling guard is closed, the runner reads
only that snapshot and the exact local identity index, revalidates every
retained file by name and size, and makes no authentication, archive-list, or
archive-media request. A directory that merely contains MP3s is not treated as
complete: if its proof is missing or invalid, the day remains safely deferred
until an eligible online pass or a hash-verified LAN completion can establish
one.

The native Library's feed-backlog view and **Resume / prioritize…** planner read
those snapshots, enabled schedule ranges, and the rolling ledger locally. They
do not authenticate or load provider metadata. Only after explicit confirmation
can selected source/network days enter the normal sequential guarded runner;
the user can instead select local processing only and spend zero archive
requests.

The desktop UI shows used, remaining, reserve, next-safe time, and a short form
of each account identity; authorized-pool mode also shows aggregate capacity.
The Windows package keeps its ledger
under the app's local data directory. Web/CLI deployments default to
`.broadcastify-archive-quota.sqlite3` in their working directory and may set
`BROADCASTIFY_QUOTA_LEDGER` to another durable installation-local path.

## What an installation cannot know

Each scope is deliberately a local safety account for one provider account. A
new scope does not create capacity by itself; only a separately authorized and
configured account does. The app cannot see:

- archive plays or downloads made manually in a browser;
- requests made before this ledger was created or while its storage was lost;
- requests from another desktop, NAS, CLI checkout, or other client using the
  same provider account; or
- server-side policy, IP-level controls, or requests counted differently by the
  provider.

Do not run two installations concurrently against the same account profile. If
a user switches between a desktop and NAS, finish or stop one before using the
other and remember that each account's provider window carries over even though
the installations' local ledgers do not. The 10-request reserve on each account
is protection for modest unseen manual activity, not enough to reconcile
multiple active clients.

On first use after an upgrade, the new ledger starts with no knowledge of older
requests. Start conservatively until the previous 24-hour provider window has
aged out. Never delete or copy a ledger to obtain more capacity.

## Why earlier tests appeared to hit a much smaller limit

Historical diagnostics found repeated requests caused by old cache-identity
assumptions, a Library split, and a short two-process overlap. In particular,
the listing's provider archive ID, its displayed source time, and the downloaded
filename token are not interchangeable; observed filename timestamps have
differed from listing timestamps by well over the old narrow fallback window.
That could make a retained block look absent. One measured run made 192 archive
endpoint calls; about 23 hours later another run made 55 before receiving 429.
Those 247 tracked calls, plus manual activity outside telemetry, are consistent
with a 250-request rolling window. The apparent 55-request limit was remaining
rolling capacity, not a separate daily quota.

The fixes now in place are:

- each retained block is indexed by the exact provider archive ID; legacy
  blocks use a bounded source-time fallback once, while complete unambiguous
  days can use conservative chronological reconciliation, and are then
  migrated;
- distinct provider IDs always retain distinct timeline filenames, including
  when authenticated responses repeat a filename or contain equal bytes;
- exact partial-day legacy matches are claimed before worker cache checks, and
  already-collapsed legacy indexes are marked for targeted repair instead of
  being accepted as complete;
- duplicate provider IDs in one listing are collapsed before acquisition;
- local and trusted-LAN cache checks occur before request admission;
- native and Web/NAS schedulers bind each claimed job to the currently selected
  Library, overriding stale saved paths that could otherwise fetch an archive
  ID again into a second local root;
- an optional historical schedule boundary survives quota deferrals; one-time
  mode clears only after every requested day is complete, while recurring mode
  retains the boundary for the next daily gap check, so released rolling slots
  can drain a backlog without a second ad-hoc downloader;
- before Broadcastify authentication, each acquisition reads the local ledger;
  a closed guard permits trusted-LAN reuse and locally proven completion
  snapshots only, never authenticates or loads a listing, and reports
  quota-limited only when a requested day is actually still missing;
- the SQLite ledger uses an atomic write transaction across local processes;
- every actual retry is separately charged;
- all 429 responses are terminal for the current run; and
- current and previous tracks are prioritized before older backlog.

## LAN behavior

Trusted-LAN peers may exchange hash-verified original archive blocks and elect
one producer for a feed/day, reducing duplicate downloads. The transient
quota result carries the producer's next-safe delay, preventing followers from
repeating the request until that rolling slot arrives. Completed old-day
manifests may still be retained for the configured 24-hour result lifetime.
The one exact provider-ID mapping for each inventoried or completed block
travels with it. Legacy peer blocks that claim several timeline identities are
rejected so a damaged cache cannot spread across the LAN.

LAN coordination does **not** merge request ledgers, credentials, account
profiles, or provider allowances. Use only one installation at a time when they
rely on the same provider account. See [Trusted-LAN archive
reuse](lan-archive-sync.md).
