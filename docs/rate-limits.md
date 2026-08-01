# Broadcastify archive request limits

Last reviewed: August 1, 2026.

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

## Application policy: 240 plus a reserve of 10

Every installed app or service creates a durable random `instance_id` in its
own SQLite request ledger. Before each `/archives/download/...` attempt it
atomically reserves one of **240 automated requests** in the previous 24 hours.
The remaining **10 of 250** are intentionally unavailable to automation so a
person can still play or download a small number of archives manually.

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

The desktop and browser UIs show used, remaining, reserve, next-safe time, and a
short form of the installation identity. The Windows package keeps its ledger
under the app's local data directory. Web/CLI deployments default to
`.broadcastify-archive-quota.sqlite3` in their working directory and may set
`BROADCASTIFY_QUOTA_LEDGER` to another durable installation-local path.

## What an installation cannot know

This is deliberately a per-installation safety account, not a claim to create a
new provider allowance. It cannot see:

- archive plays or downloads made manually in a browser;
- requests made before this ledger was created or while its storage was lost;
- requests from another desktop, NAS, CLI checkout, or other client using the
  same provider account; or
- server-side policy, IP-level controls, or requests counted differently by the
  provider.

Do not run two installations concurrently against one provider allowance. If a
user switches between a desktop and NAS, finish or stop one before using the
other and remember that the provider's rolling window carries over even though
the installations' local ledgers do not. The 10-request reserve is protection
for modest unseen manual activity, not enough to reconcile multiple active
clients.

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
- duplicate provider IDs in one listing are collapsed before acquisition;
- local and trusted-LAN cache checks occur before request admission;
- native and Web/NAS schedulers bind each claimed job to the currently selected
  Library, overriding stale saved paths that could otherwise fetch an archive
  ID again into a second local root;
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
The exact provider-ID mapping travels with an inventoried or completed block,
so receiving it from a peer does not discard its cache identity.

LAN coordination does **not** merge request ledgers, credentials, or provider
allowances. Each installed desktop or service retains its own 240-request
ledger as requested. Use only one installation at a time when they rely on the
same provider account. See [Trusted-LAN archive reuse](lan-archive-sync.md).
