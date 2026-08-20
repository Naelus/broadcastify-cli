# Trusted-LAN master and followers

Broadcastify Desktop uses a master/follower topology on a trusted LAN. The
Windows desktop is the post-download master. Browser, NAS, and other nodes are
followers.

The topology has two deliberately separate planes:

1. Provider acquisition remains globally sequential and protected by the
   persistent per-account rolling guard. An authorized follower may acquire a
   requested source day under that guard.
2. Post-download work belongs to Windows. The master clones every newly
   completed source day, then owns combination, transcription, diarization,
   analysis, indexing, and interactive questions.

A follower submits its requested feed/date range to Windows. It may help obtain
the original blocks, but it does not publish competing model results. After the
master finishes a model result, followers may pull that exact Windows-authored
artifact set.

Master loss never causes automatic promotion. Followers keep their retained
files usable and pause new master-owned work. Promoting another machine is an
explicit recovery action so a transient network fault cannot create two
authoritative pipelines.

## Incremental synchronization

Completed source days and master-authored model results append a small event to
the local pipeline journal. Each consumer stores a durable cursor for the peer
node ID and requests only later events. A bounded pass processes at most one
page and resumes from its cursor after interruption.

This replaces the old behavior that enumerated every retained day and model
fingerprint every five minutes. Normal jobs check only their requested dates.
The background worker processes only journal deltas; it does not scan the full
archive, enumerate unrelated fingerprints, or make provider requests.

Source transfer still verifies filename, feed/date containment, size, SHA-256,
provider identity, and the exact completion proof before publication. A result
transfer additionally verifies the processing fingerprint, source-audio hash,
rendered-text hash, and complete artifact set. Files are streamed to a unique
temporary path and atomically published.

The journal database and completion markers are durable. A new installation
performs one metadata-only seed from existing completion markers; it does not
hash media during that migration. Stable node IDs keep cursors valid across
restarts.

## What is shared

Acquisition nodes may serve retained original blocks matching:

```text
archives/<feed-id>/<YYYYMMDD>/<YYYYMMDDHHMM>-<source-token>-<feed-id>.mp3
```

The hidden archive identity index and completion proof establish a one-to-one
provider timeline. Equal bytes do not make two timeline positions
interchangeable.

Only the master may advertise completed derived sets:

- combined day audio when combination was used;
- the combined-audio timeline manifest;
- transcript JSON and rendered text; and
- the processing fingerprint and exact source-audio hashes binding the set.

The protocol does not copy credentials, cookies, model files, runtime caches,
analysis databases, incidents, summaries, embeddings, or evidence clips.
Interactive desktop library, search, question, and local processing paths read
only master-local state and never wait for a follower.

## Windows master

The native desktop launches its LAN service with the `master` role on port
`8766` by default. It remains eligible for the one upstream acquisition lease,
accepts follower range requests, incrementally clones follower source
completions, and publishes its completed model results.

Keep original-block sharing enabled and configure explicit numeric private peer
URLs where discovery is unreliable. The optional sync key must match every
participant. The app owns the service process and stops it during shutdown.

## Browser, Linux, and NAS followers

A follower configuration points all authority at the Windows service:

```dotenv
BROADCASTIFY_LAN_SYNC_ENABLED="true"
BROADCASTIFY_LAN_SHARING="true"
BROADCASTIFY_LAN_QUEUE_ENABLED="true"
BROADCASTIFY_LAN_BACKGROUND_SYNC="true"
BROADCASTIFY_LAN_RECONCILE_SECONDS="300"
BROADCASTIFY_LAN_ROLE="follower"
BROADCASTIFY_LAN_MASTER_URL="http://WINDOWS_PRIVATE_ADDRESS:8766"
BROADCASTIFY_LAN_COORDINATOR="http://WINDOWS_PRIVATE_ADDRESS:8766"
BROADCASTIFY_LAN_QUOTA_COORDINATOR="http://WINDOWS_PRIVATE_ADDRESS:8766"
BROADCASTIFY_LAN_ADVERTISE_URL="http://FOLLOWER_PRIVATE_ADDRESS:8765"
```

`BROADCASTIFY_LAN_MASTER_URL` is mandatory for a follower. The coordinator
settings use the same Windows address so acquisition leases and account quota
reservations cannot split from pipeline ownership. A coordinator outage pauses
only new provider requests and follower submissions; retained local content
remains usable.

`BROADCASTIFY_LAN_BACKGROUND_SYNC` now means bounded journal processing. The
interval no longer triggers feed-wide reconciliation.

### Public Web surface and private Windows execution

When a NAS hosts the authenticated browser surface, the public tunnel ends at
that NAS service. The Windows LAN service is never a tunnel origin and must not
be port-forwarded or exposed to the internet.

With the follower role and `BROADCASTIFY_LAN_MASTER_URL` configured, the Web
service relays interactive jobs, cancellation, schedules, and resumable
catch-up intent over the authenticated private-LAN channel to Windows. It also
reads compact authoritative feed spans, account allowances, and active-job
status from Windows. The browser may therefore select an explicit feed or its
entire retained span while Windows owns the actual worker and result.

The NAS keeps serving its own retained Library and media so opening the Web UI
does not trigger a full remote archive scan. If Windows is unavailable, a
relayed request fails visibly with a retryable service error; the follower does
not silently run a competing local job.

## Acquisition safety

Only one renewable acquisition lease may admit provider archive requests at a
time across feeds, dates, nodes, and authorized accounts. Each account keeps
its independent persistent allowance and manual-use reserve. Cached local
blocks, verified LAN blocks, and local processing happen before a provider
request.

A producer publishes its exact completed block manifest. The master can then
clone the day without repeating provider traffic. If neither master nor any
follower has the requested source blocks, the next eligible authorized account
may acquire them under the normal rolling guard.

## Optional shared key

On a trusted LAN, place the same high-entropy value in every participant's
private environment:

```dotenv
BROADCASTIFY_LAN_SYNC_KEY="replace-with-a-long-random-value"
```

The key is sent in `X-Radio-Archive-LAN-Key`. It is never returned by status or
inventory endpoints. Plain HTTP does not encrypt the key or audio, so never
port-forward the LAN service or expose it directly to the internet.

LAN requests ignore system HTTP proxy settings and reject redirects, public or
DNS peer addresses, path traversal, symlinks, malformed identities, oversized
objects, and hash conflicts.

## Recovery

Interrupted source copies leave the verified local destination unchanged and
resume from the same journal cursor. Interrupted model work remains resumable
from the master's existing checkpoints. A follower result never overwrites a
different master result.

If Windows is permanently unavailable, first preserve its library, analysis
database, quota ledgers, credentials, and pipeline journal. Then explicitly
configure exactly one replacement as `master` and repoint every follower. Do
not promote automatically during an ordinary outage.
