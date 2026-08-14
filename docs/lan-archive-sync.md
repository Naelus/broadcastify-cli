# Trusted-LAN archive reuse

Radio Archive Intelligence can reuse original Broadcastify archive blocks
already retained by another app instance on the same trusted LAN. This happens
before login or any Broadcastify archive request, so a household or newsroom
does not spend the same account download allowance fetching the same block on
several machines.

This is a small pull-based retained-work pool with shared acquisition and
processing queues, not a public peer-to-peer network:

1. a client asks its configured and discovered LAN peers for one feed/date
   inventory;
2. it copies only blocks missing from its own library;
3. an explicitly configured authoritative coordinator is preferred; otherwise
   all reachable queue-capable peers are considered and the same deterministic
   coordinator is selected, independent of which peer supplied a block;
4. one eligible producer receives a renewable 90-second lease and is the only
   client allowed to start new upstream archive-media requests. The active
   lease is global across every feed, day, machine, and authorized account;
5. interactive consumers may follow the active result, while scheduled jobs
   record it as deferred and immediately continue with other retained work;
6. every copy verifies the advertised byte length and SHA-256 while streaming to a
   unique temporary file;
7. the leader atomically publishes each file and reports the exact completed
   filename/size/SHA-256 manifest plus its optional provider archive identities;
   followers assemble and verify that manifest from any combination of peers,
   retain the same identity mapping, then process the day without contacting
   Broadcastify;
8. the producer acquires the newest completed track and the immediately
   previous track before older backlog, then refreshes a feed-local current-day
   listing once before publishing its manifest;
9. today/yesterday completion manifests are rolling snapshots retained for
   five minutes by default. A later job can elect one new producer to check for
   another finalized track while all followers reuse the exact snapshot;
10. a crashed producer loses its lease and another producer can take over. A
    quota-limit result suppresses retries for that exact account profile until
    its next known rolling-window release without blocking another authorized
    account from taking the next sequential turn;
11. each run also reconciles every peer-retained date for the followed feed,
    and quota-paused schedules repeat that retained-only pass every five minutes.
    Peers advertise a bounded list of every processing fingerprint with a
    complete retained artifact set. The client automatically pulls each model's
    combined audio, time-mapping manifest, transcript JSON, and rendered text as
    one hash-verified set, even when its own selected model has a different
    fingerprint. If another local model result already uses those names, the
    peer set is kept in a fingerprint/audio-hash variant directory rather than
    overwriting either result; and
12. one renewable processing lease owns each model-fingerprint/feed/day. A
    second node skips that same model/day instead of waiting or duplicating it,
    but may claim a different day and run the same model in parallel. Finished
    artifacts reconcile again before job exit and on the next scheduled pass.

Peers may introduce other explicitly configured private peers, up to a bounded
pool of 24 nodes. A filename conflict or disagreement between peers is reported
and never overwrites the local file. Redirects, DNS hostnames, public addresses,
symlinks, path traversal, oversized blocks, and malformed feed/date identities
are rejected.

## What is shared

The read-only protocol exposes retained original source blocks matching:

```text
archives/<feed-id>/<YYYYMMDD>/<YYYYMMDDHHMM>-<source-token>-<feed-id>.mp3
```

The middle filename token is provider-supplied but is not necessarily the
archive ID used by the download URL. A hidden per-day
`.broadcastify-archive-index.json` records the exact URL/listing identity for
each timeline file and is propagated with LAN inventories. One block may carry
at most one identity: equal bytes or a repeated provider filename do not make
two points on the archive timeline interchangeable. A legacy manifest that
attaches several identities to one block is rejected rather than spreading an
incomplete day. After a follower assembles and hash-verifies a completed
manifest, it writes its own `.broadcastify-archive-complete.json` proof from
those one-to-one identities. This completion file is local bookkeeping rather
than a separately shared object. Neither file contains account or credential
data.

For each verified processing fingerprint, it may also expose a complete derived
set for a retained day:

- `combined_<feed-id>_<YYYYMMDD>.mp3` when combination was used;
- its `combined_*.manifest.json`, which preserves the source-block clock and
  combined-audio offsets needed for day/time citations;
- matching transcript JSON; and
- matching rendered transcript text.

The receiver first asks each peer for its bounded complete-fingerprint list,
then verifies names, bounds, byte lengths, SHA-256 values, audio hash,
processing fingerprint, rendered-text hash, and the complete artifact set
before reuse. A partial or stale set is not advertised as completed work. A
peer from before fingerprint discovery simply returns `404`; source-block reuse
and an explicitly requested matching-model transcript remain compatible during
a rolling upgrade.

It does **not** expose or synchronize:

- Broadcastify cookies, usernames, passwords, or `.env` values;
- incidents, summaries, embeddings, SQLite data, or evidence clips;
- model files or runtime caches.

There is no archive upload, delete, credential-copy, or remote-job endpoint in
the LAN protocol. Each enabled node seeds only artifacts it already owns. The
coordinator stores bounded, transient lease/result metadata: account-scoped
quota label or model fingerprint, feed, date, producer node/URL, state, expiry,
and completion counts/manifests. The opaque lease token is returned only to its
owner. A lease does not expose credentials or cause a remote machine to start
work; it coordinates jobs users already started.

Normal Library processing detects copied blocks like any other retained source
and can finish combination, transcription, diarization, and analysis locally.

## Windows app

Archive reuse, one-hop discovery, and original-block seeding are enabled by
default for the trusted-LAN Windows workflow. Version-4 settings migrate to
producer mode when LAN reuse was enabled. The visible toggle can return a
machine to consumer-only mode.

In **Local Library → Downloads and LAN reuse**:

- leave **Reuse archives from trusted LAN peers** enabled;
- leave **Discover peers on this LAN** enabled when local firewall/network
  policy permits UDP discovery;
- add one or more explicit numeric private URLs for reliable access, such as
  `http://10.200.1.227:8765`;
- leave **Seed original blocks and join the shared LAN download queue** enabled
  when this PC should be eligible to own an upstream lease. The native
  archive/coordination node defaults to TCP port `8766` and is owned by the
  desktop app process.

Windows may request a firewall allowance the first time sharing or discovery
is enabled. An explicit peer URL is recommended even when discovery works,
especially across VLANs, Wi-Fi client isolation, or restrictive firewalls.
The standalone desktop LAN node reads both `.env` and the ignored
`.env.accounts` file before its packaged defaults, matching the worker's
private configuration precedence after an app restart.

## Browser UI, Linux, and TrueNAS

The browser service can seed from the same TCP port as its UI. Sharing is off
unless its private environment enables it:

```dotenv
BROADCASTIFY_LAN_SYNC_ENABLED="true"
BROADCASTIFY_LAN_DISCOVERY_ENABLED="true"
BROADCASTIFY_LAN_PEERS="http://10.200.1.227:8765 http://192.168.1.44:8766"
BROADCASTIFY_LAN_SHARING="true"
BROADCASTIFY_LAN_QUEUE_ENABLED="true"
BROADCASTIFY_LAN_QUOTA_SCOPE="default"
BROADCASTIFY_LAN_COORDINATOR="http://10.200.1.227:8765"
BROADCASTIFY_LAN_ADVERTISE_URL="http://10.200.1.227:8765"
BROADCASTIFY_LAN_DISCOVERY_PORT="48765"
```

`BROADCASTIFY_LAN_ADVERTISE_URL` is important behind TrueNAS Apps or another
NAT/port-publishing boundary: it must be the numeric private URL other LAN
clients can actually reach. The TrueNAS example uses host networking because
Docker port publishing does not reliably forward broadcast/multicast discovery
traffic into the App. Its HTTP server still binds only the selected private NAS
address, its discovery responder listens on UDP `48765`, and read-only sharing
is enabled. The persistent `/data/archives` dataset remains the source;
redeploying the App does not copy, move, or delete retained data.

`BROADCASTIFY_LAN_COORDINATOR` selects one stable private coordinator for both
acquisition and model/day claims. Set it to the TrueNAS service's own private
URL on TrueNAS and to that same URL on Windows. This avoids split-brain election
when discovery is asymmetric.

Windows clients that use the same provider accounts on more than one machine
also set `BROADCASTIFY_LAN_QUOTA_COORDINATOR` to the TrueNAS private URL. The
TrueNAS service owns that persistent ledger locally and therefore does not point
its own worker back through the remote-ledger variable. Account credentials and
cookies remain separate on each machine; the shared ledger contains only
non-secret profile IDs, request attempts, and rolling-limit state.

`BROADCASTIFY_LAN_QUOTA_SCOPE` is a non-secret pool label. Workers append their
non-secret account profile ID internally. That keeps one account's quota result
from blocking another account while the coordinator still enforces one global
active website stream. It does not create another provider allowance. Optional
expert timing controls are:

```dotenv
BROADCASTIFY_LAN_QUEUE_LEASE_SECONDS="90"
BROADCASTIFY_LAN_QUEUE_RESULT_SECONDS="86400"
BROADCASTIFY_LAN_QUEUE_ROLLING_RESULT_SECONDS="300"
BROADCASTIFY_LAN_QUEUE_MAX_WAIT_SECONDS="1800"
```

Active acquisition and processing leases renew in the background. If renewal
can no longer be proven, the downloader stops admitting new archive-media
requests and a model result is not published as shared completion.
`BROADCASTIFY_LAN_QUEUE_RESULT_SECONDS` applies to completed old days and
processing claims. The rolling value applies only to successful today/yesterday
archive manifests. An explicit quota result instead supplies that account
ledger's next-safe delay, bounded to the provider's 24-hour window. None of
these timers causes another media request when the exact block is already
present on a peer. Retained artifacts—not the transient queues—remain the
durable state.

For an ordinary headless machine that should share blocks without exposing the
complete browser UI:

```text
radio-archive-lan-node --host 0.0.0.0 --port 8766 --output-dir archives
```

The node refuses public or multicast bind addresses. It provides only
`/health`, the versioned read-only retained-artifact protocol, and transient
acquisition/processing coordination.

## Optional shared key

On a trusted but busy LAN, place the same high-entropy value in the private
`.env` of every participating node:

```dotenv
BROADCASTIFY_LAN_SYNC_KEY="replace-with-a-long-random-value"
```

The key is sent in `X-Radio-Archive-LAN-Key` and is never returned by status or
inventory endpoints. It prevents an uninformed LAN client from browsing or
copying source blocks, but plain HTTP does not encrypt it or the audio. This
feature is for a trusted LAN; it is not internet authentication or TLS. Never
port-forward either the browser service or native seed node.

LAN inventory, block, and lease requests explicitly ignore `HTTP_PROXY`,
`HTTPS_PROXY`, and related environment proxy settings. This preserves
source-address lease validation and keeps the optional LAN key and opaque lease
token on the trusted network. A multihomed coordinator may claim its own lease
from another private interface only when both its node ID and producer URL
exactly match the coordinator's configured advertised URL; other producers
retain strict source-address matching.

## CLI acquisition

LAN reuse is on by default for the archive CLI and can be controlled explicitly:

```text
broadcastify-cli download --feed-id 90001 --range 2026-07-18:2026-07-19 \
  --lan-sync --lan-discovery --lan-peer http://10.200.1.227:8765
```

Use `--no-lan-sync` to skip peer checks or `--no-lan-discovery` with one or
more `--lan-peer` values for deterministic explicit-peer operation. A
standalone CLI process is a queue consumer unless a reachable seed node is
also identified through `BROADCASTIFY_LAN_SELF_URL` or
`BROADCASTIFY_LAN_SELF_PORT`. LAN reuse does not increase, predict, evade, or
reset Broadcastify's account quota; it avoids duplicate requests among the
user's own trusted-LAN clients. When no authoritative quota coordinator is
configured, each installation keeps its own conservative 240-request ledger,
so clients using the same provider account must not run acquisition
concurrently. In the coordinated Windows plus TrueNAS deployment, the TrueNAS
ledger is authoritative per account and each Windows client also keeps a local
fail-safe mirror; coordinator loss pauses new archive requests rather than
risking an undercount.
