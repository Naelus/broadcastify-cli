# Trusted-LAN archive reuse

Radio Archive Intelligence can reuse original Broadcastify archive blocks
already retained by another app instance on the same trusted LAN. This happens
before login or any Broadcastify archive request, so a household or newsroom
does not spend the same account download allowance fetching the same block on
several machines.

This is a small pull-based archive pool with a shared acquisition queue, not a
public peer-to-peer network:

1. a client asks its configured and discovered LAN peers for one feed/date
   inventory;
2. it copies only blocks missing from its own library;
3. all reachable queue-capable peers are considered and the same deterministic
   coordinator is selected, independent of which peer supplied a block;
4. one eligible producer receives a renewable 90-second feed/date lease and is
   the only client allowed to start new upstream archive-media requests;
5. followers wait, poll the pool, and copy completed blocks from any peer as
   they appear;
6. every copy verifies the advertised byte length and SHA-256 while streaming to a
   unique temporary file;
7. the leader atomically publishes each file and reports the exact completed
   filename/size/SHA-256 manifest plus its optional provider archive identity;
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
    shared quota-limit result suppresses follower retries until the producer's
    installation-local ledger reaches its next known rolling-window release.

Peers may introduce other explicitly configured private peers, up to a bounded
pool of 24 nodes. A filename conflict or disagreement between peers is reported
and never overwrites the local file. Redirects, DNS hostnames, public addresses,
symlinks, path traversal, oversized blocks, and malformed feed/date identities
are rejected.

## What is shared

The read-only protocol exposes only retained original source blocks matching:

```text
archives/<feed-id>/<YYYYMMDD>/<YYYYMMDDHHMM>-<source-token>-<feed-id>.mp3
```

The middle filename token is provider-supplied but is not necessarily the
archive ID used by the download URL. A hidden per-day
`.broadcastify-archive-index.json` records that exact URL/listing identity and
is propagated with LAN inventories. It contains no account or credential data.

It does **not** expose or synchronize:

- Broadcastify cookies, usernames, passwords, or `.env` values;
- combined recordings;
- transcripts or speaker labels;
- incidents, summaries, embeddings, SQLite data, or evidence clips;
- model files or runtime caches.

There is no archive upload, delete, or remote-job endpoint in the LAN protocol.
Each enabled node seeds only original blocks it already owns. The coordinator
stores bounded, transient lease/result metadata: quota scope, feed, date,
producer node/URL, state, expiry, and the completed source-block manifest. The
opaque lease token is returned only to its owner. A lease does not expose
credentials or cause a remote machine to start work; it coordinates jobs users
already started.

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

`BROADCASTIFY_LAN_QUOTA_SCOPE` is a non-secret coordination label. Peers in one
cache/lease pool should use the same value. It does not merge installation
request ledgers or create another provider allowance. Optional expert timing
controls are:

```dotenv
BROADCASTIFY_LAN_QUEUE_LEASE_SECONDS="90"
BROADCASTIFY_LAN_QUEUE_RESULT_SECONDS="86400"
BROADCASTIFY_LAN_QUEUE_ROLLING_RESULT_SECONDS="300"
BROADCASTIFY_LAN_QUEUE_MAX_WAIT_SECONDS="1800"
```

Active leases renew in the background. If renewal can no longer be proven,
the downloader stops admitting new archive-media requests before the lease can
be reassigned. `BROADCASTIFY_LAN_QUEUE_RESULT_SECONDS` applies to completed old
days. The rolling value applies only to successful today/yesterday manifests.
An explicit quota result instead supplies the producer ledger's next-safe
delay, bounded to the provider's 24-hour window. None of these timers causes
another media request when the exact block is already present on a peer.
Completed MP3s—not the transient queue—remain the durable state.

For an ordinary headless machine that should share blocks without exposing the
complete browser UI:

```text
radio-archive-lan-node --host 0.0.0.0 --port 8766 --output-dir archives
```

The node refuses public or multicast bind addresses. It provides only
`/health`, the versioned read-only archive protocol, and transient acquisition
coordination.

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
user's own trusted-LAN clients. Each installed app/service still keeps its own
240-request ledger, so clients using the same provider account must not run
acquisition concurrently.
