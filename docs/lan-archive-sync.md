# Trusted-LAN archive reuse

Radio Archive Intelligence can reuse original Broadcastify archive blocks
already retained by another app instance on the same trusted LAN. This happens
before login or any Broadcastify archive request, so a household or newsroom
does not spend the same account download allowance fetching the same block on
several machines.

This is a small pull-based archive swarm, not a public peer-to-peer network:

1. a client asks its configured and discovered LAN peers for one feed/date
   inventory;
2. it copies only blocks missing from its own library;
3. it verifies the advertised byte length and SHA-256 while streaming to a
   unique temporary file;
4. it atomically publishes the file only after verification succeeds;
5. the normal cache-aware Broadcastify downloader requests anything still
   missing.

Peers may introduce other explicitly configured private peers, up to a bounded
pool of 24 nodes. A filename conflict or disagreement between peers is reported
and never overwrites the local file. Redirects, DNS hostnames, public addresses,
symlinks, path traversal, oversized blocks, and malformed feed/date identities
are rejected.

## What is shared

The read-only protocol exposes only retained original source blocks matching:

```text
archives/<feed-id>/<YYYYMMDD>/<YYYYMMDDHHMM>-<archive-id>-<feed-id>.mp3
```

It does **not** expose or synchronize:

- Broadcastify cookies, usernames, passwords, or `.env` values;
- combined recordings;
- transcripts or speaker labels;
- incidents, summaries, embeddings, SQLite data, or evidence clips;
- model files or runtime caches.

There is no upload, delete, mutation, or remote-job endpoint in the LAN
protocol. Each enabled node seeds the original blocks it already owns. Normal
Library processing detects copied blocks like any other retained source and can
finish combination, transcription, diarization, and analysis locally.

## Windows app

Archive reuse and one-hop discovery are enabled by default for acquisition.
They do nothing when no peer answers and do not prevent website fallback.

In **Local Library → Downloads and LAN reuse**:

- leave **Reuse archives from trusted LAN peers** enabled;
- leave **Discover peers on this LAN** enabled when local firewall/network
  policy permits UDP discovery;
- add one or more explicit numeric private URLs for reliable access, such as
  `http://10.200.1.227:8765`;
- opt into **Let this PC share original archive blocks** only on a trusted
  network. The native read-only node defaults to TCP port `8766` and is owned
  by the desktop app process.

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

For an ordinary headless machine that should share blocks without exposing the
complete browser UI:

```text
radio-archive-lan-node --host 0.0.0.0 --port 8766 --output-dir archives
```

The node refuses public or multicast bind addresses. It provides only
`/health` and the versioned read-only archive protocol.

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

## CLI acquisition

LAN reuse is on by default for the archive CLI and can be controlled explicitly:

```text
broadcastify-cli download --feed-id 90001 --range 2026-07-18:2026-07-19 \
  --lan-sync --lan-discovery --lan-peer http://10.200.1.227:8765
```

Use `--no-lan-sync` to skip peer checks or `--no-lan-discovery` with one or
more `--lan-peer` values for deterministic explicit-peer operation. LAN reuse
does not increase, predict, evade, or reset Broadcastify's account quota; it
only avoids redundant downloads of blocks already retained on the user's own
trusted network.
