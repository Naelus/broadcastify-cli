# TrueNAS Apps deployment

Use the supported **Apps → Discover Apps → Install via YAML** path for a
permanent NAS-hosted browser UI. TrueNAS 25.04 uses a Docker-backed Apps
service; this is separate from its experimental LXC **Containers** feature.

The app has two deliberately separate layers:

- a replaceable, commit-labeled application image;
- one host-path dataset containing archives, transcripts, SQLite evidence,
  model caches, runtime state, and the ignored private `.env`.

Replacing or rolling back the image therefore does not delete the library.
TrueNAS recommends host-path datasets instead of ixVolumes for production data
that needs ordinary backup and snapshot handling.

## 1. Create persistent storage

Create a dataset such as `ssd_pool/radio-archive`, then grant the selected app
UID/GID modify access. Its intended layout is:

```text
/mnt/ssd_pool/radio-archive/
  .env                         # optional, mode 0600; never put in the image
  archives/
  models/
    llama/
    speaker-diarization/
    whisper/
  runtime/
```

The Web app creates missing subdirectories but never deletes retained archives
when it starts. Snapshot or replicate this dataset like other important data.

## 2. Build or publish the immutable image

Build from a clean Git commit so ignored credentials and local archives are not
part of the context:

```bash
git archive --format=tar --prefix=radio-archive/ COMMIT > source.tar
mkdir build && tar -xf source.tar -C build --strip-components=1
docker build \
  --file build/deploy/truenas/Dockerfile \
  --build-arg SOURCE_COMMIT=COMMIT \
  --tag radio-archive-intelligence:COMMIT \
  build
```

The image combines the official Vulkan builds of whisper.cpp and llama.cpp
with the Python Web/worker package and the portable sherpa-onnx CPU speaker
preview. Community-1 and faster-whisper are intentionally not baked into this
NAS image; their much larger Python/Torch stack remains an optional future
variant. Pin the two base-image digests for a reproducible production build.

The Compose definition advertises the exact installed pipeline as the Web
app's Automatic preset: Base English Q5_1 through whisper.cpp/Vulkan with the
pinned Silero VAD speech detector, portable sherpa-onnx speakers on CPU, and
the configured local llama.cpp model. VAD bounds sparse radio traffic to
detected speech regions of at most 25 seconds and previous-text conditioning is
disabled, preventing quiet hours from collapsing into repeated Whisper output.
This is both a browser default and a server-side compatibility boundary, so a
tab that was opened before an App upgrade cannot accidentally request the
absent PyTorch/Community-1 stack. Selecting an explicit named or custom profile
still wins. The sidebar reports **Trusted LAN** from the live bootstrap scope
rather than displaying a static loopback label.

A registry-hosted image works too. A local image is sufficient when the image
is built on the TrueNAS host before installing the Custom App.

## 3. Install through TrueNAS Apps

Copy `compose.example.yaml`, replace every uppercase placeholder, and submit it
through **Apps → Discover Apps → more menu → Install via YAML**. Use:

- the exact commit tag from the image build;
- the numeric app UID/GID that owns the host-path dataset;
- the host `render` group ID and `/dev/dri/renderD128` for AMD Vulkan;
- one IP shown by TrueNAS under app IP choices;
- an unused host port;
- UDP port `48765` on the same selected LAN address when one-hop peer discovery
  is wanted;
- the absolute dataset mount path.

The container listens on `0.0.0.0` only inside its private app network. The
published port is constrained to the chosen TrueNAS LAN address. The container
runs without added Linux capabilities, with `no-new-privileges`, a read-only
root filesystem, and only `/data` writable.

The example also makes this App a read-only archive seed. It publishes UDP
discovery port `48765`, advertises the reachable
`http://TRUENAS_LAN_IP:8765` address, and serves only original source blocks
from the persistent dataset. Other app instances ask this node before using a
Broadcastify archive request. Explicitly configuring the NAS URL in each client
is more reliable than UDP discovery across firewalls, VLANs, or App networking.

To limit archive reuse to clients with the same private value, add
`BROADCASTIFY_LAN_SYNC_KEY` to the dataset `.env` and every participating
client. This is an optional LAN access check, not encrypted transport. Do not
expose the service publicly. The complete protocol and data boundary are in
[Trusted-LAN archive reuse](../../docs/lan-archive-sync.md).

## Trusted-LAN boundary

LAN mode is an explicit opt-in. Anyone who can reach the published address can
open the page and receive a fresh session cookie/token, then read retained
transcripts and audio and start supported jobs. Same-origin checks, the random
action token, path containment, CSP, and quota controls still apply, but this
is not user authentication.

Use it only on a trusted LAN. Do not port-forward the port, attach it to a
public reverse proxy, or expose it through a public tunnel without adding a
real authentication/TLS layer.

Relevant TrueNAS guidance:

- [Installing Custom Apps](https://apps.truenas.com/managing-apps/installing-custom-apps/)
- [App storage](https://apps.truenas.com/getting-started/app-storage/)

## Validated reference deployment

On July 18, 2026, exact commit
`historical-validation` was built from pinned official
whisper.cpp/llama.cpp Vulkan image digests and installed through the TrueNAS
25.04 Apps API as `radio-archive-intelligence`. The App runs as numeric user
950 with only render group 107/device `renderD128`, publishes
`10.200.1.227:8765`, and mounts `ssd_pool/radio-archive` at `/data`.

A separate Windows client reached the LAN root, trusted-LAN health, protected
bootstrap, diagnostics, and job API without an SSH tunnel. Diagnostics found
the Radeon 890M as `Vulkan0`; the installed Automatic profile is Base English
Q5_1 plus Silero VAD on whisper.cpp/Vulkan, sherpa-onnx speakers on CPU, and
Gemma through llama.cpp/Vulkan. The final image ID is
`sha256:d45d4fd56e072648923c29ca61918fd4262bd4d6e7140654b260df9c5aee7a3e`
and its OCI revision matches the source commit.

Its 2,088,960-byte source archive matched SHA-256
`3b2ba6b681b8df744407dcc33d98e20f97464e9e75e6a5be7af5e85d984d62e8`
after transfer. The final TrueNAS-managed update retained all 462 existing
feed-90001 files and the exact 10,989,568-byte SQLite store at SHA-256
`bfe6fdf7b64b9e24c4ed2614336e9c2687de94ded1b4b1a1e7ea4da375a285a6`.
Feed 90001 retained 2,440,783,509 bytes across the deployment. The explicitly
requested post-update I864 export then added one 101,312-byte evidence clip;
the database remained byte-for-byte unchanged and passed its integrity check.

Database reads rebase stale absolute Windows/Linux audio, transcript, and
manifest paths to the conventional library layout beside the database. A
same-hash reimport persists the new paths without rebuilding derived evidence.
The live I864 browser action played retained audio and downloaded
`90001_2026-07-16_I864_64701450-64726650.mp3`; both copies match SHA-256
`c7ae78be5c68972508fae8eb1f033ed34a54270b27b89c7176654269fb7c4ad9`.
The Windows originals were copied, not moved or removed.
