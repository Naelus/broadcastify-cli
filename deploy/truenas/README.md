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
- the absolute dataset mount path.

The container listens on `0.0.0.0` only inside its private app network. The
published port is constrained to the chosen TrueNAS LAN address. The container
runs without added Linux capabilities, with `no-new-privileges`, a read-only
root filesystem, and only `/data` writable.

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
