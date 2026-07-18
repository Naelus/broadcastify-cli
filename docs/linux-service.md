# Managed Linux Web service

`radio-archive-service` installs the browser companion as a per-user
systemd service. It is the managed Linux launch path for the same Library,
acquisition, transcription, diarization, analysis, and evidence workflow used
by `broadcastify-web`.

The generated configuration uses `127.0.0.1` by default. An explicit
`--host` can instead select a private/link-local address or wildcard listener
for a trusted LAN. The Web app still creates a random session token at every
launch, requires its same-site cookie for data/media, and requires both the
cookie and action token for every mutation. These controls are not user
authentication: any client that can open the LAN page can obtain a session.

TrueNAS SCALE should use the supported Apps deployment documented in
[`deploy/truenas`](../deploy/truenas/README.md), not this user-systemd path.

## Install the Python package

Python 3.10 or newer, FFmpeg, and a functioning systemd user session are
required. From a source checkout:

```bash
python3 -m venv ~/.local/share/radio-archive/venv
~/.local/share/radio-archive/venv/bin/python -m pip install --upgrade pip
~/.local/share/radio-archive/venv/bin/python -m pip install \
  -e '.[transcription,analysis]'
```

Add `openvino` to the extras only for an OpenVINO profile:

```bash
~/.local/share/radio-archive/venv/bin/python -m pip install \
  -e '.[transcription,analysis,openvino]'
```

whisper.cpp and llama.cpp are native runtimes rather than Python wheels. Set
their executable/model paths in the private service environment file as
described in [hardware-backends.md](hardware-backends.md).

## Install and open the user service

Run this with the venv that contains the package:

```bash
~/.local/share/radio-archive/venv/bin/radio-archive-service install
~/.local/share/radio-archive/venv/bin/radio-archive-service status
~/.local/share/radio-archive/venv/bin/radio-archive-service start --open
```

To opt into a trusted-LAN listener on an ordinary Linux host:

```bash
~/.local/share/radio-archive/venv/bin/radio-archive-service install \
  --host 0.0.0.0
```

Use a host firewall, keep the network trusted, and do not publicly proxy or
port-forward this unauthenticated application.

The defaults are:

- working data and session cookie:
  `~/.local/share/radio-archive`
- audio, transcripts, evidence database:
  `~/.local/share/radio-archive/archives`
- private environment file:
  `~/.config/radio-archive/.env`
- owner-readable service log:
  `~/.local/share/radio-archive/radio-archive-web.log`
- non-secret service configuration:
  `~/.config/radio-archive/service.json`
- systemd user unit:
  `~/.config/systemd/user/radio-archive-web.service`
- local address:
  `http://127.0.0.1:8765/`

The installer creates directories and a comment-only `.env` template. It does
not copy an existing repository `.env`, prompt for credentials, download a
model, or contact Broadcastify. Add private values manually and keep the file
owner-readable only:

```bash
chmod 600 ~/.config/radio-archive/.env
```

To adopt an existing retained library:

```bash
radio-archive-service install --force \
  --working-dir /absolute/path/to/private-working-data \
  --output-dir /absolute/path/to/archives \
  --database /absolute/path/to/archives/broadcastify-analysis.sqlite3 \
  --env-file /absolute/path/to/private.env
```

The exact venv Python path is retained rather than resolving its symlink to the
system interpreter. This matters because the system interpreter usually does
not contain pyannote, Torch, OpenVINO, or the installed app.

## Lifecycle and diagnostics

```bash
radio-archive-service restart --open
radio-archive-service stop
radio-archive-service logs --lines 200
radio-archive-service logs --follow
radio-archive-service print-unit
radio-archive-service uninstall
```

`status` requires both an active systemd unit and the app's exact minimal
`/health` response with the configured loopback or trusted-LAN scope. Start/restart waits up to 15 seconds for that
response before directing the user to `radio-archive-service logs`. The unit
appends stdout and stderr to the owner-only working-directory log so diagnostics
remain available on appliances where an ordinary user cannot read the system
journal.

The unit restarts on process failure, sends SIGINT for bounded cleanup, keeps a
private temporary directory, uses an owner-only umask, prevents privilege
gains, and makes system locations read-only. It deliberately does not disable
outbound networking because website sign-in, archive acquisition, and explicit
first-time model downloads require it. Archive request pacing and quota stops
remain enforced by the shared worker, not by systemd.

`uninstall` removes only the unit. It preserves archives, transcripts, models,
the SQLite evidence store, service configuration, and `.env`.

## Session and non-systemd behavior

A user service normally follows that user's systemd session. For unattended
operation after logout, ask the machine administrator whether lingering is
appropriate for that account; the app does not enable it or change host policy.

On a Linux appliance without a usable systemd user session, use the foreground
entry point under the appliance's existing supervisor:

```bash
broadcastify-web \
  --working-dir /absolute/path/to/private-working-data \
  --output-dir /absolute/path/to/archives \
  --database /absolute/path/to/archives/broadcastify-analysis.sqlite3
```

Add `--host 0.0.0.0` (or an assigned private numeric address) only when this
foreground process is intentionally serving a trusted LAN. Public and
multicast numeric addresses are rejected.

## Validation boundary

Exact commit `historical-validation` was installed from its
hash-verified 0.4.0 wheel on systemd 252 under an ordinary TrueNAS user. The
generated unit passed `systemd-analyze`, started on loopback, served the
cookie/token-protected bootstrap, restarted after a deliberate process failure,
retained useful mode-0600 logs without journal permissions, stopped with no
listener, and uninstalled while preserving every user-data path. No host
package, system service, Broadcastify request, or model download was used in
that launcher test.

Exact `historical-validation` added the explicit trusted-LAN contract and regression coverage
without changing the loopback default. The separately managed TrueNAS Apps
deployment is validated from exact `historical-validation`; it should be preferred on that
appliance because TrueNAS owns its lifecycle, LAN port, GPU device mapping, and
persistent host-path storage.
