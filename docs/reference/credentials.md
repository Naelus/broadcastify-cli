# Credential storage and precedence

The application needs a Broadcastify website login for premium archives and
may need a Hugging Face read token for gated local models. These are local
application credentials, not official Broadcastify API credentials.

## Windows desktop

**Credentials** is a one-click footer item in the native navigation.

- Broadcastify username/password use
  `BroadcastifyDesktop.Broadcastify` in Windows Credential Locker.
- The Hugging Face token uses
  `BroadcastifyDesktop.HuggingFace`.
- An explicitly remembered hosted-analysis key uses a separate resource.
- Password and token fields are blank on reopen. Status shows only a short,
  non-reversible prefix.
- The complete value is supplied only to a short-lived Python child
  environment. Scheduled jobs use the same path.

Credential Locker entries are scoped to the current Windows account. Uninstall
does not silently delete them; **Forget** is the explicit removal action.

## Browser/server UI

The browser never owns a saved secret. Its same-origin Credentials form sends a
new value to the local app server, immediately clears the field, and later
receives only status and a prefix preview.

The server store is:

- current-user DPAPI on Windows;
- AES-256-GCM on Linux/TrueNAS, with a randomly generated owner-only key file.

The encrypted payload is `.credentials.enc` under the working directory by
default. AES-GCM hosts also create `.credentials.enc.key`. Set
`BROADCASTIFY_CREDENTIAL_STORE` to select a persistent private path. TrueNAS
must place both files in its persistent app dataset. Backing up only the
encrypted payload without its key is not recoverable.

The POSIX design protects a copied payload and avoids plaintext configuration,
but an administrator or attacker who can read both the store and local key can
decrypt it. Service-account and dataset permissions remain essential.

## Precedence

For Broadcastify and Hugging Face values:

1. the native/server secure store;
2. an explicitly selected private environment file;
3. the repository `.env`;
4. a session-only value where that surface supports one.

Secure values override environment values only in the child worker. They are
not copied into `.env`, ordinary settings, schedules, analysis SQLite, logs, or
job snapshots.

## Hugging Face links

- [Create/manage user access tokens](https://huggingface.co/settings/tokens)
- [Token documentation](https://huggingface.co/docs/hub/en/security-tokens)
- [Accept Community-1 model terms](https://huggingface.co/pyannote/speaker-diarization-community-1)

Use a read token; no write-scoped token is needed for local model downloads.
