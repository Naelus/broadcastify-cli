# Credential storage and precedence

The application needs a Broadcastify website login for premium archives and
may need a Hugging Face read token for gated local models. These are local
application credentials, not official Broadcastify API credentials.

## Windows desktop

**Credentials** is a one-click footer item in the native navigation.

- The primary Broadcastify username/password retain the legacy
  `BroadcastifyDesktop.Broadcastify` Credential Locker resource so upgrades do
  not lose the existing login.
- Additional authorized accounts use isolated entries under
  `BroadcastifyDesktop.BroadcastifyProfiles`. The UI stores only a non-secret
  profile ID in schedules; username/password values never enter schedule JSON.
- The Hugging Face token uses
  `BroadcastifyDesktop.HuggingFace`.
- An explicitly remembered hosted-analysis key uses a separate resource.
- Password and token fields are blank on reopen. Status shows only a short,
  non-reversible prefix.
- The complete value is supplied only to a short-lived Python child
  environment. Each account receives a distinct cookie path and quota identity.
  Scheduled and explicit missing-day catch-ups may rotate sequentially across
  the locally authorized pool, reusing retained work between profiles.

Credential Locker entries are scoped to the current Windows account. Uninstall
does not silently delete them; **Forget** is the explicit removal action.

The desktop login follows Broadcastify's website sequence: it first opens the
login page, submits the form, follows only same-site redirects, and considers
the sign-in successful only after Broadcastify issues its premium session
cookie. An HTTP 302 alone is not treated as either success or failure because
the website also uses a 302 redirect when it rejects a login. If automatic
sign-in reports that Broadcastify rejected the login, verify the same username
or email address and password directly on the website before replacing the
saved Credential Locker entry.

Provider session files are replaced atomically. On Linux/TrueNAS each new or
refreshed session file is created owner-readable/writable only (`0600`), rather
than inheriting a service umask that could expose the cookie to other users.

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

Windows and TrueNAS store and authenticate both provider accounts separately;
credentials and cookies never synchronize over the LAN. For coordinated quota
state, the same non-secret profile ID (for example `default` or `secondary`)
must refer to the same provider account on both systems. Retained downloads and
matching transcript artifacts may then reconcile without copying either login.

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

For an authorized named profile such as `secondary`, an ignored
`.env.accounts` may alternatively contain:

```dotenv
BROADCASTIFY_AUTHORIZED_ACCOUNT_POOL=1
BROADCASTIFY_ACCOUNT_PROFILES=default,secondary
BROADCASTIFY_ACCOUNT_SECONDARY_USERNAME=
BROADCASTIFY_ACCOUNT_SECONDARY_PASSWORD=
```

The worker maps only the selected named pair to its short-lived standard
variables. If either named value is missing it does not fall back to the
primary account. `.env.accounts`, `.env`, Credential Locker contents, and
session cookies must never be staged or packaged in a public release.

## Hugging Face links

- [Create/manage user access tokens](https://huggingface.co/settings/tokens)
- [Token documentation](https://huggingface.co/docs/hub/en/security-tokens)
- [Accept Community-1 model terms](https://huggingface.co/pyannote/speaker-diarization-community-1)

Use a read token; no write-scoped token is needed for local model downloads.
