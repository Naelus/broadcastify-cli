# Broadcastify archive limits and measured behavior

Last checked: July 20, 2026.

## Public information

Broadcastify Premium advertises access to 365 days of live-feed archives, which are stored in roughly 30-minute blocks. Neither the premium page nor the archive page publishes a numeric download quota or reset schedule.

- [Broadcastify Premium](https://www.broadcastify.com/premium/)
- [Broadcastify Feed Archives](https://status.broadcastify.com/archives/)

An independent Broadcastify Archive Toolkit documents a voluntary convention of serial downloading with at least five seconds between valid MP3 requests, or about 12 files per minute. It does not claim that this is an official quota.

- [Broadcastify Archive Toolkit: Download Throttling](https://ljhopkins2.github.io/broadcastify-archtk/user-guide/downloading-audio-files.html#download-throttling)

A July 2026 search found no indexed Broadcastify or RadioReference post publishing a numeric archive-download limit. In a recent personal transcription-tool discussion, the Broadcastify/RadioReference owner said that a premium subscriber capturing a few feeds for a personal project was acceptable, but warned against large multi-instance collection and commercial activity.

- [Radio-transcription discussion and owner comments](https://www.reddit.com/r/policescanner/comments/1sajnwo/an_update_for_a_python_script_for_transcribing/)

The formal terms are controlling. They reserve the right to establish usage limits and restrict commercial use and machine-learning/AI use without a license. A newsroom or subscription product therefore needs written Broadcastify licensing regardless of technical rate-limit handling.

- [Broadcastify Terms and Conditions](https://www.broadcastify.com/terms/)

## Reference-feed experiment

Feed `90003` (Example Regional Public Safety, Example State) was tested through the authenticated website archive endpoints for July 3–9, 2026.

Observed state and responses:

- The July 3 archive listing returned 47 blocks and the feed timezone `America/Chicago`.
- Exactly 40 MP3 blocks had downloaded before the original job began returning HTTP 429. The old progress code incorrectly counted completed failed futures, which is why the UI had displayed values such as 46/47 and 47/47.
- Seven early-day blocks remained absent. The archive URL IDs use `feed_id-start_epoch`, while saved filenames use the feed-local start minute and a different middle identifier. This mismatch previously caused reruns to overlook cache entries and redownload completed audio.
- Metadata-list requests continued returning HTTP 200 while every missing audio request returned HTTP 429.
- The 429 persisted for more than seven hours. Controlled retries after approximately 32, 64, and 124 seconds were also rejected.
- The response supplied no `Retry-After`, `RateLimit-*`, or `X-RateLimit-*` headers.
- Its complete 76-byte body was: `Download limit exceeded - contact support@broadcastify.com for more details.`

The initial count of 40 did **not** prove a 40-file quota. It was only the number retained by the reference-feed run after earlier Example City requests and cache-mismatch duplicates had already consumed an unknown portion of the same account window. Broadcastify explicitly directs users to `support@broadcastify.com` for exact quota details.

### July 15 reset follow-up

Archive downloads worked again the following day. This establishes that the observed block cleared no later than the next day, but does not distinguish a midnight reset from a rolling window.

The resumable reference-feed run then recorded:

- 192 successful archive-download endpoint redirects before one explicit quota 429.
- 188 successful MP3 responses; 173 were unique newly retained files and 15 were redundant during a brief accidental two-process overlap while the monitor handoff was being corrected.
- Four complete days retained for July 3–6, plus 22 of 48 blocks for July 7.
- No transient 429 before the final explicit `Download limit exceeded` response, despite serial five-second pacing throughout.

The user had also successfully tested downloads before telemetry began. At that point a boundary near 200 successful archive requests per account window appeared plausible, but it was still an inference rather than a published or proven quota.

### July 16 availability follow-up

After archive downloads became available again, a guarded feed `90001` resume requested July 3–16. The process ran from 2:00:36 AM to 2:06:40 AM Central time and produced:

- 55 successful new media downloads at serial five-second pacing: all 48 July 3 blocks, then 7 of 48 July 4 blocks.
- One explicit `Download limit exceeded` response on the next July 4 request.
- No retry of that explicit quota response and no later media-download requests in the range.
- A combined July 3 recording, reuse of the already-complete July 11–12 caches, and safely resumable gaps for every incomplete day.

This smaller observed window is strong evidence against documenting a stable fixed 200-request daily quota. It could reflect a rolling/shared budget, only a partial reset, prior account/IP use outside this process, or server-side policy that varies by context. The only defensible operational conclusion is that download availability can return by a later day, while neither the amount restored nor the reset boundary is predictable from public information.

### July 17 retained two-day follow-up

A later guarded run acquired the two latest complete Example City Public Safety (`90001`) days at the same serial five-second pacing:

- 48 of 48 July 16 archive blocks.
- 49 of 49 July 15 archive blocks.
- 97 successful new media responses in total.
- No HTTP 429 or explicit quota response during the run.

The account had only one user-initiated archive download immediately before the earlier development work. This result is therefore another lower-bound observation, not evidence of a fixed 97-request allowance. Together, the measured 55-success/exhausted, 97-success/not-exhausted, and roughly 192-success/exhausted windows are consistent only with an unknown dynamic, rolling, shared, or policy-dependent budget. The application must continue to react to the server response rather than predict a reset or preallocate a numeric quota.

### July 20 current-tail follow-up

Feed `90001` was checked through the saved premium website login under the
normal shared `default` LAN acquisition lease. Its newest two completed
archive entries were requested sequentially at the normal five-second pace:

- current completed entry `90001-1784534106` became
  `202607200255-866789-90001.mp3`, 3,728,000 bytes;
- immediately previous entry `90001-1784532316` became
  `202607200225-389928-90001.mp3`, 3,728,000 bytes;
- both requests succeeded, with no HTTP 429 or explicit quota response.

This two-request success says nothing new about quota size. It does prove the
live-tail path and leaves both exact blocks durable on the NAS. A retained validation revision now always acquires those newest two entries before older backlog,
refreshes the feed-local current-day listing once at the end, and treats a
successful today/yesterday LAN manifest as a five-minute rolling snapshot.
The six-hour explicit-quota suppression remains unchanged.

## Implemented policy

- Before website login or archive access, ask enabled trusted-LAN peers for the exact feed/date source-block inventory. Reuse only size- and SHA-256-verified blocks and publish them atomically.
- Coordinate each quota-scope/feed/day through a renewable LAN lease. One eligible producer may issue upstream archive-media requests; followers pull blocks from any peer as they appear and skip Broadcastify entirely after assembling the producer's exact filename/size/SHA-256 completion manifest. Expired leases permit takeover, while a shared explicit quota result suppresses follower retries for a bounded period.
- Within each archive listing, acquire the newest completed track and the immediately previous track before older backlog. Refresh a feed-local current-day listing once after acquisition so a track finalized during the job is included.
- Treat a successful today/yesterday LAN completion as a five-minute rolling snapshot, then allow one new producer to check for a later block. Keep old completed days and every explicit quota result on the full shared result interval.
- Default to one download worker and space real archive requests by at least five seconds.
- Reuse cached MP3s with an exact key derived from the archive `startTs` and the feed's published IANA timezone.
- Continue exponential backoff for genuinely transient 429/5xx/network failures.
- Treat the explicit `Download limit exceeded` response as quota exhaustion: stop after that one response, block queued workers, preserve files, and report the support address.
- Acquire a requested date range before loading the GPU transcription stack. If quota exhaustion interrupts acquisition, do not call another archive-download endpoint in that job; inspect later dates through metadata only, process any days already complete in the local cache, and report incomplete dates as resumable coverage gaps.
- Keep a backend-only JSONL probe at `scripts/download_rate_probe.py` for auditable, credential-free response timing/status telemetry.
- Never use a guessed numeric quota to pre-spend a range. Radius profiles persist an approximate nearest-first feed order from Census ZCTA/county-directory matches. The shared profile runner spends in that order, persists each item, and stops all lower-priority feeds on the server's explicit limit response.
- The cross-platform Web UI enforces the same policy at its service boundary: it overwrites single-feed and area jobs to one download worker, preserves source blocks, allows only one heavy job at a time, and cannot bypass the downloader's explicit quota stop through parallel browser actions.
- Archive progress identifies every completed block as **cached locally** or
  **downloaded from Broadcastify** and includes its retained filename. A LAN
  transfer is logged separately with the exact peer-sourced block. Cache hits
  advance completion but do not consume or claim a new website media request.

The LAN data plane is deliberately read-only and limited to original archive
MP3s. Its only mutation is bounded transient acquisition-lease state; it has no
archive upload or remote-job endpoint and does not share credentials,
transcripts, analysis data, combined audio, or evidence clips. See
[Trusted-LAN archive reuse](lan-archive-sync.md).

Probe logs are runtime artifacts under `archives/` and are ignored by Git.
