# Broadcastify archive limits and measured behavior

Last checked: July 15, 2026.

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

The user had also successfully tested downloads before telemetry began. A boundary near 200 successful archive requests per account window is therefore plausible, but remains an inference rather than a published or proven quota. The observed behavior is consistent with a request budget, not a simple requests-per-minute rule.

## Implemented policy

- Default to one download worker and space real archive requests by at least five seconds.
- Reuse cached MP3s with an exact key derived from the archive `startTs` and the feed's published IANA timezone.
- Continue exponential backoff for genuinely transient 429/5xx/network failures.
- Treat the explicit `Download limit exceeded` response as quota exhaustion: stop after that one response, block queued workers, preserve files, and report the support address.
- Acquire a requested date range before loading the GPU transcription stack. If quota exhaustion interrupts acquisition, do not call another archive-download endpoint in that job; inspect later dates through metadata only, process any days already complete in the local cache, and report incomplete dates as resumable coverage gaps.
- Keep a backend-only JSONL probe at `scripts/download_rate_probe.py` for auditable, credential-free response timing/status telemetry.

Probe logs are runtime artifacts under `archives/` and are ignored by Git.
