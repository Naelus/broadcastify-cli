# Area Watch and regional leads

## Discovery and profiles

Area Watch supports either a center ZIP plus radius or an explicit ordered ZIP
list. Radius mode uses Census ZCTA centroids to choose nearby ZIP searches, then
follows Broadcastify's ZIP-to-county directories. Distances are discovery
approximations, not incident geofences.

Results are deduplicated by feed. A checked-by-default public-safety filter hides
weather/rail categories without another website request. The user reviews and
explicitly saves the feeds and nearest-first order in a named profile.

See [area coverage semantics](../area-coverage.md).

## Acquisition queue

The persisted queue records every profile/feed/date stop point. It skips
completed feeds, resumes interrupted items against exact local/LAN blocks, and
stops every lower-priority feed after the first explicit quota response. This is
important in rural areas: one feed can support a high-confidence story; multiple
feeds are not required.

Cross-feed overlap increases interest but is never called independent
confirmation because different feeds can rebroadcast the same talkgroup.

## Story construction

Only current, evidence-gated incidents participate. Clustering requires a
compatible event type, nearby time, and either a compatible reported location or
unusually strong descriptive overlap. Newsworthiness is separate from dispatch
priority, and routine single-person calls are suppressed.

Each ranked lead retains:

- exact feed and incident IDs;
- reported time and location;
- a name-preserving, identifier-masked ASR quote;
- a playable/exportable exact evidence clip;
- confidence and anonymous speaker context;
- source and clip SHA-256 values; and
- explicit coverage gaps.

Gemma writes only the readable assignment narrative. Both UIs keep that prose
collapsed by default so coverage and source evidence remain primary.

## Publication boundary

Every candidate remains `review_required`. Scanner-derived reports are noisy,
unconfirmed leads and can contain allegations or identifiers in the raw audio.
They should not be auto-published.

Neighborhood/topic tags are a foundation for a future subscription surface:
users could choose a center/radius and topics, but delivery should include only
editor-approved stories with consent and unsubscribe controls. Raw scanner clips
should remain private verification aids unless separately reviewed and licensed.
