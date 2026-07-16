# Radius coverage and feed priority

Last checked: July 16, 2026.

Area Watch supports two explicit discovery modes:

- **Radius from ZIP** expands one center ZIP into up to 20 nearby Census ZIP Code Tabulation Areas (ZCTAs), queries the same Broadcastify website ZIP/county pages used by the normal feed search, deduplicates feed IDs, and saves the results in nearest-first acquisition order.
- **Ordered ZIP list** keeps the entered order as the acquisition priority and makes no mileage claim.

Radius discovery downloads the US Census Bureau's compressed 2025 national ZCTA Gazetteer on first use and keeps it in the platform cache. The compressed file is currently about 930 KB. The reader accepts one bounded text member, validates the expected columns, and never treats the dataset as executable content. See the [Census Gazetteer files](https://www.census.gov/geographies/reference-files/time-series/geo/gazetteer-files.2025.html) and [2025 ZCTA record layout](https://www.census.gov/programs-surveys/geography/technical-documentation/records-layout/gaz-record-layouts/gaz25-record-layouts.html).

The displayed mileage is deliberately labeled approximate. It is the great-circle distance from the center ZCTA's Census internal point to the nearest queried ZCTA that returned the feed's county directory. It is not the listener's exact location, the feed receiver, a transmitter, or an agency jurisdiction boundary. ZCTAs are Census statistical areas and do not cover every USPS-only or special-purpose ZIP; use an ordered ZIP list when the center ZIP is absent from the Gazetteer.

Discovery makes no archive-media requests. A user must review the deduplicated results, select the relevant public-safety feeds, and save the profile before acquisition. Multiple feeds returned by the same county can share the same approximate distance; their stable tie-breakers are coverage count, current listener count, and name. Listener count influences only order within a geographic tie and is not evidence of news value.

The persisted profile retains its center, radius, ZIP cap, searched ZIP centroids, explicit selected feeds, approximate distance, and priority rank. Native Windows and loopback Web clients now submit that reviewed profile to the same SQLite-backed queue. The queue fingerprints the non-secret processing settings and selected feed order, forces archive concurrency to one, skips completed feeds, recovers interrupted items, resumes the first incomplete feed against the local cache, and stops the entire profile when a feed returns `download_limited`. Hugging Face and provider keys are never part of the persisted queue fingerprint or settings JSON.

The queue UI shows the latest date range, status, stop reason, and per-feed completed/requested day counts. A cached July 11 feed-90001 smoke run completed in the browser without model work; running the exact queue again emitted only `already complete` and made no authentication, archive-list, or media call.
