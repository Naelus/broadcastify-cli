# Regression scope and bounded library reads

The September 2026 review used source revision `110f2da` as its baseline.
Tests were evaluated by the product failure they detect, the boundary they
exercise, and whether existing coverage already proves the same outcome.
Test authorship is not evidence of relevance.

## Removed coverage and its replacement

| Removed tests | Reason and retained proof |
| --- | --- |
| Five location/name permutations in `test_area_watch.py` | The current quote policy preserves names and ignores location. The remaining quote test proves identifier redaction while retaining names; incident and HTTP story tests prove the rendered result. |
| Direct haversine example in `test_geography.py` | The cached radius-catalog test checks the same coordinates and distance through the actual selection path. |
| Processing-limit field parsing in `test_models.py` | The scheduled-processing workflow now parses the incoming value and proves that excess retained days are deferred. |
| Standalone scheduled-analysis selection in `test_worker.py` | The scheduled-job test proves that current transcripts are skipped and changed transcripts reach analysis. |
| Two mocked question-coverage wrappers in `test_worker.py` | The product workflow invokes real worker subprocesses for both month and entire-feed scope using retained audio, transcripts, and a real database. |
| Five web settings pass-through examples | The combined-profile test covers the common payload path; deployment-default tests cover automatic/custom selection. Worker and adapter tests retain each model stage's execution, failure, and secret-handling coverage. |
| Two Dockerfile source-string inventories in `test_linux_service.py` | Text presence and order cannot prove executable permissions or image metadata. Image construction remains a container-release responsibility; offline Python results do not claim to verify a container build. |

Seventeen test functions were removed. The nested-job acquisition guard test
also covers scheduled jobs, so the collected suite changes from 461 to 445
cases. Parameterization alone is not counted as removing redundant coverage.

## Retained boundaries

| Area | Why its coverage remains |
| --- | --- |
| Authentication, credential store, quota, downloads | Redirect restrictions, account isolation, persistent rolling limits, exact source identities, and cache reuse prevent credential exposure or excess website requests. |
| Feed search, geography, requests, schedules, area acquisition | Parser fixtures represent external response formats; validation and database transitions protect date selection, ordering, restart, and account handoff. |
| Jobs, LAN synchronization, worker and HTTP workflows | Actual local files, databases, sockets, subprocesses, and bounded external fakes prove ordering, leases, independent accounts, source/result ownership, and resume. |
| Storage and Library | Mutation, migration, file locks, deletion recovery, mixed imported paths, and stale evidence are distinct regressions. Similar fixtures do not make these outcomes interchangeable. |
| Audio, transcription, ASR, portable diarization, Qwen | Keep timestamp continuity, model identity, chunk offsets, fallback, interrupted checkpoints, and completed-artifact validation. Native/model adapters are faked at the external boundary to keep the suite offline. |
| Analysis, analysis clients, Area Watch | Keep evidence support, privacy, bounded context, external-provider consent, checkpoint reuse, coverage gaps, and stale result rejection. |
| Accelerator/runtime setup and Linux service | Keep capability selection, cache compatibility, verified installation/resume, safe service configuration, and runtime validation. |
| Native frontend, build and release | Keep compiled layout/docking workflows, release-scanner execution, version consistency, and minimal structural build-gate checks. No source-text inventory substitutes for native execution. |

## Production changes justified by the review

Single-day reports previously scanned and hashed unrelated feeds before selecting
one result. Library scanning now accepts a feed and inclusive date bounds and
applies them before inspecting retained artifacts. The database query uses the
same bounds. Day views, local-processing completion, selected-feed analysis,
scheduled analysis, and questions use those scoped reads. Full Library and
global resume operations still discover all retained days.

Range-evidence checks inspect only the database evidence the consumer can use.
They still compare live transcript and audio state on every request. No durable
cache or new schema is introduced. An unavailable unrelated feed cannot break a
single-day report; a rewritten recording must still hide its older analysis.

Completion checking previously reread a day's identity index once per archive
block. Batch resolution now shares one index snapshot within that operation,
while checking each file's identity, path, and size. The next operation rereads
the index and files. A malformed existing completion marker can be replaced
after all required retained files have been validated.

Direct, scheduled, and area web acquisitions now share the same quota-safe
payload rules instead of maintaining three copies. Account allowances,
sequential acquisition, retention defaults, and explicit LAN preferences remain
unchanged.

For a temporary fixture containing 32 days and 48 source blocks per day, the
previous full-scan path read 1,568 identity indexes before selecting a single
day; the scoped path read two (source coverage and alias detection). The selected
state was identical. A local comparison measured approximately 288 ms versus
8 ms; timing depends on storage and workload. The regression checks protect
bounded disk reads and freshness rather than asserting machine-specific timing.
