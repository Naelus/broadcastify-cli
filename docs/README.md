# Documentation

This index is the starting point for the documentation set. The root
[`README.md`](../README.md) is intentionally limited to orientation and a quick
start.

## Setup and operation

- [Windows setup and first run](guides/windows-setup.md)
- [CLI workflows](guides/cli.md)
- [Cross-platform Web UI](web-ui.md)
- [Linux user service](linux-service.md)
- [Windows publish layout](windows-publish.md)
- [TrueNAS Apps deployment](../deploy/truenas/README.md)

## Features

- [Archive acquisition, search, caching, and quota behavior](features/archive-acquisition.md)
- [Combination, transcription, and diarization](features/audio-processing.md)
- [Incident analysis, summaries, retrieval, and evidence clips](features/evidence-analysis.md)
- [Local Library and Review](features/library-and-review.md)
- [Area Watch, regional leads, and neighborhood subscriptions](features/area-watch.md)
- [Area coverage semantics](area-coverage.md)
- [Trusted-LAN archive sharing](lan-archive-sync.md)

## Design decisions

- [Why WinUI 3 and a browser companion](decisions/ui-and-platforms.md)
- [Why the model stack is split by stage](decisions/model-stack.md)
- [Evidence, identity, and privacy policy](decisions/evidence-and-privacy.md)

## Reference and validation

- [Storage layout and resumability](reference/storage-and-resume.md)
- [Hardware backends](hardware-backends.md)
- [Analysis providers](model-providers.md)
- [Archive rate limits](rate-limits.md)
- [Retained-corpus validation](validation/retained-corpus.md)

## Project records

- [`GOAL.md`](../GOAL.md) defines acceptance targets.
- [`FEATURES.md`](../FEATURES.md) is the status matrix.
- [`BUGS.md`](../BUGS.md) tracks reproducible defects and blockers.
- [`PROGRESS.md`](../PROGRESS.md) records dated implementation and validation
  evidence.

Detailed benchmark output belongs in the validation documents or
`PROGRESS.md`, not in the root README. Forward-looking work belongs in
`GOAL.md`/`FEATURES.md`; unresolved defects belong in `BUGS.md`.
