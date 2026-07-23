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

- [Archive acquisition, per-feed schedules, caching, and quota behavior](features/archive-acquisition.md)
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

## Development records

- [`FEATURES.md`](../FEATURES.md) is the durable capability/status matrix.
- [GitHub Issues](https://github.com/Naelus/broadcastify-cli/issues) tracks
  reproducible defects and actionable enhancements.
- Local ignored `GOAL.md` and `PROGRESS.md` files are temporary working notes,
  not project history.

Detailed benchmark output belongs in the validation documents, not the root
README. Completed work remains discoverable through focused documentation and
Git history rather than an ever-growing progress ledger.
