# Shared master jobs

The NAS **Master jobs** page and Windows Library **Jobs** button read the same
Windows-owned queue. Each row includes its stable ID, feed and dates, requested
work, state, saved-stage counts, current worker step, and last report. Saved
schedules are separate from actual running or queued jobs. A current-step
counter describes that stage; it is not an estimate for the whole workflow.

The persistent Library status and NAS workflow view show a separate full
catch-up estimate across retained work. Job rows may also show an estimate for
that job's requested scope; queued jobs include known work ahead of them. Both
use a finish-time range based on recent successful local stage measurements
for the active machine and profile. New profiles show a learning state until
there is enough local timing data, and estimates based on provisional input
sizes are labeled. Source publication and quota availability cannot be
predicted, so those waits are shown without a confident finish time. Pause,
failure, data older than two minutes, or master disconnection also suppress
finish times. These estimates are advisory and do not change completion checks.

## Submit and control work

Choose the followed feed, today or an explicit date range, and full workflow or
downloads only. Submission uses that feed's saved Windows processing profile.
The NAS can select several feeds; each receives its own job ID. New work queues
behind active model work, and closing the browser does not interrupt it.

- **Pause** checkpoints/stops the selected job and preserves completed files.
  Automatic guarded acquisition may continue while model jobs are paused.
- **Resume unfinished work** reuses the same identity and saved checkpoints.
  It also retries failed or canceled work.
- **Cancel** stops that job and its owned worker descendants. It does not delete
  retained evidence or disable an independent recurring schedule.
- **Run next** moves a queued job ahead of the other queued jobs. It does not
  interrupt the active job.

Acquisition remains sequential under the existing global lease and persistent
rolling account limits. Cached and LAN sources are tried even when provider
allowance is exhausted. Automatic account policy permits sequential handoff;
an explicit account policy remains explicit across recovery. Third-and-later
accounts retain the configured allowance for recent days.

## Completion and recovery

The master owns acquisition, local processing, analysis and indexing. It checks
the saved artifacts for the stages requested by the job before claiming
completion; a worker's successful exit alone is insufficient. Downloads-only
completion does not mean the audio has been analyzed.

Source freshness is evaluated for the submitted scope. A source check made
since submission satisfies that job even if model work lasts beyond the
periodic refresh interval. Later source publication belongs to another catch-up
or the recurring schedule. Confirmed unavailable days and recently empty days
awaiting publication are shown separately and are not counted as analyzed.
Quota waits retain the job and resume when eligible allowance or newly retained
source proof resolves the missing work. Newly available local processing does
not wait for provider allowance. Older native quota-wait records reconcile
against saved completion from later runs without launching another worker.
Native account handoffs and quota retries of the same saved request reuse its
original identity rather than accumulating duplicate unfinished jobs.

The existing pipeline journal stores job requests and records without embedded
credentials. Master restart resumes queued/interrupted work from retained
checkpoints. Interrupted native jobs transfer to the master queue; explicit
pauses survive restart. Recent completed/failed/canceled history is bounded in
the list. Initial native registration must succeed before model work starts,
so an unavailable companion cannot silently create competing unobserved work.

## Review from a follower

NAS day details, cited incidents, transcript search and audio playback read
saved master results through the authenticated LAN channel. Completed Windows
analysis therefore appears complete on NAS before its local copy finishes.
The local transcript/copy indicator describes storage, not model processing.
Opening or refreshing review does not launch analysis. If Windows is offline,
the UI reports the lost connection and treats displayed job data as stale.

## Verification

The paired-server offline workflow exercises submission, shared identities,
reconnect, pause/resume, saved profiles, missing-index continuation and matching
finished evidence from an empty follower, including byte-range playback and
path containment. A separate durable-workflow check covers exhausted quota,
source proof supplied by another acquisition, and explicit pause across
restart. Owned-process shutdown is checked with a real child/grandchild and a
retained checkpoint. These checks use temporary data and no provider requests
or live models. Native build/UI and installed two-machine verification remain
release gates; offline success does not establish real-world processing speed.
