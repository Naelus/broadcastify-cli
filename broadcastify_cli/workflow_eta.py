"""Measured workflow estimates. Timing never decides whether evidence is complete.

Samples live beside the durable jobs and contain only rates/configuration hashes.
Cache hits and failed attempts cannot teach the estimator an artificially fast rate.
The master alone computes estimates; followers display the same timestamped result.
"""
from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import asdict, is_dataclass
from datetime import date, datetime, timezone
from functools import wraps
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import sqlite3
import statistics
import time
import uuid

from .pipeline_sync import PIPELINE_SYNC_DATABASE

LABELS = {"combine": "Combine audio", "audio_setup": "Load audio models",
          "transcribe": "Transcription", "speakers": "Speaker labels",
          "analysis_setup": "Import and load analysis", "analysis": "Incident analysis",
          "summary": "Daily summaries", "index": "Search indexing", "download": "Downloads"}
_scope = ContextVar("workflow_timing_scope", default=None)
_timer = ContextVar("workflow_stage_timer", default=None)
_FIELDS = {
    "combine": (), "download": (),
    "audio_setup": ("model", "asr_engine", "device", "device_index", "compute_type", "diarize", "diarization_engine", "diarization_device", "asr_model_path"),
    "transcribe": ("model", "asr_engine", "device", "device_index", "compute_type", "batch_size", "asr_model_path"),
    "speakers": ("diarization_engine", "diarization_device", "device_index", "min_speakers", "max_speakers"),
    "analysis_setup": ("analysis_provider", "analysis_model", "analysis_device", "analysis_endpoint"),
    "analysis": ("analysis_provider", "analysis_model", "analysis_device", "analysis_endpoint"),
    "summary": ("analysis_provider", "analysis_model", "analysis_device", "analysis_endpoint"),
    "index": (),
}
_DEFAULTS = {"model":"turbo", "asr_engine":"auto", "device":"auto", "device_index":0,
             "compute_type":"auto", "batch_size":8, "diarization_engine":"community-1",
             "diarization_device":"auto", "analysis_provider":"local",
             "analysis_model":"ggml-org/gemma-4-12B-it-GGUF:Q4_0", "analysis_device":"auto"}


def profile_key(stage, settings):
    # A different machine/profile must calibrate independently. No keys, paths or
    # endpoint addresses are retained: only this one-way identity is stored.
    from .analysis import PROMPT_VERSION
    values = [platform.node(), platform.machine()]
    if stage in {"analysis_setup", "analysis", "summary"}:
        from .analysis_providers import AnalysisProviderConfig
        # `model` in a catch-up profile selects ASR, not the language model.
        provider = AnalysisProviderConfig.from_mapping({k:v for k,v in settings.items() if k != "model"})
        values.append(PROMPT_VERSION)
        values += [provider.provider, provider.model, provider.device, provider.endpoint]
    else:
        values += [settings.get(k, _DEFAULTS.get(k)) for k in _FIELDS[stage]]
    return hashlib.sha256(json.dumps(values, sort_keys=True, default=str).encode()).hexdigest()[:24]


class TimingStore:
    def __init__(self, root):
        self.path = Path(root) / PIPELINE_SYNC_DATABASE
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as c:
            c.execute("CREATE TABLE IF NOT EXISTS workflow_timings (id TEXT PRIMARY KEY, stage TEXT, profile TEXT, units REAL, seconds REAL, measured REAL)")
            c.execute("CREATE INDEX IF NOT EXISTS workflow_timings_profile ON workflow_timings(stage,profile,measured)")
            c.execute("CREATE TABLE IF NOT EXISTS workflow_active_timing (id TEXT PRIMARY KEY, stage TEXT, profile TEXT, feed TEXT, day TEXT, units REAL, started REAL, pid INTEGER)")

    def connect(self):
        return sqlite3.connect(self.path, timeout=2)

    def sample(self, stage, profile, units, seconds):
        if not all(math.isfinite(v) and v > 0 for v in (units, seconds)):
            return
        with self.connect() as c:
            c.execute("INSERT INTO workflow_timings VALUES (?,?,?,?,?,?)", (uuid.uuid4().hex, stage, profile, units, seconds, time.time()))
            c.execute("DELETE FROM workflow_timings WHERE stage=? AND profile=? AND id NOT IN (SELECT id FROM workflow_timings WHERE stage=? AND profile=? ORDER BY measured DESC LIMIT 40)", (stage,profile,stage,profile))

    def rates(self, stage, profile):
        with self.connect() as c:
            rows = c.execute("SELECT seconds/units FROM workflow_timings WHERE stage=? AND profile=? AND measured>? ORDER BY measured DESC LIMIT 20", (stage,profile,time.time()-30*86400)).fetchall()
        rates = [r[0] for r in rows if r[0] > 0 and math.isfinite(r[0])]
        if not rates:
            return None
        center = statistics.median(rates)
        # An interval, not a promise. One sample gets a deliberately wider band.
        return (min(min(rates),center*.65), max(max(rates),center*(2 if len(rates)<3 else 1.4)), len(rates))


def workflow_scope(function):
    """Use the request already validated by the existing worker path."""
    @wraps(function)
    def wrapped(value, *args, **kwargs):
        request = getattr(value, "request", value)
        settings = asdict(request) if is_dataclass(request) else dict(request)
        root = Path(settings.get("output_dir") or "archives")
        token = _scope.set((root, settings))
        try:
            return function(value, *args, **kwargs)
        finally:
            _scope.reset(token)
    return wrapped


class StageTimer:
    def __init__(self, stage, units=1, day=""):
        self.stage, self.units, self.day = stage, units, str(day)
        self.skip = False
        self.children = 0.0


@contextmanager
def measure(stage, units=1, day=""):
    timer = StageTimer(stage, units, day)
    scope = _scope.get()
    if scope is None:
        yield timer
        return
    root, settings = scope
    parent = _timer.get()
    token = _timer.set(timer)
    started = time.monotonic()
    identity = uuid.uuid4().hex
    store = None
    profile = profile_key(stage, settings)
    try:
        try:
            store = TimingStore(root)
            with store.connect() as c:
                c.execute("INSERT INTO workflow_active_timing VALUES (?,?,?,?,?,?,?,?)", (identity,stage,profile,str(settings.get("feed_id", "")),str(day or settings.get("archive_date") or ""),units,time.time(),os.getpid()))
        except (OSError, sqlite3.Error):
            store = None  # Optional telemetry must never stop acquisition/work.
        yield timer
        seconds = time.monotonic()-started-timer.children
        if store and not timer.skip:
            try:
                store.sample(stage, profile, float(timer.units), max(.001, seconds))
            except (OSError, sqlite3.Error):
                pass
    finally:
        if store:
            try:
                with store.connect() as c:
                    c.execute("DELETE FROM workflow_active_timing WHERE id=?", (identity,))
            except (OSError, sqlite3.Error):
                pass
        if parent:
            parent.children += time.monotonic()-started
        _timer.reset(token)


def skip_sample():
    timer = _timer.get()
    if timer:
        timer.skip = True


def timed_stage(stage):
    def decorate(function):
        @wraps(function)
        def wrapped(*args, **kwargs):
            with measure(stage):
                return function(*args, **kwargs)
        return wrapped
    return decorate


def audio_seconds(audio):
    """Read saved timeline metadata; never decode audio to draw a progress bar."""
    audio = Path(audio)
    try:
        value = json.loads(audio.with_suffix(".manifest.json").read_text(encoding="utf-8"))
        duration = sum(float(s["trimmed_duration_seconds"]) for s in value["sources"])
        if math.isfinite(duration) and duration > 0:
            return duration
    except (OSError, ValueError, KeyError, TypeError):
        pass
    try:
        value = json.loads((audio.parent/"transcripts"/(audio.stem+".json")).read_text(encoding="utf-8"))
        duration = float(value.get("duration") or 0)
        return duration if math.isfinite(duration) and duration > 0 else 0
    except (OSError, ValueError, TypeError):
        return 0


def timed_audio(stage):
    def decorate(function):
        @wraps(function)
        def wrapped(self, audio, *args, **kwargs):
            path = Path(audio)
            with measure(stage, audio_seconds(path), path.parent.name) as timer:
                result = function(self, audio, *args, **kwargs)
                if stage == "transcribe":
                    timer.units = audio_seconds(path)
                return result
        return wrapped
    return decorate


def estimate_work(root, settings, work, *, now=None):
    """Estimate only unfinished saved work, using the master's measured rates."""
    now = time.time() if now is None else now
    store = TimingStore(root)
    from .shared_jobs import process_alive
    with store.connect() as c:
        active = c.execute("SELECT stage,profile,feed,day,units,started,pid FROM workflow_active_timing").fetchall()
    active = [r for r in active if process_alive(r[6]) and r[2] == str(settings.get("feed_id") or "")]
    stages, unknown, low, high = [], [], 0.0, 0.0
    # A whole-library estimate may contain thousands of day/stage pairs. Read
    # each calibration once, rather than opening SQLite for every pair.
    profiles = {item["stage"]: profile_key(item["stage"], settings) for item in work}
    rates = {stage: store.rates(stage, profile) for stage, profile in profiles.items()}
    for item in work:
        stage, units = item["stage"], item["units"]
        rate = rates[stage]
        entry = {"stage":stage,"label":LABELS[stage],"day":item.get("day", ""),"units":units}
        if units <= 0 or rate is None:
            unknown.append(LABELS[stage])
            entry["state"] = "learning" if units > 0 else "waiting_for_input"
        else:
            lower, upper, samples = rate
            elapsed = max((now-r[5] for r in active if r[0]==stage and r[1]==profiles[stage] and r[3].replace('-','')==item.get('day','').replace('-','')), default=0)
            # This outer scope excludes nested inference from its saved sample.
            # Its wall-clock age cannot be subtracted from exclusive setup time.
            if stage == "analysis_setup":
                elapsed = 0
            if elapsed > upper*units:
                unknown.append(LABELS[stage]+" is taking longer than its measured range")
                entry["state"] = "overdue"
            else:
                a,b = max(0,lower*units-elapsed), max(0,upper*units-elapsed)
                low += a; high += b
                entry.update(state="estimated",lower_seconds=round(a),upper_seconds=round(b),samples=samples)
        stages.append(entry)
    return {"lower_seconds":round(low),"upper_seconds":round(high),"unknown_stages":sorted(set(unknown)),"stages":stages,"as_of":datetime.fromtimestamp(now,timezone.utc).isoformat()}


def remaining_work(root, database, settings, days, analyze):
    """Build work from retained artifacts, not ordinal pipeline percentages.

Untranscribed days use this feed's retained speech density, explicitly marked
provisional. Completed window checkpoints are matched by content exactly as in
the analyzer; a cached window contributes no remaining inference time.
"""
    from .analysis import build_transcript_windows, PROMPT_VERSION, WINDOW_PROMPT_VERSION, DEFAULT_EMBEDDING_MODEL
    from .analysis_providers import AnalysisProviderConfig
    from .audio import list_source_mp3s
    from .storage import AnalysisStore
    work, provisional = [], []
    def add(stage, units, day):
        work.append({"stage":stage, "units":units, "day":day})
    with AnalysisStore(database) as store:
        needs_density = analyze and settings.get("transcribe") and any(not d.get("has_transcript") and not (d.get("source_unavailable") or d.get("awaiting_source")) for d in days)
        history = store.connection.execute("SELECT id,duration_seconds FROM feed_days WHERE feed_id=? AND duration_seconds>0 ORDER BY archive_date DESC LIMIT 3", (str(settings.get("feed_id")),)).fetchall() if needs_density else []
        densities = []
        for row in history:
            segments = store.get_segments(row["id"])
            chars = sum(len(w.prompt_text()) for w in build_transcript_windows(segments))
            if chars:
                densities.append((chars/row["duration_seconds"], len(segments)/row["duration_seconds"]))
        for day in days:
            if day.get("source_unavailable") or day.get("awaiting_source"):
                continue
            label = day["archive_date"]
            folder = Path(root)/str(settings["feed_id"])/label.replace("-", "")
            audio = folder/f"combined_{settings['feed_id']}_{label.replace('-', '')}.mp3"
            duration = audio_seconds(audio)
            if not duration and day.get("raw_file_count"):
                duration = day["raw_file_count"]*1800
                provisional.append("Uncombined source lengths use approximate 30-minute blocks")
            if settings.get("combine") and not day.get("has_combined"):
                add("combine", sum(p.stat().st_size for p in list_source_mp3s(folder)), label)
            needs_asr = settings.get("transcribe") and not day.get("has_transcript")
            needs_speakers = settings.get("diarize") and not day.get("has_diarization")
            if needs_asr or needs_speakers:
                add("audio_setup", 1, label)
            if needs_speakers:
                add("speakers", duration, label)
            if needs_asr:
                add("transcribe", duration, label)
            if not analyze or (day.get("has_analysis") and day.get("has_search_index")):
                continue
            stored = store.get_day(str(settings["feed_id"]), date.fromisoformat(label))
            if needs_asr:
                chars = statistics.median(x[0] for x in densities)*duration if densities else 0
                count = statistics.median(x[1] for x in densities)*duration if densities else 0
                provisional.append("Future transcript size uses recent retained activity on this feed")
            else:
                segments = store.get_segments(stored["id"]) if stored and day.get("has_imported_transcript") else []
                if not segments:
                    try:
                        data = json.loads((folder/"transcripts"/(audio.stem+".json")).read_text(encoding="utf-8"))
                        segments = [{"segment_index":i,"start_seconds":s["start"],"end_seconds":s["end"],"text":s["text"],"speaker":s.get("speaker")}
                                    for i,s in enumerate(data.get("segments") or [])]
                    except (OSError,ValueError,KeyError,TypeError):
                        pass
                count = len(segments)
                chars = 0
                model = AnalysisProviderConfig.from_mapping({k:v for k,v in settings.items() if k != "model"}).cache_model
                for index, window in enumerate(build_transcript_windows(segments)):
                    prompt = window.prompt_text()
                    fingerprint = hashlib.sha256(f"{window.start_seconds:.3f}|{window.end_seconds:.3f}|{prompt}".encode()).hexdigest()
                    checkpoint = store.get_analysis_window_checkpoint(stored["id"], model, WINDOW_PROMPT_VERSION,
                        stored["transcript_sha256"],index,fingerprint) if stored and day.get("has_imported_transcript") else None
                    if checkpoint is None:
                        chars += len(prompt)
            if not day.get("has_analysis"):
                add("analysis_setup", 1, label)
                if chars or needs_asr:
                    add("analysis", chars, label)
                add("summary", 1, label)
            if not day.get("has_search_index"):
                # Passage packing changes with speech density. Existing imports
                # use their exact missing passages; upstream work is provisional.
                pending = len(store.passages_missing_embeddings(DEFAULT_EMBEDDING_MODEL, feed_id=str(settings["feed_id"]),
                    start_date=date.fromisoformat(label), end_date=date.fromisoformat(label))) if stored and not needs_asr else 0
                add("index", pending or max(1, count/12), label)
    return work, sorted(set(provisional))


def decorate_queue(rows, root):
    """Attach identical server timestamps and ordered queue delays to both UIs."""
    preceding_low = preceding_high = 0
    preceding_unknown = False
    for row in rows:
        scope = row.get("scope_progress") or {}
        estimate = dict(scope.get("estimate") or {})
        status = row.get("status")
        estimate.update(finish_earliest=None, finish_latest=None, queue_lower_seconds=0, queue_upper_seconds=0)
        unknown = estimate.get("unknown_stages") or []
        source = scope.get("source_days", 0)
        if status == "completed":
            estimate.update(state="complete", message="Requested workflow finished", lower_seconds=0,upper_seconds=0)
        elif status in {"paused","pausing","canceled","canceling","failed"}:
            estimate.update(state=status,message="Finish time waits for resume" if status in {"paused","pausing"} else "No active finish estimate")
        elif status == "scheduled":
            estimate.update(state="scheduled", message="Waiting for the scheduler to start remaining work", next_start_at=row.get("next_start_at"))
        elif status == "waiting_quota" or source:
            estimate.update(state="waiting_source",message="Finish time waits for source acquisition" + (" and eligible quota" if status=="waiting_quota" else ""))
        elif not estimate or "as_of" not in estimate:
            estimate.update(state="learning",message="Inspecting remaining work")
        elif unknown:
            estimate.update(state="learning",message="Measuring: "+", ".join(unknown))
        elif status == "queued" and preceding_unknown:
            estimate.update(state="waiting_queue",message="Finish time waits for the job ahead")
        else:
            low, high = estimate.get("lower_seconds",0), estimate.get("upper_seconds",0)
            queued = status == "queued"
            a,b = (preceding_low,preceding_high) if queued else (0,0)
            stamp = datetime.fromisoformat(estimate["as_of"]).timestamp()
            estimate.update(state="estimated",message="Full requested workflow" + (" · provisional input sizes" if scope.get("estimate_notes") else ""),
                queue_lower_seconds=a,queue_upper_seconds=b,
                finish_earliest=datetime.fromtimestamp(stamp+low+a,timezone.utc).isoformat(),
                finish_latest=datetime.fromtimestamp(stamp+high+b,timezone.utc).isoformat())
        estimate["notes"] = scope.get("estimate_notes", [])
        row["eta"] = estimate
        if status in {"running","queued","pausing","canceling"}:
            preceding_unknown |= estimate.get("state") != "estimated"
            preceding_low += estimate.get("lower_seconds",0)
            preceding_high += estimate.get("upper_seconds",0)
    return rows
