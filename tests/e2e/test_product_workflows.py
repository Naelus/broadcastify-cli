from __future__ import annotations

import hashlib
import http.client
import io
import json
import os
import subprocess
import sys
import threading
import time
import re
import sqlite3
from contextlib import contextmanager
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Iterator

import pytest

from broadcastify_cli.analysis import DEFAULT_EMBEDDING_MODEL, PROMPT_VERSION
from broadcastify_cli.lan_node import create_lan_node_server
from broadcastify_cli.lan_sync import LanArchiveCatalog, LanArchiveSyncClient
from broadcastify_cli.models import JobRequest
from broadcastify_cli.storage import AnalysisStore
from broadcastify_cli.web_app import create_server


def test_nas_reads_finished_master_evidence_and_rediscoverable_queue(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import requests
    from broadcastify_cli.archive_cache import remember_complete_archive_day

    for name in list(os.environ):
        if name.startswith("BROADCASTIFY_"):
            monkeypatch.delenv(name)
    monkeypatch.setenv("BROADCASTIFY_QUOTA_LEDGER", str(tmp_path / "quota.sqlite3"))
    root = tmp_path / "master"
    database = root / "broadcastify-analysis.sqlite3"
    audio = _retained_analyzed_day(root, database, "90001", date(2026,7,12), location="Main Street", start_seconds=10)
    retained = {p:hashlib.sha256(p.read_bytes()).hexdigest() for p in root.rglob("*") if p.is_file() and p.suffix in {".mp3",".json"}}
    work = tmp_path / "master-work"
    work.mkdir()
    (work / "cookies.json").write_text("{}")
    master = create_lan_node_server(root,host="127.0.0.1",port=0,working_dir=work,database_path=database,
        role="master",job_relay_enabled=True,discovery_enabled=False)
    master.quiet = True
    manager = master.job_manager
    # A native-owned job is the current model owner. Remote submissions must queue.
    observed = {"id":"0123456789abcdef","command":"run-scheduled","owner_pid":os.getpid(),"status":"running",
        "request":{"job":{"feed_id":"90001","start_date":"2026-07-12","end_date":"2026-07-12","transcribe":True,
            "analysis_api_key":"private-provider-value", "huggingface_token":"private-model-value", "analysis_api_key_env":"OPENAI_API_KEY"},"analyze":True},
        "events":[{"type":"progress","stage":"analysis","current":1,"total":2,"message":"One window saved"}]}
    manager.observe(observed)
    saved_request = manager.shared.request(observed["id"])
    assert "private-provider-value" not in json.dumps(saved_request)
    assert "private-model-value" not in json.dumps(saved_request)
    assert saved_request["job"]["analysis_api_key_env"] == "OPENAI_API_KEY"
    launches = []
    phases = []
    save_index = False
    ready, release = threading.Event(), threading.Event()
    def execute(job, arguments, payload, profile):
        launches.append(job.id)
        phases.append((list(arguments),dict(payload or {})))
        if arguments == ["run"]:
            # The fake provider completed a real persisted listing; legacy
            # combined evidence is retained even when that listing is empty.
            remember_complete_archive_day(audio.parent, "90001", date(2026,7,12), [])
        elif arguments == ["run-scheduled"] and save_index:
            with AnalysisStore(database) as store:
                store.save_embeddings("passage", [
                    (p["id"], hashlib.sha256(f"passage: {p['text']}".encode()).hexdigest(), 1, b"\0" * 4)
                    for p in store.passages_missing_embeddings(DEFAULT_EMBEDDING_MODEL)
                ], DEFAULT_EMBEDDING_MODEL)
        job.result = {"type":"complete","result":{"download_limited":False}}
        manager._append(job,{"type":"stage","stage":"analysis","message":"Resuming saved checkpoint"})
        ready.set()
        assert release.wait(3)
        job.phase_status = "completed"
        job.status = "running" if job.command == "run-scheduled" else "completed"
    monkeypatch.setattr(manager,"_execute",execute)
    threading.Thread(target=master.serve_forever,daemon=True).start()
    follower_work = tmp_path / "nas-work"
    follower_work.mkdir()
    (follower_work / ".env").write_text(f"BROADCASTIFY_LAN_SHARING=true\nBROADCASTIFY_LAN_ROLE=follower\nBROADCASTIFY_LAN_DISCOVERY_ENABLED=false\nBROADCASTIFY_LAN_MASTER_URL=http://127.0.0.1:{master.server_port}\n")
    follower = create_server(tmp_path / "nas",port=0,working_dir=follower_work,background_sync_enabled=False)
    follower.quiet = True
    threading.Thread(target=follower.serve_forever,daemon=True).start()
    base = f"http://127.0.0.1:{follower.server_port}"
    client = requests.Session()
    try:
        page = client.get(base+"/",timeout=5)
        client.headers["X-Radio-Archive-Token"] = re.search(r'name="app-token" content="([^"]+)"',page.text).group(1)
        bootstrap = client.get(base+"/api/bootstrap",timeout=5).json()
        assert bootstrap["days"][0]["is_complete"]
        assert bootstrap["days"][0]["copy_state"] == "NAS copy syncing"
        detail = client.get(base+"/api/day?feed_id=90001&date=2026-07-12",timeout=5).json()
        assert detail["state"]["is_complete"] and detail["summary"] and detail["incidents"][0]["quote"]
        transcript = client.get(base+"/api/transcript?feed_id=90001&date=2026-07-12&q=Main",timeout=5).json()
        assert transcript["total"] == 1 and "Main" in transcript["segments"][0]["text"]
        media = client.get(base+detail["audio_url"],headers={"Range":"bytes=0-2"},timeout=5)
        assert media.status_code == 206 and media.content == audio.read_bytes()[:3]
        denied = client.get(base+"/master-media?path=../master-work/cookies.json",timeout=5)
        assert denied.status_code >= 400
        assert launches == []  # Reading finished evidence never starts processing.
        response = client.post(base+"/api/jobs",json={"command":"continue-local","payload":{"feed_id":"90001","archive_date":"2026-07-12","analyze":False}},timeout=5)
        assert response.status_code == 202,response.text
        queued = response.json()
        assert queued["status"] == "queued" and launches == []
        # A fresh browser session discovers the same native and queued jobs.
        other = requests.Session()
        other.get(base+"/",timeout=5)
        rows = other.get(base+"/api/jobs",timeout=5).json()["jobs"]
        assert {r["id"] for r in rows} == {observed["id"],queued["id"]}
        assert client.post(base+f"/api/jobs/{queued['id']}/pause",json={},timeout=5).json()["status"] == "paused"
        manager.observe({**observed,"status":"completed"})
        client.post(base+f"/api/jobs/{queued['id']}/resume",json={},timeout=5).raise_for_status()
        assert ready.wait(3)
        assert len(launches)==1
        release.set()
        deadline=time.monotonic()+3
        while manager.get(queued["id"])["status"] != "completed" and time.monotonic()<deadline:
            time.sleep(.01)
        assert manager.get(queued["id"])["status"] == "completed"
        with AnalysisStore(database) as store:
            store.save_feed_schedule({"feed_id":"90001","feed_name":"Saved Windows feed","run_time_local":"00:00","lookback_days":1,
                "enabled":False,"job":{"model":"saved-master-model","device":"cuda","combine":True,"transcribe":True,"diarize":True}})
        complete_job = client.post(base+"/api/jobs",json={"command":"run-scheduled","payload":{
            "account_profile_id":"automatic","use_saved_profile":True,"analyze":True,
            "job":{"feed_id":"90001","start_date":"2026-07-12","end_date":"2026-07-12"}}},timeout=5)
        assert complete_job.status_code == 202,complete_job.text
        finished_id = complete_job.json()["id"]
        deadline=time.monotonic()+3
        while manager.get(finished_id)["status"] not in {"completed","failed"} and time.monotonic()<deadline:
            time.sleep(.01)
        assert manager.get(finished_id)["status"] == "completed"
        # The only additional child is cache/source acquisition. No ASR or
        # analysis child is launched for the already-finished Windows day.
        assert len(phases)==2 and phases[-1][0]==["run"]
        assert phases[-1][1]["model"]=="saved-master-model" and phases[-1][1]["device"]=="cuda"
        # Losing the index must launch local continuation. A zero exit code
        # without saved results must never turn an unfinished day green.
        with sqlite3.connect(database) as connection:
            connection.execute("DELETE FROM embeddings")
        unfinished = client.post(base+"/api/jobs", json={"command":"run-scheduled", "payload":{
            "account_profile_id":"automatic", "use_saved_profile":True, "analyze":True,
            "job":{"feed_id":"90001", "start_date":"2026-07-12", "end_date":"2026-07-12"}}}, timeout=5).json()
        deadline = time.monotonic()+3
        while manager.get(unfinished["id"])["status"] not in {"completed", "failed"} and time.monotonic()<deadline:
            time.sleep(.01)
        assert manager.get(unfinished["id"])["status"] == "failed"
        assert phases[-1][0] == ["run-scheduled"] and phases[-1][1]["job"]["local_only"]
        assert not client.get(base+"/api/day?feed_id=90001&date=2026-07-12", timeout=5).json()["state"]["is_complete"]
        save_index = True
        client.post(base+f"/api/jobs/{unfinished['id']}/resume", json={}, timeout=5).raise_for_status()
        deadline = time.monotonic()+3
        while manager.get(unfinished["id"])["status"] not in {"completed", "failed"} and time.monotonic()<deadline:
            time.sleep(.01)
        assert manager.get(unfinished["id"])["status"] == "completed"
        assert manager.get(unfinished["id"])["scope_progress"]["ready_days"] == 1
        assert client.get(base+"/api/day?feed_id=90001&date=2026-07-12", timeout=5).json()["state"]["is_complete"]
        assert all(hashlib.sha256(p.read_bytes()).hexdigest()==h for p,h in retained.items())
    finally:
        release.set()
        client.close()
        follower.shutdown();follower.server_close()
        master.shutdown();master.server_close()
    # Master history survives its process/service recreation; no job restarts.
    from broadcastify_cli.web_app import JobManager
    reopened = JobManager(root,database,work)
    try:
        assert reopened.get(queued["id"])["status"] == "completed"
        assert reopened.get(observed["id"])["status"] == "completed"
    finally:
        reopened.close()


def test_shared_catchup_waits_for_source_proof_and_preserves_pause(tmp_path, monkeypatch):
    from broadcastify_cli.archive_cache import remember_complete_archive_day
    from broadcastify_cli.web_app import JobManager

    for name in list(os.environ):
        if name.startswith("BROADCASTIFY_"):
            monkeypatch.delenv(name)
    root = tmp_path / "master"
    database = root / "broadcastify-analysis.sqlite3"
    audio = _retained_analyzed_day(root, database, "90001", date(2026, 7, 12), location="Main", start_seconds=10)
    # Retained analysis exists, but there is no authenticated source proof.
    # The fake exhausted provider cannot fill that gap or spend live requests.
    pool = {"authorized":True, "configured_profile_ids":["default"], "profiles":[
        {"id":"default", "configured":True, "quota":{"available":False, "remaining":0}}]}
    monkeypatch.setattr("broadcastify_cli.web_app._account_pool_profiles", lambda *a: pool)
    manager = JobManager(root, database, tmp_path)
    calls = []
    checked_waiting = threading.Event()
    def execute(job, args, payload, profile):
        calls.append((job.id, profile))
        if args == ["run-scheduled"] and (audio.parent / ".broadcastify-archive-complete.json").exists():
            with AnalysisStore(database) as store:
                store.save_embeddings("passage", [
                    (p["id"], hashlib.sha256(f"passage: {p['text']}".encode()).hexdigest(), 1, b"\0" * 4)
                    for p in store.passages_missing_embeddings(DEFAULT_EMBEDDING_MODEL)
                ], DEFAULT_EMBEDDING_MODEL)
        job.result = {"result":{"download_limited":True}}
        job.phase_status = "completed"
        job.status = "running"
    monkeypatch.setattr(manager, "_execute", execute)
    wait = manager._condition.wait
    monkeypatch.setattr(manager._condition, "wait", lambda timeout=None: wait(min(timeout or 1, .02)))
    complete = manager._scope_complete
    def checked(*args):
        result = complete(*args)
        checked_waiting.set()
        return result
    monkeypatch.setattr(manager, "_scope_complete", checked)
    payload = {"account_profile_id":"default", "analyze":False,
        "job":{"feed_id":"90001", "start_date":"2026-07-12", "end_date":"2026-07-12"}}
    try:
        job = manager.start("run-scheduled", payload)
        assert checked_waiting.wait(3)
        assert manager.get(job["id"])["status"] == "waiting_quota"
        assert manager.get(job["id"])["scope_progress"]["source_days"] == 1
        native = manager.observe({"id":"fedcba9876543210", "command":"run-scheduled",
            "owner_pid":os.getpid(), "status":"waiting_quota", "request":payload})
        deadline = time.monotonic()+3
        while native["id"] in manager._observed_scope_threads and time.monotonic()<deadline:
            time.sleep(.01)
        assert manager.get(native["id"])["status"] == "waiting_quota"
        retried = manager.observe({"id":"abcdef0123456789", "command":"run-scheduled",
            "owner_pid":os.getpid(), "status":"waiting_quota", "request":payload})
        assert retried["id"] == native["id"]
        assert retried["created_at"] == native["created_at"]
        assert manager.shared.get("abcdef0123456789") is None
        # Another authorized acquisition can satisfy the waiting job. It must
        # finish using that persisted proof without another provider worker.
        remember_complete_archive_day(audio.parent, "90001", date(2026, 7, 12), [])
        deadline = time.monotonic()+3
        while manager.get(job["id"])["status"] != "completed" and time.monotonic()<deadline:
            time.sleep(.01)
        assert manager.get(job["id"])["status"] == "completed"
        assert calls == [(job["id"], "default")]
        # Native history must also reflect later saved completion instead of
        # remaining quota-blocked forever after the desktop starts another run.
        manager._observed_scope_at[native["id"]] = 0
        manager.list()
        deadline = time.monotonic()+3
        while manager.get(native["id"])["status"] != "completed" and time.monotonic()<deadline:
            time.sleep(.01)
        assert manager.get(native["id"])["status"] == "completed"
        assert calls == [(job["id"], "default")]
        (audio.parent / ".broadcastify-archive-complete.json").unlink()
        with sqlite3.connect(database) as connection:
            connection.execute("DELETE FROM embeddings")
        checked_waiting.clear()
        processing = manager.start("run-scheduled", {**payload, "analyze":True})
        assert checked_waiting.wait(3)
        assert manager.get(processing["id"])["status"] == "waiting_quota"
        remember_complete_archive_day(audio.parent, "90001", date(2026, 7, 12), [])
        deadline = time.monotonic()+3
        while manager.get(processing["id"])["status"] != "completed" and time.monotonic()<deadline:
            time.sleep(.01)
        assert manager.get(processing["id"])["status"] == "completed"
        assert manager.get(processing["id"])["scope_progress"]["processing_days"] == 0
        (audio.parent / ".broadcastify-archive-complete.json").unlink()
        checked_waiting.clear()
        paused = manager.start("run-scheduled", payload)
        assert checked_waiting.wait(3)
        assert manager.action(paused["id"], "pause")["status"] == "paused"
        launch_count = len(calls)
    finally:
        manager.close()
    restored = JobManager(root, database, tmp_path)
    try:
        assert restored.get(paused["id"])["status"] == "paused"
        assert restored.cancel(paused["id"])["status"] == "canceled"
        assert len(calls) == launch_count  # Explicit pause did not launch work on restart.
    finally:
        restored.close()


ROOT = Path(__file__).resolve().parents[2]
SECRET_ENVIRONMENT_NAMES = {
    "BROADCASTIFY_USERNAME",
    "BROADCASTIFY_PASSWORD",
    "BROADCASTIFY_SECURE_USERNAME",
    "BROADCASTIFY_SECURE_PASSWORD",
    "HUGGINGFACE_TOKEN",
    "HUGGINGFACE_SECURE_TOKEN",
    "HF_TOKEN",
    "OPENAI_API_KEY",
    "BROADCASTIFY_ANALYSIS_API_KEY",
}

pytestmark = pytest.mark.usefixtures("fast_server_shutdown")


def _worker_environment(
    session_root: Path,
    library_root: Path,
    database: Path,
) -> dict[str, str]:
    environment = {
        key: value
        for key, value in os.environ.items()
        if key not in SECRET_ENVIRONMENT_NAMES
        and not key.startswith("BROADCASTIFY_SECURE_")
    }
    existing_python_path = environment.get("PYTHONPATH", "")
    environment.update(
        {
            "BROADCASTIFY_ANALYSIS_DB": str(database),
            "BROADCASTIFY_SECURE_ANALYSIS_DB": str(database),
            "BROADCASTIFY_LIBRARY_ROOT": str(library_root),
            "BROADCASTIFY_QUOTA_LEDGER": str(session_root / "archive-quota.sqlite3"),
            "BROADCASTIFY_CREDENTIAL_STORE": str(session_root / "credentials.enc"),
            "BROADCASTIFY_ENV_FILE": "",
            "PYTHONIOENCODING": "utf-8",
            "PYTHONUTF8": "1",
            "PYTHONPATH": os.pathsep.join(
                value for value in (str(ROOT), existing_python_path) if value
            ),
        }
    )
    return environment


def _run_worker(
    session_root: Path,
    library_root: Path,
    database: Path,
    *arguments: str,
    payload: dict[str, Any] | None = None,
    account_profile_id: str = "default",
) -> list[dict[str, Any]]:
    environment = _worker_environment(session_root, library_root, database)
    environment["BROADCASTIFY_ACCOUNT_PROFILE"] = account_profile_id
    environment["BROADCASTIFY_COOKIE_PATH"] = str(
        session_root / "account-sessions" / f"{account_profile_id}.json"
    )
    result = subprocess.run(
        [sys.executable, "-m", "broadcastify_cli.worker", *arguments],
        cwd=session_root,
        env=environment,
        input=json.dumps(payload) if payload is not None else None,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=30,
        check=False,
    )
    events: list[dict[str, Any]] = []
    for raw_line in result.stdout.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        value = json.loads(line)
        assert isinstance(value, dict), value
        events.append(value)
    assert result.returncode == 0, (
        f"Worker {' '.join(arguments)} failed with {result.returncode}.\n"
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
    assert events, f"Worker {' '.join(arguments)} emitted no JSON events."
    assert not any(value.get("type") == "error" for value in events), events
    return events


def _retained_analyzed_day(
    library_root: Path,
    database: Path,
    feed_id: str,
    archive_date: date,
    *,
    location: str,
    start_seconds: float,
) -> Path:
    day_root = library_root / feed_id / archive_date.strftime("%Y%m%d")
    transcript_root = day_root / "transcripts"
    transcript_root.mkdir(parents=True, exist_ok=True)
    stem = f"combined_{feed_id}_{archive_date:%Y%m%d}"
    audio = day_root / f"{stem}.mp3"
    transcript = transcript_root / f"{stem}.json"
    audio.write_bytes(b"ID3 retained offline e2e audio")
    transcript.write_text(
        json.dumps(
            {
                "model": "offline-e2e",
                "duration": start_seconds + 10,
                "diarization_completed": True,
                "diarization_engine": "community-1",
                "segments": [
                    {
                        "start": start_seconds,
                        "end": start_seconds + 5,
                        "speaker": "SPEAKER_00",
                        "text": (
                            "Dispatch reported possible shots near "
                            f"{location}; units are checking the radio report."
                        ),
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    with AnalysisStore(database) as store:
        imported = store.import_transcript(
            feed_id,
            archive_date,
            transcript,
            audio,
        )
        incident_ids = store.replace_incidents(
            imported.day_id,
            [
                {
                    "fingerprint": f"offline-e2e-{feed_id}-{archive_date}",
                    "event_type": "shots_fired",
                    "title": "Possible shots reported",
                    "summary": "Dispatch reported possible shots; units checked the report.",
                    "location": location,
                    "start_seconds": start_seconds,
                    "end_seconds": start_seconds + 5,
                    "priority": 4,
                    "confidence": 0.9,
                    "evidence": [
                        {
                            "segment_index": 0,
                            "start_seconds": start_seconds,
                            "end_seconds": start_seconds + 5,
                            "speaker": "SPEAKER_00",
                            "text": f"Possible shots near {location}.",
                        }
                    ],
                    "attributes": {},
                }
            ],
            model="offline-e2e",
            prompt_version=PROMPT_VERSION,
        )
        store.save_daily_summary(
            imported.day_id,
            "One possible-shots radio report was extracted.",
            incident_ids,
            model="offline-e2e",
            prompt_version=PROMPT_VERSION,
            transcript_sha256=imported.transcript_sha256,
        )
        store.save_feed_catalog(
            [{"feed_id": feed_id, "name": f"Offline feed {feed_id}"}]
        )
        store.save_embeddings("passage", [
            (p["id"], hashlib.sha256(f"passage: {p['text']}".encode()).hexdigest(), 1, b"\0" * 4)
            for p in store.passages_missing_embeddings(DEFAULT_EMBEDDING_MODEL)
        ], DEFAULT_EMBEDDING_MODEL)
    return audio


def test_activity_acknowledges_only_displayed_revisions_across_restarts(tmp_path: Path) -> None:
    library = tmp_path / "archives"
    database = library / "broadcastify-analysis.sqlite3"
    first, second = date(2026, 7, 11), date(2026, 7, 12)
    scope = {"feed_id": "90001", "start_date": first.isoformat(), "end_date": second.isoformat()}

    def seed(day: date) -> None:
        _retained_analyzed_day(library, database, "90001", day, location="Main and First", start_seconds=10)

    def report() -> dict[str, Any]:
        return _run_worker(tmp_path, library, database, "report-activity",
            "--feed-id", "90001", "--start-date", first.isoformat(), "--end-date", second.isoformat())[-1]["report"]

    def acknowledge(items: list[dict[str, Any]]) -> int:
        return _run_worker(tmp_path, library, database, "mark-activity",
            payload={**scope, "items": items})[-1]["result"]["acknowledged_count"]

    seed(first)
    initial = report()
    assert initial["coverage"]["pending_dates"] == [second.isoformat()]
    assert len(initial["items"]) == 1
    assert initial["items"][0]["evidence_quote"]
    assert acknowledge(initial["items"]) == 1
    assert report()["items"] == []
    # Replacing identical incidents changes IDs, not what the reader has seen.
    seed(first)
    assert report()["items"] == []
    with AnalysisStore(database) as store, store.transaction():
        store.connection.execute("UPDATE incidents SET summary='Dispatch updated the reported location.'")
    updated = report()
    assert updated["items"][0]["change_kind"] == "updated"
    assert acknowledge(initial["items"]) == 0  # stale clicks cannot hide updates
    seed(second)  # an arrival after the displayed batch must remain unread
    assert acknowledge(updated["items"]) == 1
    remaining = report()["items"]
    assert len(remaining) == 1 and remaining[0]["archive_date"] == second.isoformat()


def test_retained_day_recovers_index_interruption_without_repeating_other_work(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from types import SimpleNamespace
    from broadcastify_cli import worker
    from broadcastify_cli.library import scan_local_library, build_library_workflow

    library = tmp_path / "library"
    database = library / "broadcastify-analysis.sqlite3"
    day = date(2026, 7, 11)
    audio = _retained_analyzed_day(library, database, "90001", day,
        location="Main and First", start_seconds=10)
    _retained_analyzed_day(library, database, "90001", day + timedelta(days=1),
        location="Another day", start_seconds=10)
    _retained_analyzed_day(library, database, "90002", day,
        location="Another feed", start_seconds=10)
    transcript = audio.parent / "transcripts" / f"{audio.stem}.json"
    transcript.write_text(json.dumps({"duration": 21_500, "diarization_completed": True,
        "segments": [{"start": i * 330, "end": i * 330 + 2, "speaker": "SPEAKER_00",
            "text": f"Target dispatch status check {i}."} for i in range(65)]}), encoding="utf-8")
    retained_hashes = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in (audio, transcript)}
    with AnalysisStore(database) as store, store.transaction():
        store.connection.execute("DELETE FROM embeddings")

    generation_calls: list[str] = []
    server_launches: list[bool] = []
    embedded_batches: list[list[str]] = []
    model_loads: list[str] = []

    class OfflineGeneration:
        def chat_json(self, *, schema_name: str, **_kwargs: Any) -> dict[str, Any]:
            generation_calls.append(schema_name)
            return {"incidents": []} if schema_name == "police_radio_incidents" else {"summary": "Routine status checks."}

    @contextmanager
    def offline_client(provider: Any, *, launch_local_server: bool) -> Iterator[Any]:
        server_launches.append(launch_local_server)
        client = OfflineGeneration()
        client.model = provider.cache_model
        yield client

    class OfflineEmbedding:
        def __init__(self, *, model_name: str) -> None:
            model_loads.append(model_name)

        def embed(self, texts: list[str]) -> list[list[float]]:
            embedded_batches.append(texts)
            assert all("Target dispatch" in text for text in texts), "A single-day finish indexed unrelated evidence"
            if len(embedded_batches) == 2:
                raise RuntimeError("interrupted after a saved embedding batch")
            return [[1.0, 0.0] for _ in texts]

    monkeypatch.setitem(sys.modules, "fastembed", SimpleNamespace(TextEmbedding=OfflineEmbedding))
    monkeypatch.setattr(worker, "DEFAULT_DATABASE", database)
    monkeypatch.setattr(worker, "open_analysis_client", offline_client)
    events: list[dict[str, Any]] = []
    monkeypatch.setattr(worker, "emit", events.append)
    payload = {"feed_id": "90001", "archive_date": day.isoformat(), "output_dir": str(library)}

    def run() -> None:
        monkeypatch.setattr(sys, "stdin", io.StringIO(json.dumps(payload)))
        assert worker.analyze_day() == 0

    with pytest.raises(RuntimeError, match="interrupted after a saved embedding batch"):
        run()
    calls_after_analysis = list(generation_calls)
    assert calls_after_analysis
    interrupted = scan_local_library(library, database)
    selected = next(d for d in interrupted if d["feed_id"] == "90001" and d["archive_date"] == day.isoformat())
    assert selected["has_analysis"] and not selected["has_search_index"] and not selected["is_complete"]
    with AnalysisStore(database) as store:
        assert len(store.passages_missing_embeddings(DEFAULT_EMBEDDING_MODEL,
            feed_id="90001", start_date=day, end_date=day)) == 1

    # Reopens the store through the real worker entry point. The saved summary
    # and 64 committed embeddings survive the interrupted final stage.
    run()
    assert generation_calls == calls_after_analysis
    assert [len(batch) for batch in embedded_batches] == [64, 1, 1]
    finished = scan_local_library(library, database)
    workflow = build_library_workflow(finished)
    assert workflow["pending_count"] == 2 and workflow["ready_count"] == 1
    assert {(d["feed_id"], d["archive_date"]) for d in workflow["days"]} == {
        ("90001", "2026-07-12"), ("90002", "2026-07-11")}
    run()  # A fully cached retry must load neither generation nor embedding models.
    assert server_launches == [True, False, False]
    assert len(model_loads) == 2 and len(embedded_batches) == 3
    assert generation_calls == calls_after_analysis
    assert all(hashlib.sha256(p.read_bytes()).hexdigest() == digest for p, digest in retained_hashes.items())


def _audio_only_day(
    library_root: Path,
    feed_id: str,
    archive_date: date,
) -> None:
    day_root = library_root / feed_id / archive_date.strftime("%Y%m%d")
    day_root.mkdir(parents=True, exist_ok=True)
    (day_root / f"combined_{feed_id}_{archive_date:%Y%m%d}.mp3").write_bytes(
        b"ID3 retained audio awaiting local processing"
    )


def _raw_source_only_day(
    library_root: Path,
    feed_id: str,
    archive_date: date,
) -> None:
    day_root = library_root / feed_id / archive_date.strftime("%Y%m%d")
    day_root.mkdir(parents=True, exist_ok=True)
    (day_root / f"{archive_date:%Y%m%d}0000-123456-{feed_id}.mp3").write_bytes(
        b"ID3 downloaded source audio awaiting daily combination"
    )


def _shareable_model_day(
    library_root: Path,
    feed_id: str,
    archive_date: date,
    processing_fingerprint: str,
) -> None:
    day_root = library_root / feed_id / archive_date.strftime("%Y%m%d")
    transcript_root = day_root / "transcripts"
    transcript_root.mkdir(parents=True, exist_ok=True)
    audio = day_root / f"{archive_date:%Y%m%d}0000-123456-{feed_id}.mp3"
    audio.write_bytes(f"shared retained source {feed_id} {archive_date}".encode())
    rendered = f"[{archive_date} 00:00:00] Retained model result.\n"
    text_path = transcript_root / f"{audio.stem}.txt"
    text_path.write_text(rendered, encoding="utf-8")
    (transcript_root / f"{audio.stem}.json").write_text(
        json.dumps(
            {
                "audio_file": audio.name,
                "audio_sha256": hashlib.sha256(audio.read_bytes()).hexdigest(),
                "processing_fingerprint": processing_fingerprint,
                "rendered_text_sha256": hashlib.sha256(
                    text_path.read_bytes()
                ).hexdigest(),
                "segments": [],
            }
        ),
        encoding="utf-8",
    )


def test_explicit_feed_pull_is_source_only_and_never_merges_model_results(
    tmp_path: Path,
) -> None:
    """Exercise the compatibility source pull without recreating mesh behavior."""

    feed_id = "91059"
    archive_date = date(2026, 7, 12)
    fingerprint_a = "a" * 64
    fingerprint_b = "b" * 64
    node_a = tmp_path / "node-a"
    node_b = tmp_path / "node-b"
    _shareable_model_day(node_a, feed_id, archive_date, fingerprint_a)
    _shareable_model_day(node_b, feed_id, archive_date, fingerprint_b)

    server_a = create_lan_node_server(
        node_a,
        host="127.0.0.1",
        port=0,
        discovery_enabled=False,
    )
    server_b = create_lan_node_server(
        node_b,
        host="127.0.0.1",
        port=0,
        discovery_enabled=False,
    )
    server_a.quiet = True  # type: ignore[attr-defined]
    server_b.quiet = True  # type: ignore[attr-defined]
    thread_a = threading.Thread(target=server_a.serve_forever, daemon=True)
    thread_b = threading.Thread(target=server_b.serve_forever, daemon=True)
    thread_a.start()
    thread_b.start()
    url_a = f"http://127.0.0.1:{server_a.server_port}"
    url_b = f"http://127.0.0.1:{server_b.server_port}"
    try:
        result_a = LanArchiveSyncClient(
            enabled=True,
            peer_urls=(url_b,),
            discovery_enabled=False,
        ).sync_feed(
            node_a,
            feed_id,
            processing_fingerprint=fingerprint_a,
        )
        result_b = LanArchiveSyncClient(
            enabled=True,
            peer_urls=(url_a,),
            discovery_enabled=False,
        ).sync_feed(
            node_b,
            feed_id,
            processing_fingerprint=fingerprint_b,
        )

        consumer = tmp_path / "consumer"
        result_consumer = LanArchiveSyncClient(
            enabled=True,
            peer_urls=(url_a,),
            discovery_enabled=False,
        ).sync_feed(consumer, feed_id)

        assert result_a.blocks_copied == 0
        assert result_b.blocks_copied == 0
        assert result_consumer.blocks_copied == 1
        assert result_a.transcript_artifacts_copied == 0
        assert result_b.transcript_artifacts_copied == 0
        assert result_consumer.transcript_artifacts_copied == 0
        assert result_consumer.days_with_transcript_changes == 0
        assert result_a.failures == result_b.failures == result_consumer.failures == ()
        assert not list(consumer.glob("**/transcripts/*"))
        assert len(list((node_a / feed_id).glob("*/transcripts/*.json"))) == 1
        assert len(list((node_b / feed_id).glob("*/transcripts/*.json"))) == 1
    finally:
        server_a.shutdown()
        server_b.shutdown()
        server_a.server_close()
        server_b.server_close()
        thread_a.join(timeout=3)
        thread_b.join(timeout=3)


def _http_json(
    port: int,
    path: str,
    *,
    cookie: str = "",
) -> tuple[int, dict[str, Any], str]:
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
    headers = {"Cookie": cookie} if cookie else {}
    try:
        connection.request("GET", path, headers=headers)
        response = connection.getresponse()
        body = response.read()
        return (
            response.status,
            json.loads(body),
            response.getheader("Set-Cookie", "").split(";", 1)[0],
        )
    finally:
        connection.close()


def test_saved_resume_recurring_restart_and_live_session_delete(
    tmp_path: Path,
) -> None:
    """Prove the retained catch-up lifecycle through the real worker protocol."""

    feed_id = "20305"
    library_root = tmp_path / "library"
    database = library_root / "broadcastify-analysis.sqlite3"
    today = date.today()
    start_date = today - timedelta(days=2)
    missing_date = today - timedelta(days=1)
    retained_audio = _retained_analyzed_day(
        library_root,
        database,
        feed_id,
        start_date,
        location="Main and First",
        start_seconds=7_200,
    )
    _audio_only_day(library_root, feed_id, today)

    saved = _run_worker(
        tmp_path,
        library_root,
        database,
        "save-library-catch-up",
        payload={
            "feed_id": feed_id,
            "feed_name": "Offline resume feed",
            "start_date": start_date.isoformat(),
            "through_current": True,
        },
    )[-1]["catch_up"]
    assert saved["through_current"] is True
    assert saved["end_date"] == today.isoformat()

    resume = _run_worker(
        tmp_path,
        library_root,
        database,
        "library-resume-plan",
        "--output-dir",
        str(library_root),
    )[-1]
    queued = {value["archive_date"]: value for value in resume["days"]}
    assert set(queued) == {missing_date.isoformat(), today.isoformat()}
    assert queued[missing_date.isoformat()]["needs_network"] is True
    assert queued[today.isoformat()]["needs_local_processing"] is True
    assert start_date.isoformat() not in queued

    schedule = _run_worker(
        tmp_path,
        library_root,
        database,
        "save-schedule",
        payload={
            "feed_id": feed_id,
            "feed_name": "Offline resume feed",
            "run_time_local": "00:00",
            "lookback_days": 2,
            "backfill_start_date": start_date.isoformat(),
            "recurring_catch_up": True,
            "account_profile_id": "automatic",
            "enabled": True,
            "analyze": True,
            "job": {"combine": True, "transcribe": True, "diarize": True},
        },
    )[-1]["schedule"]
    assert schedule["recurring_catch_up"] is True
    assert schedule["account_profile_id"] == "automatic"
    assert schedule["job"]["download_jobs"] == 1
    assert schedule["job"]["keep_originals"] is True

    claimed = _run_worker(
        tmp_path,
        library_root,
        database,
        "claim-due-schedule",
    )[-1]["schedule"]
    assert claimed is not None
    assert claimed["job"]["start_date"] == start_date.isoformat()
    assert claimed["job"]["end_date"] == today.isoformat()
    assert claimed["job"]["newest_first"] is True
    claimed_dates = list(JobRequest.from_dict(claimed["job"]).dates())
    assert claimed_dates == [today, missing_date, start_date]
    assert claimed["account_profile_id"] == "automatic"
    assert Path(claimed["job"]["output_dir"]) == library_root.resolve()

    primary_quota = _run_worker(
        tmp_path,
        library_root,
        database,
        "quota-status",
        account_profile_id="default",
    )[-1]["status"]
    secondary_quota = _run_worker(
        tmp_path,
        library_root,
        database,
        "quota-status",
        account_profile_id="secondary",
    )[-1]["status"]
    assert primary_quota["account_profile_id"] == "default"
    assert secondary_quota["account_profile_id"] == "secondary"
    assert primary_quota["instance_id"] != secondary_quota["instance_id"]
    assert primary_quota["automated_limit"] == 248
    assert secondary_quota["automated_limit"] == 250

    recovered = _run_worker(
        tmp_path,
        library_root,
        database,
        "recover-schedules",
    )[-1]
    assert recovered["recovered"] == 1
    recovered_schedule = _run_worker(
        tmp_path,
        library_root,
        database,
        "schedules",
    )[-1]["schedules"][0]
    assert recovered_schedule["state"] == "deferred"
    assert "resuming from retained work" in recovered_schedule["message"]

    waiting = _run_worker(
        tmp_path,
        library_root,
        database,
        "finish-schedule",
        payload={
            "schedule_id": claimed["id"],
            "due_date": claimed["due_date"],
            "status": "waiting_quota",
            "message": "Waiting for the next rolling archive-request slot.",
            "next_request_at": (
                f"{(today + timedelta(days=1)).isoformat()}T23:59:00+00:00"
            ),
        },
    )[-1]["schedule"]
    assert waiting["state"] == "waiting_quota"
    assert waiting["due"] is False

    quota_recovery = _run_worker(
        tmp_path,
        library_root,
        database,
        "recover-schedules",
    )[-1]
    assert quota_recovery["recovered"] == 1
    quota_recheck = _run_worker(
        tmp_path,
        library_root,
        database,
        "schedules",
    )[-1]["schedules"][0]
    assert quota_recheck["state"] == "deferred"
    assert "retained local work" in quota_recheck["message"]

    completed = _run_worker(
        tmp_path,
        library_root,
        database,
        "finish-schedule",
        payload={
            "schedule_id": claimed["id"],
            "due_date": claimed["due_date"],
            "status": "complete",
            "message": "Offline end-to-end completion.",
        },
    )[-1]["schedule"]
    assert completed["backfill_start_date"] == start_date.isoformat()
    assert completed["recurring_catch_up"] is True

    server = create_server(
        library_root,
        database,
        port=0,
        working_dir=tmp_path,
        credential_store_path=tmp_path / "web-credentials.enc",
    )
    server.quiet = True  # type: ignore[attr-defined]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        connection = http.client.HTTPConnection(
            "127.0.0.1", server.server_port, timeout=5
        )
        connection.request("GET", "/")
        response = connection.getresponse()
        response.read()
        cookie = response.getheader("Set-Cookie", "").split(";", 1)[0]
        connection.close()
        assert response.status == 200
        assert cookie.startswith("radio_archive_session=")

        status, before, _ = _http_json(
            server.server_port,
            "/api/bootstrap",
            cookie=cookie,
        )
        assert status == 200
        assert before["summary"]["day_count"] == 2

        status, day, _ = _http_json(
            server.server_port,
            f"/api/day?feed_id={feed_id}&date={start_date.isoformat()}",
            cookie=cookie,
        )
        assert status == 200
        assert day["audio_url"]
        media = http.client.HTTPConnection(
            "127.0.0.1", server.server_port, timeout=5
        )
        media.request(
            "GET",
            day["audio_url"],
            headers={"Cookie": cookie, "Range": "bytes=0-2"},
        )
        media_response = media.getresponse()
        assert media_response.status == 206
        assert media_response.read() == retained_audio.read_bytes()[:3]
        media.close()

        deleted = _run_worker(
            tmp_path,
            library_root,
            database,
            "delete-library-feed",
            payload={
                "feed_id": feed_id,
                "output_dir": str(library_root),
                "remove_schedule": True,
            },
        )[-1]["result"]
        assert deleted["directory_deleted"] is True
        assert deleted["cleanup_pending"] is False
        assert not (library_root / feed_id).exists()

        status, after, _ = _http_json(
            server.server_port,
            "/api/bootstrap",
            cookie=cookie,
        )
        assert status == 200
        assert after["summary"]["day_count"] == 0
        assert after["schedules"] == []
        with AnalysisStore(database) as store:
            assert store.list_library_catchups() == []
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=3)


def test_month_and_entire_feed_hotspot_question_is_coverage_grounded(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Cross month/whole-feed coverage, retrieval, patterns, and Q&A audit."""

    feed_id = "141466"
    library_root = tmp_path / "library"
    database = library_root / "broadcastify-analysis.sqlite3"
    first_date = date(2026, 7, 6)
    second_date = date(2026, 7, 7)
    missing_date = date(2026, 7, 8)
    audio_only_date = date(2026, 7, 9)
    raw_source_date = date(2026, 7, 10)
    for index, archive_date in enumerate((first_date, second_date), start=1):
        _retained_analyzed_day(
            library_root,
            database,
            feed_id,
            archive_date,
            location="Main and First",
            start_seconds=7_200 * index,
        )
    _audio_only_day(library_root, feed_id, audio_only_date)
    _raw_source_only_day(library_root, feed_id, raw_source_date)

    entire = _run_worker(
        tmp_path,
        library_root,
        database,
        "question-coverage",
        "--output-dir",
        str(library_root),
        "--feed-id",
        feed_id,
        "--entire-feed",
    )[-1]["coverage"]
    assert entire["scope"] == "entire_feed"
    assert entire["start_date"] == first_date.isoformat()
    assert entire["end_date"] == raw_source_date.isoformat()
    assert entire["question_ready_dates"] == [
        first_date.isoformat(),
        second_date.isoformat(),
    ]
    assert entire["missing_audio_dates"] == [missing_date.isoformat()]
    assert entire["local_processing_dates"] == [audio_only_date.isoformat()]
    assert entire["partial_audio_dates"] == [raw_source_date.isoformat()]
    assert entire["acquisition_needed_dates"] == [
        missing_date.isoformat(),
        raw_source_date.isoformat(),
    ]

    # A new source block must not revoke the two already indexed days.
    _raw_source_only_day(library_root, feed_id, first_date)
    month = _run_worker(
        tmp_path,
        library_root,
        database,
        "question-coverage",
        "--output-dir",
        str(library_root),
        "--feed-id",
        feed_id,
        "--start-date",
        "2026-07-01",
        "--end-date",
        "2026-07-31",
    )[-1]["coverage"]
    assert month["evidence_update_dates"] == [first_date.isoformat()]
    assert "last indexed transcript" in month["summary"]
    assert month["scope"] == "range"
    assert month["requested_day_count"] == 31
    assert month["question_ready_day_count"] == 2
    assert month["audio_day_count"] == 4
    assert month["complete_coverage"] is False
    assert month["question_ready_ranges"] == [
        "2026-07-06 through 2026-07-07"
    ]

    class LocalPassageIndexer:
        def __init__(self, store: AnalysisStore, model: str) -> None:
            self.store = store
            self.model = model

        def index_missing(self, **_scope: Any) -> int:
            return 0

        def search(
            self,
            selected_feed_id: str,
            start_date: date,
            end_date: date,
            _query: str,
            *,
            limit: int,
            archive_dates: list[str] | None = None,
        ) -> list[dict[str, Any]]:
            allowed = set(archive_dates or [])
            return [
                value
                for value in self.store.get_passages(
                    selected_feed_id,
                    start_date,
                    end_date,
                )
                if not allowed or str(value["archive_date"]) in allowed
            ][:limit]

    class GroundedPatternClient:
        model = "offline-e2e-pattern-client"

        def __init__(self) -> None:
            self.system = ""
            self.user = ""
            self.location_id = ""

        def chat_json(self, **kwargs: Any) -> dict[str, Any]:
            self.system = str(kwargs["system"])
            self.user = str(kwargs["user"])
            location_line = next(
                line
                for line in self.user.splitlines()
                if "repeated extracted location Main and First=2" in line
            )
            self.location_id = location_line.split()[0]
            return {
                "answer": (
                    "Main and First is the repeated extracted radio-report cluster "
                    f"in the covered July records [{self.location_id}]."
                ),
                "evidence_ids": [self.location_id],
                "limitations": [],
            }

    client = GroundedPatternClient()

    @contextmanager
    def open_client(_config: object) -> Iterator[GroundedPatternClient]:
        yield client

    from broadcastify_cli import worker

    emitted: list[dict[str, Any]] = []
    monkeypatch.setattr(worker, "DEFAULT_DATABASE", database)
    monkeypatch.setattr(worker, "SemanticIndexer", LocalPassageIndexer)
    monkeypatch.setattr(worker, "open_analysis_client", open_client)
    monkeypatch.setattr(worker, "emit", emitted.append)
    monkeypatch.setattr(
        worker.sys,
        "stdin",
        io.StringIO(
            json.dumps(
                {
                    "feed_id": feed_id,
                    "start_date": "2026-07-01",
                    "end_date": "2026-07-31",
                    "question": "Where and when are the recurring hot spots?",
                    "output_dir": str(library_root),
                    "history": [
                        {
                            "role": "user",
                            "content": "Focus on patterns across the whole feed.",
                        },
                        {
                            "role": "assistant",
                            "content": "I will keep missing dates explicit.",
                        },
                    ],
                }
            )
        ),
    )

    assert worker.ask_archive() == 0
    answer_event = emitted[-1]
    assert answer_event["type"] == "answer"
    answer = answer_event["result"]
    assert answer["evidence_ids"] == [client.location_id]
    assert "Cited event dates and times (archive time):" in answer["answer"]
    assert "2026-07-06 at archive offset 02:00:00" in answer["answer"]
    assert "2026-07-07 at archive offset 04:00:00" in answer["answer"]
    assert any(
        value["kind"] == "location"
        and value["label"] == "Main and First"
        and value["count"] == 2
        for value in answer["patterns"]
    )
    assert answer["coverage"]["requested_day_count"] == 31
    assert answer["coverage"]["question_ready_day_count"] == 2
    assert answer["limitations"][0].startswith("Partial retained coverage:")
    assert "EARLIER CHAT:" in client.user
    assert "Question-ready date ranges: 2026-07-06 through 2026-07-07" in client.user
    assert "Never interpret an unavailable date as a day with no activity" in client.user
    assert "population-normalized crime rates" in client.system
    with AnalysisStore(database) as store:
        saved = store.connection.execute(
            "SELECT question, answer, model FROM qa_history"
        ).fetchall()
    assert len(saved) == 1
    assert saved[0]["question"] == "Where and when are the recurring hot spots?"
    assert "Cited event dates and times (archive time):" in saved[0]["answer"]
    assert saved[0]["model"] == client.model
