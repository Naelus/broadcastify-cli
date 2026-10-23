"""User boundaries: full queued finish ranges and a genuinely isolated preview."""
from datetime import date, timedelta, datetime, timezone
import hashlib
import json
import re
import threading
import time
import os

import pytest
import requests

from broadcastify_cli import workflow_eta as timing
from broadcastify_cli.storage import AnalysisStore
from broadcastify_cli.web_app import create_server
from broadcastify_cli.lan_node import create_lan_node_server
from broadcastify_cli.lan_sync import LanArchiveSyncClient
from tests.e2e.test_product_workflows import _retained_analyzed_day

pytestmark = pytest.mark.usefixtures("fast_server_shutdown")


def test_whole_backlog_eta_is_shared_and_does_not_treat_a_schedule_as_a_worker(tmp_path, monkeypatch):
    from broadcastify_cli.archive_cache import remember_complete_archive_day
    for name in list(os.environ):
        if name.startswith("BROADCASTIFY_"):
            monkeypatch.delenv(name)
    monkeypatch.setenv("BROADCASTIFY_QUOTA_LEDGER",str(tmp_path/"quota.sqlite3"))
    root=tmp_path/"master"
    database=root/"broadcastify-analysis.sqlite3"
    day=date.today()-timedelta(days=5)
    audio=_retained_analyzed_day(root,database,"90001",day,location="Main",start_seconds=10)
    remember_complete_archive_day(audio.parent,"90001",day,[])
    settings={"feed_id":"90001","combine":True,"transcribe":True,"diarize":True}
    with AnalysisStore(database) as store:
        store.connection.execute("DELETE FROM embeddings")
        store.connection.commit()
        store.save_feed_schedule({"feed_id":"90001","enabled":True,"analyze":True,"run_time_local":"23:59","lookback_days":1,"job":settings})
    timing.TimingStore(root).sample("index",timing.profile_key("index",settings),1,20)
    master=create_lan_node_server(root,host="127.0.0.1",port=0,working_dir=tmp_path,database_path=database,
        role="master",job_relay_enabled=True,discovery_enabled=False)
    master.quiet=True
    threading.Thread(target=master.serve_forever,daemon=True).start()
    manager=master.job_manager
    client=LanArchiveSyncClient(enabled=True,role="follower",master_url=f"http://127.0.0.1:{master.server_port}",discovery_enabled=False)
    def overview():
        manager._overview_at=0
        manager.overview()
        deadline=time.monotonic()+3
        while manager._overview_busy and time.monotonic()<deadline:
            time.sleep(.01)
        assert not manager._overview_busy
        return manager.overview()
    try:
        scheduled=overview()
        assert scheduled["state"]=="scheduled" and scheduled["finish_latest"] is None
        assert scheduled["next_start_at"]
        request={"analyze":True,"job":{**settings,"start_date":str(day),"end_date":str(day)}}
        record={"id":"0123456789abcdef","command":"run-scheduled","owner_pid":os.getpid(),"status":"running","request":request}
        manager.observe(record)
        running=overview()
        # The older day is unfinished although the daily schedule targets yesterday.
        assert running["pending_days"]>=1
        assert running["upper_seconds"]>=20
        client.list_master_jobs()
        assert client.last_workflow_eta==running
        manager.observe({**record,"status":"paused"})
        paused=overview()
        assert paused["state"]=="paused" and paused["finish_latest"] is None
        client.list_master_jobs()
        assert client.last_workflow_eta==paused
    finally:
        master.shutdown();master.server_close()


def test_measured_remaining_work_survives_restart_and_never_times_cache_or_pause(tmp_path, monkeypatch):
    root = tmp_path/"archive"
    database = root/"broadcastify-analysis.sqlite3"
    day = date(2026,7,12)
    _retained_analyzed_day(root,database,"90001",day,location="Main",start_seconds=10)
    with AnalysisStore(database) as store:
        store.connection.execute("DELETE FROM embeddings")
        store.connection.commit()
    clock = [100.0]
    # A controllable clock measures completed work without sleeping/GPU calls.
    monkeypatch.setattr(timing.time,"monotonic",lambda:clock[0])
    settings = {"output_dir":root,"feed_id":"90001","combine":True,"transcribe":True,"diarize":True}
    @timing.workflow_scope
    def finish_stage(request, *, cached=False, fail=False):
        with timing.measure("index",1,day):
            clock[0] += 20
            if cached: timing.skip_sample()
            if fail: raise RuntimeError("interrupted before saving")
    finish_stage(settings)
    finish_stage(settings,cached=True)
    with pytest.raises(RuntimeError): finish_stage(settings,fail=True)
    from broadcastify_cli.library import scan_local_library
    days = scan_local_library(root,database)
    work, notes = timing.remaining_work(root,database,settings,days,True)
    assert [item["stage"] for item in work] == ["index"]  # Finished ASR/analysis do not return to the budget.
    first = timing.estimate_work(root,settings,work)
    assert first["lower_seconds"] <= 20 <= first["upper_seconds"]
    assert first["stages"][0]["samples"] == 1
    # A new reader uses durable measurements; another hardware/profile must learn.
    assert timing.TimingStore(root).rates("index",timing.profile_key("index",settings))[2] == 1
    changed = timing.estimate_work(root,{**settings,"device":"cpu"},[{"stage":"transcribe","units":100,"day":str(day)}])
    assert changed["unknown_stages"]
    rows = [dict(id="first",status="running",scope_progress={"estimate":first}),
            dict(id="second",status="queued",scope_progress={"estimate":first}),
            dict(id="paused",status="paused",scope_progress={"estimate":first})]
    timing.decorate_queue(rows,root)
    assert datetime.fromisoformat(rows[1]["eta"]["finish_latest"]) > datetime.fromisoformat(rows[0]["eta"]["finish_latest"])
    assert rows[1]["eta"]["queue_upper_seconds"] == first["upper_seconds"]
    assert rows[2]["eta"]["finish_latest"] is None
    rows[0]["status"] = "waiting_quota"
    rows[0]["scope_progress"]["source_days"] = 1
    timing.decorate_queue(rows[:1],root)
    assert rows[0]["eta"]["finish_latest"] is None
    assert rows[0]["eta"]["state"] == "waiting_source"


def test_browser_preview_reviews_saved_samples_and_cannot_touch_production(tmp_path, monkeypatch):
    root = tmp_path/"production"
    database = root/"broadcastify-analysis.sqlite3"
    audio = _retained_analyzed_day(root,database,"90001",date(2026,7,12),location="Private retained place",start_seconds=10)
    original = hashlib.sha256(audio.read_bytes()).hexdigest()
    server = create_server(root,database,port=0,working_dir=tmp_path)
    server.quiet = True
    launches=[]
    monkeypatch.setattr(server.state.jobs,"start",lambda *args:launches.append(args))
    thread = threading.Thread(target=server.serve_forever,daemon=True);thread.start()
    base = f"http://127.0.0.1:{server.server_port}"
    client = requests.Session()
    try:
        page=client.get(base+"/",timeout=5)
        client.headers["X-Radio-Archive-Token"]=re.search(r'name="app-token" content="([^"]+)"',page.text).group(1)
        created=client.post(base+"/api/preview",json={},timeout=15)
        assert created.status_code == 201,created.text
        key=created.json()["url"].split("/")[2]
        prefix=f"/api/preview/{key}"
        previewpage=client.get(base+created.json()["url"],timeout=5)
        assert 'name="preview-base"' in previewpage.text
        snapshot=client.get(base+prefix+"/api/bootstrap",timeout=5).json()
        assert snapshot["days"] and all(d["feed_id"] != "90001" for d in snapshot["days"])
        params={"feed_id":"999991","start_date":str(date.today()-timedelta(days=7)),"end_date":str(date.today())}
        report=client.get(base+prefix+"/api/activity",params=params,timeout=5).json()
        assert len(report["items"])==2
        acknowledged=client.post(base+prefix+"/api/activity/seen",json={**params,"items":report["items"]},timeout=5)
        assert acknowledged.json()["acknowledged_count"]==2
        assert client.get(base+prefix+"/api/activity",params=params,timeout=5).json()["items"]==[]
        # All routes that could load a model, edit credentials, schedule work or
        # acquire provider audio are denied by the isolated backend context.
        for path in ("/api/jobs","/api/credentials","/api/schedules","/api/lan/v1/pipeline-request"):
            denied=client.post(base+prefix+path,json={"command":"run-scheduled","payload":{"job":{"feed_id":"90001"}}},timeout=5)
            assert denied.status_code == 403
        denied=client.get(base+prefix+"/media?path=../../production/broadcastify-analysis.sqlite3",timeout=5)
        assert denied.status_code == 404
        assert launches==[]
        assert hashlib.sha256(audio.read_bytes()).hexdigest()==original
        with AnalysisStore(database) as store:
            assert store.get_day("90001",date(2026,7,12)) is not None
            assert store.get_day("999991",date.today()-timedelta(days=3)) is None
    finally:
        server.shutdown();server.server_close();thread.join(2)
