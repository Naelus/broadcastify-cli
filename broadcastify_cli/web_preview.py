"""An isolated saved-data preview, served through the existing web listener.

Only the explicit read/acknowledgement routes below exist. There is no worker,
scheduler, account store, LAN client or model provider in this context.
"""
from datetime import date, timedelta
from pathlib import Path
from types import SimpleNamespace
import os
import tempfile
import uuid

from . import __version__
from .activity import activity_report, acknowledge_activity
from .preview import seed_preview


class WebPreview:
    def __init__(self):
        self.key = uuid.uuid4().hex
        paths = seed_preview(Path(tempfile.gettempdir())/f"broadcastify-preview-{self.key}")
        self.state = SimpleNamespace(output_dir=paths.library_root, database_path=paths.database_path)

    def get(self, path, query):
        from .web_app import _library_payload, _day_payload, _transcript_payload, WebRequestError
        def arg(name, default=""):
            return (query.get(name) or [default])[0]
        if path == "/api/bootstrap":
            return {**_library_payload(self.state), "profiles":[],"schedules":[],"area_runs":[],
                "runtime":{"version":__version__,"source_commit":os.getenv("BROADCASTIFY_SOURCE_COMMIT", ""),
                    "platform":"Isolated preview", "python":"No model workers", "access_scope":"Sample data only",
                    "output_dir":str(self.state.output_dir),"database_path":str(self.state.database_path),
                    "credentials":{},"lan_sync":{},"processing_defaults":{}}}
        if path == "/api/workflow":
            return {"workflow":_library_payload(self.state)["workflow"]}
        if path == "/api/day":
            result = _day_payload(self.state, arg("feed_id"), date.fromisoformat(arg("date")))
            if result.get("audio_url"):
                result["audio_url"] = f"/api/preview/{self.key}"+result["audio_url"]
            return result
        if path == "/api/transcript":
            return _transcript_payload(self.state,arg("feed_id"),date.fromisoformat(arg("date")),
                max(0,int(arg("offset",0))),min(1000,max(1,int(arg("limit",250)))),arg("q")[:200])
        if path == "/api/activity":
            return activity_report(self.state.database_path,self.state.output_dir,
                {k:arg(k) for k in ("feed_id","start_date","end_date")})
        if path == "/api/jobs":
            return {"jobs":[],"preview":True}
        if path == "/api/system-activity":
            return {"scheduler":{"active":None,"schedules":[]},"account_pool":{"profiles":[]},"preview":True}
        raise WebRequestError(403,"Preview allows saved sample review only; downloads, models, credentials, schedules and LAN controls are disabled.")

    def post(self, path, body):
        from .web_app import WebRequestError
        if path == "/api/activity/seen":
            return acknowledge_activity(self.state.database_path,self.state.output_dir,body)
        raise WebRequestError(403,"Preview cannot start or change production work. Review and acknowledge the synthetic sample incidents instead.")
