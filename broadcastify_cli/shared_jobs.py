"""Durable job records in the existing LAN journal, never model credentials."""
from __future__ import annotations

import ctypes
import json
import os
import sqlite3
from pathlib import Path
from typing import Any

from .pipeline_sync import PIPELINE_SYNC_DATABASE

WORKFLOW_COMMANDS = {"run", "run-scheduled", "continue-local", "analyze-day"}
ACTIVE_STATES = {"running", "canceling", "pausing"}


def process_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    if os.name != "nt":
        try:
            stat = Path(f"/proc/{pid}/stat")
            if stat.exists() and stat.read_text().rsplit(")",1)[-1].split()[0] == "Z":
                return False
            os.kill(pid, 0)
            return True
        except ProcessLookupError:
            return False
        except PermissionError:
            return True
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.OpenProcess.restype = ctypes.c_void_p
    kernel.OpenProcess.argtypes = [ctypes.c_uint32, ctypes.c_int, ctypes.c_uint32]
    kernel.CloseHandle.argtypes = [ctypes.c_void_p]
    kernel.GetExitCodeProcess.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint32)]
    handle = kernel.OpenProcess(0x1000, False, pid)
    if not handle:
        return ctypes.get_last_error() == 5
    try:
        code = ctypes.c_uint32()
        return not kernel.GetExitCodeProcess(handle, ctypes.byref(code)) or code.value == 259
    finally:
        kernel.CloseHandle(handle)


def safe_request(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: safe_request(v) for k, v in value.items()
                if not any(part in k.lower() for part in ("password", "token", "secret", "username"))
                and ("api_key" not in k.lower() or k.lower().endswith("_env"))}
    if isinstance(value, list):
        return [safe_request(v) for v in value]
    return value


def compact_event(event: dict[str, Any]) -> dict[str, Any]:
    keys = {"type","stage","phase","current","total","message","feed_id","date","received_at","event_index","elapsed_seconds","estimated_remaining_seconds"}
    return {k:(v[:1024] if isinstance(v,str) else v) for k,v in event.items() if k in keys}


def workflow_result(event: dict[str, Any] | None) -> dict[str, Any] | None:
    if not event:
        return None
    result = event.get("result") or {}
    keys = {"feed_id","requested_days","completed_days","download_limited","quota_message","pending_processing_days","missing_days"}
    return {"type":event.get("type"),"result":{k:v for k,v in result.items() if k in keys}}


class SharedJobs:
    def __init__(self, output_dir: Path) -> None:
        self.path = output_dir / PIPELINE_SYNC_DATABASE
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as c:
            c.execute("""CREATE TABLE IF NOT EXISTS shared_jobs (
                id TEXT PRIMARY KEY, record_json TEXT NOT NULL,
                request_json TEXT NOT NULL, position INTEGER NOT NULL, state TEXT NOT NULL
            )""")
            c.execute("CREATE INDEX IF NOT EXISTS shared_jobs_state ON shared_jobs(state,position)")

    def connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.path, timeout=15)

    def save(self, record: dict[str, Any], request: dict[str, Any] | None = None) -> None:
        with self.connect() as c:
            c.execute("BEGIN IMMEDIATE")
            old = c.execute("SELECT request_json,position FROM shared_jobs WHERE id=?", (record["id"],)).fetchone()
            position = old[1] if old else c.execute("SELECT COALESCE(MAX(position),0)+1 FROM shared_jobs").fetchone()[0]
            request_json = json.dumps(safe_request(request)) if request is not None else old[0] if old else "{}"
            c.execute("INSERT OR REPLACE INTO shared_jobs VALUES (?,?,?,?,?)",
                      (record["id"], json.dumps(record), request_json, position,record["status"]))

    def records(self) -> list[dict[str, Any]]:
        with self.connect() as c:
            rows = c.execute("""SELECT record_json FROM shared_jobs
                WHERE state NOT IN ('completed','failed','canceled') OR id IN (
                    SELECT id FROM shared_jobs WHERE state IN ('completed','failed','canceled')
                    ORDER BY position DESC LIMIT 30)
                ORDER BY position""").fetchall()
        return [json.loads(row[0]) for row in rows]

    def get(self, job_id: str) -> dict[str, Any] | None:
        with self.connect() as c:
            row = c.execute("SELECT record_json FROM shared_jobs WHERE id=?",(job_id,)).fetchone()
        return json.loads(row[0]) if row else None

    def request(self, job_id: str) -> dict[str, Any]:
        with self.connect() as c:
            row = c.execute("SELECT request_json FROM shared_jobs WHERE id=?", (job_id,)).fetchone()
        return json.loads(row[0]) if row else {}

    def move_next(self, job_id: str) -> None:
        with self.connect() as c:
            c.execute("UPDATE shared_jobs SET position=(SELECT MIN(position)-1 FROM shared_jobs) WHERE id=?", (job_id,))
