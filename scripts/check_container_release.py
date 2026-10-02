"""Exercise the shipped TrueNAS image offline before publishing it.

The container has no network, GPU, secrets, or production storage. This checks
the real unprivileged/read-only deployment boundary, not Dockerfile strings.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import sqlite3
import subprocess
import tempfile
import uuid


PROBE = r'''
import http.cookiejar, json, os, re, time, urllib.error, urllib.request
import broadcastify_cli
base = 'http://127.0.0.1:8765'
deadline = time.monotonic() + 30
while True:
    try:
        with urllib.request.urlopen(base + '/health', timeout=2) as response:
            assert json.load(response) == {'status': 'ok', 'scope': 'trusted-lan'}
        break
    except OSError:
        if time.monotonic() >= deadline:
            raise
        time.sleep(.1)
opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(http.cookiejar.CookieJar()))
html = opener.open(base + '/', timeout=5).read().decode()
token = re.search(r'<meta name="app-token" content="([^"]+)"', html).group(1)
bootstrap = json.load(opener.open(base + '/api/bootstrap', timeout=5))
assert bootstrap['runtime']['source_commit'] == os.environ['BROADCASTIFY_SOURCE_COMMIT']
assert bootstrap['runtime']['version'] == broadcastify_cli.__version__
payload = json.dumps({'command': 'diagnostics', 'payload': {}}).encode()
headers = {'Content-Type': 'application/json', 'Origin': base}
try:
    opener.open(urllib.request.Request(base + '/api/jobs', data=payload, headers=headers), timeout=5)
except urllib.error.HTTPError as error:
    assert error.code == 403
else:
    raise AssertionError('Mutation accepted without an action token')
headers['X-Radio-Archive-Token'] = token
job = json.load(opener.open(urllib.request.Request(base + '/api/jobs', data=payload, headers=headers), timeout=5))
deadline = time.monotonic() + 30
while job['status'] in {'queued', 'running', 'canceling'}:
    assert time.monotonic() < deadline, 'Packaged worker did not finish'
    time.sleep(.1)
    job = json.load(opener.open(base + '/api/jobs/' + job['id'], timeout=5))
assert job['status'] == 'completed', job.get('error')
assert job['result']['type'] == 'diagnostics'
print('Packaged bootstrap, mutation protection, and worker passed')
'''


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image")
    parser.add_argument("commit")
    args = parser.parse_args()
    if os.name == "nt" or os.getuid() == 0:
        raise RuntimeError("Run this check as an unprivileged Linux user with Docker access.")
    name = "radio-release-check-" + uuid.uuid4().hex[:12]

    def docker(*command: str, **kwargs) -> subprocess.CompletedProcess:
        return subprocess.run(["docker", *command], check=True, timeout=90, **kwargs)

    with tempfile.TemporaryDirectory(prefix="radio-container-check-") as temporary:
        data = Path(temporary)
        data.chmod(0o777)
        archives = data / "archives"
        archives.mkdir(mode=0o777)
        archives.chmod(0o777)
        retained = archives / "retained-source.mp3"
        retained.write_bytes(b"retained data must survive image replacement")
        before = retained.read_bytes()
        database = archives / "broadcastify-analysis.sqlite3"
        with sqlite3.connect(database) as connection:
            connection.execute("CREATE TABLE release_sentinel(value TEXT)")
            connection.execute("INSERT INTO release_sentinel VALUES ('retained checkpoint')")
        database.chmod(0o666)
        try:
            docker(
                "run", "--detach", "--name", name, "--network", "none",
                "--read-only", "--user", f"{os.getuid()}:{os.getgid()}", "--cap-drop", "ALL",
                "--security-opt", "no-new-privileges:true", "--tmpfs", "/tmp:rw,noexec,nosuid,nodev",
                "--volume", f"{data}:/data", "--env", "HOME=/data/runtime/home",
                "--env", "BROADCASTIFY_LAN_SHARING=false",
                "--env", "BROADCASTIFY_LAN_DISCOVERY_ENABLED=false",
                "--env", "BROADCASTIFY_LAN_BACKGROUND_SYNC=false",
                "--env", "BROADCASTIFY_QUOTA_LEDGER=/data/archive-quota.sqlite3",
                args.image,
            )
            revision = docker("inspect", "--format", '{{index .Config.Labels "org.opencontainers.image.revision"}}', name, capture_output=True, text=True).stdout.strip()
            assert revision == args.commit, (revision, args.commit)
            docker("exec", name, "/opt/radio-archive-venv/bin/python", "-c", PROBE)
            # Missing shared libraries should fail the release before a user
            # tries their first transcription or analysis job.
            for executable in ("/usr/bin/ffmpeg", "/app/build/bin/whisper-cli", "/app/llama-server"):
                docker("exec", name, executable, "-version" if executable.endswith("ffmpeg") else "--help", stdout=subprocess.DEVNULL)
            docker("stop", "--time", "25", name)
            code = docker("inspect", "--format", "{{.State.ExitCode}}", name, capture_output=True, text=True).stdout.strip()
            assert code == "0", f"Service did not exit gracefully: {code}"
            assert retained.read_bytes() == before
            with sqlite3.connect(database) as connection:
                assert connection.execute("PRAGMA integrity_check").fetchone() == ("ok",)
                assert connection.execute("SELECT value FROM release_sentinel").fetchone() == ("retained checkpoint",)
            print("Container shutdown and retained database/audio passed")
        finally:
            subprocess.run(["docker", "logs", "--tail", "30", name], timeout=10, check=False)
            subprocess.run(["docker", "rm", "--force", name], timeout=30, check=False, stdout=subprocess.DEVNULL)


if __name__ == "__main__":
    main()
