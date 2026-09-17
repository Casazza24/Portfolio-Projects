#!/usr/bin/env python3
"""Runs the pipeline's host-side jobs on behalf of the Airflow containers.

Airflow runs in Docker; the work it orchestrates does not. The Spark jobs need
a Java 11 JDK and the machine's local Parquet output, and the project keeps two
mutually incompatible virtualenvs (`.venv` for pyspark/snowflake-connector,
`dbt/.venv` for dbt-core -- see dbt/requirements-dbt.txt for why they cannot
share one). Neither exists inside an Airflow image.

This is the bridge: a fixed registry of named jobs, each pinned to the
interpreter and working directory it needs, exposed over HTTP so a task in a
container can start one and stream its output back into the Airflow log.

ponytail: the alternatives, and why not.
  * Mount the project into the Airflow container and run the venvs there. Does
    not work at all -- `.venv` holds macOS arm64 wheels and resolves JAVA_HOME
    through /usr/libexec/java_home, neither of which exists on Linux.
  * DockerOperator. Moves the problem rather than solving it: the spawned
    container still needs Spark, Java, both venvs and the 363 MB of local
    Parquet, so it means building and maintaining a second image.
  * Run Airflow itself on the host, no containers. Genuinely simpler, and what
    a solo laptop build would normally do -- but the deliverable here is a
    containerised LocalExecutor deployment, and containerised Airflow reaching
    host-resident work is the situation worth being able to talk about.
  * SSHOperator back to the host. The standard production answer, and closest
    to this in spirit -- a fixed remote command run over a authenticated
    channel. It needs macOS Remote Login enabled and a key provisioned into the
    image, which is more moving parts than this for the same shape.

What replaces this in production is a worker that has the runtime baked in --
a Celery/Kubernetes executor whose image ships Spark and dbt, so the "where
does the code live" question stops existing.

Security: the socket binds 0.0.0.0 because Docker Desktop's
host.docker.internal cannot reach a loopback-bound port on the host. What makes
that acceptable is that no request can name a command -- callers pick from
JOBS, and an unrecognised name is a 404. A shared token guards it beyond that.

Usage:
    python3 airflow/host_runner.py
    HOST_RUNNER_TOKEN=... HOST_RUNNER_PORT=8171 python3 airflow/host_runner.py
"""
from __future__ import annotations

import os
import signal
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
VENV_PYTHON = PROJECT_ROOT / ".venv" / "bin" / "python"
DBT_DIR = PROJECT_ROOT / "dbt"
DBT = DBT_DIR / ".venv" / "bin" / "dbt"

DEFAULT_PORT = 8171

# Shared with the DAG, which reads the same variable. The default is a
# placeholder, not a secret -- it exists so a fresh clone starts without
# configuration, on a port that only this machine's containers can reach.
TOKEN = os.environ.get("HOST_RUNNER_TOKEN", "local-dev")

# Emitted as the last line of every response. HTTP has no way to signal failure
# once a 200 has been sent and the body has started streaming, and streaming is
# the point -- the Spark job runs for minutes and its progress should appear in
# the Airflow log while it happens, not all at once at the end.
EXIT_SENTINEL = "__host_runner_exit__"

# Every child currently running, so a shutdown can take them with it. Found the
# hard way: killing the runner mid-backfill left the Spark driver and its JVM
# alive with no parent, still holding data/clean/events open -- and the next
# run would then have had two drivers overwriting the same Parquet dataset.
# A crashed orchestrator leaving live writers behind is worse than the crash.
_children: set[subprocess.Popen] = set()
_children_lock = threading.Lock()

# The registry is the interface. Adding a pipeline step means adding a line
# here and a task in the DAG; nothing else about the bridge changes.
JOBS = {
    "extract_postgres": {
        "argv": [str(VENV_PYTHON), "warehouse/extract_postgres.py"],
        "cwd": PROJECT_ROOT,
    },
    "spark_backfill": {
        "argv": [str(VENV_PYTHON), "spark/batch_backfill.py"],
        "cwd": PROJECT_ROOT,
    },
    "build_dim_date": {
        "argv": [str(VENV_PYTHON), "warehouse/build_dim_date.py"],
        "cwd": PROJECT_ROOT,
    },
    "load_snowflake": {
        # dim_date is deliberately absent from the dimension load: the canonical
        # calendar is dbt's `dim_date` model, generated in-warehouse.
        "argv": [str(VENV_PYTHON), "warehouse/load_snowflake.py"],
        "cwd": PROJECT_ROOT,
    },
    # dbt runs from its own directory and its own venv. DBT_PROFILES_DIR points
    # at dbt/ so the profile stays in the repo instead of ~/.dbt.
    "dbt_run": {
        "argv": [str(DBT), "run"],
        "cwd": DBT_DIR,
        "env": {"DBT_PROFILES_DIR": str(DBT_DIR)},
    },
    "dbt_test": {
        "argv": [str(DBT), "test"],
        "cwd": DBT_DIR,
        "env": {"DBT_PROFILES_DIR": str(DBT_DIR)},
    },
}


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def do_GET(self):
        if self.path != "/health":
            self.send_error(404)
            return
        self._respond_text("ok\n")

    def do_POST(self):
        if not self.path.startswith("/run/"):
            self.send_error(404)
            return
        if self.headers.get("X-Runner-Token") != TOKEN:
            self.send_error(403, "bad or missing X-Runner-Token")
            return

        job = JOBS.get(self.path[len("/run/") :])
        if job is None:
            self.send_error(404, "unknown job")
            return

        self.send_response(200)
        self.send_header("Content-Type", "text/plain; charset=utf-8")
        # Chunked rather than a length: the body is a live process's output.
        self.send_header("Transfer-Encoding", "chunked")
        self.end_headers()

        code = self._stream(job)
        self._chunk(f"{EXIT_SENTINEL} {code}\n")
        self.wfile.write(b"0\r\n\r\n")

    def _stream(self, job) -> int:
        env = {**os.environ, **job.get("env", {})}
        # Unbuffered, or a Python child's stdout arrives in 8 KB blocks and the
        # Airflow log shows nothing for minutes and then everything at once.
        env["PYTHONUNBUFFERED"] = "1"

        proc = subprocess.Popen(
            job["argv"],
            cwd=str(job["cwd"]),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            # Its own process group, so terminating the child takes the JVM
            # pyspark spawns underneath it as well. Killing the Python driver
            # alone leaves a 4 GB Spark JVM running.
            start_new_session=True,
        )
        with _children_lock:
            _children.add(proc)
        try:
            for line in proc.stdout:
                sys.stdout.write(line)
                try:
                    self._chunk(line)
                except (BrokenPipeError, ConnectionResetError):
                    # The caller hung up -- an Airflow task hitting its timeout
                    # or being cleared. Kill the child rather than orphaning a
                    # Spark driver that holds the Kafka consumer group open.
                    _terminate(proc)
                    break
            return proc.wait()
        finally:
            with _children_lock:
                _children.discard(proc)

    def _chunk(self, text: str) -> None:
        payload = text.encode("utf-8")
        self.wfile.write(f"{len(payload):x}\r\n".encode())
        self.wfile.write(payload + b"\r\n")
        self.wfile.flush()

    def _respond_text(self, text: str) -> None:
        payload = text.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/plain; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, fmt, *args):
        sys.stderr.write(f"{self.address_string()} {fmt % args}\n")


def _terminate(proc: subprocess.Popen) -> None:
    """Kills a child and everything it spawned, via its process group."""
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        pass  # already gone


def _shutdown(signum, frame):
    with _children_lock:
        running = list(_children)
    for proc in running:
        _terminate(proc)
    raise SystemExit(128 + signum)


def main():
    port = int(os.environ.get("HOST_RUNNER_PORT", DEFAULT_PORT))
    missing = [n for n, j in JOBS.items() if not Path(j["argv"][0]).exists()]
    if missing:
        # Fail at startup, not on the first task: a missing interpreter is a
        # setup mistake, and finding out about it eight minutes into a DAG run
        # is worse than finding out now.
        raise SystemExit(
            f"interpreter missing for: {', '.join(missing)} -- "
            "create .venv and dbt/.venv first (see README)"
        )

    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, _shutdown)

    # Threading, so /health still answers while the Spark job holds a request
    # open for minutes.
    server = ThreadingHTTPServer(("0.0.0.0", port), Handler)
    print(f"host runner on :{port}  jobs: {', '.join(JOBS)}")
    server.serve_forever()


if __name__ == "__main__":
    main()
