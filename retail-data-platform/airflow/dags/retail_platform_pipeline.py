"""Sequences the full pipeline: Postgres -> Spark -> warehouse -> dbt.

Every task is a named job on the host runner (see airflow/host_runner.py for
why the work runs on the host rather than in this container). The runner owns
which interpreter and working directory each job needs; this file owns the
order they run in and what happens when one fails. That split is deliberate --
changing a venv path should not touch the DAG, and changing the dependency
graph should not touch the runner.

ponytail: task granularity is one task per script. Finer (a task per Olist
table) buys nothing -- extract_postgres holds one Postgres connection and
either completes or does not, so nine tasks would always go green or red
together while adding eight rows of scheduler overhead. Coarser (one task for
the whole pipeline) is just a shell script with a web UI: the reason to
orchestrate at all is that "which stage broke, and can I resume from there"
is answerable, and per-script is the smallest unit at which it stays true.
"""
from __future__ import annotations

import os
import urllib.error
import urllib.request
from datetime import datetime, timedelta

from airflow.decorators import task
from airflow.models.dag import DAG

# host.docker.internal is Docker Desktop's route back to the host; the compose
# file maps it explicitly via host-gateway so this also works on plain Linux.
RUNNER_URL = os.environ.get("HOST_RUNNER_URL", "http://host.docker.internal:8171")
RUNNER_TOKEN = os.environ.get("HOST_RUNNER_TOKEN", "local-dev")

EXIT_SENTINEL = "__host_runner_exit__"


def run_on_host(job: str, timeout: int = 3600) -> None:
    """Runs a named host job, echoing its output into this task's log.

    The exit status arrives as the last line of the body rather than as an HTTP
    status, because the response starts streaming long before the process is
    finished -- see host_runner.py. A body that ends without the sentinel means
    the connection dropped mid-job, which is a failure even though the status
    was 200.
    """
    request = urllib.request.Request(
        f"{RUNNER_URL}/run/{job}",
        method="POST",
        headers={"X-Runner-Token": RUNNER_TOKEN},
    )

    exit_code = None
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            for raw in response:
                line = raw.decode("utf-8", errors="replace").rstrip("\n")
                if line.startswith(EXIT_SENTINEL):
                    exit_code = int(line.split()[1])
                    continue
                print(line)
    except urllib.error.URLError as exc:
        # The host runner being down is the most likely failure in this setup,
        # so it gets a message that says so instead of a bare socket error.
        raise RuntimeError(
            f"could not reach the host runner at {RUNNER_URL} ({exc}) -- "
            "start it with `python3 airflow/host_runner.py` on the host"
        ) from exc

    if exit_code is None:
        raise RuntimeError(f"{job}: connection ended before the job reported an exit code")
    if exit_code != 0:
        raise RuntimeError(f"{job}: exited {exit_code}")


with DAG(
    dag_id="retail_platform_pipeline",
    description="Postgres extract -> Spark clean -> dim_date -> Snowflake load -> dbt",
    # Manual trigger only. The sources are static Kaggle downloads, so a
    # schedule would produce identical runs on a timer; the interval this would
    # carry in production is the Postgres sync's, not the event stream's.
    schedule=None,
    start_date=datetime(2026, 1, 1),
    catchup=False,
    # The Spark job overwrites data/clean/events in place, so two concurrent
    # runs would read a dataset the other is rewriting.
    max_active_runs=1,
    default_args={
        "retries": 2,
        "retry_delay": timedelta(seconds=30),
    },
    tags=["retail-platform", "phase-5"],
) as dag:

    @task
    def extract_postgres():
        run_on_host("extract_postgres", timeout=900)

    @task(
        # ponytail: no retries here. The backfill takes ~8 minutes over the full
        # topic, and its realistic failures -- a transform bug, an OOM driver,
        # Redpanda down -- are all deterministic. Retrying twice costs 16
        # minutes to learn the same thing, so this one asks for a human.
        retries=0,
        execution_timeout=timedelta(minutes=45),
    )
    def spark_backfill():
        run_on_host("spark_backfill", timeout=2700)

    @task
    def build_dim_date():
        # Genuinely downstream of both branches: the span is derived from the
        # order dates in Postgres and from the event_date= partition names
        # Spark just wrote. Its CSV is no longer loaded into raw -- dbt's
        # dim_date model is the canonical calendar -- but the derived span is
        # what dbt_project.yml's date_spine bounds are checked against, so a
        # source that grows past them shows up here first.
        run_on_host("build_dim_date", timeout=300)

    @task
    def load_snowflake():
        run_on_host("load_snowflake", timeout=3600)

    @task
    def dbt_run():
        run_on_host("dbt_run", timeout=1800)

    @task
    def dbt_test():
        run_on_host("dbt_test", timeout=1800)

    # The two extracts are independent -- one reads Postgres, one reads the
    # Redpanda topic -- so LocalExecutor runs them in parallel and dim_date
    # waits on both.
    [extract_postgres(), spark_backfill()] >> build_dim_date() >> load_snowflake() >> dbt_run() >> dbt_test()
