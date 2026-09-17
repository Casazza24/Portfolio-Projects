# Phase 5 (Airflow) — handoff

Status as of 2026-08-11. The build was **interrupted mid-session**, not finished.
Nothing here is known to be broken; it is unfinished and partly unverified.

## What exists

| File | State |
|---|---|
| `airflow/docker-compose.yml` | Complete. Airflow 2.10.5, LocalExecutor, own metadata Postgres, webserver on **:8081** (8080 is the Redpanda Console). |
| `airflow/dags/retail_platform_pipeline.py` | Complete, parses, registered, **zero import errors**. |
| `airflow/host_runner.py` | Complete and syntactically valid, but **modified after the last DAG run** — the version on disk has not been exercised end to end. |
| `airflow/.env.example`, `.gitignore` | Present. |
| `warehouse/load_snowflake.py` | Modified: retires the duplicate `raw.dim_date` load. Reviewed, correct, matches the decision that dbt's `dim_date` model is canonical. |

Containers currently running: `retail_platform_airflow_postgres`, `_scheduler`,
`_webserver` (`_init` exited 0).

## The architecture, in one paragraph

Airflow runs in Docker; the work does not. The Spark jobs need a Java 11 JDK,
local Parquet, and two mutually incompatible host virtualenvs (`.venv` for
pyspark/snowflake-connector, `dbt/.venv` for dbt-core). None of that exists in
an Airflow image. `host_runner.py` bridges the gap: a fixed registry of six
named jobs, each pinned to its interpreter and working directory, exposed over
HTTP on **:8171** so a container task can start one and stream stdout back into
the Airflow log. Exit status arrives as a sentinel line (`__host_runner_exit__ N`)
because the response starts streaming long before the process finishes.

**This means `host_runner.py` must be running on the host before any DAG run.**
That is the single most important operational fact in this phase.

## What actually ran

Two DAG runs exist, both ending `failed`.

`phase5_real_run_01`:

| Task | State | Note |
|---|---|---|
| `extract_postgres` | **success** | Real run against the live Olist Postgres, 15s. |
| `spark_backfill` | attempt 1 **exit 0**, attempt 2 **failed** | Attempt 1 genuinely succeeded — 18:09:54→18:20:35, ~11 min, consistent with the known 8-min job. |
| `build_dim_date` | upstream_failed | Never executed. |
| `load_snowflake` | upstream_failed | Never executed. |
| `dbt_run` | upstream_failed | Never executed. |
| `dbt_test` | upstream_failed | Never executed. |

`phase5_broken_run_01` (the deliberately-broken-run deliverable): same shape,
`extract_postgres` success, `spark_backfill` failed, rest upstream_failed.

## What is wrong

**One root cause, and it is not a code defect.**

Both `spark_backfill` failures are identical:

```
RuntimeError: spark_backfill: connection ended before the job reported an exit code
```

That is the DAG's own guard for "the HTTP body ended without the exit sentinel" —
i.e. the host runner died mid-job. `host_runner.py` was running in the
foreground of the interrupted agent's session, so killing the agent killed the
runner, which dropped the connection. Attempt 1 exiting 0 is the proof: the same
task, same code, same data, succeeded when the runner stayed up.

So the guard worked correctly. It caught a real dropped connection and refused
to report success. The failure is an artifact of how the session ended.

## What is unverified

This is the important part — the headline Phase 5 claim is **unproven**:

1. **`load_snowflake`, `dbt_run` and `dbt_test` have never executed once.** In
   both runs they were `upstream_failed`. The expected-and-correct outcome —
   the pipeline running green until it fails at Snowflake for want of an
   account — has not been demonstrated.
2. **The on-disk `host_runner.py` postdates the last run.** It was edited at
   14:31, after the 14:09 and 14:27 runs started. Whatever was fixed in that
   edit is untested.
3. **The deliberately-broken run demonstrates the wrong failure.** It shows the
   host runner dying, which is an artifact of this session, not a designed
   demo. The scope doc wants a failure that shows Airflow surfacing a *pipeline*
   problem.

## What another agent needs to do

In order:

1. **Start the host runner as a detached process**, not in the foreground of the
   agent session — `nohup .venv/bin/python airflow/host_runner.py &` or
   equivalent, verified with `curl localhost:8171/health`. Every failure in this
   phase so far traces back to this. Confirm it survives independently before
   triggering anything.
2. **Clear and re-run `phase5_real_run_01`** (or trigger a fresh run). Expected
   correct outcome: `extract_postgres` success → `spark_backfill` success (~8–11
   min) → `build_dim_date` success → `load_snowflake` **fails at the Snowflake
   connection**, because no account exists → `dbt_run`/`dbt_test` upstream_failed.
   That is the honest Phase 5 deliverable. **Do not mock or stub Snowflake to
   manufacture a green run.**
   - Note `dbt/profiles.yml` exists and reads credentials from env vars that are
     unset, so `dbt_run` would fail on connection too — confirm it fails for
     that reason and not a missing-profile reason.
3. **Replace the broken-run demo** with a failure that is about the pipeline
   rather than the harness. Best candidate: a dbt test failing, once Snowflake
   exists. Until then, a deliberately bad input to `spark_backfill` (a
   quarantine-triggering payload, or a missing input path) shows Airflow
   surfacing a real task failure, retry behaviour, and downstream tasks
   correctly not running.
4. **Document Phase 5 in `README.md`.** It currently says only "*(Phase 5)*"
   next to Airflow in two places. Needs: how to start Airflow, the :8081 port
   and why it is not 8080, the mandatory host-runner step, how to trigger the
   DAG, and what currently passes vs. what is blocked on Snowflake.
5. **Verify `spark_backfill`'s retry policy.** The DAG sets `retries=0` on that
   task with a documented rationale (8-minute deterministic job), but attempt 2
   exists in the logs, meaning it was manually cleared. Confirm the policy is
   what's actually wanted.

## Constraints to respect

- **Disk is at 92%, ~15 GB free.** The Airflow images cost ~3 GB. Do not pull
  more images without checking; do not run `docker system prune`.
- Do not disturb `retail_platform_postgres`, `retail_platform_redpanda`,
  `retail_platform_redpanda_console` — those hold the source data and the event
  topic. Airflow lives in its own compose file specifically so a `docker compose
  down` cannot take them with it.
- `spark/batch_backfill.py` overwrites `data/clean/events` in place, and the DAG
  sets `max_active_runs=1` for that reason. Do not run two DAG runs at once.
- There is no Snowflake account and one cannot be obtained. Everything from
  `load_snowflake` onward is expected to fail at connection.