#!/usr/bin/env bash
# Creates the Redpanda topics this project publishes to. Idempotent -- safe to
# re-run; `rpk topic create` reports an existing topic without failing the script.
set -euo pipefail

CONTAINER="${REDPANDA_CONTAINER:-retail_platform_redpanda}"
TOPIC="${EVENTS_TOPIC:-retail.events}"
PARTITIONS="${EVENTS_PARTITIONS:-3}"

# 3 partitions is deliberate: it gives the Spark job real parallelism, and it
# guarantees cross-partition ordering is *not* global -- which is exactly the
# out-of-order condition the Phase 3 watermark logic has to handle.
docker exec "$CONTAINER" rpk topic create "$TOPIC" -p "$PARTITIONS" -r 1 || true

docker exec "$CONTAINER" rpk topic list
