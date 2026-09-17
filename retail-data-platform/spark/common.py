"""Shared Spark session setup and project paths.

Both the streaming job and the batch backfill build their session through
here, so they run on identical configuration -- which is the only way the two
code paths can be trusted to produce the same output.
"""
from __future__ import annotations

import os
import subprocess
from pathlib import Path

from pyspark.sql import SparkSession

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
CLEAN_DIR = DATA_DIR / "clean"
QUARANTINE_DIR = DATA_DIR / "quarantine"
CHECKPOINT_DIR = PROJECT_ROOT / "spark" / "checkpoints"

EVENTS_OUT = CLEAN_DIR / "events"
EVENTS_QUARANTINE = QUARANTINE_DIR / "events"
DIM_ITEMS_OUT = CLEAN_DIR / "dim_items"

DEFAULT_BROKERS = "localhost:19092"
DEFAULT_TOPIC = "retail.events"

# Spark 3.5 officially supports Java 8/11/17 only. This machine's default JDK
# is 26, which fails at startup on module-access restrictions, so we pin to an
# installed supported JDK rather than relying on whatever `java` resolves to.
SUPPORTED_JDK_VERSIONS = ("11", "17")

KAFKA_CONNECTOR = "org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.3"


def resolve_java_home() -> str:
    """Returns a JAVA_HOME Spark can actually run on.

    Honours an explicitly-set JAVA_HOME if it is already a supported version;
    otherwise asks macOS's java_home for one of the supported releases.
    """
    current = os.environ.get("JAVA_HOME")
    if current and _java_major(Path(current)) in SUPPORTED_JDK_VERSIONS:
        return current

    for version in SUPPORTED_JDK_VERSIONS:
        try:
            found = subprocess.run(
                ["/usr/libexec/java_home", "-v", version],
                capture_output=True,
                text=True,
                check=True,
            ).stdout.strip()
        except (subprocess.CalledProcessError, FileNotFoundError):
            continue
        if found:
            return found

    raise RuntimeError(
        "No supported JDK found. Spark 3.5 needs Java 11 or 17; install one "
        "(e.g. Amazon Corretto 11) or set JAVA_HOME to a supported JDK."
    )


def _java_major(java_home: Path) -> str | None:
    release = java_home / "release"
    if not release.exists():
        return None
    for line in release.read_text().splitlines():
        if line.startswith("JAVA_VERSION="):
            version = line.split("=", 1)[1].strip().strip('"')
            major = version.split(".")[0]
            return "8" if major == "1" else major
    return None


def build_spark(app_name: str, with_kafka: bool = False, shuffle_partitions: int = 8) -> SparkSession:
    """Builds a local SparkSession.

    shuffle_partitions defaults well below Spark's 200 -- on a single laptop
    core count, 200 shuffle tasks over this data volume is mostly scheduler
    overhead.
    """
    os.environ["JAVA_HOME"] = resolve_java_home()
    # Ensure the driver and executors run the same interpreter as the caller.
    os.environ.setdefault("PYSPARK_PYTHON", os.environ.get("PYSPARK_PYTHON", _current_python()))

    builder = (
        SparkSession.builder.appName(app_name)
        .master("local[*]")
        .config("spark.sql.session.timeZone", "UTC")
        .config("spark.sql.shuffle.partitions", str(shuffle_partitions))
        .config("spark.driver.memory", "4g")
        .config("spark.sql.adaptive.enabled", "true")
        .config("spark.sql.parquet.compression.codec", "snappy")
        # Keep the local run quiet enough that job output is readable.
        .config("spark.ui.showConsoleProgress", "false")
    )

    if with_kafka:
        builder = builder.config("spark.jars.packages", KAFKA_CONNECTOR)

    spark = builder.getOrCreate()
    spark.sparkContext.setLogLevel("WARN")
    return spark


def _current_python() -> str:
    import sys

    return sys.executable
