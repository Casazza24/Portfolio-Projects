"""Event cleaning logic, shared by the streaming job and the batch backfill.

Everything here is a pure DataFrame -> DataFrame function with no I/O and no
SparkSession of its own. That is what lets the streaming path and the batch
path be the same logic rather than two implementations that drift: the only
difference between them is how the DataFrame is obtained and where it is
written.

The cleaning stages, in order:
    1. parse       -- bytes -> JSON, tolerating garbage
    2. classify    -- tag every row valid/invalid with a specific reason
    3. cast_valid  -- string fields -> real types (safe, post-validation)
    4. dedupe      -- collapse at-least-once redeliveries on event_id
    5. enrich      -- broadcast join to the item/category dimension
"""
from __future__ import annotations

from pyspark.sql import Column, DataFrame
from pyspark.sql import functions as F
from pyspark.sql.types import StringType, StructField, StructType

# Every field is read as a string first, then cast explicitly. Declaring
# event_time as a LongType here would let Spark's PERMISSIVE parser null it
# out, making "the field was absent" and "the field held an ISO date string"
# indistinguishable -- and those are different data quality defects with
# different upstream owners.
RAW_EVENT_SCHEMA = StructType(
    [
        StructField("event_id", StringType(), True),
        StructField("event_time", StringType(), True),
        StructField("visitor_id", StringType(), True),
        StructField("event_type", StringType(), True),
        StructField("item_id", StringType(), True),
        StructField("transaction_id", StringType(), True),
        StructField("ingested_at", StringType(), True),
        StructField("producer_id", StringType(), True),
        # Written by the replay script purely so detection can be *scored*
        # after the fact. Deliberately not consulted by classify() -- the
        # cleaning logic has to stand on its own against a real stream that
        # would never carry such a marker.
        StructField("_fault", StringType(), True),
    ]
)

VALID_EVENT_TYPES = ("view", "addtocart", "transaction")

# Kimball "unknown member". Events whose item has no metadata resolve to this
# rather than to a null foreign key -- see enrich() for why.
UNKNOWN_KEY = -1

# RetailRocket covers May-Sep 2015. Anything outside a generous window around
# that is a broken clock, not a real event.
MIN_PLAUSIBLE_MS = 1_262_304_000_000  # 2010-01-01
MAX_PLAUSIBLE_MS = 1_893_456_000_000  # 2030-01-01


def parse(df: DataFrame) -> DataFrame:
    """Turns the Kafka `value` column into a parsed struct.

    Keeps the Kafka metadata (partition/offset) alongside -- quarantined rows
    are much easier to trace back to the broker with it.
    """
    return (
        df.select(
            F.col("key").cast("string").alias("kafka_key"),
            F.col("value").cast("string").alias("raw_value"),
            F.col("topic").alias("kafka_topic"),
            F.col("partition").alias("kafka_partition"),
            F.col("offset").alias("kafka_offset"),
            F.col("timestamp").alias("kafka_timestamp"),
        )
        .withColumn("e", F.from_json(F.col("raw_value"), RAW_EVENT_SCHEMA))
    )


def _numeric(col: Column) -> Column:
    """True when a string column holds an unsigned integer."""
    return col.rlike(r"^\d+$")


def classify(df: DataFrame) -> DataFrame:
    """Adds `dq_reason` (null when the row is good) and `is_valid`.

    Ordered most-fundamental-first so each row reports the root defect rather
    than a downstream symptom -- an unparseable payload is reported as such,
    not as five separate missing fields.
    """
    e = F.col("e")

    # from_json in PERMISSIVE mode does not reliably return a null struct for
    # malformed input -- for a payload truncated mid-write it returns a struct
    # with every field null instead. Checking only e.isNull() therefore reports
    # a truncated payload as `missing_event_id`, which points a reader at the
    # wrong upstream owner. An all-null struct from a non-empty payload is the
    # dependable signal that parsing failed.
    all_fields_null = (
        e["event_id"].isNull()
        & e["event_time"].isNull()
        & e["visitor_id"].isNull()
        & e["event_type"].isNull()
        & e["item_id"].isNull()
    )
    unparseable = e.isNull() | (all_fields_null & (F.length(F.trim(F.col("raw_value"))) > 0))

    reason = (
        F.when(unparseable, F.lit("unparseable_json"))
        .when(F.length(F.trim(F.col("raw_value"))) == 0, F.lit("empty_payload"))
        .when(e["event_id"].isNull(), F.lit("missing_event_id"))
        .when(e["event_time"].isNull(), F.lit("missing_event_time"))
        .when(~_numeric(e["event_time"]), F.lit("event_time_not_numeric"))
        .when(
            (e["event_time"].cast("long") < MIN_PLAUSIBLE_MS)
            | (e["event_time"].cast("long") > MAX_PLAUSIBLE_MS),
            F.lit("event_time_implausible"),
        )
        .when(e["event_type"].isNull(), F.lit("missing_event_type"))
        .when(~e["event_type"].isin(*VALID_EVENT_TYPES), F.lit("unknown_event_type"))
        .when(e["visitor_id"].isNull(), F.lit("missing_visitor_id"))
        .when(~_numeric(e["visitor_id"]), F.lit("visitor_id_not_numeric"))
        .when(e["item_id"].isNull(), F.lit("missing_item_id"))
        .when(~_numeric(e["item_id"]), F.lit("item_id_not_numeric"))
        .otherwise(F.lit(None).cast("string"))
    )

    return df.withColumn("dq_reason", reason).withColumn("is_valid", F.col("dq_reason").isNull())


def split(df: DataFrame) -> tuple[DataFrame, DataFrame]:
    """Splits a classified DataFrame into (valid, quarantined)."""
    return df.filter(F.col("is_valid")), quarantine_frame(df.filter(~F.col("is_valid")))


def quarantine_frame(df: DataFrame) -> DataFrame:
    """Shapes rejected rows for the dead-letter sink.

    The raw payload is kept verbatim: a quarantine record that has already
    been reshaped is useless for diagnosing why it was rejected, and useless
    for replaying once the upstream bug is fixed.
    """
    return df.select(
        F.col("dq_reason"),
        F.col("raw_value"),
        F.col("kafka_topic"),
        F.col("kafka_partition"),
        F.col("kafka_offset"),
        F.col("kafka_timestamp"),
        F.col("e")["_fault"].alias("injected_fault"),
        F.current_timestamp().alias("quarantined_at"),
    )


def cast_valid(df: DataFrame) -> DataFrame:
    """Projects validated rows onto their real types.

    Safe only after classify() has guaranteed the string forms are castable,
    which is why this is a separate stage.
    """
    e = F.col("e")
    return df.select(
        e["event_id"].alias("event_id"),
        e["event_time"].cast("long").alias("event_time_ms"),
        F.timestamp_millis(e["event_time"].cast("long")).alias("event_time"),
        e["visitor_id"].cast("long").alias("visitor_id"),
        e["event_type"].alias("event_type"),
        e["item_id"].cast("int").alias("item_id"),
        e["transaction_id"].cast("long").alias("transaction_id"),
        F.timestamp_millis(e["ingested_at"].cast("long")).alias("ingested_at"),
        e["producer_id"].alias("producer_id"),
        F.col("kafka_partition"),
        F.col("kafka_offset"),
    )


def dedupe(df: DataFrame, streaming: bool, watermark: str = "3 days") -> DataFrame:
    """Collapses at-least-once redeliveries, keyed on the producer's event_id.

    Batch can dedupe globally because it sees the whole dataset at once.
    Streaming cannot -- holding every event_id ever seen would grow state
    without bound. `dropDuplicatesWithinWatermark` bounds that state to the
    watermark interval: a redelivery arriving more than `watermark` after the
    original's event time will slip through as a duplicate.

    That is a real, deliberate trade-off (bounded memory vs perfect dedupe),
    and the watermark is sized from the producer's observed lateness -- see
    docs/data_quality.md.
    """
    if not streaming:
        return df.dropDuplicates(["event_id"])

    return df.withWatermark("event_time", watermark).dropDuplicatesWithinWatermark(["event_id"])


def enrich(df: DataFrame, dim_items: DataFrame) -> DataFrame:
    """Left-joins the item/category dimension and adds partition columns.

    Broadcast because the dimension is ~417k rows (a few MB) -- and because a
    stream-static join has to broadcast anyway; a shuffle join against a
    streaming side is not something Spark will plan.

    Left, not inner: an event referencing an item with no metadata is still a
    real event and must not be silently dropped by the join.

    Unmatched rows resolve to the UNKNOWN_KEY sentinel rather than to null.
    RetailRocket's item_properties feed simply never emitted a categoryid for
    some items, so there is no true value to recover -- imputing a plausible
    category would move real revenue into categories those items were never
    in. The sentinel keeps the fact, keeps the key resolvable (so the Phase 4
    `relationships` test can run strict instead of tolerating nulls), and
    keeps the gap visible in BI, where a null would instead be dropped from
    slicers and quietly break totals.

    `has_item_metadata` preserves the distinction so an analyst can exclude
    unknowns as a deliberate choice rather than having that choice baked in
    here irreversibly.
    """
    # A literal on the dimension side is a more reliable match indicator than
    # checking category_id for null -- it stays correct even if a matched row
    # legitimately holds a null category.
    dim = dim_items.withColumn("_dim_matched", F.lit(True))

    return (
        df.join(F.broadcast(dim), on="item_id", how="left")
        .withColumn("has_item_metadata", F.coalesce(F.col("_dim_matched"), F.lit(False)))
        .withColumn("category_id", F.coalesce(F.col("category_id"), F.lit(UNKNOWN_KEY)))
        .withColumn(
            "parent_category_id", F.coalesce(F.col("parent_category_id"), F.lit(UNKNOWN_KEY))
        )
        .withColumn(
            "root_category_id", F.coalesce(F.col("root_category_id"), F.lit(UNKNOWN_KEY))
        )
        .withColumn("event_date", F.to_date(F.col("event_time")))
        .select(
            "event_id",
            "event_time",
            "event_time_ms",
            "event_date",
            "visitor_id",
            "event_type",
            "item_id",
            "transaction_id",
            "category_id",
            "parent_category_id",
            "root_category_id",
            "is_available",
            "has_item_metadata",
            "producer_id",
            "ingested_at",
            "kafka_partition",
            "kafka_offset",
        )
    )


def clean_events(df: DataFrame, dim_items: DataFrame, streaming: bool, watermark: str = "3 days"):
    """The whole pipeline. Returns (clean, quarantined)."""
    classified = classify(parse(df))
    valid, quarantined = split(classified)
    clean = enrich(dedupe(cast_valid(valid), streaming, watermark), dim_items)
    return clean, quarantined
