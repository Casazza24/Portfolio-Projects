#!/usr/bin/env python3
"""Builds the item/category dimension the event stream joins against.

RetailRocket ships item metadata as an EAV change-log (~20M rows across two
files): one row per (timestamp, itemid, property, value). Most property names
are hashed integers; the two that are readable and useful are `categoryid` and
`available`.

This flattens that change-log into a current-state dimension, and resolves the
category tree into parent/root levels so the Phase 4 marts can roll up.

Design note -- point-in-time vs current-state: joining each event against the
category that was in effect *at that event's timestamp* would be the strictly
correct SCD-2 treatment. This builds a current-state (latest known) dimension
instead, because it can then be broadcast into a streaming join, whereas a
range join against a versioned dimension cannot. The cost is that events
before a re-categorisation get the item's newer category; ~2.8% of items ever
change category, so the distortion is small and bounded. Called out in
docs/data_quality.md.

Usage:
    python3 spark/build_item_metadata.py
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

from pyspark.sql import Window
from pyspark.sql import functions as F
from pyspark.sql.types import IntegerType, LongType, StringType, StructField, StructType

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import DATA_DIR, DIM_ITEMS_OUT, build_spark  # noqa: E402
from transforms import UNKNOWN_KEY  # noqa: E402

RAW_RR = DATA_DIR / "raw" / "retailrocket"

PROPERTIES_SCHEMA = StructType(
    [
        StructField("timestamp", LongType(), True),
        StructField("itemid", IntegerType(), True),
        StructField("property", StringType(), True),
        StructField("value", StringType(), True),
    ]
)

MAX_TREE_DEPTH = 20


def load_category_tree(path: Path) -> dict[int, int | None]:
    """category_tree.csv is ~1.7k rows -- small enough to resolve on the driver
    rather than paying for an iterative Spark self-join."""
    parents: dict[int, int | None] = {}
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            category = int(row["categoryid"])
            parent = row["parentid"].strip()
            parents[category] = int(float(parent)) if parent else None
    return parents


def resolve_root(category: int, parents: dict[int, int | None]) -> int:
    """Walks to the top of the tree, with a depth cap so a cycle in the source
    data degrades to 'stop here' instead of hanging the job."""
    seen = set()
    current = category
    for _ in range(MAX_TREE_DEPTH):
        if current in seen:
            break  # cycle
        seen.add(current)
        parent = parents.get(current)
        if parent is None:
            break
        current = parent
    return current


def main():
    spark = build_spark("build_item_metadata")

    property_files = sorted(RAW_RR.glob("item_properties_part*.csv"))
    if not property_files:
        raise SystemExit(f"no item_properties_part*.csv found in {RAW_RR}")

    props = (
        spark.read.csv(
            [str(p) for p in property_files], header=True, schema=PROPERTIES_SCHEMA
        )
        .filter(F.col("property").isin("categoryid", "available"))
        .filter(F.col("itemid").isNotNull())
    )

    # The change-log holds every historical value; keep only the most recent
    # per (item, property). ties broken by value for determinism -- without a
    # tiebreak, two snapshots at the same millisecond would make the job
    # non-reproducible.
    latest = Window.partitionBy("itemid", "property").orderBy(
        F.col("timestamp").desc(), F.col("value").desc()
    )
    current = (
        props.withColumn("_rn", F.row_number().over(latest))
        .filter(F.col("_rn") == 1)
        .drop("_rn")
    )

    wide = (
        current.groupBy("itemid")
        .pivot("property", ["categoryid", "available"])
        .agg(F.first("value"))
        .withColumnRenamed("itemid", "item_id")
        # `value` is a free-text column in the source; anything non-numeric is
        # unusable as a key, and cast() nulls it rather than failing the job.
        .withColumn("category_id", F.col("categoryid").cast(IntegerType()))
        .withColumn("is_available", F.col("available").cast(IntegerType()) == 1)
        .drop("categoryid", "available")
    )

    # Also record when the current category took effect -- Phase 4 can use it
    # to reason about how stale a given item's category is.
    category_since = (
        current.filter(F.col("property") == "categoryid")
        .select(F.col("itemid").alias("item_id"), F.col("timestamp").alias("category_valid_from"))
    )
    wide = wide.join(category_since, on="item_id", how="left")

    parents = load_category_tree(RAW_RR / "category_tree.csv")
    tree_rows = [
        (category, parents.get(category), resolve_root(category, parents)) for category in parents
    ]
    tree = spark.createDataFrame(
        tree_rows,
        schema=StructType(
            [
                StructField("category_id", IntegerType(), True),
                StructField("parent_category_id", IntegerType(), True),
                StructField("root_category_id", IntegerType(), True),
            ]
        ),
    )

    dim = (
        wide.join(F.broadcast(tree), on="category_id", how="left")
        .select(
            "item_id",
            "category_id",
            "parent_category_id",
            "root_category_id",
            "is_available",
            F.col("category_valid_from").cast(LongType()),
        )
    )

    # Unknown member. Events referencing an item with no metadata keep their
    # real item_id but resolve their category keys to UNKNOWN_KEY, so this row
    # is what makes those keys point at something real. Without it, a strict
    # `relationships` test in Phase 4 would fail on every unknown-item event.
    unknown = spark.createDataFrame(
        [(UNKNOWN_KEY, UNKNOWN_KEY, UNKNOWN_KEY, UNKNOWN_KEY, None, None)],
        schema=dim.schema,
    )
    dim = dim.unionByName(unknown).orderBy("item_id")

    DIM_ITEMS_OUT.parent.mkdir(parents=True, exist_ok=True)
    dim.write.mode("overwrite").parquet(str(DIM_ITEMS_OUT))

    total = dim.count()
    with_category = dim.filter(F.col("category_id").isNotNull()).count()
    orphan = dim.filter(
        F.col("category_id").isNotNull() & F.col("root_category_id").isNull()
    ).count()

    print(f"\nwrote {DIM_ITEMS_OUT}")
    print(f"  items                     : {total:,}")
    print(f"  with a category           : {with_category:,} ({with_category / total:.1%})")
    print(f"  category not in tree file : {orphan:,}")
    print(f"  categories in tree        : {len(parents):,}")

    spark.stop()


if __name__ == "__main__":
    main()
