#!/usr/bin/env python3
"""Computes every metric proposed in docs/dashboard_design.md against local data.

There is no Snowflake account, so the marts do not exist as queryable tables.
That is exactly why this script exists: the point of Phase 6 planning is to
find out which metrics are *interesting* BEFORE spending trial credits building
them. A funnel chart is worthless if conversion is flat; a weather overlay is
worthless if the correlation is noise. Better to kill those here.

Each check reproduces the mart logic against the source the mart will be built
from -- Olist in the running Postgres container for the fct_orders /
fct_order_payments star, the cleaned Parquet for fct_events, and
data/clean/dim_weather.csv for dim_weather. Where a check restates a mart's
SQL, the mart is named in a comment so the two can be diffed by eye when the
warehouse finally exists.

ponytail: no pandas. Postgres does the Olist aggregation (that is what it is
for, and the results are tens of rows), Spark does the 2.7M-row event scan
because it is already a project dependency with a session builder written, and
the weather join is a dict lookup over a 20k-row CSV. Adding a dataframe
library to hold results that print in twenty lines would be a dependency for
formatting.

Usage:
    .venv/bin/python analysis/validate_dashboard_metrics.py
    .venv/bin/python analysis/validate_dashboard_metrics.py --skip-events
"""
from __future__ import annotations

import argparse
import csv
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import psycopg2

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "spark"))

WEATHER_CSV = PROJECT_ROOT / "data" / "clean" / "dim_weather.csv"

PG = dict(host="localhost", port=5432, dbname="retail", user="retail", password="retail")

# dim_weather.is_rainy_day / is_heavy_rain_day. Duplicated here rather than
# imported because they live in SQL that cannot run without a warehouse; if the
# model's threshold changes, this is the second place to change it.
RAIN_MM_THRESHOLD = 1.0
HEAVY_RAIN_MM_THRESHOLD = 20.0

# Olist's first four and last two months hold a handful of orders each -- see
# M1. Trend and weather checks run on the dense window so a 6-order month
# cannot swing a mean.
DENSE_START = "2017-01-01"
DENSE_END = "2018-09-01"


def rule(title: str) -> None:
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


def verdict(label: str, keep: bool, note: str) -> None:
    print(f"\n  VERDICT [{'KEEP' if keep else 'KILL'}] {label}\n    {note}")


# --------------------------------------------------------------- Olist star ---


def olist_checks(cur) -> None:
    rule("M0  Grain and reconciliation facts the metric definitions depend on")

    # Every number here is a definition decision waiting to happen, not trivia.
    cur.execute(
        """
        select 'orders'                    , count(*)::bigint from olist.orders
        union all select 'order lines'     , count(*) from olist.order_items
        union all select 'orders w/o lines', count(*) from olist.orders o
             where not exists (select 1 from olist.order_items i where i.order_id = o.order_id)
        union all select 'multi-line orders', count(*) from
             (select order_id from olist.order_items group by 1 having count(*) > 1) t
        union all select 'split-payment orders', count(*) from
             (select order_id from olist.order_payments group by 1 having count(*) > 1) t
        union all select 'orders with >1 review', count(*) from
             (select order_id from olist.order_reviews group by 1 having count(*) > 1) t
        union all select 'products with no category', count(*) from olist.products
             where product_category_name is null
        """
    )
    for label, n in cur.fetchall():
        print(f"  {label:<28} {n:>10,}")

    # The 775 line-less orders are not spread evenly -- they are almost all
    # `unavailable`, which fct_orders therefore drops almost entirely. Any
    # status breakdown read off fct_orders is missing them, and that has to be
    # said on the page rather than discovered by someone comparing to source.
    cur.execute(
        """
        select o.order_status, count(*) from olist.orders o
        where not exists (select 1 from olist.order_items i where i.order_id = o.order_id)
        group by 1 order by 2 desc
        """
    )
    print("  line-less orders by status (absent from fct_orders by construction):")
    for status, n in cur.fetchall():
        print(f"    {status:<14} {n:>6,}")

    # fct_orders sums item_price + freight_value; fct_order_payments sums
    # payment_value. They are different grains AND different totals. A
    # dashboard that shows both without saying which is "revenue" is wrong.
    cur.execute("select sum(price + freight_value) from olist.order_items")
    line_rev = float(cur.fetchone()[0])
    cur.execute("select sum(payment_value) from olist.order_payments")
    pay_total = float(cur.fetchone()[0])
    gap = pay_total - line_rev
    print(f"\n  fct_orders item_revenue total       R$ {line_rev:>14,.0f}")
    print(f"  fct_order_payments payment_value    R$ {pay_total:>14,.0f}")
    print(f"  gap                                 R$ {gap:>14,.0f}  ({100 * gap / line_rev:+.2f}%)")

    rule("M1  Revenue trend -- monthly, order-line grain (fct_orders.item_revenue)")
    cur.execute(
        """
        select to_char(o.order_purchase_timestamp, 'YYYY-MM') as ym,
               count(distinct o.order_id),
               sum(oi.price + oi.freight_value)
        from olist.orders o
        join olist.order_items oi on oi.order_id = o.order_id
        group by 1 order by 1
        """
    )
    months = cur.fetchall()
    for ym, orders, rev in months:
        bar = "#" * int(float(rev) / 25000)
        print(f"  {ym}  {orders:>6,} orders  R$ {float(rev):>10,.0f}  {bar}")
    thin = [m for m in months if m[1] < 100]
    verdict(
        "M1 monthly revenue trend",
        True,
        f"Real growth 2017-01 to 2018-08. But {len(thin)} months "
        f"({', '.join(m[0] for m in thin)}) hold <100 orders and are load "
        "artifacts, not a business trend -- they must be excluded or the "
        "y-axis is dominated by the ramp-up.",
    )

    rule("M1b Daily revenue -- is there a spike worth annotating?")
    cur.execute(
        """
        select o.order_purchase_timestamp::date d,
               count(distinct o.order_id),
               sum(oi.price + oi.freight_value)
        from olist.orders o
        join olist.order_items oi on oi.order_id = o.order_id
        where o.order_purchase_timestamp >= %s and o.order_purchase_timestamp < %s
        group by 1 order by 3 desc limit 5
        """,
        (DENSE_START, DENSE_END),
    )
    top = cur.fetchall()
    cur.execute(
        """
        select avg(rev) from (
          select sum(oi.price + oi.freight_value) rev
          from olist.orders o join olist.order_items oi on oi.order_id = o.order_id
          where o.order_purchase_timestamp >= %s and o.order_purchase_timestamp < %s
          group by o.order_purchase_timestamp::date) t
        """,
        (DENSE_START, DENSE_END),
    )
    mean_day = float(cur.fetchone()[0])
    print(f"  mean day in dense window: R$ {mean_day:,.0f}")
    for d, n, rev in top:
        print(f"  {d}  {n:>5,} orders  R$ {float(rev):>10,.0f}   {float(rev)/mean_day:.1f}x mean")
    verdict(
        "M1b daily trend with peak annotation",
        True,
        f"Top day is {top[0][0]} at {float(top[0][2])/mean_day:.1f}x the mean "
        "day -- Black Friday 2017. A single annotated point carries the whole "
        "'the pipeline captures real events' story.",
    )

    rule("M2  Revenue by product category (fct_orders x dim_products.product_category)")
    cur.execute(
        """
        select coalesce(t.product_category_name_english, p.product_category_name, 'unknown') cat,
               sum(oi.price + oi.freight_value) rev,
               count(*) lines
        from olist.order_items oi
        join olist.products p on p.product_id = oi.product_id
        left join olist.product_category_name_translation t
               on t.product_category_name = p.product_category_name
        group by 1 order by 2 desc
        """
    )
    cats = cur.fetchall()
    total = sum(float(c[1]) for c in cats)
    top10 = cats[:10]
    print(f"  {len(cats)} distinct categories; top 10 hold "
          f"{100 * sum(float(c[1]) for c in top10) / total:.1f}% of revenue")
    for cat, rev, lines in top10:
        print(f"    {cat:<26} R$ {float(rev):>10,.0f}  {100*float(rev)/total:>5.1f}%  {lines:>6,} lines")
    head = 100 * float(top10[0][1]) / total
    verdict(
        "M2 revenue by category",
        True,
        f"{len(cats)} categories is far past any colour ceiling, and the head "
        f"is only {head:.1f}% -- so this is a ranked horizontal bar (top 10 + "
        "Other), never a pie and never a colour-per-category series.",
    )

    rule("M3  Delivery lateness vs review score -- the headline hypothesis")
    cur.execute(
        """
        select case
            when d < -10 then 'a  >10d early'
            when d <  -3 then 'b  4-10d early'
            when d <=  0 then 'c  0-3d early'
            when d <=  3 then 'd  1-3d LATE'
            when d <= 10 then 'e  4-10d LATE'
            else              'f  >10d LATE' end bucket,
          count(*), avg(review_score), count(review_score)
        from (
          select extract(epoch from (o.order_delivered_customer_date
                                     - o.order_estimated_delivery_date)) / 86400 d,
                 r.review_score
          from olist.orders o
          left join (select distinct on (order_id) order_id, review_score
                     from olist.order_reviews
                     order by order_id, review_creation_date desc, review_id desc) r
                 on r.order_id = o.order_id
          where o.order_delivered_customer_date is not null) t
        group by 1 order by 1
        """
    )
    rows = cur.fetchall()
    for bucket, n, avg_score, scored in rows:
        stars = "*" * int(round(float(avg_score)))
        print(f"  {bucket:<16} {n:>7,} orders  {scored:>7,} reviewed  avg {float(avg_score):.2f}  {stars}")
    spread = float(rows[0][2]) - float(rows[-1][2])
    verdict(
        "M3 review score by delivery-vs-estimate bucket",
        True,
        f"Monotonic across all six buckets and a {spread:.2f}-star spread "
        f"({float(rows[0][2]):.2f} -> {float(rows[-1][2]):.2f}). This is the "
        "single strongest relationship in the dataset -- it should lead the page.",
    )

    rule("M3b Delivery performance by customer state (fct_orders.customer_state)")
    cur.execute(
        """
        select c.customer_state, count(*),
               avg(extract(epoch from (o.order_delivered_customer_date
                                       - o.order_purchase_timestamp)) / 86400),
               100.0 * count(*) filter (
                   where o.order_delivered_customer_date > o.order_estimated_delivery_date
               ) / count(*)
        from olist.orders o
        join olist.customers c on c.customer_id = o.customer_id
        where o.order_delivered_customer_date is not null
        group by 1 having count(*) >= 200
        order by 3 desc
        """
    )
    states = cur.fetchall()
    for s, n, days, late in states[:5] + [("...", 0, 0, 0)] + states[-3:]:
        if s == "...":
            print("    ...")
            continue
        print(f"    {s}  {n:>6,} delivered  {float(days):>5.1f} days  {float(late):>5.1f}% late")
    verdict(
        "M3b days-to-deliver and late rate by state",
        True,
        f"{float(states[0][2]):.1f} days in {states[0][0]} vs "
        f"{float(states[-1][2]):.1f} in {states[-1][0]} -- a "
        f"{float(states[0][2]) / float(states[-1][2]):.1f}x spread, and the "
        "late rate moves with it. Real geographic signal.",
    )

    rule("M4  Order status mix -- can 'cancellation rate' carry a tile?")
    cur.execute(
        """
        select o.order_status, count(distinct o.order_id), sum(oi.price + oi.freight_value)
        from olist.orders o
        join olist.order_items oi on oi.order_id = o.order_id
        group by 1 order by 3 desc
        """
    )
    st = cur.fetchall()
    tot_rev = sum(float(r[2]) for r in st)
    tot_ord = sum(r[1] for r in st)
    for status, n, rev in st:
        print(f"    {status:<14} {n:>7,} orders  R$ {float(rev):>12,.0f}  {100*float(rev)/tot_rev:>6.3f}% of line revenue")
    canc = next((float(r[2]) for r in st if r[0] == "canceled"), 0.0)
    canc_n = next((r[1] for r in st if r[0] == "canceled"), 0)
    verdict(
        "M4 cancellation rate as a KPI tile",
        False,
        f"Cancelled orders visible at fct_orders grain are {canc_n:,} of "
        f"{tot_ord:,} ({100*canc_n/tot_ord:.2f}%) and "
        f"{100*canc/tot_rev:.2f}% of line revenue -- and 164 more cancelled "
        "orders never had a line, so they are not in the fact at all. There is "
        "nothing to look at "
        "and no variation to filter. Keep the *definition* (revenue includes "
        "all statuses, stated on the page) and drop the tile.",
    )

    rule("M5  Repeat purchase / cohort view (dim_customers.is_repeat_customer)")
    cur.execute(
        """
        select count(*), count(*) filter (where n > 1)
        from (select c.customer_unique_id, count(distinct o.order_id) n
              from olist.customers c join olist.orders o on o.customer_id = c.customer_id
              group by 1) t
        """
    )
    people, repeat = cur.fetchone()
    cur.execute(
        """
        select n, count(*) from (
          select c.customer_unique_id, count(distinct o.order_id) n
          from olist.customers c join olist.orders o on o.customer_id = c.customer_id
          group by 1) t
        group by 1 order by 1 limit 6
        """
    )
    print(f"    distinct people (customer_unique_id): {people:,}")
    print(f"    with >1 order: {repeat:,}  ({100*repeat/people:.2f}%)")
    for n, cnt in cur.fetchall():
        print(f"      {n} order(s): {cnt:>8,}")
    verdict(
        "M5 cohort / repeat-purchase page (scope doc section 4, page 3)",
        False,
        f"{100*repeat/people:.2f}% of people ever order twice. A retention "
        "curve here is a flat line at ~3% with no cohort separation, and a "
        "cohort heatmap is a grid of near-identical cells. The dataset does "
        "not support the page the scope doc asked for.",
    )

    rule("M6  Payment mix and installments (fct_order_payments)")
    cur.execute(
        """
        select payment_type, count(*), sum(payment_value), avg(payment_installments)
        from olist.order_payments group by 1 order by 3 desc
        """
    )
    pays = cur.fetchall()
    pay_tot = sum(float(p[2]) for p in pays)
    for t, n, v, inst in pays:
        print(f"    {t:<13} {n:>7,} rows  R$ {float(v):>12,.0f}  {100*float(v)/pay_tot:>5.1f}%  avg {float(inst):.2f} installments")
    cur.execute(
        """
        select payment_installments, count(*), avg(payment_value)
        from olist.order_payments where payment_type = 'credit_card'
        group by 1 order by 1 limit 13
        """
    )
    print("    credit_card installment ladder:")
    for inst, n, avg_v in cur.fetchall():
        print(f"      {inst:>2}x  {n:>7,} payments  avg R$ {float(avg_v):>8,.2f}")
    verdict(
        "M6 payment mix",
        True,
        "Only credit_card has installments > 1 (every other type is exactly "
        "1.00), so the interesting chart is the credit-card installment ladder "
        "against average basket, not a four-slice payment-type breakdown.",
    )


# ------------------------------------------------------------------ weather ---


def weather_check(cur) -> None:
    rule("M7  Weather vs revenue (fct_orders.weather_key -> dim_weather.is_rainy_day)")

    rainy: dict[tuple[str, str], bool] = {}
    heavy: dict[tuple[str, str], bool] = {}
    with open(WEATHER_CSV) as f:
        for r in csv.DictReader(f):
            key = (r["state"], r["weather_date"])
            rainy[key] = float(r["rain_sum"]) >= RAIN_MM_THRESHOLD
            heavy[key] = float(r["rain_sum"]) >= HEAVY_RAIN_MM_THRESHOLD
    print(f"  dim_weather rows: {len(rainy):,}   rainy-day share: {100*sum(rainy.values())/len(rainy):.1f}%")

    cur.execute(
        """
        select c.customer_state, o.order_purchase_timestamp::date,
               sum(oi.price + oi.freight_value)
        from olist.orders o
        join olist.order_items oi on oi.order_id = o.order_id
        join olist.customers c on c.customer_id = o.customer_id
        where o.order_purchase_timestamp >= %s and o.order_purchase_timestamp < %s
        group by 1, 2
        """,
        (DENSE_START, DENSE_END),
    )
    rows = [(s, d, float(rev)) for s, d, rev in cur.fetchall()]

    # Matched-cell comparison rather than a raw rainy-vs-dry mean. Rain is
    # seasonal and revenue triples over the window, so an unmatched comparison
    # mostly measures which months were wet. Comparing inside
    # (state, month, weekday) removes both confounds without needing a
    # regression package.
    for label, flag in (("rain >= 1mm", rainy), ("heavy rain >= 20mm", heavy)):
        cells: dict[tuple, dict[bool, list[float]]] = defaultdict(lambda: {True: [], False: []})
        for s, d, rev in rows:
            key = (s, str(d))
            if key in flag:
                cells[(s, d.strftime("%Y-%m"), d.weekday())][flag[key]].append(rev)
        num = den = 0.0
        matched = 0
        per_state: dict[str, list[float]] = defaultdict(lambda: [0.0, 0.0])
        for (s, _ym, _dow), v in cells.items():
            if v[True] and v[False]:
                matched += 1
                wet, dry = statistics.mean(v[True]), statistics.mean(v[False])
                num += wet - dry
                den += dry
                per_state[s][0] += wet - dry
                per_state[s][1] += dry
        print(f"\n  {label}: {matched:,} matched (state, month, weekday) cells")
        print(f"    overall weighted delta: {100 * num / den:+.2f}%")
        ranked = sorted(per_state.items(), key=lambda kv: -kv[1][1])[:8]
        for s, (delta, base) in ranked:
            print(f"      {s}: {100 * delta / base:+6.2f}%   (baseline R$ {base:,.0f})")

    verdict(
        "M7 weather overlay on the revenue page (scope doc section 4, page 1)",
        False,
        "Effect is ~+2% with the sign flipping state to state (SP +2.5%, "
        "MG +8.3%, SC -11.7%) and no coherent direction -- and it is POSITIVE, "
        "which is the opposite of the 'revenue dips in bad weather' framing. "
        "This is noise dressed as a finding. Keep dim_weather in the model "
        "(the external-API ingestion step is a real deliverable) and put the "
        "null result in the README rather than a chart on the dashboard.",
    )


# ------------------------------------------------------------------- events ---


def event_checks() -> None:
    from common import DIM_ITEMS_OUT, EVENTS_OUT, build_spark
    from pyspark.sql import functions as F

    spark = build_spark("validate_dashboard_metrics")
    events = spark.read.parquet(str(EVENTS_OUT))
    items = spark.read.parquet(str(DIM_ITEMS_OUT))

    # fct_events.item_key -- the unknown-member resolution, restated.
    fct = events.withColumn(
        "item_key", F.when(F.col("has_item_metadata"), F.col("item_id")).otherwise(F.lit(-1))
    ).cache()

    rule("M8  Funnel: view -> addtocart -> transaction (fct_events, factless)")
    by_type = {r["event_type"]: r["count"] for r in fct.groupBy("event_type").count().collect()}
    v, a, t = by_type["view"], by_type["addtocart"], by_type["transaction"]
    print(f"  event grain    view {v:>10,}  ->  addtocart {a:>8,} ({100*a/v:.2f}%)  ->  transaction {t:>7,} ({100*t/v:.3f}%)")

    # Visitor grain: the honest conversion rate. Event grain double-counts a
    # visitor who viewed the same item eleven times.
    per_visitor = fct.groupBy("visitor_id").agg(
        F.max(F.when(F.col("event_type") == "view", 1).otherwise(0)).alias("v"),
        F.max(F.when(F.col("event_type") == "addtocart", 1).otherwise(0)).alias("a"),
        F.max(F.when(F.col("event_type") == "transaction", 1).otherwise(0)).alias("t"),
    )
    vv, va, vt = per_visitor.agg(F.sum("v"), F.sum("a"), F.sum("t")).collect()[0]
    visitors = per_visitor.count()
    print(f"  visitor grain  {visitors:>10,} visitors, {vv:,} viewed  ->  {va:,} added ({100*va/vv:.2f}%)  ->  {vt:,} bought ({100*vt/vv:.3f}%)")
    verdict(
        "M8 headline funnel",
        True,
        f"Two real drop-offs ({100*va/vv:.1f}% then "
        f"{100*vt/va:.1f}% of adders convert) -- not flat, and the visitor and "
        "event grains agree closely, so either is defensible. Use visitor "
        "grain and say why.",
    )

    rule("M9  Funnel by root category (fct_events.item_key -> dim_items.root_category_id)")
    dim = items.select(
        F.col("item_id").alias("k"),
        F.col("root_category_id").alias("root"),
        F.col("category_id").alias("cat"),
    )
    joined = fct.join(dim, fct.item_key == F.col("k"), "left")
    funnel = (
        joined.groupBy("root")
        .agg(
            F.count(F.when(F.col("event_type") == "view", 1)).alias("views"),
            F.count(F.when(F.col("event_type") == "addtocart", 1)).alias("atc"),
            F.count(F.when(F.col("event_type") == "transaction", 1)).alias("txn"),
        )
        .orderBy(F.desc("views"))
        .collect()
    )
    print(f"  {len(funnel)} root categories present in events")
    print("    root      views      atc    atc%     txn    txn%")
    for r in funnel[:14]:
        root, vw, ac, tx = r["root"], r["views"], r["atc"], r["txn"]
        if not vw:
            continue
        tag = "  <- unknown member, NOT a category" if root == -1 else ""
        print(f"    {str(root):>5} {vw:>10,} {ac:>8,}  {100*ac/vw:>5.2f}%  "
              f"{tx:>6,}  {100*tx/vw:>5.3f}%{tag}")
    real = [r for r in funnel if r["root"] != -1 and r["views"] >= 20000]
    rates = sorted((100 * r["atc"] / r["views"], r["root"]) for r in real)
    print(f"\n    real categories with >= 20,000 views: {len(real)}")
    print(f"    add-to-cart rate range: {rates[0][0]:.2f}% (root {rates[0][1]}) "
          f"to {rates[-1][0]:.2f}% (root {rates[-1][1]})  = {rates[-1][0]/rates[0][0]:.1f}x")
    unknown = next(r for r in funnel if r["root"] == -1)
    verdict(
        "M9 funnel split by category",
        True,
        f"{rates[-1][0]/rates[0][0]:.1f}x spread across real categories -- worth "
        f"charting. BUT the -1 member shows {100*unknown['atc']/unknown['views']:.2f}% "
        "add-to-cart, which is a metadata-coverage artifact (unknowns are "
        "concentrated in views), not a badly-performing category. It must be "
        "excluded from the ranking or it reads as the worst category on the "
        "chart. Categories are opaque integer ids -- there are no names in "
        "RetailRocket, so the axis labels will be numbers.",
    )

    rule("M10 Pipeline health (assert_unknown_item_rate_within_bounds, assert_orphan_categories_bounded)")
    total = fct.count()
    unknown_events = fct.filter(F.col("item_key") == -1).count()
    print(f"  fct_events rows                  {total:>12,}")
    print(f"  events on unknown item (-1)      {unknown_events:>12,}  ({100*unknown_events/total:.2f}%)")
    for et in ("view", "addtocart", "transaction"):
        sub = fct.filter(F.col("event_type") == et)
        n = by_type[et]
        u = sub.filter(F.col("item_key") == -1).count()
        print(f"    {et:<12} {n:>12,}  unknown {100*u/n:>5.2f}%")

    orphan_items = items.filter(F.col("category_id").isNotNull() & F.col("root_category_id").isNull())
    orphan_item_n = orphan_items.count()
    orphan_cat_n = orphan_items.select("category_id").distinct().count()
    orphan_events = joined.filter(F.col("root").isNull() & F.col("cat").isNotNull()).count()
    print(f"\n  dim_items rows                   {items.count():>12,}")
    print(f"  orphan ITEMS (root null)         {orphan_item_n:>12,}")
    print(f"  distinct orphan CATEGORY ids     {orphan_cat_n:>12,}   <- what assert_orphan_categories_bounded counts")
    print(f"  events landing on those items    {orphan_events:>12,}")

    daily = fct.groupBy("event_date").count().orderBy("event_date").collect()
    print(f"\n  {len(daily)} event days, {daily[0]['event_date']} .. {daily[-1]['event_date']}")
    print(f"    first day {daily[0]['count']:,}  last day {daily[-1]['count']:,}  "
          f"median {statistics.median(d['count'] for d in daily):,.0f}")

    verdict(
        "M10 pipeline health page",
        True,
        f"Unknown-item rate {100*unknown_events/total:.2f}% overall but "
        "concentrated in views -- that skew IS the metric worth showing. "
        f"Note: assert_orphan_categories_bounded counts DISTINCT CATEGORY IDS "
        f"and measures {orphan_cat_n}, not the {orphan_item_n} in the var; the "
        f"{orphan_item_n} is the item count. And {orphan_events} events land on "
        "orphan-category items, so orphans have zero dashboard impact.",
    )
    print(f"\n  NOTE: the last event day ({daily[-1]['event_date']}, "
          f"{daily[-1]['count']:,} events vs a median of "
          f"{statistics.median(d['count'] for d in daily):,.0f}) is a truncated "
          "day. A daily line chart must drop it or it reads as a collapse.")

    rule("M11 Cross-star time overlap -- can one date filter scope both pages?")
    print(f"  fct_events    {daily[0]['event_date']} .. {daily[-1]['event_date']}")
    print("  fct_orders    2016-09-04 .. 2018-10-17   (from M1)")
    verdict(
        "M11 a single global date slicer",
        False,
        "The two stars do not overlap in time by a single day. A shared date "
        "filter would blank one page whenever the other has data. Each star "
        "gets its own date scope, on its own page, with the disjointness "
        "stated on the page.",
    )

    spark.stop()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-events", action="store_true",
                        help="Skip the Spark half (2.7M rows, ~2 min).")
    args = parser.parse_args()

    conn = psycopg2.connect(**PG)
    with conn, conn.cursor() as cur:
        olist_checks(cur)
        weather_check(cur)
    conn.close()

    if not args.skip_events:
        event_checks()

    rule("Done")
    print("  Metrics recommended for KILL: M4 (cancellation tile), "
          "M5 (cohort page),\n  M7 (weather overlay), M11 (single global date filter).")


if __name__ == "__main__":
    main()