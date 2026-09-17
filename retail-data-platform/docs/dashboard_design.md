# Dashboard design (Phase 6)

Design specification for the BI layer. **No dashboard is built yet, and the BI
tool is not chosen.** Everything here is expressed as metric definitions,
grains, chart forms and layout so it can be implemented unchanged in Power BI,
Tableau Public, Metabase or Streamlit. Where a choice genuinely differs by
tool, it is marked **[branch]**.

Every number quoted below was computed against the local source data by
`analysis/validate_dashboard_metrics.py` on 2026-08-11. Re-run it to
regenerate them:

```bash
.venv/bin/python analysis/validate_dashboard_metrics.py
.venv/bin/python analysis/validate_dashboard_metrics.py --skip-events   # Olist half only, ~5s
```

---

## 0. Constraints this design is built around

**The BI tool is undecided.** The scope doc locked Power BI, but Power BI
Desktop is Windows-only and this is a macOS build. Live options:

| Option | Cost | Public link | Note |
|---|---|---|---|
| Power BI in a Windows VM | VM licence/setup | no (screen recording only) | keeps the scope doc's resume line |
| Tableau Public | free | **yes, cold public URL** | native macOS; workbook is public by definition |
| Metabase (local Docker) | free | no | fastest to stand up; weakest resume signal |
| Streamlit | free | yes (Community Cloud) | most control, but it stops being a "BI tool" line on a CV |

Nothing below depends on which one wins. **This decision needs Matt's
sign-off** — see §7.

**There is no Snowflake account**, so the marts do not exist as queryable
tables. Everything here was validated against the sources the marts are built
from: Olist in the running `retail_platform_postgres` container, the cleaned
event Parquet in `data/clean/events/`, `data/clean/dim_items/`, and
`data/clean/dim_weather.csv`.

**The two stars do not overlap in time.** `fct_events` spans
**2015-05-03 .. 2015-09-18** (139 days). `fct_orders` spans
**2016-09-04 .. 2018-10-17**. Not one shared day. This is the single most
important layout constraint in the document — see Q4 and §7.

---

## 1. The questions the dashboard answers

Five. Deliberately five. A portfolio dashboard that answers five questions
well beats one with twenty tiles, and each of these survived a validation pass
that killed four others (§5).

| # | Question a real stakeholder asks | Star | Page |
|---|---|---|---|
| **Q1** | Is revenue growing, and what actually moves it? | Olist | 1 |
| **Q2** | Does missing our delivery promise cost us customer satisfaction — and by how much? | Olist | 2 |
| **Q3** | Where is delivery worst, and is it where we sell most? | Olist | 2 |
| **Q4** | Where does browse → cart → buy leak, and does the leak differ by category? | RetailRocket | 3 |
| **Q5** | Can I trust these numbers today? | both | 4 |

Q2 is the headline. It has the strongest measured effect in the dataset
(a 2.63-star swing) and it is the one a business would act on.

Q5 is the differentiator. Most portfolio dashboards show only the business
numbers; showing the pipeline's own health is what signals data *engineer*
rather than analyst.

---

## 2. Metric definitions

Precision here is what stops a dashboard being quietly wrong. Each metric
states grain, numerator, denominator, and how it treats the four known
landmines: **cancelled orders**, **the `-1` unknown item member**, **the
orphan categories**, and **line-less orders**.

### Global definitional decisions

These apply everywhere and must be stated once, visibly, on the dashboard.

**D1 — "Revenue" means `sum(fct_orders.item_revenue)`, at order-line grain,
across all order statuses.**
`item_revenue = item_price + freight_value`. Validated total:
**R$ 15,843,553**.

There is a second, different number available — `sum(fct_order_payments
.payment_value)` = **R$ 16,008,872**, a **+R$ 165,319 (+1.04%)** gap. Two
revenue numbers on one dashboard that do not agree is the classic BI failure.
The rule: **`fct_orders` is revenue; `fct_order_payments` is never used as a
revenue total, only for payment-mix composition** (Q1c), where it is expressed
as a *share*, never as an absolute R$ alongside the D1 figure.

**D2 — cancelled and non-delivered orders are INCLUDED in revenue.**
Rationale: this is gross booked revenue, and excluding them changes the total
by 0.67%, which is less than the D1 reconciliation gap. Excluding them would
also mean the revenue trend and the order count disagree with the order-status
breakdown for no visible reason. Validated status mix at fct_orders grain:

| status | orders | line revenue | share |
|---|---:|---:|---:|
| delivered | 96,478 | R$ 15,419,774 | 97.325% |
| shipped | 1,106 | R$ 177,129 | 1.118% |
| canceled | 461 | R$ 105,886 | 0.668% |
| processing | 301 | R$ 69,394 | 0.438% |
| invoiced | 312 | R$ 68,989 | 0.435% |
| unavailable | 6 | R$ 2,140 | 0.014% |
| approved | 2 | R$ 241 | 0.002% |

**D3 — 775 orders have no lines and are absent from `fct_orders` by
construction.** Not a filter — that is what line grain means. They are not
evenly distributed: **603 `unavailable`, 164 `canceled`, 5 `created`,
2 `invoiced`, 1 `shipped`**. So `unavailable` reads as 6 orders in the fact
and 609 in the source. **Any page showing order status must carry a footnote
saying so**, or someone reconciling against the source finds a 100x
discrepancy and stops trusting the whole thing. `fct_orders` distinct orders =
**98,666**; source orders = **99,441**.

**D4 — the `-1` unknown item member is shown, never silently dropped, and
never ranked against real categories.** 254,336 events (9.27%) resolve to it.
Rule: it appears as its own labelled `Unknown` bar in *volume* charts (Q5),
and is **excluded from any rate or ranking** (Q4b), because its 0.33%
add-to-cart rate is a metadata-coverage artifact, not a category performing
badly. See D6.

**D5 — orphan categories are a non-issue for the dashboard.** 132 `dim_items`
rows carry a `category_id` with no resolvable `root_category_id`, spanning
**30 distinct category ids**. **Zero events land on those items.** No metric
below needs to handle them. (This also corrects a mislabel — see §6.)

**D6 — every rate has its denominator named on the visual.** No exceptions.

---

### Q1 — Is revenue growing, and what moves it?

**M1 · Revenue by month**
- Grain: one point per `dim_date.year_month`.
- Numerator: `sum(fct_orders.item_revenue)`. Denominator: none (absolute).
- Edge cases: **restrict to 2017-01 .. 2018-08 by default.** Three months
  outside that window hold under 100 orders each — 2016-09 (3 orders,
  R$ 355), 2016-12 (1 order, R$ 20), 2018-09 (1 order, R$ 166). Those are
  data-load artifacts, not a business ramp, and including them makes the
  chart's left third an uninformative crawl along the axis. The default date
  scope must be a *visible, changeable filter*, not a hardcoded `where`, so a
  viewer can see the artifact months exist.
- Validated: R$ 137,188 (2017-01) → R$ 1,003,308 (2018-08); peak
  R$ 1,179,144 in 2017-11.

**M2 · Revenue by day, with the peak annotated**
- Grain: one point per `fct_orders.order_date`.
- Validated: mean day in the dense window **R$ 26,223**; top day
  **2017-11-24 = R$ 178,378 across 1,166 orders — 6.8× the mean**. That is
  Black Friday 2017. Second is 2017-11-25 at 2.7×; no other day exceeds 2.5×.
- One annotation, on that one point. Not a label per point.

**M3 · Revenue by product category**
- Grain: `dim_products.product_category`.
- Numerator: `sum(fct_orders.item_revenue)` grouped by category. Denominator
  for the share label: the D1 total.
- Edge cases: 610 products carry no category and land in the `'unknown'`
  bucket that `dim_products` creates deliberately (nulls get dropped from
  slicers, and a dropped slice makes filtered totals disagree with the
  unfiltered one). `is_uncategorised` flags them.
- Validated: **74 distinct categories**; top 10 hold **62.3%** of revenue;
  the largest is `health_beauty` at **9.1%**.
- Consequence for form: 74 classes is an order of magnitude past any colour
  ceiling and the head is only 9.1%, so this is a **ranked horizontal bar,
  top 10 + explicit "Other (64 categories)"** — never a pie, never a
  colour-per-category series.

**M4 · Payment mix and the installment ladder**
- Grain: `fct_order_payments`, one row per `(order_id, payment_sequential)`.
- Validated share of `payment_value`: credit_card **78.3%**, boleto **17.9%**,
  voucher **2.4%**, debit_card **1.4%**, not_defined 0.0% (3 rows, R$ 0).
- The real finding: **only `credit_card` ever has installments > 1.** Every
  other type averages exactly 1.00. So the chart is not a four-slice type
  breakdown (which is one dominant slice and three slivers) — it is the
  **credit-card installment ladder against average payment value**:
  1× R$ 95.87 (25,455 payments) → 10× R$ 415.09 (5,328) → 12× R$ 321.68 (133).
  Basket size rises with installment count, with a visible jump at the 8× and
  10× marketing tiers.
- Edge case: 2 credit-card payments record `payment_installments = 0`. Fold
  into the 1× bucket **[branch]** or show as its own `0×` bar; either is fine
  at n=2, but pick one and say which.

### Q2 — Does missing the delivery promise cost satisfaction? (headline)

**M5 · Average review score by delivery-timeliness band**
- Grain: **one row per order** — filter `fct_orders.is_order_header_line`
  (new column, §6) so a three-line order counts once, not three times.
- Numerator: `avg(fct_orders.review_score)`. Denominator: orders in the band
  that have a review.
- Bands: `fct_orders.delivery_timeliness_band` (new column, §6), sorted by
  `delivery_timeliness_sort`.
- Edge cases: `review_score` is null where no review was filed (~1-3% per
  band) — those orders are in the band's count but out of the average, and
  the visual must show both `n orders` and `n reviewed`. Undelivered orders
  have a null band and are excluded entirely rather than forming a phantom
  seventh band.
- **Validated — this is the strongest relationship in the dataset:**

| band | orders | reviewed | avg score |
|---|---:|---:|---:|
| >10d early | 57,208 | 56,910 | **4.32** |
| 4-10d early | 26,698 | 26,553 | 4.26 |
| 0-3d early | 4,743 | 4,705 | 4.13 |
| 1-3d late | 2,662 | 2,636 | 3.77 |
| 4-10d late | 2,865 | 2,793 | 2.13 |
| >10d late | 2,300 | 2,233 | **1.70** |

  Monotonic across all six bands, a **2.63-star spread**, and the cliff is
  between "1-3d late" (3.77) and "4-10d late" (2.13) — being a few days late
  is survivable; being a week late is not. That cliff is the insight, and it
  is why six bands beat a binary on-time/late split.

### Q3 — Where is delivery worst?

**M6 · Days to deliver and late rate by customer state**
- Grain: `fct_orders.customer_state`, one row per delivered order
  (`is_order_header_line` + `delivered_to_customer_at is not null`).
- `avg(days_to_deliver)`; late rate = `count(is_late_delivery) / count(delivered orders)`.
- Edge case: **minimum 200 delivered orders per state**, or the small northern
  states produce a ±10-day mean off a few dozen orders and dominate the
  ranking on noise. 23 of 27 states clear the bar.
- Validated: **AL 24.5 days / 23.9% late** at the worst, **SP 8.8 days /
  5.9% late** at the best — a **2.8× spread** in delivery time, with the late
  rate moving with it. SP is also 41% of all orders, so the volume-weighted
  average hides this entirely — which is exactly why the state cut earns a
  chart.

### Q4 — Where does the funnel leak?

**M7 · Headline funnel: view → addtocart → transaction**
- Grain: **visitor** — `count(distinct fct_events.visitor_id)` reaching each
  step, using `funnel_step` (new column, §6) for ordering.
- Validated (visitor grain): **1,402,623 visitors → 1,399,193 viewed →
  37,571 added to cart (2.69% of viewers) → 11,675 transacted (0.83% of
  viewers, 31.1% of adders)**.
- Event grain agrees closely (2.60% / 0.842%), so either is defensible.
  **Visitor grain is chosen** because event grain counts a visitor who
  reloaded the same item eleven times as eleven views, which inflates the
  denominator with behaviour that is not eleven people. Say this on the page.
- Edge case: 3,430 visitors appear with no `view` event at all (added to cart
  or transacted without a recorded view). They are inside the funnel's total
  visitor count but outside the view step — the visual's first bar is
  "viewed", not "visitors", for that reason.
- Context worth stating: median visitor has **1 active day and ~2 events**;
  1,259,550 of 1,402,623 visitors appear on exactly one day. This is a
  browse-heavy, low-repeat log — which is why there is no session/retention
  metric here.

**M8 · Add-to-cart rate by root category**
- Grain: `dim_items.root_category_id`, joined via `fct_events.item_key`.
- Numerator: `count(*) where event_type = 'addtocart'`. Denominator:
  `count(*) where event_type = 'view'`, same category.
- Edge cases, both load-bearing:
  1. **Exclude `root_category_id = -1`** from the ranking (D4). Its 0.33%
     add-to-cart rate would sit at the bottom of the chart reading as the
     worst-performing category, when it is really "these items have no
     metadata". Its *volume* still appears on the Q5 page.
  2. **Minimum 20,000 views per category.** Four categories (431, 1394, 1452,
     755) have literally zero add-to-cart events off small view counts and
     would otherwise anchor the chart at 0%.
- Validated: **24 root categories present in events; 11 clear the 20,000-view
  bar; add-to-cart rate ranges 1.19% (root 378) to 4.86% (root 1224) — a
  4.1× spread.** Root 1224 also converts to transaction at 1.906% vs a 0.84%
  overall — it is genuinely the standout, on 80,264 views.
- **Known weakness:** RetailRocket ships **no category names**, only integer
  ids. The axis labels are literally "1224", "140", "378". Nothing in the
  source can fix this. Options in §7.

### Q5 — Can I trust these numbers?

**M9 · Unknown-item rate, by event type**
- Numerator: `count(*) where not fct_events.has_item_metadata`. Denominator:
  `count(*)`, within event type.
- Validated: **9.27% overall (254,336 of 2,742,263)** — but
  **view 9.54%, transaction 2.12%, addtocart 1.20%**. *The skew is the
  metric.* The gap is concentrated in top-of-funnel browsing and barely
  touches commercially significant events, which is what makes the
  revenue-side analysis defensible. Guarded by
  `assert_unknown_item_rate_within_bounds` (band 7%–12%).

**M10 · Mart row counts and freshness**
- `fct_events` 2,742,263 · `dim_items` 417,054 · `fct_orders` 112,650 lines /
  98,666 orders · `fct_order_payments` 103,886 · `dim_weather` 20,898 ·
  `dim_date` 1,826.
- Plus min/max date per fact, which is what makes the two-star disjointness
  (§0) visible instead of a trap.

**M11 · dbt test results over time** — **this data does not exist yet.**
See §6, "Required but not yet built".

**M12 · Payment-vs-line reconciliation** — the +1.04% / R$ 165,319 gap from
D1, shown as a tracked number rather than hidden. This is what
`assert_order_payments_reconcile` is for, and its `error_if: >500` threshold
should be re-set from the first real run.

---

## 3. Layout

Four pages. One filter row per page, above everything it scopes — never
per-chart filters, never a filter inside a chart card.

### Information hierarchy, per page

Hero/KPI row first (the number you'd say out loud), then the one chart that
answers the page's question, then the supporting cuts. Not a grid of equal
tiles: equal visual weight means no story.

```
┌──────────────────────────────────────────────────────────────────────────┐
│  PAGE 1 — REVENUE                                                         │
│  ┌────────────────────────────────────────────────────────────────────┐  │
│  │ FILTERS:  [ date range: 2017-01 .. 2018-08 ▾ ]  [ state ▾ ]        │  │
│  │           [ category ▾ ]                                            │  │
│  └────────────────────────────────────────────────────────────────────┘  │
│                                                                           │
│   R$ 15.8M            98,666            R$ 160.6            74            │
│   total revenue       orders            avg order value     categories    │
│   ▁▂▃▄▅▆▇█ sparkline  (stat tiles: value + label, no chart chrome)        │
│                                                                           │
│  ┌─────────────────────────────────────────────────────────────────────┐ │
│  │  Revenue by day                                    [line, 1 series] │ │
│  │                                        ● 2017-11-24 · Black Friday  │ │
│  │                                        │ R$178k — 6.8× a normal day │ │
│  │   ╱╲    ╱╲  ╱╲    ╱╲╱╲   ╱╲  ╱╲       ╱│╲  ╱╲╱╲   ╱╲╱╲╱╲  ╱╲       │ │
│  │  ╱  ╲╱╲╱  ╲╱  ╲╱╲╱    ╲╱╲╱ ╲╱  ╲╱╲╱╲╱  │  ╲╱    ╲╱      ╲╱  ╲╱╲    │ │
│  │  └─────────────────────────────────────────────────────────────────┘ │
│  │  crosshair + tooltip on hover; ONE annotation, not a label per point │ │
│  └─────────────────────────────────────────────────────────────────────┘ │
│                                                                           │
│  ┌────────────────────────────────┐ ┌───────────────────────────────────┐ │
│  │ Revenue by category  (top 10)  │ │ Credit-card installments vs       │ │
│  │                                │ │ average payment value             │ │
│  │ health_beauty     ████████ 9.1%│ │                                   │ │
│  │ watches_gifts     ███████  8.2%│ │  R$420 ┤                    ▄     │ │
│  │ bed_bath_table    ███████  7.8%│ │        │                 ▄  █     │ │
│  │ sports_leisure    ██████   7.3%│ │        │           ▄  ▄  █  █  ▄  │ │
│  │ computers_acc.    ██████   6.7%│ │  R$100 ┤ ▄  ▄  ▄  █  █  █  █  █  █│ │
│  │ furniture_decor   █████    5.7%│ │        └─1──2──3──4──6──8─10─12──│ │
│  │ housewares        ████     4.9%│ │          installments             │ │
│  │ cool_stuff        ████     4.5%│ │                                   │ │
│  │ auto              ████     4.3%│ │  bar width ∝ nothing; height =    │ │
│  │ garden_tools      ███      3.7%│ │  avg payment value. n labelled.   │ │
│  │ Other (64 cats)   ██████████    │ │                                   │ │
│  │                        37.7%   │ │                                   │ │
│  └────────────────────────────────┘ └───────────────────────────────────┘ │
└──────────────────────────────────────────────────────────────────────────┘
```

```
┌──────────────────────────────────────────────────────────────────────────┐
│  PAGE 2 — DELIVERY & SATISFACTION            ← the page that earns the    │
│                                                 dashboard                 │
│  FILTERS:  [ date range ▾ ]  [ state ▾ ]  [ category ▾ ]                  │
│                                                                           │
│   ┌────────────────────────────────────────────────────────────────────┐ │
│   │  HERO:   2.63 stars                                                │ │
│   │  the review-score gap between an order that arrives >10 days early │ │
│   │  and one that arrives >10 days late.   4.32 ★  →  1.70 ★           │ │
│   │  (≥48px, same sans as everything else, proportional figures)       │ │
│   └────────────────────────────────────────────────────────────────────┘ │
│                                                                           │
│  ┌─────────────────────────────────────────────────────────────────────┐ │
│  │  Average review score by how late delivery was    [ordinal bars]    │ │
│  │                                                                     │ │
│  │  >10d early  ████████████████████████████  4.32   n=56,910          │ │
│  │  4-10d early ███████████████████████████   4.26   n=26,553          │ │
│  │  0-3d early  ██████████████████████████    4.13   n= 4,705          │ │
│  │  1-3d late   ████████████████████████      3.77   n= 2,636          │ │
│  │  4-10d late  █████████████                 2.13   n= 2,793  ◄ cliff │ │
│  │  >10d late   ██████████                    1.70   n= 2,233          │ │
│  │              └─1───2───3───4───5─ avg review score                  │ │
│  │  ONE hue, monotone lightness light→dark (ordinal ramp — the bands   │ │
│  │  are ordered, so the order must be visible in the colour).          │ │
│  │  Direct-labelled: every bar, because there are only six and the     │ │
│  │  value IS the finding.                                              │ │
│  └─────────────────────────────────────────────────────────────────────┘ │
│                                                                           │
│  ┌──────────────────────────────────┐ ┌──────────────────────────────────┐│
│  │ Days to deliver by state         │ │ Late rate by state               ││
│  │ (≥200 delivered orders)          │ │                                  ││
│  │  AL ██████████████████ 24.5      │ │  AL ████████████ 23.9%           ││
│  │  PA █████████████████  23.8      │ │  MA █████████    19.7%           ││
│  │  MA ███████████████    21.6      │ │  PI ████████     16.0%           ││
│  │  ...                             │ │  ...                             ││
│  │  MG ████████           12.0      │ │  MG ███          5.6%            ││
│  │  PR ████████           12.0      │ │  SP ███          5.9%            ││
│  │  SP ██████              8.8      │ │  PR ███          5.0%            ││
│  │                                  │ │                                  ││
│  │ SP emphasised (accent hue), rest │ │ Same emphasis, same entity→hue   ││
│  │ in de-emphasis gray: SP is 41%   │ │ mapping. NOT two y-axes on one   ││
│  │ of volume and the fastest.       │ │ plot — two charts, one measure   ││
│  └──────────────────────────────────┘ └──────────────────────────────────┘│
└──────────────────────────────────────────────────────────────────────────┘
```

```
┌──────────────────────────────────────────────────────────────────────────┐
│  PAGE 3 — BROWSE → BUY  (RetailRocket clickstream)                        │
│  ⚠ Different retailer, different period: 2015-05-03 .. 2015-09-18.        │
│    Nothing on this page relates to the Olist orders on pages 1-2.         │
│  FILTERS:  [ date range: 2015-05-03 .. 2015-09-17 ▾ ]  [ category ▾ ]     │
│            [ ☐ include items with no metadata ]                           │
│                                                                           │
│   1,402,623          2.69%              0.83%             31.1%           │
│   visitors           view → cart        view → buy        cart → buy      │
│                                                                           │
│  ┌─────────────────────────────────────────────────────────────────────┐ │
│  │  Visitor funnel                          [ordinal, 3 stages]        │ │
│  │                                                                     │ │
│  │  viewed      ████████████████████████████████████  1,399,193        │ │
│  │  added       █                                        37,571  2.69% │ │
│  │  transacted  ▏                                        11,675  0.83% │ │
│  │                                                                     │ │
│  │  ponytail: horizontal bars, NOT a tapered funnel graphic. At a      │ │
│  │  2.7% first drop the funnel shape degenerates into a spike and a    │ │
│  │  hairline; bars keep the second-stage number readable.              │ │
│  │  A log axis would make the drop look mild — it isn't. Linear, and   │ │
│  │  the stage-to-stage % direct-labelled beside each bar.              │ │
│  └─────────────────────────────────────────────────────────────────────┘ │
│                                                                           │
│  ┌─────────────────────────────────────────────────────────────────────┐ │
│  │  Add-to-cart rate by category   (≥20,000 views; excludes Unknown)   │ │
│  │                                                                     │ │
│  │  cat 1224  ██████████████████  4.86%   ← 1.91% buy rate, 2.3× mean  │ │
│  │  cat 1482  ████████████        3.29%                                │ │
│  │  cat  679  ███████████         3.15%                                │ │
│  │  cat  140  ███████████         3.02%   (largest: 692,860 views)     │ │
│  │  cat 1600  ██████████          2.94%                                │ │
│  │  cat 1532  █████████           2.55%                                │ │
│  │  cat  653  ████████            2.39%                                │ │
│  │  cat  395  ████████            2.31%                                │ │
│  │  cat 1490  ███████             2.22%                                │ │
│  │  cat  378  ████                1.19%   ← worst real category        │ │
│  │            └─ sorted by rate; 11 categories clear the view floor    │ │
│  │                                                                     │ │
│  │  ONE hue for every bar (nominal categories — bar length already     │ │
│  │  encodes the value; a value-ramp here would double-encode it).      │ │
│  │  Table view alongside, because integer category ids are opaque.     │ │
│  └─────────────────────────────────────────────────────────────────────┘ │
└──────────────────────────────────────────────────────────────────────────┘
```

```
┌──────────────────────────────────────────────────────────────────────────┐
│  PAGE 4 — PIPELINE HEALTH        (the data-engineer differentiator)       │
│  no filters — this page is about the pipeline, not a slice of it          │
│                                                                           │
│   ┌───────────────┐ ┌───────────────┐ ┌───────────────┐ ┌──────────────┐ │
│   │ — not yet     │ │ 9.27%         │ │ 2,742,263     │ │ +1.04%       │ │
│   │   instrumented│ │ unknown-item  │ │ fct_events    │ │ payment vs   │ │
│   │ dbt tests (M11│ │ band 7–12% ✓  │ │ rows          │ │ line revenue │ │
│   │ — see §6)     │ │ [status: good]│ │               │ │              │ │
│   └───────────────┘ └───────────────┘ └───────────────┘ └──────────────┘ │
│   status colours (good/warn/critical) with icon + label, never colour     │
│   alone, and never reused as a series hue anywhere else in the report     │
│                                                                           │
│  ┌────────────────────────────────┐ ┌───────────────────────────────────┐ │
│  │ Unknown-item rate by event type│ │ Row counts per mart               │ │
│  │                                │ │  (table, not a chart — 9 numbers  │ │
│  │  view        █████████  9.54%  │ │   of wildly different magnitude)  │ │
│  │  transaction ██         2.12%  │ │                                   │ │
│  │  addtocart   █          1.20%  │ │  fct_events          2,742,263    │ │
│  │                                │ │  dim_items             417,054    │ │
│  │  THE finding: the gap is a     │ │  fct_orders (lines)    112,650    │ │
│  │  browse-side problem, not a    │ │  fct_orders (orders)    98,666    │ │
│  │  revenue-side one.             │ │  fct_order_payments    103,886    │ │
│  └────────────────────────────────┘ │  dim_weather            20,898    │ │
│                                     │  dim_date                1,826    │ │
│  ┌────────────────────────────────┐ │                                   │ │
│  │ dbt test pass/fail per run     │ └───────────────────────────────────┘ │
│  │ ⚠ NOT YET AVAILABLE — needs    │                                       │
│  │   run_results.json persisted;  │  ┌──────────────────────────────────┐ │
│  │   see §6                       │  │ Fact coverage windows            │ │
│  └────────────────────────────────┘  │  fct_orders  2016-09-04→2018-10-17│ │
│                                      │  fct_events  2015-05-03→2015-09-18│ │
│                                      │  ⚠ no overlap — by design         │ │
│                                      └──────────────────────────────────┘ │
└──────────────────────────────────────────────────────────────────────────┘
```

### Chart-form rationale (per the dataviz form heuristic)

| Visual | Form | Colour job | Why not the obvious alternative |
|---|---|---|---|
| KPI row | stat tiles | none (ink only) | A one-bar bar chart per KPI is the classic mistake — the number *is* the chart |
| Revenue by day | line, 1 series | sequential / single hue | 1 series → **no legend box**; the title names it |
| Revenue by category | ranked horizontal bar, top 10 + Other | **one hue, all bars** | 74 classes; a pie is unreadable past ~6 and a colour-per-category needs 74 hues |
| Installment ladder | column, ordered x | ordinal ramp | x is an ordered integer scale, so the order must be visible in the colour |
| Review by band | horizontal ordinal bar | **ordinal ramp**, light→dark | Six ordered bands: swapping their order changes the meaning, so this is ordinal, not nominal |
| Delivery by state | two separate bars, **emphasis** | 1 accent hue + de-emphasis gray | **Never a dual axis** — days and % are different scales; two charts, one measure each |
| Funnel | horizontal bars | ordinal, 3 stages | A tapered funnel graphic degenerates at a 2.7% first drop |
| ATC rate by category | horizontal bar, sorted | **one hue** | Nominal ids — colouring darker-where-bigger double-encodes bar length |
| Pipeline health | stat tiles + table | **status palette**, icon + label | Status colours are reserved and never reused as series hues |

### Interaction and accessibility rules (all tools)

- Hover tooltip on every plotted mark; crosshair on the time series. Tooltips
  **enhance, never gate** — every value is also reachable from a direct label
  or the table view.
- One filter row per page, above the charts. All charts on the page re-render
  against the same slice.
- **Colour follows the entity, never its rank.** Filtering states out must not
  repaint the survivors.
- Thin marks, hairline solid grid/axes (never dashed), 2px surface gap between
  adjacent fills, generous padding.
- A **table view** exists for every chart. Non-negotiable for page 3, where the
  category axis is opaque integers.
- Text wears text tokens, never the series colour.
- **[branch] Dark mode.** Power BI/Tableau: pick one theme and commit — neither
  gives a viewer-side toggle on a published report, so a two-mode palette is
  wasted effort. Metabase/Streamlit: honour the viewer's theme, and validate
  the palette against *both* surfaces, not by flipping the light one.

### Colour: what must be done once the tool is picked

The palette is not specified here because each tool ships its own theme
defaults and the swap point differs. What is fixed:

1. Every chart's colour does **exactly one** of the four jobs named in the
   table above.
2. Categorical hues are assigned in fixed order and **never cycled**. No chart
   in this design needs more than three categorical slots — that is deliberate.
3. Before shipping, run the validator on whatever palette the chosen tool's
   theme produces:
   `node scripts/validate_palette.js "<hex,hex,…>" --mode light` (and again
   `--mode dark` if the tool has a dark surface). Adjacent-pair CVD ΔE ≥ 8,
   normal-vision worst pair ≥ 15, contrast ≥ 3:1. Fix any FAIL before
   publishing. Do not eyeball this.
4. Ordinal ramps (review bands, funnel stages, installments) are **one hue,
   monotone lightness** — validate with `--ordinal`.
5. Status colours on page 4 are reserved and appear nowhere else in the report.

---

## 4. Column-level dependency map

Every metric, and exactly which mart column it reads. Columns marked **NEW**
were added by this design — see §6.

| Metric | Mart | Columns read |
|---|---|---|
| D1, M1, M2 revenue | `fct_orders` | `item_revenue`, `order_date` |
| M1 month grouping | `dim_date` | `date_day`, `year_month`, `month_start_date` |
| M2 day grouping | `dim_date` | `date_day` |
| KPI: orders | `fct_orders` | `order_id` (distinct) |
| KPI: avg order value | `fct_orders` | `item_revenue`, `order_id` |
| D2 status mix | `fct_orders` | `order_status`, `order_id`, `item_revenue` |
| M3 revenue by category | `fct_orders` → `dim_products` | `fct_orders.product_id`, `fct_orders.item_revenue`; `dim_products.product_id`, `product_category`, `is_uncategorised` |
| M4 payment mix | `fct_order_payments` | `payment_type`, `payment_value`, `payment_installments`, `order_date` |
| M5 review by band | `fct_orders` | `review_score`, **`delivery_timeliness_band`** (NEW), **`delivery_timeliness_sort`** (NEW), **`is_order_header_line`** (NEW), `order_id` |
| M5 hero figure | `fct_orders` | same as M5 |
| M6 delivery by state | `fct_orders` | `customer_state`, `days_to_deliver`, `is_late_delivery`, `delivered_to_customer_at`, **`is_order_header_line`** (NEW), `order_id` |
| M7 funnel | `fct_events` | `visitor_id`, `event_type`, **`funnel_step`** (NEW), `event_date` |
| M8 ATC rate by category | `fct_events` → `dim_items` | `fct_events.item_key`, `event_type`; `dim_items.item_id`, `root_category_id`, `is_unknown_item` |
| M8 unknown filter | `fct_events` | `has_item_metadata` |
| M9 unknown-item rate | `fct_events` | `has_item_metadata`, `event_type` |
| M10 row counts | all facts + dims | `count(*)`; `fct_orders.order_date`, `fct_events.event_date` for windows |
| M11 dbt test results | **does not exist** | see §6 |
| M12 reconciliation | `fct_orders` + `fct_order_payments` | `item_revenue`; `payment_value` |
| Page 1–2 state filter | `fct_orders` | `customer_state` |
| Page 1–2 category filter | `dim_products` | `product_category` |
| Page 3 category filter | `dim_items` | `root_category_id` |

**Marts read by nothing in this design:** `dim_sellers` (kept — it is what
makes `fct_orders.seller_id` a testable key rather than a dangling one),
`dim_weather` (metric killed, see §5 — the model and the external-API
ingestion step stay as pipeline deliverables), and `dim_customers` (the cohort
page it existed for is killed; `customer_state` is denormalised onto
`fct_orders` already, so the dimension is unused by the dashboard but still
carries the `relationships` guard).

---

## 5. Metrics killed, and why

Killing a metric before spending Snowflake trial credits on it is the point of
this exercise. Four proposals did not survive contact with the data. Two of
them came from the scope doc.

### KILLED — Weather overlay on the revenue page *(scope doc §4, page 1)*

The scope doc proposed overlaying weather on the revenue trend to answer "does
revenue dip on certain weather conditions?" **It does not.**

Method: revenue per `(customer_state, order_date)` compared rainy vs dry
*within* matched `(state, calendar month, weekday)` cells — matching is
necessary because rain is seasonal and revenue tripled over the window, so an
unmatched comparison mostly measures which months happened to be wet.

| treatment | matched cells | weighted delta |
|---|---:|---:|
| rain ≥ 1mm | 1,595 | **+1.86%** |
| heavy rain ≥ 20mm | 331 | **+3.18%** |

Per state (rain ≥ 1mm): SP +2.45%, RJ +6.22%, MG +8.30%, RS +3.81%,
PR −5.16%, SC −11.68%, BA −3.96%, ES −0.58%. **The sign flips**, the magnitude
is inside the noise, and the aggregate is *positive* — the opposite of the
hypothesis the chart was meant to illustrate.

Publishing a "+1.9% revenue on rainy days" line would be presenting noise as a
finding, which is worse than having no weather chart at all.

**What survives:** `dim_weather`, `ingestion/fetch_weather.py`, and the
`weather_key` join stay. They are a real external-API ingestion step and a
real composite-key join, and the null result is itself worth one paragraph in
the README — "I built the join, tested the hypothesis, and it wasn't there" is
a stronger interview answer than a chart of noise.

### KILLED — Customer cohort / repeat-purchase page *(scope doc §4, page 3)*

**3.12% of people ever place a second order** (2,997 of 96,096 distinct
`customer_unique_id`). The distribution: 93,099 people with 1 order, 2,745
with 2, 203 with 3, 30 with 4, 8 with 5, 6 with 6.

A retention curve on that is a flat line pinned near zero with no cohort
separation; a cohort heatmap is a grid of near-identical near-empty cells.
Olist mints a fresh `customer_id` per order, and `dim_customers` documents
this honestly — the dataset simply is not a repeat-purchase dataset.

**What survives:** `dim_customers.orders_by_person` / `is_repeat_customer`
stay in the model. The 3.12% is worth one sentence on page 1 as a stated
property of the dataset, not a page.

### KILLED — Cancellation-rate KPI tile

461 cancelled orders visible at `fct_orders` grain out of 98,666 (**0.47%**),
**0.668% of line revenue** — plus 164 more cancelled orders that never had a
line and so are not in the fact at all. Nothing to look at, no variation to
filter, and the split across the grain boundary makes it actively misleading
as a headline number.

**What survives:** the *definition* (D2 — revenue includes all statuses) and
the D3 footnote. Both stay, stated on the page. The tile goes.

### KILLED — A single global date filter across all pages

`fct_events` covers 2015-05-03 .. 2015-09-18; `fct_orders` covers 2016-09-04
.. 2018-10-17. **Zero overlapping days.** A shared date slicer would blank
page 3 whenever pages 1–2 have data and vice versa — a dashboard that looks
broken to anyone who touches the filter.

**What survives:** per-page date scope, and an explicit banner on page 3
stating that the two stars describe different retailers in different periods.
`_marts.yml` already makes this argument for the model layer; the dashboard
has to make it visibly, in the UI, or the disjointness reads as a bug.

### Kept, but reshaped by the data

- **Payment mix (M4)** — the obvious four-slice type breakdown is one 78%
  slice and three slivers. Reshaped into the credit-card installment ladder,
  which is where the actual variation is.
- **Funnel by category (M8)** — kept, but the `-1` member had to be excluded
  from the ranking and a 20,000-view floor imposed, or the chart is four
  zero-rate bars and an artifact at the bottom.
- **Revenue trend (M1)** — kept, but defaulted to 2017-01 .. 2018-08 because
  three months hold under 100 orders.

---

## 6. Required mart changes

### Made (additive columns only — no grain or structure changed)

| Model | Column | Type | Why |
|---|---|---|---|
| `fct_orders` | `delivery_timeliness_band` | varchar | The six ordered bands are the x-axis of M5, the dashboard's headline chart. Their boundaries are a *definition*, not a presentation choice — left to the BI tool, the same bands get written three different ways across the three candidate tools and stop agreeing. Null until delivered. |
| `fct_orders` | `delivery_timeliness_sort` | int 1–6 | A banded label is ordinal and every BI tool sorts text alphabetically by default, which would interleave "early" and "late" and destroy the monotonic gradient that is the finding. Sort key, never a measure. |
| `fct_orders` | `is_order_header_line` | boolean | `order_item_id = 1`, so exactly one true row per order. At line grain, `avg(review_score)` weights a three-line order three times. A plain boolean filter makes header averages order-weighted in *any* tool, instead of needing a distinct-aware measure written three different ways. Not a substitute for `count(distinct order_id)`. |
| `fct_events` | `funnel_step` | int 1–3 | `event_type` is nominal text and sorts alphabetically — addtocart, transaction, view — drawing the funnel with the outcome in the middle and the entry point last. Fixing that is a different tool-specific manoeuvre in each of Power BI, Tableau and Metabase, so the order belongs in the model where there is one of it. |

`_marts.yml` documents all four, with an `accepted_values` test on
`delivery_timeliness_band` and on `funnel_step`, and `not_null` on
`is_order_header_line`.

**Explicitly not changed:** no grain, no existing column, no model structure.
`is_order_header_line` is derived from an existing column rather than
introducing a new order-header fact.

### Required but not yet built — M11, dbt test results over time

The scope doc's page-4 differentiator ("dbt test pass/fail counts over time,
row counts loaded per run, last successful pipeline run timestamp") **has no
data source anywhere in this project.** dbt writes `target/run_results.json`
on every invocation and then overwrites it on the next one. Nothing persists
it.

What is needed, precisely:

| Need | Where it should live |
|---|---|
| A table `raw.dbt_run_results` — one row per `(invocation_id, node_id)` with `status`, `execution_time`, `message`, `run_started_at` | A post-run step that parses `target/run_results.json` and appends to Snowflake. Belongs alongside `warehouse/load_snowflake.py`, not in `dbt/models/`. |
| A `fct_pipeline_runs` mart — one row per invocation: tests passed/failed/errored, models built, total runtime | `dbt/models/marts/` |
| A `fct_pipeline_test_results` mart — one row per `(invocation_id, test_name)` | `dbt/models/marts/` |
| The Airflow DAG to call the persist step after `dbt build` | `airflow/` |

This was **not** built here — it needs a new ingestion path and a change to the
Airflow DAG, both outside the additive-mart-edit boundary of this task, and it
should be scoped as its own piece of work.

Until it exists, page 4 shows M9, M10 and M12 (all computable from existing
marts) and carries a visible placeholder where the test-history chart goes.
Do not fake it with a hardcoded "62/62 passing" tile.

### Bug found — `assert_orphan_categories_bounded` baseline is mislabelled

`dbt/tests/assert_orphan_categories_bounded.sql` counts **distinct
`category_id`** where `root_category_id is null`, and asserts it is ≤ the
`orphan_category_max` var, set to **132** with the comment "measured exactly
132".

Measured locally against `data/clean/dim_items/`:

- **132** = the number of orphan **items**.
- **30** = the number of distinct orphan **category ids** — what the test
  actually counts.

`docs/data_quality.md` §3.5 repeats the same mislabel ("132 category IDs
referenced by items do not appear in `category_tree.csv`"). The test still
passes (30 ≤ 132), but the bound is 4.4× looser than intended, so it would not
catch the category tree degrading from 30 orphan categories to 100.

**Not fixed here** — the task scope is additive mart edits, and changing a
test threshold and a data-quality doc is a separate, deliberate call. Recommend
setting `orphan_category_max: 35` and correcting the §3.5 wording.

Also validated while checking this: **zero events land on orphan-category
items**, so orphan categories have no effect on any dashboard metric.

---

## 7. Design calls needing Matt's sign-off

1. **Which BI tool.** Nothing above depends on it, but the build cannot start
   without it. The trade-off is Microsoft-stack resume signal (Power BI, needs
   a Windows VM, no public link) versus a free cold-shareable URL (Tableau
   Public, native macOS). Given that the Phase 3A dashboard *also* still has
   no public URL and that hosting is deferred to Phase 5, a Tableau Public
   link would solve two problems at once — but it drops the Power BI line the
   scope doc chose deliberately. **Recommendation: Tableau Public**, with the
   Power BI version as the stretch goal the scope doc already lists in reverse.

2. **The opaque category axis on page 3.** RetailRocket ships no category
   names — the labels are integers. Options: (a) leave them as ids with a
   table view and a note, honest but forgettable; (b) label them by their
   most-viewed item id, no better; (c) hand-label the ~11 charted roots by
   inferring from item behaviour, which is fabrication and should not be done.
   **Recommendation: (a)**, and say on the page that the source is anonymised
   — that is a real property of the dataset, not a shortcoming of the build.

3. **Whether page 4 ships without test history.** M11 needs a new ingestion
   path (§6). Either build it, or ship page 4 with three real metrics and a
   visible "not yet instrumented" placeholder. **Recommendation: ship the
   placeholder**, then build M11 as the next scoped piece — a fabricated
   passing-tests tile on a page whose entire purpose is trustworthiness would
   be self-defeating.

4. **The killed weather metric.** Confirm the null result goes in the README
   rather than becoming a chart. The temptation to show *something* for the
   weather work is real, and should be resisted.

5. **Whether the four-page structure is right at all**, or whether pages 1–2
   should merge. Page 2 is the strongest content in the project; page 1 is the
   most conventional. An argument exists for leading with delivery-vs-
   satisfaction and demoting revenue to a supporting page.