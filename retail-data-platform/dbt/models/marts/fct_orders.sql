-- GRAIN: one row per order line -- (order_id, order_item_id).
--
-- ponytail: the obvious alternative is order-header grain, one row per order.
-- Line grain wins because price and freight are recorded per line, and because
-- product and seller are line-level attributes -- at header grain there is no
-- honest place to put a product key, so dim_products and dim_sellers would
-- have nothing to join to and the star would collapse into a single flat
-- order table. Header-level metrics are a `group by order_id` away from this
-- grain; the reverse direction is lossy. The cost is that anything genuinely
-- order-level repeats down the lines of a multi-line order, which is why the
-- non-additive columns are named and marked as such below.
--
-- Order-level MEASURES are deliberately not here. Payment value lives in
-- fct_order_payments at its own grain, because repeating a R$200 order payment
-- across three lines and letting someone sum the column is a revenue number
-- wrong by 3x that nobody catches until a meeting.
--
-- Orders with no lines (a small number in Olist, all cancelled or unavailable)
-- do not appear. That is what line grain means, not a filter -- an order that
-- never had a line has nothing to record at this grain.

with lines as (

    select * from {{ ref('stg_olist__order_items') }}

),

orders as (

    select * from {{ ref('stg_olist__orders') }}

),

customers as (

    select * from {{ ref('stg_olist__customers') }}

),

-- Some orders carry more than one review. Joining reviews in directly would
-- duplicate those orders' lines and silently inflate revenue -- the exact
-- failure tests/assert_fct_orders_line_count_matches_source.sql watches for.
-- Latest review wins, tie broken on review_id so the result is deterministic
-- across runs rather than dependent on scan order.
order_review as (

    select
        order_id,
        review_score,
        review_created_at
    from (
        select
            order_id,
            review_score,
            review_created_at,
            row_number() over (
                partition by order_id
                order by review_created_at desc, review_id desc
            ) as rn
        from {{ ref('stg_olist__order_reviews') }}
    )
    where rn = 1

)

select
    l.order_line_key,
    l.order_id,
    l.order_item_id,

    -- Foreign keys.
    o.customer_id,
    l.product_id,
    l.seller_id,
    cast(o.ordered_at as date)                                  as order_date,

    -- Composite key into dim_weather. The customer's state, not the seller's:
    -- the question this supports is whether weather stopped people ordering,
    -- and the buyer is the one deciding.
    c.customer_state || '-' || to_char(cast(o.ordered_at as date), 'YYYY-MM-DD')
                                                                as weather_key,
    c.customer_state,

    -- Degenerate / header attributes. Correct on any row, non-additive across
    -- the lines of one order: order_status, the timestamps, review_score.
    o.order_status,
    o.ordered_at,
    o.approved_at,
    o.delivered_to_carrier_at,
    o.delivered_to_customer_at,
    o.estimated_delivery_at,
    l.shipping_limit_at,
    r.review_score,

    -- Additive measures.
    l.item_price,
    l.freight_value,
    l.item_price + l.freight_value                              as item_revenue,

    -- Delivery performance. Null until the order is actually delivered, which
    -- keeps undelivered orders out of the average instead of counting them as
    -- a zero-day delivery.
    datediff('day', o.ordered_at, o.delivered_to_customer_at)    as days_to_deliver,
    datediff('day', o.delivered_to_customer_at, o.estimated_delivery_at)
                                                                 as days_early_vs_estimate,
    case
        when o.delivered_to_customer_at is null then null
        else o.delivered_to_customer_at > o.estimated_delivery_at
    end                                                         as is_late_delivery,

    -- Banded here, not in the BI tool. The dashboard's strongest finding is
    -- review score against how late delivery was, and the bands ARE the
    -- x-axis of that chart -- so their boundaries are a definition, not a
    -- presentation choice. Left to the tool, the same six bands get written
    -- three different ways across Power BI / Tableau / Metabase and stop
    -- agreeing the first time someone edits one of them. Null until
    -- delivered, so undelivered orders stay out of the chart rather than
    -- forming a seventh band that means "no data".
    --
    -- The sort key is a separate integer column because a banded label is
    -- ordinal and every BI tool sorts text alphabetically by default -- which
    -- here would interleave "early" and "late" and destroy the monotonic
    -- gradient that is the whole point.
    case
        when o.delivered_to_customer_at is null then null
        when datediff('day', o.delivered_to_customer_at, o.estimated_delivery_at) > 10  then '>10d early'
        when datediff('day', o.delivered_to_customer_at, o.estimated_delivery_at) > 3   then '4-10d early'
        when datediff('day', o.delivered_to_customer_at, o.estimated_delivery_at) >= 0  then '0-3d early'
        when datediff('day', o.delivered_to_customer_at, o.estimated_delivery_at) >= -3 then '1-3d late'
        when datediff('day', o.delivered_to_customer_at, o.estimated_delivery_at) >= -10 then '4-10d late'
        else '>10d late'
    end                                                         as delivery_timeliness_band,
    case
        when o.delivered_to_customer_at is null then null
        when datediff('day', o.delivered_to_customer_at, o.estimated_delivery_at) > 10  then 1
        when datediff('day', o.delivered_to_customer_at, o.estimated_delivery_at) > 3   then 2
        when datediff('day', o.delivered_to_customer_at, o.estimated_delivery_at) >= 0  then 3
        when datediff('day', o.delivered_to_customer_at, o.estimated_delivery_at) >= -3 then 4
        when datediff('day', o.delivered_to_customer_at, o.estimated_delivery_at) >= -10 then 5
        else 6
    end                                                         as delivery_timeliness_sort,

    -- Exactly one true row per order. The grain here is the line, so
    -- averaging any header attribute -- review_score, days_to_deliver -- over
    -- the raw fact weights a three-line order three times. Filtering to this
    -- flag makes those averages order-weighted in any BI tool with a plain
    -- filter, instead of needing a distinct-aware measure written three
    -- different ways in three different tools. It is not a substitute for
    -- count(distinct order_id), which every tool does correctly.
    (l.order_item_id = 1)                                       as is_order_header_line,

    (o.order_status = 'delivered')                              as is_delivered

from lines l
join orders o    on o.order_id = l.order_id
join customers c on c.customer_id = o.customer_id
left join order_review r on r.order_id = l.order_id
