-- Grain: one row per customer_id -- which in Olist is one row per *order*,
-- not per person. That is the source's design, and the dimension has to keep
-- it, because customer_id is the key fct_orders carries.
--
-- The person-level facts (how many orders, first/last purchase, repeat status)
-- are computed against customer_unique_id and then repeated onto every
-- customer_id belonging to that person. So `orders_by_person` is correct on
-- any single row and must never be summed across rows -- summing it counts
-- each order once per order the person ever placed. It is here rather than
-- left to Power BI so that "repeat customer" has one definition in one place;
-- the alternative (a separate person-grain dimension, snowflaked behind this
-- one) is more correct and more joins, and buys nothing at this size.

with customers as (

    select * from {{ ref('stg_olist__customers') }}

),

orders as (

    select * from {{ ref('stg_olist__orders') }}

),

per_person as (

    select
        c.customer_unique_id,
        count(distinct o.order_id) as orders_by_person,
        min(o.ordered_at)          as first_ordered_at,
        max(o.ordered_at)          as last_ordered_at
    from customers c
    join orders o on o.customer_id = c.customer_id
    group by 1

)

select
    c.customer_id,
    c.customer_unique_id,

    c.customer_city,
    c.customer_state,
    c.customer_zip_code_prefix,

    -- Zero for a customer_id that exists but has no order. Olist mints
    -- customer_ids per order so this should be empty; coalesce rather than
    -- assume, because an inner join here would silently delete such a row and
    -- break the relationships test on fct_orders instead of reporting it.
    coalesce(p.orders_by_person, 0)     as orders_by_person,
    (coalesce(p.orders_by_person, 0) > 1) as is_repeat_customer,

    p.first_ordered_at,
    p.last_ordered_at

from customers c
left join per_person p on p.customer_unique_id = c.customer_unique_id
