-- GRAIN: one row per payment instrument on an order --
-- (order_id, payment_sequential). An order paid half on a voucher and half on
-- a card is two rows.
--
-- Separate from fct_orders because the grains genuinely differ. Folding
-- payments into the line fact would require either allocating each payment
-- across lines (inventing a split the source does not record) or repeating the
-- order total on every line (a column that is wrong the moment anyone sums
-- it). A second fact at its own grain costs one more model and keeps
-- payment_value additive.
--
-- order_date is denormalised on so that "revenue by payment type over time"
-- does not need a join back to fct_orders -- a date is 4 bytes and this is the
-- only fact that would otherwise have no date key of its own.

with payments as (

    select * from {{ ref('stg_olist__order_payments') }}

),

orders as (

    select * from {{ ref('stg_olist__orders') }}

)

select
    p.order_payment_key,
    p.order_id,
    p.payment_sequential,

    cast(o.ordered_at as date) as order_date,
    o.customer_id,

    p.payment_type,
    p.payment_installments,
    p.payment_value

from payments p
join orders o on o.order_id = p.order_id
