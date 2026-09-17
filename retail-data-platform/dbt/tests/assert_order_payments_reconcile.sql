{{ config(severity='error', warn_if='>0', error_if='>500') }}

-- Warns when what an order was paid does not match what its lines cost.
--
-- The two facts are built from independent source tables and joined to nothing
-- in common, so a fan-out or a dropped join in either one moves this apart --
-- it is the cross-check that neither fact's own tests can perform on itself.
--
-- Thresholded rather than all-or-nothing, because Olist is a real operational
-- system and a residue of genuinely mismatched orders exists in the source:
-- partial refunds, order-level discounts and vouchers applied outside the line
-- total. A handful of those is business reality; hundreds means the model
-- broke. `warn_if: >0` surfaces every one of them, `error_if: >500` is where it
-- stops being data and starts being a bug -- so the test still fails the build
-- when it should, instead of being a warning everyone learns to scroll past.
-- The 500 is a first guess and should be re-set from the real number the first
-- time this runs against Snowflake.
--
-- The 2% band is on the ratio, not an absolute cent difference: a R$3 gap on a
-- R$2,000 order is rounding, the same R$3 gap on a R$12 order is not.
-- Cancelled and unavailable orders are excluded -- an order that never shipped
-- has no reason to have been paid in full, and including them would fill the
-- output with rows that are correct.

with line_totals as (

    select
        order_id,
        sum(item_revenue) as line_total
    from {{ ref('fct_orders') }}
    where order_status not in ('canceled', 'unavailable')
    group by 1

),

payment_totals as (

    select
        order_id,
        sum(payment_value) as payment_total
    from {{ ref('fct_order_payments') }}
    group by 1

)

select
    l.order_id,
    l.line_total,
    p.payment_total,
    p.payment_total - l.line_total as difference
from line_totals l
join payment_totals p on p.order_id = l.order_id
where abs(p.payment_total - l.line_total) > 0.02 * l.line_total
