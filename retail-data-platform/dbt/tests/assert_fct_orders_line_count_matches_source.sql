-- Fails when fct_orders does not have exactly one row per source order line.
--
-- fct_orders joins four tables. Three of those joins are safe by construction;
-- the reviews join is not -- some Olist orders carry more than one review, so
-- joining reviews in directly duplicates every line of those orders and
-- inflates revenue by a few percent. The model deduplicates reviews to one per
-- order to prevent that, and this test is what proves the deduplication is
-- still working after someone edits the model.
--
-- A `unique` test on order_line_key catches the same fan-out. This catches
-- something that one cannot: rows going *missing*. Change a left join to an
-- inner join and unique still passes on a fact that has quietly lost every
-- line whose order has no review.
--
-- Equality, not a tolerance. There is exactly one right answer here.

with modeled as (

    select count(*) as row_count from {{ ref('fct_orders') }}

),

source_lines as (

    select count(*) as row_count from {{ ref('stg_olist__order_items') }}

)

select
    modeled.row_count       as fct_orders_rows,
    source_lines.row_count  as source_order_item_rows
from modeled
cross join source_lines
where modeled.row_count != source_lines.row_count
