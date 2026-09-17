-- GRAIN: one row per clickstream event -- unique on event_id, which the
-- Phase 3 Spark layer already guarantees by deduplicating at-least-once
-- redeliveries. The `unique` test here is not redundant with that: it is what
-- catches the load itself double-copying a Parquet file into Snowflake, a
-- failure the Spark job cannot see.
--
-- Factless: the measure is the row. Funnel conversion is count(*) filtered by
-- event_type, so there is no numeric column to carry.
--
-- item_key vs item_id -- the unknown-member resolution:
--   item_id  is the real RetailRocket item, always populated, joins to
--            nothing when the item has no metadata.
--   item_key is the surrogate that joins to dim_items, resolving to -1 for
--            the 9.3% of events whose item never appeared in the metadata
--            feed.
-- Keeping both means the relationships test on item_key runs strict (no
-- `where item_key is not null` escape hatch that would stop it ever catching
-- a genuinely broken join), while the real item id survives for anyone who
-- needs to chase an individual event upstream.
--
-- Category is NOT carried here -- it comes from dim_items via item_key. The
-- Spark layer put it on the event because a broadcast join was the only way to
-- resolve the sentinel inside a stream; in the warehouse it would just be a
-- second copy of the dimension's answer, free to drift the first time an item
-- is re-categorised.

with events as (

    select * from {{ ref('stg_retailrocket__events') }}

)

select
    event_id,

    case when has_item_metadata then item_id else -1 end as item_key,
    item_id,

    event_date,
    event_at,

    visitor_id,
    event_type,

    -- The funnel's step order. event_type is a nominal string and sorts
    -- alphabetically in every BI tool -- addtocart, transaction, view -- which
    -- draws the funnel with the outcome in the middle and the entry point
    -- last. Fixing that is a different, tool-specific manoeuvre in each of
    -- Power BI, Tableau and Metabase, so the order belongs in the model where
    -- there is one of it. It is a sort key, never a measure: summing it is
    -- meaningless.
    case event_type
        when 'view'        then 1
        when 'addtocart'   then 2
        when 'transaction' then 3
    end                                 as funnel_step,

    -- Populated only on `transaction` events. Degenerate: RetailRocket ships
    -- no transaction table, so there is nothing to join it to. It is still the
    -- only way to group the lines of one basket together.
    transaction_id,

    has_item_metadata,

    -- Load provenance. Kept on the fact because when a duplicate or a gap
    -- shows up, the first question is which partition and offset it came from,
    -- and that answer is unrecoverable if it is dropped here.
    ingested_at,
    kafka_partition,
    kafka_offset

from events
