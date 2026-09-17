-- Grain: one row per event_id. Already deduped and validated by the Phase 3
-- Spark layer, so this model does not re-clean anything -- it renames, and it
-- drops columns that would otherwise give the warehouse two answers to the
-- same question.
--
-- Dropped on purpose:
--   event_time_ms       -- the same instant as event_at, in a second unit.
--                          Two representations of one value is how they drift.
--   category_id,
--   parent_category_id,
--   root_category_id,
--   is_available        -- these arrived on the event because Spark broadcast
--                          the item dimension into the stream to resolve the
--                          unknown-member sentinel. They are attributes of the
--                          *item*, not of the event, and they live in
--                          dim_items. Carrying them on the fact as well means
--                          a re-categorisation upstream leaves the fact
--                          disagreeing with the dimension, permanently.
--   producer_id         -- pipeline provenance, not an analytical attribute.
--
-- `has_item_metadata` is kept: whether an event's item resolved is a property
-- of the event, and it is the column the pipeline-health page tracks.

select
    event_id,
    cast(event_time as timestamp_ntz)   as event_at,
    cast(event_date as date)            as event_date,

    cast(visitor_id as number(18, 0))   as visitor_id,
    lower(trim(event_type))             as event_type,
    cast(item_id as integer)            as item_id,

    -- Only populated on `transaction` events. Not a foreign key to anything in
    -- Olist despite the name -- the two datasets describe different retailers.
    cast(transaction_id as number(18, 0)) as transaction_id,

    cast(has_item_metadata as boolean)  as has_item_metadata,

    cast(ingested_at as timestamp_ntz)  as ingested_at,
    cast(kafka_partition as integer)    as kafka_partition,
    cast(kafka_offset as number(18, 0)) as kafka_offset

from {{ source('retailrocket', 'events') }}
