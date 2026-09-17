-- Grain: one row per item_id, including the synthetic item_id = -1
-- unknown-member row. That row is not filtered out anywhere in this project:
-- it is what makes fct_events.item_key resolvable for the 9.3% of events whose
-- item never appeared in RetailRocket's metadata feed, and therefore what lets
-- the relationships test on that key run strict instead of tolerating nulls.
--
-- Current-state, not point-in-time: the upstream builder takes the latest
-- known value per (item, property) from an EAV change-log. Events before a
-- re-categorisation are attributed to the item's newer category. See
-- docs/data_quality.md 3.5.

select
    item_id,
    category_id,

    -- Null for the 132 category ids that items reference but category_tree.csv
    -- never defines. Left null rather than folded into the -1 sentinel -- see
    -- the model description in _staging.yml for why those are different
    -- defects with different owners.
    parent_category_id,
    root_category_id,

    is_available,

    -- Milliseconds since epoch in the source, because the whole RetailRocket
    -- feed is millisecond-based. When the item's current category took effect,
    -- which is how stale a given row's category can be shown to be.
    to_timestamp_ntz(category_valid_from, 3) as category_valid_from

from {{ source('retailrocket', 'dim_items') }}
