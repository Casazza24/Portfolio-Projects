-- Fails when more category ids are referenced by items than category_tree.csv
-- defines. Phase 3 measured exactly 132.
--
-- These orphans are the reason dim_items keeps null parent/root instead of
-- folding them into the -1 unknown member: the two gaps have different
-- upstream owners. Tolerating a known gap is only defensible if the size of it
-- is pinned -- otherwise "we decided to allow nulls there" becomes cover for
-- an unbounded and growing hole, and nobody notices when 132 becomes 1,300.
--
-- Distinct categories, not items: one newly-undefined category can affect
-- thousands of items, and the defect is in the tree file, not in the items.

with orphans as (

    select distinct category_id
    from {{ ref('dim_items') }}
    where category_id is not null
      and root_category_id is null

)

select
    count(*)                              as orphan_category_count,
    {{ var('orphan_category_max') }}      as expected_max
from orphans
having count(*) > {{ var('orphan_category_max') }}
