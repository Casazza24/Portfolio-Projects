-- Grain: one row per item_id in the RetailRocket catalogue, plus one synthetic
-- unknown-member row at item_id = -1.
--
-- The category hierarchy is carried here as attributes rather than snowflaked
-- into a dim_categories. Category has no attributes of its own in this source
-- beyond its own id and its parent's -- a dimension of nothing but keys is a
-- join that answers no question.
--
-- ORPHAN CATEGORIES -- the decision this model exists to make explicit.
-- 132 category ids are referenced by items but never defined in
-- category_tree.csv, so those items have a real category_id and no resolvable
-- parent or root. Two options:
--
--   (a) roll them into the -1 unknown member alongside items that have no
--       metadata at all;
--   (b) leave parent/root null and flag them.
--
-- (b) is chosen. (a) conflates two different defects with two different
-- upstream owners: "the item_properties feed never described this item" and
-- "the category_tree file is incomplete". Merged into one bucket, a fix to
-- either one is invisible, and a regression in either looks like a regression
-- in the other. Nulls are tolerable *here specifically* because parent_category
-- and root_category are descriptive attributes of a dimension row, not
-- foreign keys leaving a fact table -- nothing joins on them, so no
-- relationships test has to be weakened to accommodate them, which is exactly
-- the argument that made the -1 sentinel right for the fact keys.
--
-- The count is pinned at 132 by tests/assert_orphan_categories_bounded.sql, so
-- if the tree file degrades further the build says so.

with items as (

    select * from {{ ref('stg_retailrocket__items') }}

)

select
    item_id,

    category_id,
    parent_category_id,
    root_category_id,

    -- False for the 132 orphan categories described above, and for the -1
    -- unknown member. The column that lets an analyst exclude them as a
    -- deliberate choice instead of discovering the gap in a total that does
    -- not add up.
    (root_category_id is not null)              as has_category_hierarchy,

    (item_id = -1)                              as is_unknown_item,

    is_available,
    category_valid_from

from items
