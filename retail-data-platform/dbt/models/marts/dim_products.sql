-- Grain: one row per product_id (Olist catalogue).
--
-- No unknown-member row here, unlike dim_items. Postgres enforces
-- order_items.product_id -> products.product_id, so an unresolvable product
-- key is impossible by construction rather than merely unobserved. A sentinel
-- row nothing can point at is clutter that makes a reader think orphans exist.
-- The strict relationships test on fct_orders.product_id is the guard.
--
-- Two real gaps in the source, both handled by falling back rather than
-- nulling, because Power BI drops nulls from slicers and a dropped slice makes
-- filtered totals disagree with the unfiltered total for no visible reason:
--   * ~600 products carry no category at all           -> 'unknown'
--   * a couple of categories are missing from the
--     Portuguese->English translation table            -> keep the Portuguese

with products as (

    select * from {{ ref('stg_olist__products') }}

),

categories as (

    select * from {{ ref('stg_olist__product_categories') }}

)

select
    p.product_id,

    coalesce(p.product_category_name, 'unknown')            as product_category_name,
    coalesce(
        c.product_category_name_english,
        p.product_category_name,
        'unknown'
    )                                                       as product_category,

    (p.product_category_name is null)                       as is_uncategorised,

    p.product_weight_g,
    p.product_length_cm,
    p.product_height_cm,
    p.product_width_cm,

    -- Null propagates if any dimension is missing, which is the honest answer:
    -- a product with no recorded height has no known volume, and coalescing
    -- the missing side to zero would report a volume of 0 cm3 for a real box.
    p.product_length_cm * p.product_height_cm * p.product_width_cm
                                                            as product_volume_cm3,

    p.product_photos_qty,
    p.product_name_length,
    p.product_description_length

from products p
left join categories c on c.product_category_name = p.product_category_name
