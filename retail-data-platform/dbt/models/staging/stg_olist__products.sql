-- The source column names `product_name_lenght` / `product_description_lenght`
-- are misspelled in the Kaggle CSVs. Corrected here rather than in the load,
-- so the raw layer stays a byte-for-byte mirror of what arrived and every
-- rename is visible in one diffable place.

select
    product_id,
    nullif(trim(product_category_name), '')          as product_category_name,

    cast(product_name_lenght as integer)             as product_name_length,
    cast(product_description_lenght as integer)      as product_description_length,
    cast(product_photos_qty as integer)              as product_photos_qty,

    cast(product_weight_g as number(12, 2))          as product_weight_g,
    cast(product_length_cm as number(12, 2))         as product_length_cm,
    cast(product_height_cm as number(12, 2))         as product_height_cm,
    cast(product_width_cm as number(12, 2))          as product_width_cm

from {{ source('olist', 'products') }}
