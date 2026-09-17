-- Grain: one row per seller_id.
--
-- Thin, and kept anyway: without it, fct_orders.seller_id is a key pointing at
-- nothing, which means no relationships test covers it and a broken seller
-- sync would surface as a blank slicer rather than a failing build.

select
    seller_id,
    seller_city,
    seller_state,
    seller_zip_code_prefix

from {{ ref('stg_olist__sellers') }}
