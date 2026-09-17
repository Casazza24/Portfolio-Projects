-- Nulls in the later lifecycle timestamps are left as nulls. An order that
-- was cancelled before shipping genuinely has no delivery date; coalescing to
-- a sentinel date here would make it look delivered on 1970-01-01 in every
-- downstream average.

select
    order_id,
    customer_id,
    lower(trim(order_status))                                as order_status,

    cast(order_purchase_timestamp as timestamp_ntz)          as ordered_at,
    cast(order_approved_at as timestamp_ntz)                 as approved_at,
    cast(order_delivered_carrier_date as timestamp_ntz)      as delivered_to_carrier_at,
    cast(order_delivered_customer_date as timestamp_ntz)     as delivered_to_customer_at,
    cast(order_estimated_delivery_date as timestamp_ntz)     as estimated_delivery_at

from {{ source('olist', 'orders') }}
