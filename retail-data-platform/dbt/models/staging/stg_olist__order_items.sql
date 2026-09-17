-- Grain: (order_id, order_item_id) -- one physical line on an order.
--
-- `order_item_id` is a per-order sequence number (1, 2, 3...), not a global
-- key, which is why the surrogate key below concatenates both parts. Two
-- different orders both have a line 1.

select
    order_id || '-' || cast(order_item_id as varchar) as order_line_key,
    order_id,
    cast(order_item_id as integer)                    as order_item_id,
    product_id,
    seller_id,

    cast(shipping_limit_date as timestamp_ntz)        as shipping_limit_at,

    -- Money as fixed-point, never float. These get summed into revenue
    -- totals, and binary floating point does not add up to what a finance
    -- person expects when you sum 100k of them.
    cast(price as number(12, 2))                      as item_price,
    cast(freight_value as number(12, 2))              as freight_value

from {{ source('olist', 'order_items') }}
