-- Grain: (order_id, payment_sequential) -- one payment instrument per row.

select
    order_id || '-' || cast(payment_sequential as varchar) as order_payment_key,
    order_id,
    cast(payment_sequential as integer)                    as payment_sequential,
    lower(trim(payment_type))                              as payment_type,

    -- A one-off payment is recorded as 1 installment in some rows and 0 in
    -- others. Both mean the same thing; normalising here stops "average
    -- installments" being dragged down by a data entry convention.
    greatest(cast(payment_installments as integer), 1)     as payment_installments,

    cast(payment_value as number(12, 2))                   as payment_value

from {{ source('olist', 'order_payments') }}
