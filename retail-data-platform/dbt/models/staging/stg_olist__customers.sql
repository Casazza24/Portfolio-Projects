-- Renaming and typing only. The customer_id / customer_unique_id distinction
-- is preserved exactly as the source has it; collapsing to one key here would
-- destroy the only signal repeat-purchase analysis has.

select
    customer_id,
    customer_unique_id,

    -- Zip prefixes are identifiers, not quantities: leading zeros matter and
    -- arithmetic on them never does. Kept as text and left-padded, because a
    -- CSV load will have already eaten the zeros on some of them.
    lpad(cast(customer_zip_code_prefix as varchar), 5, '0') as customer_zip_code_prefix,

    initcap(trim(customer_city))                            as customer_city,
    upper(trim(customer_state))                             as customer_state

from {{ source('olist', 'customers') }}
