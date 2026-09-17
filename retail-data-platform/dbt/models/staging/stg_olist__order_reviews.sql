-- Grain: one row per review. NOT one row per order -- some orders carry more
-- than one review, and `review_id` itself repeats where a single review was
-- filed against multiple orders. Neither column is a key on its own, so this
-- model deliberately exposes no surrogate key: the deduplication decision
-- belongs to the mart that needs one order-shaped answer (fct_orders), where
-- the choice of *which* review wins can be stated explicitly.

select
    review_id,
    order_id,
    cast(review_score as integer)                    as review_score,

    nullif(trim(review_comment_title), '')           as review_comment_title,
    nullif(trim(review_comment_message), '')         as review_comment_message,

    cast(review_creation_date as timestamp_ntz)      as review_created_at,
    cast(review_answer_timestamp as timestamp_ntz)   as review_answered_at

from {{ source('olist', 'order_reviews') }}
