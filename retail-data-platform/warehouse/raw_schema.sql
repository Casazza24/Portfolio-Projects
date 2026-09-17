-- Snowflake raw layer for the retail data platform.
-- Idempotent: run top to bottom on a fresh trial account to stand the warehouse
-- up in one shot, or re-run any time to reconcile drift.
--
-- Nothing here is modelled. The Olist tables mirror Postgres 1:1 (which mirrors
-- the Kaggle CSVs 1:1, see ingestion/olist_schema.sql) and the event/dimension
-- tables mirror what Spark wrote. Typing, renaming and business logic are dbt's
-- job in STAGING and MARTS -- keeping RAW a faithful copy of the source is what
-- makes a reload reproducible and a discrepancy attributable.

-- ---------------------------------------------------------------------------
-- Database, warehouse, schemas
-- ---------------------------------------------------------------------------

create database if not exists retail_platform;
use database retail_platform;

-- XSMALL with a 60s auto-suspend: the whole dataset is ~215 MB, so compute size
-- buys nothing here, and idle credits are the only real way to burn a trial.
create warehouse if not exists retail_wh
    warehouse_size = 'XSMALL'
    auto_suspend = 60
    auto_resume = true
    initially_suspended = true;

create schema if not exists raw;      -- landed source data, untouched
create schema if not exists staging;  -- dbt: typed, renamed, deduped
create schema if not exists marts;    -- dbt: facts and conformed dimensions

use schema raw;

-- ---------------------------------------------------------------------------
-- File formats and stage
-- ---------------------------------------------------------------------------

-- EMPTY_FIELD_AS_NULL with an empty NULL_IF is what preserves Postgres's
-- distinction between NULL and the empty string: `COPY TO ... FORMAT csv`
-- writes NULL as a bare empty field and '' as a quoted empty field, and this
-- combination reads exactly that back. Setting NULL_IF = ('') instead would
-- collapse both to NULL. COMPRESSION = AUTO covers the gzipped Olist extracts
-- and the plain-CSV dimensions with one format.
create or replace file format raw.csv_format
    type = csv
    compression = auto
    skip_header = 1
    field_optionally_enclosed_by = '"'
    empty_field_as_null = true
    null_if = ()
    trim_space = false;

create or replace file format raw.parquet_format
    type = parquet;

create stage if not exists raw.retail_stage
    file_format = raw.csv_format
    comment = 'Internal stage for PUT/COPY loads from warehouse/load_snowflake.py';

-- ---------------------------------------------------------------------------
-- Olist (from Postgres, via warehouse/extract_postgres.py)
-- ---------------------------------------------------------------------------
-- Column names, order and types match olist.<table> in Postgres exactly, so a
-- row count or a checksum can be compared across the two without a mapping.

create or replace table raw.customers (
    customer_id              varchar,
    customer_unique_id       varchar,
    customer_zip_code_prefix varchar,
    customer_city            varchar,
    customer_state           varchar
);

create or replace table raw.sellers (
    seller_id              varchar,
    seller_zip_code_prefix varchar,
    seller_city            varchar,
    seller_state           varchar
);

create or replace table raw.products (
    product_id                 varchar,
    product_category_name      varchar,
    product_name_lenght        number,
    product_description_lenght number,
    product_photos_qty         number,
    product_weight_g           number,
    product_length_cm          number,
    product_height_cm          number,
    product_width_cm           number
);

create or replace table raw.orders (
    order_id                      varchar,
    customer_id                   varchar,
    order_status                  varchar,
    order_purchase_timestamp      timestamp_ntz,
    order_approved_at             timestamp_ntz,
    order_delivered_carrier_date  timestamp_ntz,
    order_delivered_customer_date timestamp_ntz,
    order_estimated_delivery_date timestamp_ntz
);

create or replace table raw.order_items (
    order_id            varchar,
    order_item_id       number,
    product_id          varchar,
    seller_id           varchar,
    shipping_limit_date timestamp_ntz,
    price               number(12, 2),
    freight_value       number(12, 2)
);

create or replace table raw.order_payments (
    order_id             varchar,
    payment_sequential   number,
    payment_type         varchar,
    payment_installments number,
    payment_value        number(12, 2)
);

create or replace table raw.order_reviews (
    review_id               varchar,
    order_id                varchar,
    review_score            number,
    review_comment_title    varchar,
    review_comment_message  varchar,
    review_creation_date    timestamp_ntz,
    review_answer_timestamp timestamp_ntz
);

create or replace table raw.geolocation (
    geolocation_zip_code_prefix varchar,
    geolocation_lat             number(12, 6),
    geolocation_lng             number(12, 6),
    geolocation_city            varchar,
    geolocation_state           varchar
);

create or replace table raw.product_category_name_translation (
    product_category_name         varchar,
    product_category_name_english varchar
);

-- ---------------------------------------------------------------------------
-- Cleaned event stream (from Spark, data/clean/events)
-- ---------------------------------------------------------------------------
-- event_date is a Hive partition column on disk, so it is absent from the
-- Parquet files themselves; load_snowflake.py rederives it -- see the note
-- there. Kafka partition/offset are carried through because they are what makes
-- a warehouse row traceable back to a position in the topic.

create or replace table raw.events (
    event_id          varchar,
    event_time        timestamp_ntz,
    event_time_ms     number,
    event_date        date,
    visitor_id        number,
    event_type        varchar,
    item_id           number,
    transaction_id    number,
    category_id       number,
    parent_category_id number,
    root_category_id  number,
    is_available      boolean,
    has_item_metadata boolean,
    producer_id       varchar,
    ingested_at       timestamp_ntz,
    kafka_partition   number,
    kafka_offset      number
);

-- Item/category dimension the event fact joins to. Contains the -1 unknown
-- member, so the Phase 4 relationships test on category_id can run strict.
create or replace table raw.dim_items (
    item_id             number,
    category_id         number,
    parent_category_id  number,
    root_category_id    number,
    is_available        boolean,
    category_valid_from number
);

-- ---------------------------------------------------------------------------
-- Generated dimensions
-- ---------------------------------------------------------------------------

-- One row per (state, date), from ingestion/fetch_weather.py.
create or replace table raw.dim_weather (
    state               varchar,
    weather_date        date,
    latitude            number(9, 4),
    longitude           number(9, 4),
    orders_in_state     number,
    temperature_2m_mean number(6, 2),
    temperature_2m_max  number(6, 2),
    temperature_2m_min  number(6, 2),
    precipitation_sum   number(8, 2),
    rain_sum            number(8, 2),
    wind_speed_10m_max  number(6, 2),
    weather_code        number
);

-- One row per calendar day plus the -1 unknown member, from
-- warehouse/build_dim_date.py. date_day is null on the unknown row only.
create or replace table raw.dim_date (
    date_key           number,
    date_day           date,
    year               number,
    quarter            number,
    month              number,
    day_of_month       number,
    day_of_year        number,
    week_of_year       number,
    iso_year           number,
    day_of_week        number,
    day_name           varchar,
    day_abbr           varchar,
    month_name         varchar,
    month_abbr         varchar,
    year_month         varchar,
    year_quarter       varchar,
    first_day_of_month date,
    last_day_of_month  date,
    is_weekend         boolean,
    is_month_start     boolean,
    is_month_end       boolean,
    is_quarter_end     boolean,
    is_year_end        boolean
);
