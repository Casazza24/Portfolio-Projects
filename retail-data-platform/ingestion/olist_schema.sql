-- Olist Brazilian E-Commerce dataset — raw OLTP-style schema
-- Mirrors the Kaggle CSVs 1:1; cleanup/typing happens later in dbt staging models.

create schema if not exists olist;

drop table if exists olist.order_reviews;
drop table if exists olist.order_payments;
drop table if exists olist.order_items;
drop table if exists olist.orders;
drop table if exists olist.customers;
drop table if exists olist.products;
drop table if exists olist.sellers;
drop table if exists olist.geolocation;
drop table if exists olist.product_category_name_translation;

create table olist.customers (
    customer_id varchar primary key,
    customer_unique_id varchar,
    customer_zip_code_prefix varchar,
    customer_city varchar,
    customer_state varchar
);

create table olist.sellers (
    seller_id varchar primary key,
    seller_zip_code_prefix varchar,
    seller_city varchar,
    seller_state varchar
);

create table olist.products (
    product_id varchar primary key,
    product_category_name varchar,
    product_name_lenght numeric,
    product_description_lenght numeric,
    product_photos_qty numeric,
    product_weight_g numeric,
    product_length_cm numeric,
    product_height_cm numeric,
    product_width_cm numeric
);

create table olist.orders (
    order_id varchar primary key,
    customer_id varchar references olist.customers(customer_id),
    order_status varchar,
    order_purchase_timestamp timestamp,
    order_approved_at timestamp,
    order_delivered_carrier_date timestamp,
    order_delivered_customer_date timestamp,
    order_estimated_delivery_date timestamp
);

create table olist.order_items (
    order_id varchar references olist.orders(order_id),
    order_item_id integer,
    product_id varchar references olist.products(product_id),
    seller_id varchar references olist.sellers(seller_id),
    shipping_limit_date timestamp,
    price numeric,
    freight_value numeric,
    primary key (order_id, order_item_id)
);

create table olist.order_payments (
    order_id varchar references olist.orders(order_id),
    payment_sequential integer,
    payment_type varchar,
    payment_installments integer,
    payment_value numeric,
    primary key (order_id, payment_sequential)
);

create table olist.order_reviews (
    review_id varchar,
    order_id varchar references olist.orders(order_id),
    review_score integer,
    review_comment_title varchar,
    review_comment_message text,
    review_creation_date timestamp,
    review_answer_timestamp timestamp
);

create table olist.geolocation (
    geolocation_zip_code_prefix varchar,
    geolocation_lat numeric,
    geolocation_lng numeric,
    geolocation_city varchar,
    geolocation_state varchar
);

create table olist.product_category_name_translation (
    product_category_name varchar primary key,
    product_category_name_english varchar
);
