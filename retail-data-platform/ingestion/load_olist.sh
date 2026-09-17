#!/usr/bin/env bash
# Loads the raw Olist Kaggle CSVs into the local Postgres instance (olist schema).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
DATA_DIR="$PROJECT_ROOT/data/raw/olist"

export PGHOST=localhost
export PGPORT=5432
export PGUSER=retail
export PGPASSWORD=retail
export PGDATABASE=retail

psql -v ON_ERROR_STOP=1 -f "$SCRIPT_DIR/olist_schema.sql"

copy_csv() {
    local table=$1
    local file=$2
    psql -v ON_ERROR_STOP=1 -c "\copy $table from '$DATA_DIR/$file' with (format csv, header true, encoding 'UTF8')"
}

copy_csv olist.customers olist_customers_dataset.csv
copy_csv olist.sellers olist_sellers_dataset.csv
copy_csv olist.products olist_products_dataset.csv
copy_csv olist.orders olist_orders_dataset.csv
copy_csv olist.order_items olist_order_items_dataset.csv
copy_csv olist.order_payments olist_order_payments_dataset.csv
copy_csv olist.order_reviews olist_order_reviews_dataset.csv
copy_csv olist.geolocation olist_geolocation_dataset.csv
copy_csv olist.product_category_name_translation product_category_name_translation.csv

echo "Row counts:"
psql -v ON_ERROR_STOP=1 -c "
select 'customers' as table_name, count(*) from olist.customers
union all select 'sellers', count(*) from olist.sellers
union all select 'products', count(*) from olist.products
union all select 'orders', count(*) from olist.orders
union all select 'order_items', count(*) from olist.order_items
union all select 'order_payments', count(*) from olist.order_payments
union all select 'order_reviews', count(*) from olist.order_reviews
union all select 'geolocation', count(*) from olist.geolocation
union all select 'category_translation', count(*) from olist.product_category_name_translation
order by 1;
"
