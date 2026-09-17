-- Not consumed by any mart: the weather dimension's state centroids are
-- computed upstream in ingestion/fetch_weather.py, against Postgres, because
-- the API pull needs the coordinates before anything reaches Snowflake.
--
-- Modeled anyway, and kept deliberately: it makes the provenance of
-- dim_weather's coordinates visible in the lineage graph instead of appearing
-- to arrive from nowhere, and it is the table any "where do our customers
-- actually live" question would start from. The out-of-Brazil rows are
-- flagged, not deleted -- the same bounding box the ingestion script clips to.

select
    lpad(cast(geolocation_zip_code_prefix as varchar), 5, '0') as zip_code_prefix,
    cast(geolocation_lat as number(11, 7))                     as latitude,
    cast(geolocation_lng as number(11, 7))                     as longitude,
    initcap(trim(geolocation_city))                            as city,
    upper(trim(geolocation_state))                             as state,

    (
        cast(geolocation_lat as number(11, 7)) between -34.0 and 5.5
        and cast(geolocation_lng as number(11, 7)) between -74.0 and -34.0
    ) as is_within_brazil

from {{ source('olist', 'geolocation') }}
