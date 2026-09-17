-- Grain: one row per (state, date).
--
-- `orders_in_state` is dropped: the fetch script carries it only to order its
-- API batches by importance. It is an order count masquerading as a weather
-- attribute, and leaving it here would let someone sum it across 774 days and
-- get a number 774x too big.

select
    upper(trim(state))                              as state,
    cast(weather_date as date)                      as weather_date,

    cast(latitude as number(11, 7))                 as latitude,
    cast(longitude as number(11, 7))                as longitude,

    cast(temperature_2m_mean as number(6, 2))       as temp_mean_c,
    cast(temperature_2m_max as number(6, 2))        as temp_max_c,
    cast(temperature_2m_min as number(6, 2))        as temp_min_c,

    cast(precipitation_sum as number(8, 2))         as precipitation_mm,
    cast(rain_sum as number(8, 2))                  as rain_mm,
    cast(wind_speed_10m_max as number(6, 2))        as wind_speed_max_kmh,

    -- WMO code. An integer that looks ordinal but isn't -- 95 (thunderstorm)
    -- is not "more" than 71 (snow). Decoded into bands in dim_weather.
    cast(weather_code as integer)                   as weather_code

from {{ source('open_meteo', 'dim_weather') }}
