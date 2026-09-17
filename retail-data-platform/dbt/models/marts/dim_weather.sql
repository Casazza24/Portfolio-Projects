-- Grain: one row per (state, date).
--
-- The key is a concatenation rather than a hash. fct_orders joins on it, and
-- 'SP-2017-03-14' can be read straight off a failing test or a Power BI
-- tooltip; an md5 has to be reverse-engineered before anyone can say which
-- state and day went wrong. Hashing earns its place when the key parts are
-- many or wide -- two short columns is neither.

with weather as (

    select * from {{ ref('stg_open_meteo__weather') }}

)

select
    state || '-' || to_char(weather_date, 'YYYY-MM-DD') as weather_key,

    state,
    weather_date,
    latitude,
    longitude,

    temp_mean_c,
    temp_max_c,
    temp_min_c,
    precipitation_mm,
    rain_mm,
    wind_speed_max_kmh,
    weather_code,

    -- 1mm, not "> 0". Sub-millimetre readings are drizzle or measurement
    -- noise; calling those a rainy day makes ~40% of days in the southeast
    -- rainy and destroys the contrast the "did revenue dip in bad weather"
    -- question depends on.
    (rain_mm >= 1.0)                                   as is_rainy_day,
    (rain_mm >= 20.0)                                  as is_heavy_rain_day,

    -- WMO weather codes, banded. The raw code looks ordinal and is not, so
    -- charting it directly implies an ordering that does not exist.
    case
        when weather_code between 0 and 3   then 'clear_or_cloudy'
        when weather_code between 45 and 48 then 'fog'
        when weather_code between 51 and 57 then 'drizzle'
        when weather_code between 61 and 67 then 'rain'
        when weather_code between 71 and 77 then 'snow'
        when weather_code between 80 and 82 then 'rain_showers'
        when weather_code between 95 and 99 then 'thunderstorm'
        else 'other'
    end                                                as weather_condition

from weather
