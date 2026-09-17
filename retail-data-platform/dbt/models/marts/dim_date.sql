-- Generated, not loaded. A calendar has no upstream owner and no failure mode,
-- so making it a hand-loaded raw table only adds a file to keep in sync with
-- the fact tables' actual span.
--
-- ponytail: dbt_utils.date_spine would do this in one macro call, and this is
-- the only place in the project that would have used dbt_utils. Snowflake's
-- native GENERATOR does the same job in three lines, so the package would be
-- carrying its own version pinning, `dbt deps` step and upgrade surface to
-- save two. If a second use appeared, adding it would be the right call.
--
-- The key is the DATE itself, not an integer like 20160904. Integer date keys
-- exist because 1990s row stores compared 4-byte ints faster than dates;
-- Snowflake stores a DATE in 4 bytes and prunes micro-partitions on it
-- natively, so the integer buys nothing and costs readability plus a cast in
-- every ad-hoc query.

{%- set start_date = var('date_spine_start') -%}
{%- set end_date = var('date_spine_end') -%}
{%- set day_count = (
        modules.datetime.datetime.strptime(end_date, '%Y-%m-%d')
        - modules.datetime.datetime.strptime(start_date, '%Y-%m-%d')
    ).days + 1 %}

with spine as (

    select dateadd(day, seq4(), to_date('{{ start_date }}')) as date_day
    from table(generator(rowcount => {{ day_count }}))

)

select
    date_day,

    year(date_day)                                  as calendar_year,
    quarter(date_day)                               as calendar_quarter,
    month(date_day)                                 as calendar_month,
    monthname(date_day)                             as month_name,
    to_char(date_day, 'YYYY-MM')                    as year_month,

    day(date_day)                                   as day_of_month,
    dayofweekiso(date_day)                          as day_of_week,   -- 1 = Monday
    dayname(date_day)                               as day_name,
    weekiso(date_day)                               as week_of_year,
    dayofyear(date_day)                             as day_of_year,

    (dayofweekiso(date_day) in (6, 7))              as is_weekend,
    (date_day = last_day(date_day))                 as is_month_end,

    date_trunc('week', date_day)                    as week_start_date,
    date_trunc('month', date_day)                   as month_start_date

from spine
