-- Fails when the share of events whose item has no metadata leaves the band
-- measured in Phase 3 (9.27% -- docs/data_quality.md 3.4).
--
-- This is the test that earns its keep. The unknown-member design means a
-- broken upstream metadata feed produces no nulls, no orphan keys and no
-- failing relationships test -- every event still resolves, just to -1. The
-- pipeline stays green while `revenue by category` quietly moves from real
-- categories into an Unknown bucket. Rate-of-unknowns is the only signal that
-- moves, so it is the one worth asserting on.
--
-- Bounded on both sides on purpose. A rate that collapses to near zero is not
-- good news either: it means the join key changed shape, or the events table
-- was loaded from a filtered extract.
--
-- Compared against a pinned baseline rather than against the previous run,
-- because the previous run is not available to a stateless `dbt test`, and
-- because a slow drift is exactly what a run-over-run comparison misses.

with measured as (

    select
        count_if(not has_item_metadata)::float / nullif(count(*), 0) as unknown_item_rate,
        count(*)                                                    as event_count
    from {{ ref('fct_events') }}

)

select
    unknown_item_rate,
    event_count,
    {{ var('unknown_item_rate_min') }} as expected_min,
    {{ var('unknown_item_rate_max') }} as expected_max
from measured
where unknown_item_rate not between {{ var('unknown_item_rate_min') }}
                                and {{ var('unknown_item_rate_max') }}
