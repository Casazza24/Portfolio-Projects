# Does Weather Actually Move the Subway? A NYC Ridership Post-Mortem

I started this project with a simple, plausible question: **does bad weather keep New Yorkers off the subway?** It seemed like an easy win — everyone *knows* people stay home when it rains. What I found instead is a small case study in why "everyone knows" and "the data says" are not the same thing, and why a good-looking result on a small sample is exactly the moment to distrust yourself.

## Why this question matters

For a system that moves ~4 million riders on a normal weekday, even a few percent of demand swing is hundreds of thousands of trips. If weather genuinely drove ridership, that would be operationally useful: you could pre-position service, staff stations, and manage crowding around the forecast. But if the real drivers are calendar structure — which day of the week it is, whether it's a holiday — then planning around the weather is chasing noise. The whole point of the analysis is to figure out which lever is real and which just *looks* real.

## The data

Two public, real-world sources, both pulled fresh via API into local CSVs:

- **MTA Open Data (Socrata / data.ny.gov)** — daily subway ridership totals, plus a separate monthly dataset of delay incidents broken out by line and cause.
- **NOAA Climate Data Online** — daily weather for the Central Park station (precipitation, max/min temperature).

I joined ridership to weather on date across a **13-month window (May 2025 – May 2026, 396 daily observations)**. The delay data is monthly, so it lives as a separate, secondary analysis rather than something I could bolt onto the daily model — a limitation I'll come back to.

## The trap: a naive rain comparison

My first pass compared average ridership on rainy days versus dry days. On the initial two-month sample this looked like a slam dunk: rain days ran roughly **9% below** dry days. That is a big, headline-worthy number, and it's exactly the kind of finding that's tempting to write up and walk away from.

Two things stopped me. First, when I expanded to the full year, that gap shrank to **-3.5%** (correlation between ridership and precipitation of just -0.13). Second, and more importantly: rain days don't fall evenly across the week. If it happens to rain on a few extra weekends, a naive rain-vs-dry comparison quietly absorbs the *weekend* drop and mislabels it as a *weather* drop. The rain effect and the day-of-week effect were tangled together, and I had no way to tell how much of my -9%/-3.5% was actually about rain.

## Untangling it: day-of-week is the real story

So I looked at the confound directly. The day-of-week pattern turned out to dwarf everything else:

| Split | Mean daily ridership |
|---|---|
| Weekday | ~4.00M |
| Weekend | ~2.44M |
| **Gap** | **-39%** |

A weekend loses roughly *1.5 million riders* versus a weekday. Against a number that large, a rain effect of a few percent is a rounding error. Season mattered too, but far less — ridership peaked in April (~3.76M) and bottomed out in January (~3.23M), a **16.5%** peak-to-trough swing. Season nudges the baseline; day-of-week sets it.

The moment I re-ran the rain comparison **within weekdays only** — removing the weekend confound — the rain effect collapsed to about **-1%**. Most of that scary original number was never about weather at all.

## The honest version: one model, everything at once

Comparing factors one at a time still isn't enough, because they're all correlated (cold days cluster in low-ridership winter months, and so on). The clean way to ask "what does rain do, *holding everything else constant*" is a single regression with every factor in it at once:

`ridership ~ weekday + month + rain_day + max_temperature + holiday`

This model explains **88% of the daily variance (R² = 0.88, n = 396)**. Ranked by absolute effect on daily ridership:

| Driver | Effect (riders/day) | Significant? |
|---|---|---|
| Sunday | **-1.76M** | *** |
| **Holiday** | **-1.52M** | *** |
| Saturday | -1.16M | *** |
| Month / season shifts | ±250k–370k | mostly *** |
| Rain (any measurable) | **-123k** (~3% of a weekday) | *** |
| Max temperature | +2,751 / °F | **no** (p ≈ 0.14) |

Two things jump out. The **surprise is holidays**: a federal holiday drops ridership by ~1.5M — landing *between* Saturday and Sunday in magnitude. A holiday behaves like an extra weekend day dropped into the workweek, and none of the one-factor-at-a-time views could see it. And the original hero of the story, weather, is relegated to the edges: rain is real but small (-123k/day, a genuine ~3% weekday dip), and **temperature isn't statistically distinguishable from zero** once season is in the model. The intuition that "cold or hot days empty the subway" simply doesn't survive the controls — rain beats temperature, and the calendar beats both.

## The side quest: which lines break down, and why

Separately, I ranked subway lines by total delay incidents over the trailing 12 months. The worst offenders were the **6, 2, N, A, and F**. The single largest cause system-wide was **Police & Medical incidents (20,415)**, ahead of **Infrastructure & Equipment (17,220)** — a useful reminder that a lot of "delays" are people-and-emergency events, not just aging tracks and signals.

## Event spikes

Because the model tells me what each weekday *normally* looks like, I could flag the days that ran far above their own weekday's average — candidate event days. Rather than guess, I surfaced them as a table to cross-reference against the real NYC calendar (Halloween on Oct 31 showed up, along with clusters in November, December, and May that likely map to playoff runs and big events). A spike on a *rainy* day is doubly notable, since rain usually suppresses ridership.

## What this means, and what it doesn't

The clean takeaway: **calendar structure drives NYC subway ridership; weather only trims the edges.** If you're planning capacity, the day of the week and the holiday calendar are worth far more than the forecast.

A few honest limitations. This is **subway-wide daily total** ridership — it says nothing about individual stations or lines, where weather could plausibly bite harder (think outdoor platforms or beach-bound routes in summer). The **delay analysis is monthly**, so I couldn't join it to the daily model; linking specific delays to same-day ridership would need finer data. And a regression coefficient is an *association*, not proof of cause — the holiday effect, for instance, bundles together tourism, school schedules, and remote work, which this data can't pull apart.

*For the full interactive breakdown — day-of-week, seasonal, and driver-ranking charts you can explore directly — see the accompanying dashboard (`dashboard.html`).*
