# Methodology (Free-Data CC CAPE Proxy)

Date: 2026-02-13

This document describes the methodology implemented in this repository for a **free-data proxy** of Current Constituents CAPE (CC CAPE). It is designed for internal research and prototyping, not as a production valuation benchmark.

## 1) Definitions

- **Company CAPE** (proxy): `price / avg(real EPS over lookback window)`
- **CC CAPE** (proxy): weighted average of company CAPE across the **current constituent set**
- **Shiller CAPE** (benchmark proxy): pulled from an external published time series
- **CAPE Spread**: `CC CAPE - Shiller CAPE`

## 2) Universe (Current Constituents)

- Universe is the latest ingested S&P 500 constituent snapshot from Wikipedia.
- Stored in: `sp500_constituents` keyed by `(symbol, as_of_date)`.

## 3) Prices

- Source: Stooq daily CSV endpoint.
- For “latest” runs: uses the latest available daily close per symbol.
- For monthly backfill: uses the last close on or before the month-end observation date.

Stored in: `daily_prices`.

## 4) Earnings (EPS)

- Source: SEC XBRL company facts (`company_facts_values`).
- Tags used (preference by availability):
  - `EarningsPerShareBasic`
  - `EarningsPerShareDiluted`
- Annual-ish filtering:
  - SEC company facts often include multiple contexts for the same `end_date` (quarter, YTD, annual)
  - filter to "annual-ish" periods by requiring `start_date` and `end_date` span at least ~11 months (`period_days >= 330`)
  - dedupe by `(tag, end_date)` and keep the row with the **longest period** (then latest `filed_date` as a restatement proxy)
  - if reported EPS is missing, a computed fallback is used:
    - `NetIncomeLoss / WeightedAverageNumberOfSharesOutstandingBasic` (annual-ish periods only)

## 5) Inflation Adjustment (Real EPS)

- Source: FRED CPI (`CPIAUCSL`), monthly.
- Real EPS at an EPS end-date is approximated as:

`real_eps = nominal_eps * (CPI_at_price_date / CPI_at_eps_end_date)`

## 6) Company CAPE Construction

- Lookback window: `--lookback-years` (default 10).
- Minimum observations: `--min-eps-points` (default 8).
- A company is **excluded** if:
  - insufficient EPS observations in the window
  - average real EPS is not strictly positive
  - missing price as-of the observation date

Company CAPE is:

`company_cape = close_price / avg_real_eps`

## 7) Weighting and CC CAPE

Market-cap proxy:

- `market_cap ≈ close_price * shares_outstanding`
- Shares come from SEC facts tags (latest available on/before observation date for series; latest overall for “latest” runs):
  - `EntityCommonStockSharesOutstanding`
  - `CommonStockSharesOutstanding`
  - `WeightedAverageNumberOfSharesOutstandingBasic`

Weighting rule:

- if market-cap coverage among valid constituents >= `--market-cap-min-coverage` (default 0.8):
  - use market-cap weighting (impute missing market caps with the median of available)
- else:
  - fall back to equal weight

CC CAPE:

`cc_cape = Σ_i (weight_i * company_cape_i)`

## 8) Shiller CAPE and Spread

- Shiller CAPE is ingested from the Multpl Shiller PE monthly table into `shiller_cape_observations`.
- For a given observation date, the benchmark is the latest Shiller value on/before that date.

Spread:

`cape_spread = cc_cape - shiller_cape`

## 9) Percentiles and Z-Scores

Two contexts:

1. Per-run history (`cc_cape_runs`)
- percentiles/z-scores computed across the stored run history

2. Monthly series (`cc_cape_series_monthly`)
- percentiles/z-scores computed across the stored monthly series points

These are **internal relative** measures and depend on how much history you’ve accumulated.

## 10) Known Limitations (Important)

- Free inputs are not a substitute for licensed constituent and price feeds.
- Shares-outstanding alignment to price date is approximate.
- EPS per share tags are not perfectly comparable across issuers and time; SEC facts quality varies.
- Corporate actions (splits/spinoffs) can distort prices/EPS if the free source doesn’t fully adjust.
- Negative earnings handling is simplistic (exclusion via non-positive average real EPS).

## 11) Implementation References

- Ingestion: `scripts/free_data_pipeline.py`
- Latest calculation: `scripts/calc_cc_cape_free.py`
- Monthly backfill: `scripts/backfill_cc_cape_series_free.py`
