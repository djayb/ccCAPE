# Data Sources and Usage Notes (Free-Data Mode)

Date: 2026-02-13

This tool is designed to operate without paid vendor feeds. It uses public/free sources and should be treated as **internal research only** unless legal review confirms acceptable terms for your intended use.

## Primary Sources (Used in Code)

### Constituents (Current S&P 500)

- Source: Wikipedia S&P 500 constituents table
- Used for: universe + CIK mapping bootstrap
- Caveat: unofficial; not a licensed S&P membership feed

### Fundamentals (Earnings + Shares)

- Source: SEC EDGAR XBRL Company Facts API (`data.sec.gov`)
- Used for:
  - EPS tags (annual-ish)
  - shares outstanding tags for market-cap proxy
- Caveat:
  - coverage varies by filer/tag
  - automated use must follow SEC fair-access policies
  - require a real `User-Agent` contact string

### Inflation (Real Earnings Deflation)

- Source: FRED CPI series `CPIAUCSL`
- Used for: real EPS conversion
- Caveat: monthly CPI only; approximation for day-level alignment

### Prices

- Source: Stooq daily price CSV endpoint
- Used for: close price series
- Caveat:
  - rate limits (“Exceeded the daily hits limit”)
  - coverage may differ by symbol/corporate actions

### Shiller CAPE Benchmark

- Source: Multpl “Shiller PE” monthly table
- Used for: benchmark CAPE + spread
- Caveat:
  - scraping/publishing restrictions may apply
  - for academic source-of-truth consider Shiller/Yale data (requires XLS parsing + mapping)

## Legal / Licensing Notes (Non-Authoritative)

- SEC: follow fair-access rules and identify your client via `User-Agent`.
- FRED: terms vary by series; CPIAUCSL is broadly used, but verify intended redistribution/commercial use.
- Wikipedia: content is under Creative Commons; verify reuse requirements if redistributing derived datasets.
- Stooq / Multpl: verify terms and whether scraping is permitted for your context.

## Recommended Next Step for Compliance

Create a short internal memo covering:

- intended use (internal research vs client-facing)
- whether derived datasets will be redistributed
- acceptable sources and limitations
- data retention policy

