# External Fundamentals Import (Extend History)

Date: 2026-02-14

## Goal

Extend CC CAPE history beyond what is available from SEC XBRL "company facts" by importing an external fundamentals dataset into `data/free_data.db`.

## Reality Check (Free Data vs 50-Year History)

- Price history can often be extended far back (e.g. decades).
- Inflation (CPI) can be extended far back (FRED).
- **Per-company fundamentals history is the limiting factor** for a true CC CAPE backfill.

The SEC XBRL company-facts era usually limits fundamentals coverage to roughly the last ~10-20 years (varies by filer and tag). A 50-year CC CAPE series for current constituents generally requires a licensed fundamentals dataset, or an internal historical dataset you can import.

## Where The Data Goes

We store imported rows in the existing table:

- `company_facts_values`

We distinguish sources via:

- `taxonomy` (for example: `external`, `simfin`, `compustat`)

The calculation engine reads `company_facts_values` by `tag`, regardless of taxonomy.

## CSV Import

Script:

- `scripts/import_external_fundamentals_csv.py`

Example:

```bash
python3 scripts/import_external_fundamentals_csv.py \
  --data-db data/free_data.db \
  --csv data/external_fundamentals.csv \
  --taxonomy external
```

### Expected CSV columns

Required:

- `end_date` (YYYY-MM-DD) OR `fiscal_year` (YYYY)
- `symbol` or `cik`

Optional:

- `start_date` (YYYY-MM-DD)
- `fiscal_year`, `fiscal_period`

Fundamentals (import what you have):

- `net_income` (USD)
- `shares_basic` (shares)
- `shares_outstanding` (shares)
- `eps_basic` (USD/share)
- `eps_diluted` (USD/share)

## SimFin Import (Optional Adapter)

Script:

- `scripts/import_simfin_bulk.py`

This uses SimFin’s bulk download endpoint and requires a SimFin API key.

```bash
SIMFIN_API_KEY="your_key_here" \
python3 scripts/import_simfin_bulk.py \
  --data-db data/free_data.db \
  --taxonomy simfin \
  --universe sp500
```

Notes:

- SimFin licensing applies.
- The amount of historical fundamentals available depends on your SimFin plan.

## After Import: Recompute

Once external fundamentals are in `company_facts_values`, re-run:

```bash
python3 scripts/calc_cc_cape_free.py --update-tracker
python3 scripts/backfill_cc_cape_series_free.py --series-years 50 --replace-existing --update-tracker
```

