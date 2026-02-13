# Pilot Plan (Internal)

Date: 2026-02-13

This is a template for running a short internal pilot of the ccCAPE tool.

## 1) Pilot Scope

- Audience: internal PMs + 1-2 quant/analyst users
- Duration: 2 weeks
- Cadence: weekly refresh (free-data mode)
- Key pages:
  - `/metrics/cc-cape`
  - `/metrics/health`
  - `/metrics/cc-cape/contributors`

## 2) Success Criteria

- Users can answer (without engineer help):
  - “What is the latest CC CAPE and spread?”
  - “Which sectors/constituents are driving it?”
  - “How fresh is the underlying data?”
- Ops:
  - weekly run completes without manual intervention
  - warnings are visible and interpretable

## 3) What To Collect

Qualitative:

- confusing terminology/fields
- missing exports
- trust issues (why a value changed)
- feature requests

Quantitative:

- page/API usage counts (from `/admin/audit`)
- pipeline run counts/statuses
- coverage trend (valid CAPE count, price coverage)

## 4) Feedback Template

Ask each pilot user:

1. What decisions would you make with CC CAPE/spread?
2. What output do you not trust, and why?
3. What workflow step is too slow or confusing?
4. What export/report format do you need?

## 5) Triage Process

Categories:

- P0: incorrect calculation / broken pipeline
- P1: missing/unclear key metric, major usability blocker
- P2: nice-to-have / polish

## 6) Exit Deliverables

- A short summary doc:
  - what worked
  - what failed and why
  - recommended next phase changes

