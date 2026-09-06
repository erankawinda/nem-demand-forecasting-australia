# Australian NEM Demand Forecasting

A complete, reproducible pipeline for forecasting total electricity demand in
Australia's National Electricity Market (NEM) 30 minutes ahead. It builds a
regular half-hour demand panel from public AEMO records and compares XGBoost and
random-forest models with three transparent forecasting baselines.

The deliberately limited research question is:

> How accurately can NEM demand at `t + 30 minutes` be forecast using demand
> observed at or before `t` and calendar information known in advance?

This project does not estimate AI or data-centre electricity use, assess grid
security or reserve margins, make long-term projections, or identify causal
drivers of demand.

## Evaluation design

Every example is indexed by a forecast origin `t`. Its target is demand one
half-hour later. Inputs contain only information available at `t` or calendar
information already known for the target time:

- current and one-step-lagged demand;
- the demand observed at the target slot one day and one week earlier;
- trailing 24-hour and seven-day demand mean and standard deviation; and
- cyclical target-time, target-week, target-month, and weekend indicators.

The day and week offsets are 47 and 335 half-hours from the origin because the
target itself is one interval ahead. This alignment is covered by automated
tests.

Examples are split in time order: the first 70% for training, the next 15% for
validation, and the final 15% as an untouched test period. The fixed model
settings are first evaluated on validation data, then each model is refitted on all
pre-test observations and evaluated once on the final block. MAE and RMSE are
reported alongside:

- persistence: demand at the forecast origin;
- previous day: demand at the same target slot one day earlier; and
- previous week: demand at the same target slot one week earlier.

The test period is a rolling one-step evaluation: observed demand up to each
test origin is available. It is not a recursive multi-step forecast.

## Data preparation

`src/make_panel.py` reads `clean.parquet` files produced by
[`nemdata`](https://github.com/ADGEfficiency/nem-data) from public Australian
Energy Market Operator (AEMO) [NEM
records](https://www.aemo.com.au/energy-systems/electricity/national-electricity-market-nem/data-nem).
It:

1. resolves exact duplicate source rows and rejects conflicting revisions;
2. converts five-minute regional demand readings to half-hour means;
3. retains a regional half-hour only when all six readings are present; and
4. retains total NEM demand only when all five NEM regions are complete.

The core forecasting panel depends only on demand data. With
`--include-generation`, the same script also aggregates unit-SCADA readings to
average MW and half-hour MWh and left-joins them as optional descriptive
columns. Generation coverage never removes an otherwise complete demand row,
and generation is not used as a forecasting input.

Raw and processed datasets are excluded from Git because they are large and
retain their source terms. `data/README.md` documents the expected input and
output schema.

## Reproduce

The tested environment is Python 3.10.19 with the complete resolved environment
in `requirements-lock.txt`; `requirements.txt` lists only the direct runtime
dependencies. Python 3.10 is required by the pinned `nemdata==0.3.7`.

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-lock.txt
```

Download the same month range for NEM demand. For example:

```bash
nemdata -t demand --start 2020-01 --end 2024-12
```

By default `nemdata` writes under `~/nem-data/data`. Build the demand panel and
run the chronological evaluation:

```bash
python src/make_panel.py --cache ~/nem-data/data \
  --start 2020-01-01 --end 2024-12-31
python src/analyze_nem.py
```

To include the optional descriptive generation columns, first download the
same month range for `unit-scada`, then enable the flag:

```bash
nemdata -t unit-scada --start 2020-01 --end 2024-12
python src/make_panel.py --cache ~/nem-data/data \
  --start 2020-01-01 --end 2024-12-31 --include-generation
```

The analysis uses two worker threads by default. Use `--n-jobs 1` on a small
machine or increase it deliberately on a larger one.

## Outputs

Generated data are written to `data/processed/`, and analysis artifacts to
`reports/`. Both directories are ignored by Git. The outputs are:

- `panel_nem_30min.csv` — validated demand panel, plus optional generation;
- `demand_region_30min.csv` and `demand_nem_30min.csv` — prepared demand data;
- `generation_nem_30min.csv` — written only when generation is requested;
- `metrics.csv` — validation and test MAE/RMSE for all baselines and models;
- `test_predictions.csv` — timestamped final-period predictions;
- `split_manifest.json` — data hash, software versions, split boundaries,
  feature names, horizon, and seed;
- `forecast_sample.png` — final seven test days; and
- `xgboost_feature_importance.*` — model-specific, non-causal importance.

No performance number is copied into this README because the result is only
valid for the exact downloaded data snapshot recorded by the generated
manifest. Running the pipeline produces the auditable metrics and predictions.

## Tests

The tests use deterministic synthetic five-minute and half-hour data, so they
do not download external files. They check interval alignment, completeness,
duplicate handling, NEM aggregation, forecast-target alignment, chronological
splits, and the absence of future demand in model features.

```bash
python -m pip install -r requirements-lock.txt
python -m pytest -q
```

The same suite runs automatically on GitHub Actions with Python 3.10.

## Repository structure

```text
src/make_panel.py       Five-minute to half-hour data preparation
src/analyze_nem.py      Features, models, baselines, metrics, and artifacts
tests/                  Deterministic methodological and smoke tests
data/README.md          Input/output schema and provenance notes
requirements*.txt       Direct, development, and fully resolved dependencies
```

## Interpretation limits

- One chronological holdout does not establish performance across every
  weather regime or market condition.
- Weather, prices, outages, public holidays, and other useful predictors are
  outside this intentionally autoregressive benchmark.
- Calendar and demand-history features predict demand; they do not explain why
  it changes.
- XGBoost feature importance is not evidence of causation.
- AEMO revises its systems and reports over time; preserve the generated input
  hash and manifest when reporting a particular run.

## Licence

Project code is provided under the BSD 3-Clause licence in `LICENSE`.
Downloaded AEMO data are not redistributed here. Their use remains subject to
AEMO's [privacy and legal notices](https://www.aemo.com.au/privacy-and-legal-notices)
and [copyright permissions](https://www.aemo.com.au/privacy-and-legal-notices/copyright-permissions).
