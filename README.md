# Australian NEM Operational Demand Forecasting

Forecasting electricity demand for the next half hour in Australia's
**National Electricity Market (NEM)**. This project uses measured operational
demand published by the Australian Energy Market Operator (AEMO), checks the
source data, and compares two machine-learning models with three simple
baselines.

The question is:

> How accurately can the next half-hour mean operational demand be predicted
> from recent demand and calendar information?

Operational demand measures demand supplied through the power system covered
by AEMO's definition. It excludes electricity supplied by sources such as
household rooftop solar. The target here is the sum of the measured half-hour
values for NSW/ACT, Queensland, South Australia, Tasmania, and Victoria,
expressed in megawatts (MW). See [AEMO's definition](https://www.aemo.com.au/energy-systems/electricity/national-electricity-market-nem/data-nem/operational-demand-data).

## What the project does

1. Downloads AEMO's monthly measured operational-demand archives.
2. Validates timestamps, values, duplicate records, and five-region coverage.
3. Builds demand-history and calendar features using completed intervals.
4. Evaluates XGBoost and random forest against persistence, previous-day,
   and previous-week forecasts using chronological splits.
5. Saves predictions, errors, plots, and manifests that identify the exact
   source snapshot, code, and model settings.

## Historical pilot

The first reproducible pilot covers October–December 2024. Its
[protocol](results/operational_demand_2024q4/protocol.md) records the source,
features, split, and fixed model settings before model results were inspected.
The [results folder](results/operational_demand_2024q4/) contains the full
comparison, source checks, and supporting artifacts. All 4,416 half hours have
complete five-region coverage. The final test has 612 forecasts, covering
19–31 December 2024.

| Method | Test MAE (MW) | Test RMSE (MW) |
|---|---:|---:|
| Persistence | 487.30 | 625.06 |
| Previous day | 1026.53 | 1441.11 |
| Previous week | 2134.83 | 3274.83 |
| XGBoost | 183.37 | 237.88 |
| Random forest | 177.94 | 236.74 |

Both models improved on the baselines in this test period. Random forest had
the lowest test MAE; XGBoost had the lowest validation MAE. The results folder
reports both splits so the comparison can be inspected in context.

## Run the project

The tested environment is Python 3.10.19. The lock file records the resolved
runtime and test dependencies; it also retains dependencies for the older
Parquet reader.

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-lock.txt

python src/prepare_operational_demand.py --start 2024-10 --end 2024-12
python src/analyze_nem.py \
  --data data/processed/panel_nem_30min.csv \
  --source-manifest data/processed/source_manifest.json \
  --reports reports/operational_demand_2024q4 --n-jobs 1

python -m pytest -q
```

The month arguments are inclusive. Downloads are cached under
`data/raw/operational_demand/`. Use `--cache` and `--output` to choose other
locations. The analysis checks that the source manifest matches the input
panel's hash. Keep the original archives and manifests when comparing runs,
since AEMO may revise historical records.

## Data and forecast timing

The source is `DEMANDOPERATIONALACTUAL.OPERATIONAL_DEMAND`, AEMO's original
half-hour measured series. The reader uses this field directly, without
adding the separate adjustment or wholesale demand-response estimates.
It retains finite negative regional readings and only sums intervals with
all five regions present. Missing data are not filled.

AEMO timestamps identify interval ends. The prepared panel uses interval
starts, in fixed AEST (UTC+10), with no daylight-saving shift. For example:

| Forecast issued | Latest completed observation | Interval being predicted |
|---|---|---|
| 10:00 | 09:30–10:00 | 10:00–10:30 |

The forecast features use the latest two completed half hours, the same target
slot one day and one week earlier, trailing 24-hour and seven-day statistics,
and calendar information known in advance. See [data documentation](data/README.md)
for the schema and timestamp checks.

## Evaluation

Usable examples are split in time order: 70% training, 15% validation, and 15%
test. Fixed model settings are evaluated on validation data, then each model
is fitted on training plus validation data for the final test. The saved
pilot uses seed 42 and one worker thread, with no hyperparameter search.

Each method is evaluated on the same examples using **MAE** (average absolute
error) and **RMSE** (an error measure that gives more weight to large misses),
both in MW. Lower is better. The three baselines predict:

- **Persistence:** demand stays at the latest completed half-hour value.
- **Previous day:** demand matches the same half hour yesterday.
- **Previous week:** demand matches the same half hour one week ago.

The test advances one half hour at a time, using newly completed observations
without retraining the models. This is a rolling one-step evaluation.

## Outputs and tests

The preparation step writes the regional data, `panel_nem_30min.csv`, and
`source_manifest.json`. Analysis writes:

- `metrics.csv` and `test_predictions.csv` for all five methods;
- `split_manifest.json` with source and code hashes, model settings, software
  versions, split boundaries, and the forecast timing assumption;
- `forecast_sample.png`, showing the final seven days of test predictions;
- `xgboost_feature_importance.csv` and `.png` for model interpretation.

Raw data and routine generated reports are ignored by Git. The selected pilot
artifacts are retained under `results/` for review.

Tests use synthetic fixtures and run without network access. They cover archive
parsing, missing regions, duplicate conflicts, time alignment, chronological
splits, and the rule that future readings cannot change earlier forecast
inputs. GitHub Actions runs the suite with Python 3.10.

## Evaluation scope

This pilot evaluates one quarter using demand history and calendar features.
A useful next study would test additional seasons and add weather information.
The historical backtest uses archived values and assumes each observation is
available when its interval ends; it does not reconstruct publication delays
or later corrections. Feature importance describes how the fitted model uses
its inputs, rather than why electricity demand changes.

## Repository guide

```text
src/prepare_operational_demand.py  Download and validate measured demand
src/analyze_nem.py                 Features, models, baselines, and reports
src/make_panel.py                  Legacy five-minute dispatch/SCADA reader
tests/                            Data and forecasting tests
data/README.md                    Source definitions and data schema
results/operational_demand_2024q4/  Reproducible historical pilot
requirements*.txt                 Pinned dependencies
```

The legacy reader is retained for existing Parquet datasets. It works with a
different demand quantity and is not used for the measured-demand pilot.

## Licence and data attribution

Project code uses the BSD 3-Clause licence in [LICENSE](LICENSE).
Source data are published by AEMO. Raw archives are downloaded directly from
AEMO and are not included in this repository. Derived pilot results identify
AEMO as their source; the code licence does not replace AEMO's
[copyright permissions](https://www.aemo.com.au/privacy-and-legal-notices/copyright-permissions)
or [legal notices](https://www.aemo.com.au/privacy-and-legal-notices).
