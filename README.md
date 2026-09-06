# Australian NEM Demand Forecasting

This is an exploratory data-science project for forecasting total electricity
demand in Australia's National Electricity Market (NEM) 30 minutes ahead. It
builds a reproducible 30-minute dataset and compares two tree-based
models with three simple forecasting baselines.

The repository is being maintained as a transparent research portfolio project.
It does **not** estimate the electricity use of AI or data centres, assess grid
security, predict reserve margins, or identify causal drivers of demand.

## Research question

How accurately can NEM demand at time `t + 30 minutes` be forecast using demand
observed at or before `t` and calendar information known in advance?

The analysis reports mean absolute error (MAE) and root mean squared error
(RMSE) on a final chronological test period. XGBoost and random-forest forecasts
are compared with:

- persistence: demand observed at the forecast origin;
- previous day: demand at the same target-time slot one day earlier; and
- previous week: demand at the same target-time slot one week earlier.

No model-performance numbers are stated in this README. They should be produced
from the data and code version being evaluated.

## Method

`src/make_panel.py` reads `nemdata` demand and unit-SCADA parquet files. It:

1. reduces overlapping records to one observation per region or generating unit
   and five-minute timestamp;
2. converts five-minute demand and generation observations into half-hour means;
3. keeps a regional half hour only when all six five-minute observations exist;
4. keeps NEM demand only when all five NEM regions are complete;
5. keeps generation half hours only when all six observations exist; and
6. joins demand and generation on their common timestamps.

The panel retains both average generation power (`NEM_GEN_AVG_MW`) and the
corresponding half-hour energy (`NEM_GEN_ENERGY_MWH`). Generation is preserved
for descriptive follow-up work but is deliberately not used as an input to the
demand-forecasting models.

`src/analyze_nem.py` first reindexes the panel to a regular 30-minute grid so
missing periods remain missing. It constructs a target exactly one half hour
ahead and uses only past/current demand values plus target-time calendar
features. The observations are split in time order:

- first 70%: training;
- next 15%: validation; and
- final 15%: untouched test period.

Model settings are fixed in the script. After the validation check, each model
is refitted on the combined training and validation periods and evaluated once
on the final test block. The generated split manifest records exact dates and
feature names.

## Data

The input data are obtained through [`nemdata`](https://github.com/ADGEfficiency/nem-data),
which prepares public Australian Energy Market Operator (AEMO) records. Follow
that project's current instructions to create `clean.parquet` files under:

```text
<cache>/demand/
<cache>/unit-scada/
```

Raw and processed datasets are intentionally excluded from Git because they are
large and are governed by their source terms. Check AEMO documentation and the
downloaded files before interpreting timestamps, revisions, or measurement
fields. The preparation script uses `interval-start` when the source provides
it. As a fallback, it treats `SETTLEMENTDATE` as the end of a five-minute
dispatch interval and labels the containing half hour by its start.

## Reproduce

Use Python 3.10 (required by the current `nemdata` release), create a virtual
environment, and install the dependencies:

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Build the panel (the default cache is `~/nem-data/data`):

```bash
python src/make_panel.py --cache /path/to/nem-data/data
```

Run the chronological evaluation:

```bash
python src/analyze_nem.py
```

The analysis uses two worker threads by default to keep memory use predictable.
Use `--n-jobs 1` on a small machine or increase it deliberately on a larger one.

Generated files are written to `data/processed/` and `reports/`. Both folders
are ignored by Git. The main outputs are:

- `metrics.csv` — validation and test MAE/RMSE for every baseline and model;
- `test_predictions.csv` — timestamped out-of-sample predictions;
- `split_manifest.json` — exact split boundaries and feature list;
- `forecast_sample.png` — final seven test days; and
- `xgboost_feature_importance.*` — model feature importance, which must not be
  interpreted as a causal effect.

## Interpretation and limitations

- This is a one-step forecasting study, not a long-term demand projection.
- It is a rolling one-step evaluation: at each test origin, observed demand up
  to that origin is available. It is not a recursive multi-step forecast.
- The split is chronological, but one holdout period does not establish
  performance across every weather regime or market condition.
- Calendar and autoregressive demand features do not explain why demand changes.
- Model feature importance describes use within one fitted model; it is not
  evidence of causation.
- Weather, prices, outages, holidays, and other potentially relevant variables
  are not included.
- Operational reserve and system-security assessments require additional AEMO
  datasets and domain-specific definitions; they cannot be inferred by simply
  subtracting demand from unit-SCADA generation.

The project should therefore be read as a reproducible forecasting exercise,
not as an operational or policy assessment of the NEM.
