# Measured operational-demand data

The main pipeline reads AEMO's monthly `DEMANDOPERATIONALACTUAL` archives.
Use the original 30-minute `OPERATIONAL_DEMAND` field in MW. The separate
`OPERATIONAL_DEMAND_ADJUSTMENT` and `WDR_ESTIMATE` fields are retained in the
regional data for audit but are not added to the forecast target.

[AEMO's operational-demand documentation](https://www.aemo.com.au/energy-systems/electricity/national-electricity-market-nem/data-nem/operational-demand-data)
defines the measured quantity, publication process, and possible revisions.
Its five-minute operational-demand files are interpolated from half-hour
values; the main pipeline does not use those files.

## Download and validation

```bash
python src/prepare_operational_demand.py --start 2024-10 --end 2024-12
```

The inclusive month range selects half-hour interval starts from 1 October
2024 00:00 through 31 December 2024 23:30. Each monthly ZIP is downloaded from
AEMO's [historical MMSDM archive](https://www.nemweb.com.au/Data_Archive/Wholesale_Electricity/MMSDM/)
and cached with its URL, retrieval time, byte count, SHA-256, and HTTP
last-modified value. Cached files are checked against their receipts.

AEMO CSV files contain `C` metadata, `I` schema, and `D` data records. The
reader validates the table identity and schema before reading values. It
checks half-hour alignment, finite demand, known regions, and duplicate keys.
Exact duplicate rows are removed; conflicting records require resolution
instead of silently selecting a revision. A half hour enters the NEM panel
only when NSW1, QLD1, SA1, TAS1, and VIC1 are all present.

Negative regional demand is valid and retained. Missing intervals are reported
in the source manifest and are not filled. The forecasting code restores a
regular time grid before constructing features, so gaps cannot be mistaken
for adjacent half hours.

## Timestamps

`INTERVAL_DATETIME` is treated as the half-hour **end**. The panel subtracts
30 minutes to obtain `interval_start`. All times use fixed NEM market time:
AEST, UTC+10, without daylight saving. AEMO defines market time in section 4.2
of its [data specification](https://www.aemo.com.au/-/media/files/initiatives/der/2023/project-edge---data-specification-part-a.pdf?la=en).

The [table definition](https://visualisations.aemo.com.au/aemo/nemweb/mmsdatamodelreport/electricity/mms%20data%20model%20report_files/MMS_95.htm)
names the time and region keys but does not itself state the start/end
convention. The pilot checks archive timestamps and values against the
explicit half-hour-ending extrema in AEMO's [Q4 2024 report](https://aemo.com.au/-/media/files/major-publications/qed/2024/qed-q4-2024.pdf), pages 12–13.
The results folder records this independent check.

For a forecast issued at 10:30, the latest observation covers 10:00–10:30 and
the target covers 10:30–11:00. Prediction files record all three timestamps:
`forecast_origin`, `target_interval_start`, and `target_interval_end`.
The source archive is retrospective: the evaluation assumes zero publication
delay and does not reconstruct original releases or later corrections.

## Prepared files

- `panel_nem_30min.csv`: `interval_start` and `NEM_DEMAND_MW`, the sum of the
  five measured regional means.
- `regional_operational_demand_30min.csv`: regional readings and source audit
  fields, including adjustment, WDR estimate, and last-changed metadata.
- `source_manifest.json`: source quantity, archive receipts, region and time
  conventions, coverage checks, and output hashes.

The panel keeps its earlier two-column interface, but the source quantity has
changed from dispatch demand to measured operational demand. Always use the
matching source manifest when running or reporting the new evaluation.
The analysis report schema is version 3. Regenerate older reports: those made
before the timing correction labelled origins 30 minutes before the last
observation was complete.

## Legacy Parquet reader

`src/make_panel.py` is retained for existing `nemdata` caches containing
`clean.parquet` files under `demand/` and optionally `unit-scada/`. It accepts
`interval-start` or five-minute-ending `SETTLEMENTDATE`, `REGIONID` or `REGION`,
and `DEMAND` or `TOTALDEMAND`. It requires six readings per regional half hour.
Optional generation columns are descriptive and are not forecast inputs.

Dispatch `TOTALDEMAND` includes forecast adjustments and differs from the
measured target. See [AEMO's demand terms](https://aemo.com.au/-/media/Files/Electricity/NEM/Security_and_Reliability/Dispatch/Policy_and_Process/Demand-terms-in-EMMS-Data-Model.pdf).
Legacy outputs default to `data/processed/legacy_dispatch/`; they are separate
from the canonical measured-demand panel. The old downloader is not part of
the reproduction commands above.
