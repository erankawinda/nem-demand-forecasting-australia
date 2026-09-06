# Data interface

The repository does not redistribute [AEMO NEM
records](https://www.aemo.com.au/energy-systems/electricity/national-electricity-market-nem/data-nem).
Use `nemdata==0.3.7` to create monthly `clean.parquet` files under the cache
passed to `src/make_panel.py`.

## Required demand fields

The preparation script accepts the `nemdata` `interval-start` field (preferred)
or `SETTLEMENTDATE`, a region field (`REGIONID` or `REGION`), and a demand field
(`DEMAND` or `TOTALDEMAND`). It keeps the five NEM region identifiers `NSW1`,
`QLD1`, `SA1`, `TAS1`, and `VIC1`.

If only `SETTLEMENTDATE` is present, the script treats it as a five-minute
interval end and subtracts five minutes before forming half-hour bins. With
`interval-start`, no shift is applied.

## Optional generation fields

When `--include-generation` is used, unit-SCADA files must contain
`interval-start` (preferred) or `SETTLEMENTDATE`, `DUID`, and `SCADAVALUE`.
Unit values are summed for each five-minute timestamp, then the six values in a
complete half hour are averaged to MW and multiplied by 0.5 hours to obtain
MWh.

## Generated columns

- `interval_start` — start of the half-hour bin in the source's displayed NEM
  market time;
- `NEM_DEMAND_MW` — sum of the five complete regional half-hour demand means;
- `NEM_GEN_AVG_MW` — optional mean of six five-minute aggregate-SCADA values;
- `NEM_GEN_ENERGY_MWH` — optional half-hour energy corresponding to that mean.

The pipeline rejects conflicting duplicate values rather than choosing a
revision silently. It also reports missing half-hours in the final window.
