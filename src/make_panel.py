#!/usr/bin/env python3
"""Build a clean 30-minute NEM demand and generation panel.

The input layout follows ``nemdata``: parquet files named ``clean.parquet``
under ``<cache>/demand`` and ``<cache>/unit-scada``. Only half-hour bins with
all six five-minute observations are retained.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CACHE = Path.home() / "nem-data" / "data"
DEFAULT_OUTPUT = REPO_ROOT / "data" / "processed"
NEM_REGIONS = ("NSW1", "QLD1", "SA1", "TAS1", "VIC1")
EXPECTED_OBSERVATIONS = 6


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--start", type=str, default=None)
    parser.add_argument("--end", type=str, default=None)
    return parser.parse_args()


def load_parquets(root: Path) -> pd.DataFrame:
    files = sorted(root.rglob("clean.parquet"))
    if not files:
        raise FileNotFoundError(
            f"No clean.parquet files found under {root}. "
            "Download and prepare the dataset with nemdata first."
        )
    return pd.concat((pd.read_parquet(path) for path in files), ignore_index=True)


def find_column(frame: pd.DataFrame, candidates: tuple[str, ...]) -> str:
    by_name = {
        str(column).strip().lower().replace("_", "-"): column
        for column in frame.columns
    }
    for candidate in candidates:
        key = candidate.strip().lower().replace("_", "-")
        if key in by_name:
            return str(by_name[key])
    raise ValueError(
        f"None of {candidates} was found. Available columns: {frame.columns.tolist()}"
    )


def parse_timestamp(values: pd.Series) -> pd.Series:
    timestamps = pd.to_datetime(values, errors="coerce")
    # Compare timestamps using their displayed NEM market-time values.
    if getattr(timestamps.dt, "tz", None) is not None:
        timestamps = timestamps.dt.tz_localize(None)
    return timestamps


def resolve_duplicate_measurements(
    frame: pd.DataFrame, key_columns: list[str], value_column: str
) -> pd.DataFrame:
    """Remove exact repeats but refuse ambiguous values for the same key."""
    frame = frame.drop_duplicates()
    repeated = frame.duplicated(subset=key_columns, keep=False)
    if repeated.any():
        value_counts = frame.loc[repeated].groupby(key_columns)[value_column].nunique()
        if (value_counts > 1).any():
            n_conflicts = int((value_counts > 1).sum())
            raise ValueError(
                f"Found {n_conflicts} timestamp/key combinations with conflicting "
                f"{value_column} values. Resolve source revisions before aggregation."
            )
        frame = frame.drop_duplicates(subset=key_columns, keep="first")
    return frame


def apply_window(
    frame: pd.DataFrame, start: str | None, end: str | None
) -> pd.DataFrame:
    result = frame
    if start is not None:
        result = result[result["interval_start"] >= pd.Timestamp(start)]
    if end is not None:
        result = result[result["interval_start"] <= pd.Timestamp(end)]
    return result.copy()


def prepare_demand(raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    timestamp_column = find_column(
        raw, ("interval-start", "interval_start", "SETTLEMENTDATE")
    )
    region_column = find_column(raw, ("REGIONID", "REGION"))
    value_column = find_column(raw, ("DEMAND", "TOTALDEMAND"))

    demand = raw[[timestamp_column, region_column, value_column]].rename(
        columns={
            timestamp_column: "reported_time",
            region_column: "REGIONID",
            value_column: "DEMAND_MW",
        }
    )
    demand["reported_time"] = parse_timestamp(demand["reported_time"])
    demand["REGIONID"] = demand["REGIONID"].astype(str).str.upper().str.strip()
    demand["DEMAND_MW"] = pd.to_numeric(demand["DEMAND_MW"], errors="coerce")
    demand = demand.dropna(subset=["reported_time", "DEMAND_MW"])
    demand = demand[demand["REGIONID"].isin(NEM_REGIONS)]
    demand = resolve_duplicate_measurements(
        demand, ["reported_time", "REGIONID"], "DEMAND_MW"
    )

    is_interval_end = timestamp_column.strip().upper() == "SETTLEMENTDATE"
    demand["five_minute_start"] = demand["reported_time"]
    if is_interval_end:
        demand["five_minute_start"] -= pd.Timedelta(minutes=5)

    # Reduce the five-minute readings to one regional value before counting
    # observations in each half hour.
    demand_5min = (
        demand.groupby(["five_minute_start", "REGIONID"], as_index=False)[
            "DEMAND_MW"
        ]
        .mean()
        .sort_values(["five_minute_start", "REGIONID"])
    )
    demand_5min["interval_start"] = demand_5min["five_minute_start"].dt.floor(
        "30min"
    )
    regional = (
        demand_5min.groupby(["interval_start", "REGIONID"], as_index=False)
        .agg(
            DEMAND_MW=("DEMAND_MW", "mean"),
            OBSERVATIONS=("five_minute_start", "nunique"),
        )
    )
    regional = regional[regional["OBSERVATIONS"] == EXPECTED_OBSERVATIONS].copy()

    complete = regional.groupby("interval_start")["REGIONID"].agg(
        lambda values: set(values) == set(NEM_REGIONS)
    )
    regional = regional[regional["interval_start"].isin(complete[complete].index)]
    nem = (
        regional.groupby("interval_start", as_index=False)["DEMAND_MW"]
        .sum()
        .rename(columns={"DEMAND_MW": "NEM_DEMAND_MW"})
    )
    return regional.sort_values(["interval_start", "REGIONID"]), nem


def prepare_generation(raw: pd.DataFrame) -> pd.DataFrame:
    timestamp_column = find_column(
        raw, ("interval-start", "interval_start", "SETTLEMENTDATE")
    )
    value_column = find_column(raw, ("SCADAVALUE",))
    duid_column = find_column(raw, ("DUID",))

    selected = [timestamp_column, value_column, duid_column]
    generation = raw[selected].rename(
        columns={
            timestamp_column: "reported_time",
            value_column: "SCADA_MW",
            duid_column: "DUID",
        }
    )
    generation["reported_time"] = parse_timestamp(generation["reported_time"])
    generation["SCADA_MW"] = pd.to_numeric(generation["SCADA_MW"], errors="coerce")
    generation = generation.dropna(subset=["reported_time", "SCADA_MW"])
    generation["DUID"] = generation["DUID"].astype(str).str.strip()
    generation = resolve_duplicate_measurements(
        generation, ["reported_time", "DUID"], "SCADA_MW"
    )

    is_interval_end = timestamp_column.strip().upper() == "SETTLEMENTDATE"
    generation["five_minute_start"] = generation["reported_time"]
    if is_interval_end:
        generation["five_minute_start"] -= pd.Timedelta(minutes=5)

    generation_5min = (
        generation.groupby("five_minute_start", as_index=False)["SCADA_MW"]
        .sum()
        .rename(columns={"SCADA_MW": "NEM_GEN_MW_5MIN"})
    )
    generation_5min["interval_start"] = generation_5min[
        "five_minute_start"
    ].dt.floor("30min")
    generation_30min = (
        generation_5min.groupby("interval_start", as_index=False)
        .agg(
            NEM_GEN_AVG_MW=("NEM_GEN_MW_5MIN", "mean"),
            OBSERVATIONS=("five_minute_start", "nunique"),
        )
    )
    generation_30min = generation_30min[
        generation_30min["OBSERVATIONS"] == EXPECTED_OBSERVATIONS
    ].copy()
    generation_30min["NEM_GEN_ENERGY_MWH"] = (
        generation_30min["NEM_GEN_AVG_MW"] * 0.5
    )
    return generation_30min.sort_values("interval_start")


def validate_panel(panel: pd.DataFrame) -> None:
    if panel.empty:
        raise ValueError("The demand and generation datasets have no complete overlap.")
    if panel["interval_start"].duplicated().any():
        raise ValueError("Duplicate half-hour timestamps remain in the final panel.")
    if not panel["interval_start"].is_monotonic_increasing:
        raise ValueError("Final panel timestamps are not in chronological order.")
    if (panel["NEM_DEMAND_MW"] <= 0).any():
        raise ValueError("Non-positive NEM demand values remain in the final panel.")


def main() -> None:
    args = parse_args()
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)

    print("Loading demand data...")
    regional_demand, nem_demand = prepare_demand(
        load_parquets(args.cache.expanduser() / "demand")
    )
    print("Loading unit-SCADA data...")
    generation = prepare_generation(
        load_parquets(args.cache.expanduser() / "unit-scada")
    )

    regional_demand = apply_window(regional_demand, args.start, args.end)
    nem_demand = apply_window(nem_demand, args.start, args.end)
    generation = apply_window(generation, args.start, args.end)
    panel = (
        nem_demand.merge(generation, on="interval_start", how="inner")
        .sort_values("interval_start")
        .reset_index(drop=True)
    )
    validate_panel(panel)

    paths = {
        "regional": output / "demand_region_30min.csv",
        "demand": output / "demand_nem_30min.csv",
        "generation": output / "generation_nem_30min.csv",
        "panel": output / "panel_nem_30min.csv",
    }
    regional_demand.to_csv(paths["regional"], index=False)
    nem_demand.to_csv(paths["demand"], index=False)
    generation.to_csv(paths["generation"], index=False)
    panel.to_csv(paths["panel"], index=False)

    expected_grid = pd.date_range(
        panel["interval_start"].iloc[0], panel["interval_start"].iloc[-1], freq="30min"
    )
    print(
        "Final common window:",
        panel["interval_start"].iloc[0],
        "to",
        panel["interval_start"].iloc[-1],
    )
    print(f"Complete half-hours: {len(panel):,}")
    print(f"Missing half-hours inside that window: {len(expected_grid) - len(panel):,}")
    for path in paths.values():
        print("Wrote:", path)


if __name__ == "__main__":
    main()
