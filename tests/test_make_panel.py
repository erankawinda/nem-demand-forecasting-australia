from __future__ import annotations

import pandas as pd
import pytest

from src.make_panel import (
    NEM_REGIONS,
    apply_window,
    prepare_demand,
    prepare_generation,
    resolve_duplicate_measurements,
)


def demand_rows(timestamp_name: str = "interval-start") -> pd.DataFrame:
    starts = pd.date_range("2024-01-01 00:00", periods=12, freq="5min")
    reported = starts + pd.Timedelta(minutes=5) if timestamp_name == "SETTLEMENTDATE" else starts
    rows = []
    for region_index, region in enumerate(NEM_REGIONS, start=1):
        for step, timestamp in enumerate(reported):
            rows.append(
                {
                    timestamp_name: timestamp,
                    "REGIONID": region,
                    "TOTALDEMAND": region_index * 1_000 + step,
                }
            )
    return pd.DataFrame(rows)


def test_demand_aggregation_requires_all_regions_and_six_readings() -> None:
    raw = demand_rows()
    # Make the second half hour incomplete for one region.
    raw = raw.drop(
        raw[(raw["REGIONID"] == "TAS1") & (raw["interval-start"] == "2024-01-01 00:35")].index
    )

    regional, nem = prepare_demand(raw)

    assert regional["interval_start"].nunique() == 1
    assert len(regional) == len(NEM_REGIONS)
    assert nem.iloc[0]["interval_start"] == pd.Timestamp("2024-01-01 00:00")
    # Each region contributes its base plus the mean of steps 0..5 (2.5).
    assert nem.iloc[0]["NEM_DEMAND_MW"] == pytest.approx(15_012.5)


def test_settlementdate_is_treated_as_interval_end() -> None:
    regional, nem = prepare_demand(demand_rows("SETTLEMENTDATE"))

    assert len(nem) == 2
    assert nem["interval_start"].tolist() == [
        pd.Timestamp("2024-01-01 00:00"),
        pd.Timestamp("2024-01-01 00:30"),
    ]
    assert (regional["OBSERVATIONS"] == 6).all()


def test_generation_sums_units_then_averages_time() -> None:
    timestamps = pd.date_range("2024-01-01 00:00", periods=6, freq="5min")
    raw = pd.DataFrame(
        [
            {"interval-start": timestamp, "DUID": duid, "SCADAVALUE": value}
            for timestamp in timestamps
            for duid, value in (("UNIT_A", 10.0), ("UNIT_B", 20.0))
        ]
    )

    generation = prepare_generation(raw)

    assert len(generation) == 1
    assert generation.iloc[0]["NEM_GEN_AVG_MW"] == pytest.approx(30.0)
    assert generation.iloc[0]["NEM_GEN_ENERGY_MWH"] == pytest.approx(15.0)


def test_conflicting_duplicate_measurements_are_rejected() -> None:
    frame = pd.DataFrame(
        {
            "time": ["2024-01-01 00:00", "2024-01-01 00:00"],
            "region": ["NSW1", "NSW1"],
            "value": [1.0, 2.0],
        }
    )

    with pytest.raises(ValueError, match="conflicting"):
        resolve_duplicate_measurements(frame, ["time", "region"], "value")


def test_date_only_end_includes_the_complete_day() -> None:
    frame = pd.DataFrame(
        {
            "interval_start": pd.to_datetime(
                ["2024-01-01 23:30", "2024-01-02 00:00"]
            )
        }
    )

    selected = apply_window(frame, None, "2024-01-01")

    assert selected["interval_start"].tolist() == [pd.Timestamp("2024-01-01 23:30")]


def test_irregular_five_minute_timestamp_is_rejected() -> None:
    raw = demand_rows()
    raw.loc[raw.index[0], "interval-start"] = pd.Timestamp("2024-01-01 00:01")

    with pytest.raises(ValueError, match="five-minute grid"):
        prepare_demand(raw)
