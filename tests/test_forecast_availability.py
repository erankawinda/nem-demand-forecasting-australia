"""Check information availability across data preparation and forecasting."""

import numpy as np
import pandas as pd
import pytest

from src.analyze_nem import build_examples, load_panel
from src.make_panel import NEM_REGIONS, prepare_demand


@pytest.mark.parametrize("timestamp_column", ["interval-start", "SETTLEMENTDATE"])
def test_future_five_minute_readings_cannot_change_forecast_features(
    tmp_path, timestamp_column
) -> None:
    starts = pd.date_range("2024-01-01", periods=1_400 * 6, freq="5min")
    raw = pd.DataFrame(
        {
            "interval-start": np.repeat(starts, len(NEM_REGIONS)),
            "REGIONID": np.tile(NEM_REGIONS, len(starts)),
            "TOTALDEMAND": np.repeat(4_000 + np.arange(len(starts)), len(NEM_REGIONS)),
        }
    )
    if timestamp_column == "SETTLEMENTDATE":
        raw = raw.rename(columns={"interval-start": timestamp_column})
        raw[timestamp_column] += pd.Timedelta(minutes=5)

    # At Monday midnight, the previous half hour is complete. The forecast
    # covers Monday 00:00–00:30, so its calendar features must describe Monday.
    origin = pd.Timestamp("2024-01-15 00:00")
    reading_ends = raw[timestamp_column]
    if timestamp_column == "interval-start":
        reading_ends = reading_ends + pd.Timedelta(minutes=5)

    def examples_from_readings(readings, filename):
        _, panel = prepare_demand(readings)
        panel_path = tmp_path / filename
        panel.to_csv(panel_path, index=False)
        return build_examples(load_panel(panel_path))

    examples, features = examples_from_readings(raw, "original.csv")
    last_complete = raw[
        (reading_ends > origin - pd.Timedelta(minutes=30)) & (reading_ends <= origin)
    ]
    expected_observed = last_complete.groupby("REGIONID")["TOTALDEMAND"].mean().sum()
    target_readings = raw[
        (reading_ends > origin) & (reading_ends <= origin + pd.Timedelta(minutes=30))
    ]
    expected_target = target_readings.groupby("REGIONID")["TOTALDEMAND"].mean().sum()

    assert examples.loc[origin, "demand_last_complete_half_hour_mw"] == expected_observed
    assert examples.loc[origin, "baseline_persistence_mw"] == expected_observed
    assert examples.loc[origin, "target_demand_mw"] == expected_target
    assert examples.loc[origin, "target_interval_start"] == origin
    assert examples.loc[origin, "target_interval_end"] == origin + pd.Timedelta(minutes=30)
    assert examples.loc[origin, "target_is_weekend"] == 0
    assert examples.loc[origin, "target_week_cos"] == pytest.approx(1.0)

    changed = raw.copy()
    changed.loc[reading_ends > origin, "TOTALDEMAND"] += 10_000
    changed_examples, changed_features = examples_from_readings(changed, "changed.csv")

    assert changed_features == features
    baseline_columns = [name for name in examples if name.startswith("baseline_")]
    pd.testing.assert_series_equal(
        examples.loc[origin, features + baseline_columns],
        changed_examples.loc[origin, features + baseline_columns],
    )
    assert changed_examples.loc[origin, "target_demand_mw"] != expected_target
