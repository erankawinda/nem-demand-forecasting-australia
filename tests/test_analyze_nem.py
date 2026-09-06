from __future__ import annotations

import hashlib
import json
import sys

import numpy as np
import pandas as pd
import pytest

from src import analyze_nem
from src.analyze_nem import build_examples, chronological_split, load_panel


def regular_panel(periods: int = 1_400) -> pd.DataFrame:
    index = pd.date_range("2023-01-01", periods=periods, freq="30min")
    values = 20_000 + 700 * np.sin(np.arange(periods) * 2 * np.pi / 48)
    return pd.DataFrame({"NEM_DEMAND_MW": values}, index=index)


def test_target_and_previous_day_week_features_are_aligned() -> None:
    panel = regular_panel()
    examples, feature_columns = build_examples(panel)
    origin = examples.index[10]

    assert examples.loc[origin, "target_time"] == origin + pd.Timedelta(minutes=30)
    assert examples.loc[origin, "target_demand_mw"] == pytest.approx(
        panel.loc[origin + pd.Timedelta(minutes=30), "NEM_DEMAND_MW"]
    )
    assert examples.loc[origin, "demand_same_slot_previous_day_mw"] == pytest.approx(
        panel.loc[origin - pd.Timedelta(hours=23, minutes=30), "NEM_DEMAND_MW"]
    )
    assert examples.loc[origin, "demand_same_slot_previous_week_mw"] == pytest.approx(
        panel.loc[origin - pd.Timedelta(days=6, hours=23, minutes=30), "NEM_DEMAND_MW"]
    )
    assert "target_demand_mw" not in feature_columns
    assert all("baseline" not in name for name in feature_columns)


def test_chronological_split_is_ordered_and_disjoint() -> None:
    examples, _ = build_examples(regular_panel())
    train, validation, test = chronological_split(examples)

    assert len(train) + len(validation) + len(test) == len(examples)
    assert train.index.max() < validation.index.min() < test.index.min()
    assert (len(train), len(validation), len(test)) == (744, 160, 160)


@pytest.mark.parametrize(
    ("timestamp", "demand", "message"),
    [
        ("not-a-time", 20_000, "invalid timestamps"),
        ("2024-01-01 00:05", 20_000, "align"),
        ("2024-01-01 00:00", 0, "non-positive"),
    ],
)
def test_load_panel_rejects_invalid_inputs(
    tmp_path, timestamp: str, demand: float, message: str
) -> None:
    path = tmp_path / "panel.csv"
    pd.DataFrame(
        {"interval_start": [timestamp], "NEM_DEMAND_MW": [demand]}
    ).to_csv(path, index=False)

    with pytest.raises(ValueError, match=message):
        load_panel(path)


def test_end_to_end_analysis_writes_auditable_artifacts(
    tmp_path, monkeypatch
) -> None:
    periods = 1_600
    interval_start = pd.date_range("2024-01-01", periods=periods, freq="30min")
    step = np.arange(periods)
    demand = (
        25_000
        + 3_000 * np.sin(2 * np.pi * step / 48)
        + 900 * np.cos(2 * np.pi * step / 336)
        + 0.25 * step
    )
    panel_path = tmp_path / "panel.csv"
    reports = tmp_path / "reports"
    pd.DataFrame(
        {"interval_start": interval_start, "NEM_DEMAND_MW": demand}
    ).to_csv(panel_path, index=False)

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "analyze_nem.py",
            "--data",
            str(panel_path),
            "--reports",
            str(reports),
            "--n-jobs",
            "1",
        ],
    )
    analyze_nem.main()

    expected_files = {
        "forecast_sample.png",
        "metrics.csv",
        "split_manifest.json",
        "test_predictions.csv",
        "xgboost_feature_importance.csv",
        "xgboost_feature_importance.png",
    }
    assert {path.name for path in reports.iterdir()} == expected_files

    metrics = pd.read_csv(reports / "metrics.csv")
    assert len(metrics) == 10
    assert set(metrics["split"]) == {"validation", "test"}
    assert set(metrics["method"]) == {
        "persistence",
        "previous_day",
        "previous_week",
        "xgboost",
        "random_forest",
    }

    manifest = json.loads((reports / "split_manifest.json").read_text())
    expected_hash = hashlib.sha256(panel_path.read_bytes()).hexdigest()
    assert manifest["forecast_horizon_minutes"] == 30
    assert manifest["input"]["sha256"] == expected_hash
    assert manifest["total_examples"] == sum(
        manifest[split]["n"] for split in ("train", "validation", "test")
    )
