from __future__ import annotations

import csv
import io
import json
from pathlib import Path
import zipfile

import pandas as pd
import pytest

from src import prepare_operational_demand as operational


def source_frame(times: tuple[str, ...] = ("2024-10-01 00:30:00",)) -> pd.DataFrame:
    return pd.DataFrame([
        {
            "INTERVAL_DATETIME": timestamp,
            "REGIONID": region,
            "OPERATIONAL_DEMAND": index * 1000,
            "LASTCHANGED": timestamp,
            "OPERATIONAL_DEMAND_ADJUSTMENT": 100,
            "WDR_ESTIMATE": 200,
        }
        for timestamp in times
        for index, region in enumerate(operational.NEM_REGIONS, start=1)
    ])


def archive_bytes(frame: pd.DataFrame, table: tuple[str, ...] = operational.TABLE) -> bytes:
    """Create a small, invented fixture in the official MMS CSV structure."""
    csv_buffer = io.StringIO(newline="")
    writer = csv.writer(csv_buffer)
    writer.writerow(["C", "SETP.WORLD", "DVD_DEMANDOPERATIONALACTUAL", "AEMO", "PUBLIC"])
    writer.writerow(["I", *table, *operational.SOURCE_COLUMNS])
    for values in frame[list(operational.SOURCE_COLUMNS)].itertuples(index=False, name=None):
        values = list(values)
        for index in (0, 3):
            values[index] = pd.Timestamp(values[index]).strftime("%Y/%m/%d %H:%M:%S")
        writer.writerow(["D", *table, *values])
    writer.writerow(["C", "END OF REPORT", len(frame) + 3])
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("PUBLIC_ARCHIVE#DEMANDOPERATIONALACTUAL#FILE01#202410010000.CSV", csv_buffer.getvalue())
    return output.getvalue()


def write_archive(tmp_path: Path, frame: pd.DataFrame, **kwargs) -> Path:
    path = tmp_path / "sample.zip"
    path.write_bytes(archive_bytes(frame, **kwargs))
    return path


def test_original_half_hours_sum_regions_and_exclude_adjustments(tmp_path: Path) -> None:
    frame = source_frame(("2024-10-01 00:30:00", "2024-10-01 01:00:00"))
    raw = operational.read_archive(write_archive(tmp_path, frame))
    regional, panel, coverage = operational.prepare_panel(raw, "2024-10", "2024-10")

    assert panel["interval_start"].tolist() == [
        pd.Timestamp("2024-10-01 00:00:00"), pd.Timestamp("2024-10-01 00:30:00"),
    ]
    assert panel["NEM_DEMAND_MW"].tolist() == [15_000, 15_000]
    assert len(regional) == 10
    assert coverage["complete_intervals"] == 2
    assert coverage["expected_intervals"] == 31 * 48
    assert coverage["imputation"] == "none"


def test_month_end_midnight_belongs_to_previous_day() -> None:
    frame = source_frame((
        "2024-10-01 00:00:00", "2024-10-01 00:30:00", "2024-11-01 00:00:00", "2024-11-01 00:30:00",
    ))
    _, panel, coverage = operational.prepare_panel(frame, "2024-10", "2024-10")

    assert panel["interval_start"].tolist() == [
        pd.Timestamp("2024-10-01 00:00:00"), pd.Timestamp("2024-10-31 23:30:00"),
    ]
    assert coverage["rows_outside_requested_window"] == 10


def test_missing_region_removes_only_its_interval_and_reports_the_gap() -> None:
    frame = source_frame(("2024-10-01 00:30:00", "2024-10-01 01:00:00"))
    frame = frame.drop(frame.index[-1])
    regional, panel, coverage = operational.prepare_panel(frame, "2024-10", "2024-10")

    assert len(regional) == 9
    assert len(panel) == 1
    assert coverage["missing_or_incomplete_intervals"] == 31 * 48 - 1
    assert coverage["missing_region_observations"] == 31 * 48 * 5 - 9
    assert coverage["missing_intervals"][0] == {
        "interval_start": "2024-10-01T00:30:00", "missing_regions": ["VIC1"],
    }


def test_exact_duplicates_deduplicate_but_revision_conflicts_fail() -> None:
    frame = source_frame()
    duplicate = pd.concat([frame, frame.iloc[[0]]], ignore_index=True)
    _, panel, coverage = operational.prepare_panel(duplicate, "2024-10", "2024-10")
    assert panel.iloc[0]["NEM_DEMAND_MW"] == 15_000
    assert coverage["exact_duplicate_rows_removed"] == 1

    duplicate.loc[duplicate.index[-1], "OPERATIONAL_DEMAND"] = 999
    with pytest.raises(ValueError, match="Conflicting duplicate"):
        operational.prepare_panel(duplicate, "2024-10", "2024-10")


def test_negative_regional_measurement_is_retained() -> None:
    frame = source_frame()
    frame.loc[frame["REGIONID"] == "SA1", "OPERATIONAL_DEMAND"] = -205
    _, panel, _ = operational.prepare_panel(frame, "2024-10", "2024-10")
    assert panel.iloc[0]["NEM_DEMAND_MW"] == 11_795


def test_offset_aware_input_converts_to_fixed_aest_during_daylight_saving() -> None:
    frame = source_frame(("2024-10-06 03:30:00+11:00",))
    _, panel, _ = operational.prepare_panel(frame, "2024-10", "2024-10")
    assert panel.iloc[0]["interval_start"] == pd.Timestamp("2024-10-06 02:00:00")


def test_future_measurements_cannot_change_features_at_the_forecast_origin(tmp_path: Path) -> None:
    from src.analyze_nem import build_examples, load_panel

    ends = pd.date_range("2024-10-01 00:30:00", periods=31 * 48, freq="30min")
    original = source_frame(tuple(ends.astype(str)))
    changed = original.copy()
    origin = pd.Timestamp("2024-10-20 12:00:00")
    future = pd.to_datetime(changed["INTERVAL_DATETIME"]) > origin
    changed.loc[future, "OPERATIONAL_DEMAND"] += 10_000

    examples = []
    for filename, source in (("original.csv", original), ("changed.csv", changed)):
        _, panel, _ = operational.prepare_panel(source, "2024-10", "2024-10")
        path = tmp_path / filename
        panel.to_csv(path, index=False)
        frame, feature_columns = build_examples(load_panel(path))
        examples.append(frame)

    baseline_columns = [column for column in examples[0] if column.startswith("baseline_")]
    pd.testing.assert_frame_equal(
        examples[0].loc[:origin, feature_columns + baseline_columns],
        examples[1].loc[:origin, feature_columns + baseline_columns],
    )
    assert examples[0].loc[origin, "target_interval_start"] == origin
    assert examples[0].loc[origin, "target_interval_end"] == origin + pd.Timedelta(minutes=30)
    assert examples[1].loc[origin, "target_demand_mw"] - examples[0].loc[origin, "target_demand_mw"] == 50_000


@pytest.mark.parametrize("column,value,match", [
    ("OPERATIONAL_DEMAND", float("nan"), "null"),
    ("OPERATIONAL_DEMAND", float("inf"), "Nonfinite"),
    ("OPERATIONAL_DEMAND", "not a number", "Invalid numeric"),
    ("OPERATIONAL_DEMAND_ADJUSTMENT", float("inf"), "Nonfinite"),
    ("WDR_ESTIMATE", "", "empty or null"),
    ("REGIONID", "WA1", "Unknown NEM regions"),
    ("INTERVAL_DATETIME", "2024-10-01 00:31:00", "half-hour grid"),
    ("INTERVAL_DATETIME", "2024-10-01 00:30:00.000000001", "half-hour grid"),
    ("LASTCHANGED", "not a date", "Invalid LASTCHANGED"),
])
def test_invalid_source_records_fail(column: str, value: object, match: str) -> None:
    frame = source_frame()
    frame.loc[frame.index[0], column] = value
    with pytest.raises(ValueError, match=match):
        operational.prepare_panel(frame, "2024-10", "2024-10")


@pytest.mark.parametrize("table", [
    ("DISPATCH", "REGIONSUM", "3"), ("OPERATIONAL_DEMAND", "ACTUAL", "2"),
])
def test_wrong_archive_table_or_version_fails(tmp_path: Path, table: tuple[str, ...]) -> None:
    path = write_archive(tmp_path, source_frame(), table=table)
    with pytest.raises(ValueError, match="Unexpected table/version"):
        operational.read_archive(path)


@pytest.mark.parametrize("change,match", [
    (lambda text: "\r\n".join(text.split("\r\n")[1:]), "opening control metadata"),
    (lambda text: text.replace("WDR_ESTIMATE", "UNKNOWN_FIELD"), "source header"),
    (lambda text: text.replace("C,END OF REPORT,8", "C,END OF REPORT,7"), "row count"),
    (lambda text: text.replace("C,END OF REPORT,8\r\n", ""), "report footer"),
    (lambda text: text.replace("2024/10/01 00:30:00", "bad timestamp"), "timestamp"),
    (lambda text: text.replace(",100,200", ",100,200,extra"), "Malformed data row"),
])
def test_malformed_archive_fails(tmp_path: Path, change, match: str) -> None:
    original = zipfile.ZipFile(io.BytesIO(archive_bytes(source_frame())))
    content = original.read(original.namelist()[0]).decode()
    path = tmp_path / "malformed.zip"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("sample.csv", change(content))
    with pytest.raises(ValueError, match=match):
        operational.read_archive(path)


class FakeResponse(io.BytesIO):
    def __init__(self, body: bytes):
        super().__init__(body)
        self.headers = {"Content-Length": str(len(body)), "Last-Modified": "Mon, 19 May 2025 12:00:00 GMT"}


def test_download_receipt_cache_hash_and_output_manifest(tmp_path: Path, monkeypatch) -> None:
    frame = source_frame()
    body = archive_bytes(pd.concat([frame, frame.iloc[[0]]], ignore_index=True))
    calls = []

    def fake_urlopen(request, timeout):
        calls.append(request.full_url)
        assert timeout == 30
        return FakeResponse(body)

    monkeypatch.setattr(operational, "urlopen", fake_urlopen)
    cache, output = tmp_path / "cache", tmp_path / "processed"
    manifest = operational.build_dataset("2024-10", "2024-10", cache, output)
    archive, receipt = operational.download_month("2024-10", cache)

    assert calls == [operational.archive_url("2024-10")]
    assert receipt["size_bytes"] == len(body)
    assert receipt["http_last_modified"] == "Mon, 19 May 2025 12:00:00 GMT"
    assert manifest["quantity"] == "measured_operational_demand_unadjusted_mw"
    assert manifest["archives"][0]["csv_member"].endswith(".CSV")
    assert len(manifest["archives"][0]["csv_sha256"]) == 64
    assert manifest["archives"][0]["raw_data_rows"] == 6
    assert manifest["archives"][0]["exact_duplicate_rows"] == 1
    assert manifest["coverage"]["exact_duplicate_rows_removed"] == 1
    assert manifest["adjustment_fields_added"] == []
    assert manifest["outputs"]["panel_nem_30min.csv"]["sha256"] == operational.sha256_file(output / "panel_nem_30min.csv")
    assert json.loads((output / "source_manifest.json").read_text()) == manifest
    assert (cache / "2024-10.json").is_file()

    archive.write_bytes(body + b"changed")
    with pytest.raises(ValueError, match="receipt/hash validation failed"):
        operational.download_month("2024-10", cache)
    assert len(calls) == 1


def test_invalid_download_leaves_no_partial_cache(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(operational, "urlopen", lambda *args, **kwargs: FakeResponse(b"not a zip"))
    with pytest.raises(ValueError, match="Invalid CSV ZIP"):
        operational.download_month("2024-10", tmp_path)
    assert not list(tmp_path.iterdir())


def test_unreceipted_archive_is_not_silently_adopted(tmp_path: Path) -> None:
    (tmp_path / "2024-10.zip").write_bytes(archive_bytes(source_frame()))
    with pytest.raises(ValueError, match="Incomplete cache pair"):
        operational.download_month("2024-10", tmp_path)


@pytest.mark.parametrize("start,end", [
    ("2024-1", "2024-12"), ("2024-13", "2024-13"), ("2024-12", "2024-10"),
])
def test_invalid_month_range_fails(start: str, end: str) -> None:
    with pytest.raises(ValueError):
        operational.month_range(start, end)
