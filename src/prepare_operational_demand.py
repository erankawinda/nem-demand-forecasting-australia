#!/usr/bin/env python3
"""Download AEMO's measured half-hour operational demand and build a NEM panel.

``--start`` and ``--end`` are inclusive calendar months, for example
``--start 2024-10 --end 2024-12``. Each monthly archive includes the interval
ending at midnight on the first day of the following month. The output labels
that observation with its interval start, in fixed AEST (UTC+10, without DST).
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timedelta, timezone
import hashlib
import io
import json
import os
from pathlib import Path
import re
import tempfile
from urllib.request import Request, urlopen
import zipfile

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
NEM_REGIONS = ("NSW1", "QLD1", "SA1", "TAS1", "VIC1")
TABLE = ("OPERATIONAL_DEMAND", "ACTUAL", "3")
SOURCE_COLUMNS = (
    "INTERVAL_DATETIME", "REGIONID", "OPERATIONAL_DEMAND", "LASTCHANGED",
    "OPERATIONAL_DEMAND_ADJUSTMENT", "WDR_ESTIMATE",
)
QUANTITY = "measured_operational_demand_unadjusted_mw"
MARKET_TIMEZONE = timezone(timedelta(hours=10), name="AEST")
HALF_HOUR = pd.Timedelta(minutes=30)
SOURCE_DOCUMENTATION = (
    "https://www.aemo.com.au/energy-systems/electricity/"
    "national-electricity-market-nem/data-nem/operational-demand-data"
)


def month_range(start: str, end: str) -> pd.PeriodIndex:
    """Validate strict YYYY-MM inputs and return both endpoint months."""
    for value in (start, end):
        if not re.fullmatch(r"\d{4}-(0[1-9]|1[0-2])", value):
            raise ValueError(f"Expected a YYYY-MM month, received {value!r}.")
    if start > end:
        raise ValueError("The start month must not follow the end month.")
    return pd.period_range(start, end, freq="M")


def archive_url(month: str) -> str:
    month_range(month, month)
    year, number = month.split("-")
    return (
        "https://www.nemweb.com.au/Data_Archive/Wholesale_Electricity/MMSDM/"
        f"{year}/MMSDM_{year}_{number}/MMSDM_Historical_Data_SQLLoader/DATA/"
        f"PUBLIC_ARCHIVE%23DEMANDOPERATIONALACTUAL%23FILE01%23{year}{number}010000.zip"
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_write(path: Path, body: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(body)
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _write_json(path: Path, value: dict) -> None:
    _atomic_write(path, (json.dumps(value, indent=2, allow_nan=False) + "\n").encode())


def download_month(month: str, cache: Path) -> tuple[Path, dict]:
    """Reuse a receipt-verified snapshot or download and validate one atomically.

    An existing archive without a valid receipt is rejected. Move an incomplete
    cache pair aside before retrying; this avoids silently replacing a snapshot.
    """
    url = archive_url(month)
    archive = cache / f"{month}.zip"
    receipt_path = cache / f"{month}.json"
    if archive.exists() or receipt_path.exists():
        if not archive.is_file() or not receipt_path.is_file():
            raise ValueError(f"Incomplete cache pair for {month}; archive and receipt are required.")
        try:
            receipt = json.loads(receipt_path.read_text())
            valid = (
                isinstance(receipt, dict)
                and receipt["source_url"] == url
                and receipt["sha256"] == sha256_file(archive)
                and receipt["size_bytes"] == archive.stat().st_size
                and datetime.fromisoformat(receipt["retrieved_at_utc"]).utcoffset()
                == timedelta(0)
            )
        except (KeyError, TypeError, ValueError):
            valid = False
        if not valid:
            raise ValueError(f"Cache receipt/hash validation failed for {month}.")
        # Also check CSV/schema integrity rather than trusting a receipt alone.
        read_archive(archive)
        return archive, receipt

    request = Request(url, headers={"User-Agent": "nem-demand-forecasting-australia/1.0"})
    with urlopen(request, timeout=30) as response:
        body = response.read()
        expected_size = response.headers.get("Content-Length")
        if expected_size is not None and len(body) != int(expected_size):
            raise ValueError(f"Incomplete HTTP download for {month}.")
        last_modified = response.headers.get("Last-Modified")
    # Validate in memory before writing either cache file.
    _read_archive_bytes(body, str(archive))
    receipt = {
        "month": month,
        "source_url": url,
        "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
        "sha256": hashlib.sha256(body).hexdigest(),
        "size_bytes": len(body),
        "http_last_modified": last_modified,
    }
    _atomic_write(archive, body)
    _write_json(receipt_path, receipt)
    return archive, receipt


def _read_archive_bytes(body: bytes, label: str) -> pd.DataFrame:
    try:
        with zipfile.ZipFile(io.BytesIO(body)) as archive:
            members = [member for member in archive.infolist() if not member.is_dir()]
            if len(members) != 1 or not members[0].filename.lower().endswith(".csv"):
                raise ValueError(f"Expected exactly one CSV in {label}.")
            member_name = members[0].filename
            member_body = archive.read(members[0])
            content = member_body.decode("utf-8-sig")
    except (zipfile.BadZipFile, UnicodeError) as error:
        raise ValueError(f"Invalid CSV ZIP archive: {label}.") from error

    data: list[list[str]] = []
    header_seen = False
    footer_seen = False
    row_count = 0
    control_header: list[str] = []
    try:
        for row_count, row in enumerate(csv.reader(io.StringIO(content), strict=True), start=1):
            if not row or footer_seen:
                raise ValueError(f"Unexpected empty row or content after footer in {label}.")
            if row_count == 1:
                if row[0] != "C" or len(row) < 5 or row[1] == "END OF REPORT":
                    raise ValueError(f"Missing opening control metadata in {label}.")
                control_header = row
            if row[0] == "C":
                if len(row) >= 2 and row[1] == "END OF REPORT":
                    if len(row) != 3 or int(row[2]) != row_count:
                        raise ValueError(f"Invalid report row count in {label}.")
                    footer_seen = True
                elif row_count != 1:
                    raise ValueError(f"Unexpected control row in {label}.")
                continue
            if tuple(row[1:4]) != TABLE:
                raise ValueError(f"Unexpected table/version in {label}: {row[1:4]}.")
            if row[0] == "I":
                if header_seen or tuple(row[4:]) != SOURCE_COLUMNS:
                    raise ValueError(f"Unexpected or repeated source header in {label}.")
                header_seen = True
            elif row[0] == "D":
                if not header_seen or len(row) != len(SOURCE_COLUMNS) + 4:
                    raise ValueError(f"Malformed data row in {label} at line {row_count}.")
                data.append(row[4:])
            else:
                raise ValueError(f"Unexpected row type in {label}: {row[0]!r}.")
    except csv.Error as error:
        raise ValueError(f"Malformed CSV in {label}.") from error
    if not header_seen or not footer_seen or not data:
        raise ValueError(f"Archive requires a header, data and report footer: {label}.")

    frame = pd.DataFrame(data, columns=SOURCE_COLUMNS)
    for column in ("INTERVAL_DATETIME", "LASTCHANGED"):
        try:
            frame[column] = pd.to_datetime(
                frame[column], format="%Y/%m/%d %H:%M:%S", errors="raise", exact=True,
            )
        except (ValueError, TypeError) as error:
            raise ValueError(f"Invalid {column} timestamp in {label}.") from error
    validated = _validate_frame(frame, deduplicate=False)
    validated.attrs = {
        "csv_member": member_name,
        "csv_sha256": hashlib.sha256(member_body).hexdigest(),
        "csv_control_header": control_header,
        "raw_data_rows": len(validated),
        "exact_duplicate_rows": int(validated.duplicated().sum()),
    }
    return validated


def read_archive(path: Path) -> pd.DataFrame:
    """Read one original MMS archive without extracting its member to disk.

    Returns all six source fields, with timestamps in naive fixed-AEST market
    time and numeric demand fields. Exact duplicate records are retained here
    so prepare_panel can count and remove them. DataFrame attrs record the CSV
    member, hash, control header and original row counts.
    """
    return _read_archive_bytes(path.read_bytes(), str(path))


def _market_timestamp(value: object) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is not None:
        timestamp = timestamp.tz_convert(MARKET_TIMEZONE).tz_localize(None)
    return timestamp


def _validate_frame(frame: pd.DataFrame, *, deduplicate: bool = True) -> pd.DataFrame:
    missing = set(SOURCE_COLUMNS) - set(frame.columns)
    if missing:
        raise ValueError(f"Missing required source columns: {sorted(missing)}.")
    result = frame[list(SOURCE_COLUMNS)].copy()
    if result.empty or result.isna().any().any() or result.eq("").any().any():
        raise ValueError("Operational demand source contains empty or null fields.")
    for column in ("INTERVAL_DATETIME", "LASTCHANGED"):
        try:
            result[column] = result[column].map(_market_timestamp)
        except (TypeError, ValueError) as error:
            raise ValueError(f"Invalid {column} timestamp.") from error
        if result[column].isna().any():
            raise ValueError(f"Null {column} timestamp.")
    times = result["INTERVAL_DATETIME"]
    if not (times == times.dt.floor("30min")).all():
        raise ValueError("Operational demand timestamps must lie on the exact half-hour grid.")
    unknown = set(result["REGIONID"]) - set(NEM_REGIONS)
    if unknown:
        raise ValueError(f"Unknown NEM regions: {sorted(unknown)}.")
    for column in ("OPERATIONAL_DEMAND", "OPERATIONAL_DEMAND_ADJUSTMENT", "WDR_ESTIMATE"):
        try:
            result[column] = pd.to_numeric(result[column], errors="raise")
        except (TypeError, ValueError) as error:
            raise ValueError(f"Invalid numeric {column}.") from error
        if not np.isfinite(result[column]).all():
            raise ValueError(f"Nonfinite {column} values.")
    # Negative regional measurements are valid, for example during high solar output.
    unique = result.drop_duplicates()
    if unique.duplicated(["INTERVAL_DATETIME", "REGIONID"]).any():
        raise ValueError("Conflicting duplicate interval/region records; resolve source revisions first.")
    if deduplicate:
        result = unique
    return result.sort_values(["INTERVAL_DATETIME", "REGIONID"]).reset_index(drop=True)


def prepare_panel(
    frame: pd.DataFrame, start: str, end: str,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Return regional rows, complete five-region totals and a coverage audit.

    Regional rows retain adjustment and WDR fields for inspection. Neither is
    added to measured operational demand. Missing intervals are listed, never
    interpolated, and cannot appear as complete totals in the NEM panel.
    """
    months = month_range(start, end)
    start_time = months[0].start_time
    end_time = (months[-1] + 1).start_time
    source_rows = len(frame)
    regional = _validate_frame(frame)
    duplicate_rows_removed = source_rows - len(regional)
    regional["interval_start"] = regional["INTERVAL_DATETIME"] - HALF_HOUR
    in_window = regional["interval_start"].between(start_time, end_time, inclusive="left")
    outside_rows = int((~in_window).sum())
    regional = regional.loc[in_window].copy()
    expected = pd.date_range(start_time, end_time, freq="30min", inclusive="left")
    pivot = regional.pivot(
        index="interval_start", columns="REGIONID", values="OPERATIONAL_DEMAND",
    ).reindex(index=expected, columns=NEM_REGIONS)
    missing = pivot.isna()
    complete = ~missing.any(axis=1)
    panel = pivot.loc[complete].sum(axis=1).rename("NEM_DEMAND_MW").rename_axis("interval_start").reset_index()
    regional = regional[["interval_start", *SOURCE_COLUMNS]].reset_index(drop=True)
    missing_intervals = [
        {"interval_start": timestamp.isoformat(), "missing_regions": list(missing.columns[row])}
        for timestamp, row in missing.loc[~complete].iterrows()
    ]
    coverage = {
        "requested_start_month": start,
        "requested_end_month_inclusive": end,
        "interval_start_inclusive": start_time.isoformat(),
        "interval_start_exclusive_end": end_time.isoformat(),
        "expected_intervals": len(expected),
        "complete_intervals": int(complete.sum()),
        "missing_or_incomplete_intervals": int((~complete).sum()),
        "missing_region_observations": int(missing.to_numpy().sum()),
        "observed_regional_rows": len(regional),
        "exact_duplicate_rows_removed": duplicate_rows_removed,
        "rows_outside_requested_window": outside_rows,
        "missing_intervals": missing_intervals,
        "imputation": "none",
    }
    return regional, panel, coverage


def build_dataset(start: str, end: str, cache: Path, output: Path) -> dict:
    frames = []
    receipts = []
    for month in month_range(start, end).astype(str):
        path, receipt = download_month(month, cache)
        frame = read_archive(path)
        frames.append(frame)
        receipts.append({"cache_file": path.name, **receipt, **frame.attrs})
    regional, panel, coverage = prepare_panel(pd.concat(frames, ignore_index=True), start, end)
    if panel.empty:
        raise ValueError("No complete five-region intervals are available in the requested months.")
    output.mkdir(parents=True, exist_ok=True)
    output_frames = {
        "panel_nem_30min.csv": panel,
        "regional_operational_demand_30min.csv": regional,
    }
    output_records = {}
    for filename, frame in output_frames.items():
        path = output / filename
        _atomic_write(path, frame.to_csv(index=False, date_format="%Y-%m-%d %H:%M:%S").encode())
        output_records[filename] = {"sha256": sha256_file(path), "rows": len(frame)}
    manifest = {
        "schema_version": 1,
        "quantity": QUANTITY,
        "unit": "MW",
        "source_table": "DEMANDOPERATIONALACTUAL",
        "source_csv_table": list(TABLE),
        "source_field": "OPERATIONAL_DEMAND",
        "source_documentation": SOURCE_DOCUMENTATION,
        "regions": list(NEM_REGIONS),
        "aggregation": "Sum of the five regional measured 30-minute mean demand values.",
        "adjustment_fields_added": [],
        "excluded_adjustment_fields": ["OPERATIONAL_DEMAND_ADJUSTMENT", "WDR_ESTIMATE"],
        "time_contract": {
            "timezone": "AEST (UTC+10), fixed throughout the year; no daylight saving",
            "source_timestamp": "INTERVAL_DATETIME labels the end of the measured half hour",
            "panel_timestamp": "interval_start = INTERVAL_DATETIME minus 30 minutes",
            "interval_minutes": 30,
        },
        "revision_scope": "Historical archive snapshots may include later corrections; publication-time vintages are not reconstructed.",
        "archives": receipts,
        "coverage": coverage,
        "outputs": output_records,
    }
    _write_json(output / "source_manifest.json", manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", required=True, help="First month, YYYY-MM (inclusive).")
    parser.add_argument("--end", required=True, help="Last month, YYYY-MM (inclusive).")
    parser.add_argument("--cache", type=Path, default=REPO_ROOT / "data/raw/operational_demand")
    parser.add_argument("--output", type=Path, default=REPO_ROOT / "data/processed")
    args = parser.parse_args()
    manifest = build_dataset(args.start, args.end, args.cache, args.output)
    coverage = manifest["coverage"]
    print(f"Saved {coverage['complete_intervals']} complete intervals to {args.output}.")
    print(f"Missing or incomplete intervals: {coverage['missing_or_incomplete_intervals']}.")


if __name__ == "__main__":
    main()
