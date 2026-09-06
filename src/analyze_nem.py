#!/usr/bin/env python3
"""Evaluate 30-minute-ahead NEM demand forecasts chronologically.

Each example is indexed by a forecast origin ``t``. Its target is demand at
``t + 30 minutes``. Every input is observed at or before ``t`` or is a calendar
value already known for the target time.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
import xgboost
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = REPO_ROOT / "data" / "processed" / "panel_nem_30min.csv"
DEFAULT_REPORTS = REPO_ROOT / "reports"
HALF_HOUR = pd.Timedelta(minutes=30)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--reports", type=Path, default=DEFAULT_REPORTS)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=2,
        help="Worker threads used by both models (default: 2).",
    )
    return parser.parse_args()


def load_panel(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {"interval_start", "NEM_DEMAND_MW"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    timestamps = pd.to_datetime(frame["interval_start"], errors="coerce")
    if timestamps.isna().any():
        raise ValueError("The panel contains invalid timestamps.")
    if getattr(timestamps.dt, "tz", None) is not None:
        timestamps = timestamps.dt.tz_localize(None)
    frame["interval_start"] = timestamps
    demand = pd.to_numeric(frame["NEM_DEMAND_MW"], errors="coerce")
    if demand.isna().any():
        raise ValueError("The panel contains missing or non-numeric demand values.")
    if (demand <= 0).any():
        raise ValueError("The panel contains non-positive demand values.")
    frame["NEM_DEMAND_MW"] = demand
    frame = frame.sort_values("interval_start")
    if frame["interval_start"].duplicated().any():
        raise ValueError("The panel contains duplicate timestamps.")
    aligned = (
        frame["interval_start"].dt.minute.isin((0, 30))
        & frame["interval_start"].dt.second.eq(0)
        & frame["interval_start"].dt.microsecond.eq(0)
    )
    if not aligned.all():
        raise ValueError("Panel timestamps must align to 00 or 30 minutes past the hour.")

    # Reindexing exposes missing half hours. No imputation is performed: rows
    # whose required history crosses a gap are removed during feature building.
    return frame.set_index("interval_start").asfreq("30min")


def add_cyclical_time_features(
    features: pd.DataFrame, target_time: pd.DatetimeIndex
) -> None:
    minute_of_day = target_time.hour * 60 + target_time.minute
    features["target_time_sin"] = np.sin(2 * np.pi * minute_of_day / 1440)
    features["target_time_cos"] = np.cos(2 * np.pi * minute_of_day / 1440)
    features["target_week_sin"] = np.sin(
        2 * np.pi * (target_time.dayofweek * 48 + minute_of_day / 30) / 336
    )
    features["target_week_cos"] = np.cos(
        2 * np.pi * (target_time.dayofweek * 48 + minute_of_day / 30) / 336
    )
    features["target_month_sin"] = np.sin(2 * np.pi * target_time.month / 12)
    features["target_month_cos"] = np.cos(2 * np.pi * target_time.month / 12)
    features["target_is_weekend"] = (target_time.dayofweek >= 5).astype(int)


def build_examples(panel: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    demand = panel["NEM_DEMAND_MW"]
    examples = pd.DataFrame(index=panel.index)
    examples.index.name = "forecast_origin"
    examples["target_time"] = examples.index + HALF_HOUR
    examples["target_demand_mw"] = demand.shift(-1)

    # All values below are known at the forecast origin. The offsets of 47 and
    # 335 half hours line up with the target slot one day/week earlier.
    examples["demand_now_mw"] = demand
    examples["demand_lag_1_mw"] = demand.shift(1)
    examples["demand_same_slot_previous_day_mw"] = demand.shift(47)
    examples["demand_same_slot_previous_week_mw"] = demand.shift(335)
    examples["demand_mean_previous_24h_mw"] = demand.rolling(48).mean()
    examples["demand_std_previous_24h_mw"] = demand.rolling(48).std()
    examples["demand_mean_previous_7d_mw"] = demand.rolling(336).mean()
    examples["demand_std_previous_7d_mw"] = demand.rolling(336).std()

    add_cyclical_time_features(
        examples, pd.DatetimeIndex(examples["target_time"])
    )
    examples["baseline_persistence_mw"] = demand
    examples["baseline_previous_day_mw"] = demand.shift(47)
    examples["baseline_previous_week_mw"] = demand.shift(335)

    feature_columns = [
        "demand_now_mw",
        "demand_lag_1_mw",
        "demand_same_slot_previous_day_mw",
        "demand_same_slot_previous_week_mw",
        "demand_mean_previous_24h_mw",
        "demand_std_previous_24h_mw",
        "demand_mean_previous_7d_mw",
        "demand_std_previous_7d_mw",
        "target_time_sin",
        "target_time_cos",
        "target_week_sin",
        "target_week_cos",
        "target_month_sin",
        "target_month_cos",
        "target_is_weekend",
    ]
    required_columns = feature_columns + [
        "target_time",
        "target_demand_mw",
        "baseline_persistence_mw",
        "baseline_previous_day_mw",
        "baseline_previous_week_mw",
    ]
    examples = examples.dropna(subset=required_columns)
    if len(examples) < 1_000:
        raise ValueError(
            "Fewer than 1,000 complete forecast examples remain; "
            "a meaningful chronological evaluation is not possible."
        )
    return examples, feature_columns


def chronological_split(
    examples: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_end = int(len(examples) * 0.70)
    validation_end = int(len(examples) * 0.85)
    train = examples.iloc[:train_end]
    validation = examples.iloc[train_end:validation_end]
    test = examples.iloc[validation_end:]
    if min(len(train), len(validation), len(test)) == 0:
        raise ValueError("Chronological split produced an empty partition.")
    return train, validation, test


def make_models(seed: int, n_jobs: int) -> dict[str, object]:
    return {
        "xgboost": XGBRegressor(
            objective="reg:squarederror",
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.9,
            random_state=seed,
            n_jobs=n_jobs,
        ),
        "random_forest": RandomForestRegressor(
            n_estimators=160,
            max_depth=18,
            min_samples_leaf=2,
            random_state=seed,
            n_jobs=n_jobs,
        ),
    }


def metric_row(
    split: str, method: str, actual: pd.Series, prediction: np.ndarray
) -> dict:
    return {
        "split": split,
        "method": method,
        "n_examples": len(actual),
        "mae_mw": mean_absolute_error(actual, prediction),
        "rmse_mw": mean_squared_error(actual, prediction) ** 0.5,
    }


def evaluate_baselines(
    split_name: str, frame: pd.DataFrame
) -> tuple[list[dict], dict[str, np.ndarray]]:
    actual = frame["target_demand_mw"]
    predictions = {
        "persistence": frame["baseline_persistence_mw"].to_numpy(),
        "previous_day": frame["baseline_previous_day_mw"].to_numpy(),
        "previous_week": frame["baseline_previous_week_mw"].to_numpy(),
    }
    rows = [
        metric_row(split_name, name, actual, prediction)
        for name, prediction in predictions.items()
    ]
    return rows, predictions


def fit_and_predict(
    training: pd.DataFrame,
    evaluation: pd.DataFrame,
    feature_columns: list[str],
    seed: int,
    n_jobs: int,
) -> tuple[dict[str, object], dict[str, np.ndarray]]:
    models = make_models(seed, n_jobs)
    predictions: dict[str, np.ndarray] = {}
    for name, model in models.items():
        model.fit(training[feature_columns], training["target_demand_mw"])
        predictions[name] = model.predict(evaluation[feature_columns])
    return models, predictions


def save_forecast_plot(predictions: pd.DataFrame, path: Path) -> None:
    # A fixed final seven-day window keeps the plot readable and deterministic.
    sample = predictions.tail(7 * 48)
    plt.figure(figsize=(12, 5.5))
    plt.plot(
        sample["target_time"],
        sample["actual_demand_mw"],
        color="#1f2937",
        linewidth=1.8,
        label="Observed",
    )
    plt.plot(
        sample["target_time"],
        sample["persistence_mw"],
        color="#9ca3af",
        linewidth=1.1,
        label="Persistence baseline",
    )
    plt.plot(
        sample["target_time"],
        sample["xgboost_mw"],
        color="#0f766e",
        linewidth=1.2,
        label="XGBoost",
    )
    plt.xlabel("Target time")
    plt.ylabel("NEM demand (MW)")
    plt.title("Thirty-minute-ahead demand forecasts: final seven test days")
    plt.legend(frameon=False, ncol=3)
    plt.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()


def save_feature_importance(
    model: XGBRegressor, feature_columns: list[str], reports: Path
) -> None:
    importance = pd.DataFrame(
        {"feature": feature_columns, "model_feature_importance": model.feature_importances_}
    ).sort_values("model_feature_importance", ascending=False)
    importance.to_csv(reports / "xgboost_feature_importance.csv", index=False)

    displayed = importance.head(12).sort_values("model_feature_importance")
    plt.figure(figsize=(9, 5.5))
    plt.barh(
        displayed["feature"], displayed["model_feature_importance"], color="#0f766e"
    )
    plt.xlabel("XGBoost feature importance")
    plt.title("Model feature importance (not a causal effect)")
    plt.grid(axis="x", alpha=0.2)
    plt.tight_layout()
    plt.savefig(reports / "xgboost_feature_importance.png", dpi=200, bbox_inches="tight")
    plt.close()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    reports = args.reports.expanduser().resolve()
    reports.mkdir(parents=True, exist_ok=True)

    data_path = args.data.expanduser().resolve()
    panel = load_panel(data_path)
    examples, feature_columns = build_examples(panel)
    train, validation, test = chronological_split(examples)

    metrics: list[dict] = []
    validation_baseline_rows, _ = evaluate_baselines("validation", validation)
    metrics.extend(validation_baseline_rows)
    _, validation_model_predictions = fit_and_predict(
        train, validation, feature_columns, args.seed, args.n_jobs
    )
    for name, prediction in validation_model_predictions.items():
        metrics.append(
            metric_row("validation", name, validation["target_demand_mw"], prediction)
        )

    # Hyperparameters are fixed above. Refit each model on all pre-test data,
    # then evaluate exactly once on the untouched final chronological block.
    pre_test = pd.concat([train, validation])
    test_baseline_rows, test_baseline_predictions = evaluate_baselines("test", test)
    metrics.extend(test_baseline_rows)
    final_models, test_model_predictions = fit_and_predict(
        pre_test, test, feature_columns, args.seed, args.n_jobs
    )
    for name, prediction in test_model_predictions.items():
        metrics.append(metric_row("test", name, test["target_demand_mw"], prediction))

    metrics_frame = pd.DataFrame(metrics)
    metrics_frame.to_csv(reports / "metrics.csv", index=False)
    prediction_frame = pd.DataFrame(
        {
            "forecast_origin": test.index,
            "target_time": test["target_time"].to_numpy(),
            "actual_demand_mw": test["target_demand_mw"].to_numpy(),
            "persistence_mw": test_baseline_predictions["persistence"],
            "previous_day_mw": test_baseline_predictions["previous_day"],
            "previous_week_mw": test_baseline_predictions["previous_week"],
            "xgboost_mw": test_model_predictions["xgboost"],
            "random_forest_mw": test_model_predictions["random_forest"],
        }
    )
    prediction_frame.to_csv(reports / "test_predictions.csv", index=False)

    split_manifest = {
        "forecast_horizon_minutes": 30,
        "random_seed": args.seed,
        "input": {
            "filename": data_path.name,
            "sha256": sha256_file(data_path),
            "regular_grid_rows": len(panel),
            "observed_demand_rows": int(panel["NEM_DEMAND_MW"].notna().sum()),
            "start": str(panel.index.min()),
            "end": str(panel.index.max()),
        },
        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
            "xgboost": xgboost.__version__,
        },
        "total_examples": len(examples),
        "train": {
            "n": len(train),
            "start": str(train.index.min()),
            "end": str(train.index.max()),
        },
        "validation": {
            "n": len(validation),
            "start": str(validation.index.min()),
            "end": str(validation.index.max()),
        },
        "test": {
            "n": len(test),
            "start": str(test.index.min()),
            "end": str(test.index.max()),
        },
        "feature_columns": feature_columns,
    }
    (reports / "split_manifest.json").write_text(
        json.dumps(split_manifest, indent=2) + "\n", encoding="utf-8"
    )

    save_forecast_plot(prediction_frame, reports / "forecast_sample.png")
    save_feature_importance(final_models["xgboost"], feature_columns, reports)

    print("\nChronological evaluation")
    print(metrics_frame.to_string(index=False, float_format=lambda value: f"{value:.2f}"))
    print("\nWrote results to", reports)


if __name__ == "__main__":
    main()
