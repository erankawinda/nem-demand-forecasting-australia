# October–December 2024 pilot protocol

Recorded on 6 September 2026, before inspecting model metrics for this pilot.

The target is the next half-hour mean of measured operational demand summed
across NSW1, QLD1, SA1, TAS1, and VIC1. Use the original half-hour
`DEMANDOPERATIONALACTUAL.OPERATIONAL_DEMAND` field, with neither
`OPERATIONAL_DEMAND_ADJUSTMENT` nor `WDR_ESTIMATE` added.

- Source window: interval starts from 1 October 2024 00:00 through
  31 December 2024 23:30, in fixed AEST (UTC+10).
- Preserve archive hashes and retrieval metadata. Use complete five-region
  intervals; do not fill missing demand. Allow finite negative regional values.
- Forecast at the end of the latest completed interval, with a 30-minute
  horizon. This retrospective evaluation assumes zero publication delay and
  uses the values in the downloaded archive, including any source revisions.
- Inputs: two latest completed intervals, target-slot demand one day and one
  week earlier, trailing 24-hour and seven-day means and standard deviations,
  and known calendar features. No weather or other external predictors.
- Split usable examples in time order: 70% training, 15% validation, 15% test.
  Discard examples lacking a complete seven-day history or next target.
- Compare persistence, previous-day and previous-week baselines with XGBoost
  and random forest. Report MAE and RMSE in MW for every method on both splits.
- Keep the existing fixed model settings in `src/analyze_nem.py`, seed 42,
  and one worker thread. Do not tune models on this pilot. Fit on training
  data for validation; refit on training plus validation for the final test.
- Save every test prediction, source and split manifests, exact code hashes,
  and plots. Describe the result as one quarterly historical pilot.

The month range was chosen for a small, manageable reproduction before the
target decision. It was not selected based on forecast accuracy. Changes to
this protocol after evaluation must be recorded explicitly.
