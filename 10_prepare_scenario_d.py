import argparse
import json
from pathlib import Path
from typing import Optional

import pandas as pd


FORECAST_SHEET = "Forecasts"
DATE_COL = "DATE"
ATM_COL = "CASHP_ID_ATM"
TARGET_COL = "WITHDRWLS_ATM"
PRED_COL = "Y_PRED_WITHDRWLS_ATM"


def load_forecasts(path: Path) -> pd.DataFrame:
    df = pd.read_excel(path, sheet_name=FORECAST_SHEET)
    required = {"WEEK_START", "TRAIN_END", "FORECAST_DATE", ATM_COL, PRED_COL}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Forecast sheet is missing required columns: {sorted(missing)}")

    df["WEEK_START"] = pd.to_datetime(df["WEEK_START"], errors="coerce").dt.normalize()
    df["TRAIN_END"] = pd.to_datetime(df["TRAIN_END"], errors="coerce").dt.normalize()
    df["FORECAST_DATE"] = pd.to_datetime(df["FORECAST_DATE"], errors="coerce").dt.normalize()
    df[ATM_COL] = df[ATM_COL].astype(str)
    df[PRED_COL] = pd.to_numeric(df[PRED_COL], errors="coerce")
    df = df.dropna(subset=["WEEK_START", "TRAIN_END", "FORECAST_DATE", PRED_COL]).copy()
    return df.sort_values(["WEEK_START", ATM_COL, "FORECAST_DATE"]).reset_index(drop=True)


def load_thresholds(path: Path, quantile: float) -> pd.Series:
    df = pd.read_excel(path)
    required = {DATE_COL, ATM_COL, TARGET_COL}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Historical data is missing required columns: {sorted(missing)}")

    df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors="coerce")
    df[ATM_COL] = df[ATM_COL].astype(str)
    df[TARGET_COL] = pd.to_numeric(df[TARGET_COL], errors="coerce")
    df = df.dropna(subset=[DATE_COL, ATM_COL, TARGET_COL]).copy()
    thresholds = df.groupby(ATM_COL)[TARGET_COL].quantile(quantile)
    thresholds.name = "threshold"
    return thresholds


def build_run_payload(
    run_df: pd.DataFrame,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    train_end: pd.Timestamp,
    week_start: pd.Timestamp,
    trigger_type: str,
    trigger_source_date: Optional[pd.Timestamp],
) -> dict:
    run_slice = run_df[(run_df["FORECAST_DATE"] >= start_date) & (run_df["FORECAST_DATE"] <= end_date)].copy()
    if run_slice.empty:
        raise ValueError(f"No forecast rows found for run {start_date.date()} -> {end_date.date()}")

    unique_dates = sorted(run_slice["FORECAST_DATE"].unique())
    t_to_date = {str(i + 1): pd.Timestamp(d).strftime("%d.%m.%Y") for i, d in enumerate(unique_dates)}
    date_to_t = {pd.Timestamp(d): i + 1 for i, d in enumerate(unique_dates)}
    run_slice["_t"] = run_slice["FORECAST_DATE"].map(date_to_t)

    dup = run_slice.duplicated(subset=[ATM_COL, "_t"], keep=False)
    if dup.any():
        bad = run_slice.loc[dup, [ATM_COL, "FORECAST_DATE", "_t", PRED_COL]].sort_values([ATM_COL, "_t"])
        raise ValueError(
            "Duplicate rows detected for the same (ATM, t).\n"
            f"Examples:\n{bad.head(20).to_string(index=False)}"
        )

    run_tag = "weekly" if trigger_type == "weekly" else "peak"
    r_json = {
        f"{row[ATM_COL]}|{int(row['_t'])}": float(row[PRED_COL])
        for _, row in run_slice.iterrows()
    }

    prefix = f"ScenarioD_{run_tag}_start_{start_date.strftime('%Y%m%d')}_end_{end_date.strftime('%Y%m%d')}"
    meta = {
        "scenario_prefix": prefix,
        "scenario_name": "ScenarioD",
        "trigger_type": trigger_type,
        "trigger_source_date": trigger_source_date.strftime("%d.%m.%Y") if trigger_source_date is not None else "",
        "planning_start_date": start_date.strftime("%d.%m.%Y"),
        "planning_end_date": end_date.strftime("%d.%m.%Y"),
        "reoptimization_date": train_end.strftime("%d.%m.%Y"),
        "base_week_start": week_start.strftime("%d.%m.%Y"),
        "horizon_days": len(unique_dates),
        "t_index_start": 1,
        "t_to_date": t_to_date,
        "source_forecast_sheet": FORECAST_SHEET,
    }

    return {
        "prefix": prefix,
        "r_json": r_json,
        "meta": meta,
        "rows": len(run_slice),
        "atms": run_slice[ATM_COL].nunique(),
    }


def build_runs(df_forecasts: pd.DataFrame, thresholds: pd.Series) -> list[dict]:
    runs: list[dict] = []

    for week_start, week_df in df_forecasts.groupby("WEEK_START", sort=True):
        week_df = week_df.copy()
        week_end = week_df["FORECAST_DATE"].max()
        train_end = week_df["TRAIN_END"].iloc[0]

        # Baseline weekly run.
        runs.append(
            build_run_payload(
                run_df=week_df,
                start_date=week_start,
                end_date=week_end,
                train_end=train_end,
                week_start=week_start,
                trigger_type="weekly",
                trigger_source_date=None,
            )
        )

        week_df = week_df.merge(thresholds, left_on=ATM_COL, right_index=True, how="left")
        peaks = week_df[week_df[PRED_COL] > week_df["threshold"]].copy()
        if peaks.empty:
            continue

        trigger_starts = sorted((peaks["FORECAST_DATE"] - pd.Timedelta(days=1)).drop_duplicates())
        for trigger_start in trigger_starts:
            trigger_start = pd.Timestamp(trigger_start).normalize()
            if trigger_start <= train_end or trigger_start < week_start or trigger_start > week_end:
                continue

            runs.append(
                build_run_payload(
                    run_df=week_df,
                    start_date=trigger_start,
                    end_date=week_end,
                    train_end=trigger_start - pd.Timedelta(days=1),
                    week_start=week_start,
                    trigger_type="peak_threshold",
                    trigger_source_date=trigger_start + pd.Timedelta(days=1),
                )
            )

    # Deduplicate exact same run window if multiple peaks create the same trigger.
    unique_runs = {}
    for run in runs:
        unique_runs[run["prefix"]] = run

    return [unique_runs[k] for k in sorted(unique_runs)]


def write_outputs(runs: list[dict], output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_rows = []

    for run in runs:
        run_dir = output_dir / run["prefix"]
        run_dir.mkdir(parents=True, exist_ok=True)

        r_path = run_dir / f"{run['prefix']}_Pred.json"
        meta_path = run_dir / f"{run['prefix']}_meta.json"

        r_path.write_text(json.dumps(run["r_json"], indent=2, ensure_ascii=False), encoding="utf-8")
        meta_path.write_text(json.dumps(run["meta"], indent=2, ensure_ascii=False), encoding="utf-8")

        summary_rows.append(
            {
                "prefix": run["prefix"],
                "trigger_type": run["meta"]["trigger_type"],
                "trigger_source_date": run["meta"]["trigger_source_date"],
                "planning_start_date": run["meta"]["planning_start_date"],
                "planning_end_date": run["meta"]["planning_end_date"],
                "reoptimization_date": run["meta"]["reoptimization_date"],
                "base_week_start": run["meta"]["base_week_start"],
                "horizon_days": run["meta"]["horizon_days"],
                "rows": run["rows"],
                "atms": run["atms"],
            }
        )

    summary_path = output_dir / "scenario_d_run_summary.csv"
    pd.DataFrame(summary_rows).to_csv(summary_path, index=False)
    return summary_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare Scenario D optimization inputs from old XGBoost forecasts.")
    parser.add_argument(
        "--forecast-file",
        default="yeni parametrelerle old/XGBoost_pred_weekend_filled_old.xlsx",
        help="Prediction workbook with Forecasts sheet.",
    )
    parser.add_argument(
        "--historical-file",
        default="ATM_Branch_Data_Final_filled.xlsx",
        help="Historical ATM data used to compute peak thresholds.",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs_scenario_d_fixed",
        help="Directory where Scenario D Pred/meta files will be written.",
    )
    parser.add_argument(
        "--quantile",
        type=float,
        default=0.90,
        help="Historical ATM-level quantile used as the demand peak threshold.",
    )
    args = parser.parse_args()

    forecast_file = Path(args.forecast_file)
    historical_file = Path(args.historical_file)
    output_dir = Path(args.output_dir)

    if not forecast_file.exists():
        raise FileNotFoundError(f"Forecast file not found: {forecast_file}")
    if not historical_file.exists():
        raise FileNotFoundError(f"Historical file not found: {historical_file}")
    if not (0 < args.quantile < 1):
        raise ValueError("--quantile must be between 0 and 1")

    df_forecasts = load_forecasts(forecast_file)
    thresholds = load_thresholds(historical_file, args.quantile)
    runs = build_runs(df_forecasts, thresholds)
    summary_path = write_outputs(runs, output_dir)

    weekly_runs = sum(1 for run in runs if run["meta"]["trigger_type"] == "weekly")
    peak_runs = sum(1 for run in runs if run["meta"]["trigger_type"] == "peak_threshold")

    print(f"Forecast file : {forecast_file}")
    print(f"Historical file: {historical_file}")
    print(f"Threshold q    : {args.quantile:.2f}")
    print(f"Weekly runs    : {weekly_runs}")
    print(f"Peak runs      : {peak_runs}")
    print(f"Total runs     : {len(runs)}")
    print(f"Summary        : {summary_path}")


if __name__ == "__main__":
    main()
