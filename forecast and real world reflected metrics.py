from __future__ import annotations

import re
from pathlib import Path
import pandas as pd
import numpy as np

# =========================
# CONFIG
# =========================
RESULT_DIR = Path("/Users/porsa/Desktop/result")

SCENARIO_ORDER = [
    "Scenario0",
    "ScenarioA",
    "ScenarioE",
    "ScenarioF",
    "ScenarioF_5pct",
]

SCENARIO_LABEL_MAP = {
    "Scenario0": "Scenario 0",
    "ScenarioA": "Scenario 1",
    "ScenarioE": "Scenario 2",
    "ScenarioF": "Scenario 3",
    "ScenarioF_5pct": "Scenario 4",
}

DEMAND_FILES = {
    "Scenario0": Path("/Users/porsa/Desktop/result/scenario0_avg_demands_from_nextplanning.xlsx"),
    "ScenarioA": Path("/Users/porsa/Desktop/result/XGBoost_pred_weekend_filled_old (1).xlsx"),
    "ScenarioE": Path("/Users/porsa/Desktop/result/scenario_E_fixed_3day_reopt_forecasts_XGBoost (2).xlsx"),
    "ScenarioF": Path("/Users/porsa/Desktop/result/scenario_E_daily_reopt_forecasts_XGBoost (1).xlsx"),
    "ScenarioF_5pct": Path("/Users/porsa/Desktop/result/scenario_E_daily_reopt_forecasts_XGBoost (1).xlsx"),
}

# real daily demand for Scenario 0
SCENARIO0_REAL_FILE = Path("/Users/porsa/Desktop/result/XGBoost_pred_weekend_filled_old (1).xlsx")

OUTPUT_FILE = RESULT_DIR / "real_world_metrics1.csv"

# =========================
# REGEX
# =========================
SERVICE_PATTERN = re.compile(
    r"t=\d+ \| date=(\d{2}\.\d{2}\.\d{4}) \| ATM=([A-Za-z0-9_]+)"
)

# =========================
# SCENARIO DETECTION
# =========================
def detect_scenario_from_filename(filename: str) -> str | None:
    name = filename.lower()

    if "scenariof_5pct" in name:
        return "ScenarioF_5pct"
    if "scenario0" in name:
        return "Scenario0"
    if "scenarioa" in name or "scenario1" in name:
        return "ScenarioA"
    if "scenarioe" in name or "scenario2" in name:
        return "ScenarioE"
    if "scenariof" in name or "scenario3" in name:
        return "ScenarioF"
    return None

# =========================
# PARSE SERVICE
# =========================
def parse_result_file(file_path: Path) -> pd.DataFrame:
    rows = []

    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            m = SERVICE_PATTERN.search(line)
            if m:
                date_str, atm = m.groups()
                rows.append({
                    "atm_id": str(atm).strip(),
                    "date": pd.to_datetime(date_str, dayfirst=True).date(),
                    "served": 1,
                })

    if not rows:
        return pd.DataFrame(columns=["atm_id", "date", "served"])

    return pd.DataFrame(rows)

def load_all_services() -> pd.DataFrame:
    all_rows = []

    for f in RESULT_DIR.glob("*.txt"):
        scenario = detect_scenario_from_filename(f.name)
        if scenario is None:
            continue

        df = parse_result_file(f)
        if df.empty:
            continue

        df["scenario"] = scenario
        all_rows.append(df)

    if not all_rows:
        return pd.DataFrame(columns=["atm_id", "date", "served", "scenario"])

    out = pd.concat(all_rows, ignore_index=True)
    out["atm_id"] = out["atm_id"].astype(str).str.strip()
    return out

# =========================
# LOAD DEMAND
# =========================
def load_regular_demand(file_path: Path) -> pd.DataFrame:
    df = pd.read_excel(file_path)
    df.columns = df.columns.str.strip()

    df["atm_id"] = df["CASHP_ID_ATM"].astype(str).str.strip()
    df["date"] = pd.to_datetime(df["FORECAST_DATE"]).dt.date
    df["real_demand"] = pd.to_numeric(df["Y_TRUE_WITHDRWLS_ATM"], errors="coerce")
    df["forecast_demand"] = pd.to_numeric(df["Y_PRED_WITHDRWLS_ATM"], errors="coerce")

    return df[["atm_id", "date", "real_demand", "forecast_demand"]]

def load_scenario0_demand() -> pd.DataFrame:
    f0 = pd.read_excel(DEMAND_FILES["Scenario0"])
    f0.columns = f0.columns.astype(str).str.strip()

    rows = []
    for _, r in f0.iterrows():
        if pd.isna(r["date_range"]):
            continue

        parts = str(r["date_range"]).split(" - ")
        if len(parts) != 2:
            continue

        start_str, end_str = parts
        dates = pd.date_range(start=start_str, end=end_str, freq="D")

        atm_id = str(r["atm_id"]).strip()
        forecast_val = pd.to_numeric(r["avg_withdrawal"], errors="coerce")

        for d in dates:
            rows.append({
                "atm_id": atm_id,
                "date": d.date(),
                "forecast_demand": forecast_val
            })

    scen0_forecast = pd.DataFrame(rows)

    real_df = pd.read_excel(SCENARIO0_REAL_FILE)
    real_df.columns = real_df.columns.astype(str).str.strip()

    real_df["atm_id"] = real_df["CASHP_ID_ATM"].astype(str).str.strip()
    real_df["date"] = pd.to_datetime(real_df["FORECAST_DATE"]).dt.date
    real_df["real_demand"] = pd.to_numeric(real_df["Y_TRUE_WITHDRWLS_ATM"], errors="coerce")
    real_df = real_df[["atm_id", "date", "real_demand"]]

    out = scen0_forecast.merge(real_df, on=["atm_id", "date"], how="left")
    return out[["atm_id", "date", "real_demand", "forecast_demand"]]

def load_demand(scenario: str) -> pd.DataFrame:
    if scenario == "Scenario0":
        return load_scenario0_demand()
    return load_regular_demand(DEMAND_FILES[scenario])

# =========================
# METRICS
# =========================
def compute_metrics(service_df: pd.DataFrame, demand_df: pd.DataFrame, scenario_name: str = "") -> dict:
    df = demand_df.merge(
        service_df[["atm_id", "date", "served"]],
        on=["atm_id", "date"],
        how="left"
    )
    df["served"] = df["served"].fillna(0)

    # keep only rows with valid demand values for forecast metrics
    metric_df = df.copy()
    metric_df["real_demand"] = pd.to_numeric(metric_df["real_demand"], errors="coerce")
    metric_df["forecast_demand"] = pd.to_numeric(metric_df["forecast_demand"], errors="coerce")
    metric_df = metric_df.dropna(subset=["real_demand", "forecast_demand"])

    # -------------------------
    # Forecast metrics
    # -------------------------
    if len(metric_df) > 0:
        abs_err = (metric_df["real_demand"] - metric_df["forecast_demand"]).abs()
        err = metric_df["forecast_demand"] - metric_df["real_demand"]

        mae = abs_err.mean()
        bias = err.mean()

        total_real = metric_df["real_demand"].sum()
        wmape = abs_err.sum() / total_real if total_real > 0 else np.nan

        real_peak_threshold = metric_df["real_demand"].quantile(0.9)
        real_peak_days = metric_df[metric_df["real_demand"] >= real_peak_threshold]

        forecast_peak_threshold = metric_df["forecast_demand"].quantile(0.9)
        forecast_peak_days = metric_df[metric_df["forecast_demand"] >= forecast_peak_threshold]

        if len(real_peak_days) > 0:
            real_peak_index = set(zip(real_peak_days["atm_id"], real_peak_days["date"]))
            forecast_peak_index = set(zip(forecast_peak_days["atm_id"], forecast_peak_days["date"]))
            correctly_captured_peaks = len(real_peak_index & forecast_peak_index)
            peak_capture = correctly_captured_peaks / len(real_peak_index)
        else:
            peak_capture = np.nan
    else:
        mae = np.nan
        bias = np.nan
        wmape = np.nan
        peak_capture = np.nan

    # -------------------------
    # Realized operational metrics
    # -------------------------
    total_demand_realized = pd.to_numeric(df["real_demand"], errors="coerce").sum()
    served_demand_realized = pd.to_numeric(
        df.loc[df["served"] == 1, "real_demand"], errors="coerce"
    ).sum()

    service_level_real = (
        served_demand_realized / total_demand_realized
        if total_demand_realized > 0 else np.nan
    )

    real_series = pd.to_numeric(df["real_demand"], errors="coerce")
    if real_series.notna().any():
        real_threshold = real_series.quantile(0.9)
        peak_real = df[real_series >= real_threshold]
        peak_capture_real = peak_real["served"].mean() if len(peak_real) > 0 else np.nan
    else:
        peak_capture_real = np.nan

    forecast_series = pd.to_numeric(df["forecast_demand"], errors="coerce")
    forecast_served = pd.to_numeric(
        df.loc[df["served"] == 1, "forecast_demand"], errors="coerce"
    ).sum()
    total_forecast = forecast_series.sum()
    timing_accuracy_forecast = (
        forecast_served / total_forecast if total_forecast > 0 else np.nan
    )

    if forecast_series.notna().any():
        forecast_threshold = forecast_series.quantile(0.9)
        peak_forecast = df[forecast_series >= forecast_threshold]
        peak_capture_forecast = peak_forecast["served"].mean() if len(peak_forecast) > 0 else np.nan
    else:
        peak_capture_forecast = np.nan

    return {
        # forecast metrics
        "MAE": mae,
        "wMAPE": wmape,
        "Bias": bias,
        "Peak_Capture": peak_capture,

        # realized operational metrics
        "service_level_real": service_level_real,
        "peak_capture_real": peak_capture_real,
        "timing_accuracy_forecast": timing_accuracy_forecast,
        "peak_capture_forecast": peak_capture_forecast,
    }

# =========================
# MAIN
# =========================
def main():
    service_all = load_all_services()
    results = []

    for scenario in SCENARIO_ORDER:
        print(f"Processing {scenario}...")
        service_df = service_all[service_all["scenario"] == scenario].copy()
        demand_df = load_demand(scenario)

        metrics = compute_metrics(service_df, demand_df, scenario)
        metrics["scenario"] = SCENARIO_LABEL_MAP[scenario]
        results.append(metrics)

    out_df = pd.DataFrame(results)

    # optional column order
    desired_cols = [
        "scenario",
        "MAE",
        "wMAPE",
        "Bias",
        "Peak_Capture",
        "service_level_real",
        "peak_capture_real",
        "timing_accuracy_forecast",
        "peak_capture_forecast",
    ]
    out_df = out_df[[c for c in desired_cols if c in out_df.columns]]

    out_df.to_csv(OUTPUT_FILE, index=False)

    print("\nSaved to:", OUTPUT_FILE)
    print(out_df)

if __name__ == "__main__":
    main()