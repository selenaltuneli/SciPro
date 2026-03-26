from __future__ import annotations

import re
from pathlib import Path
import pandas as pd

# =========================================================
# CONFIG
# =========================================================
RESULT_DIR = Path("/Users/porsa/Desktop/result")

RUN_LEVEL_CSV = RESULT_DIR / "run_level_kpis_final.csv"
SCENARIO_SUMMARY_CSV = RESULT_DIR / "scenario_summary_final.csv"
FAILED_PARSING_CSV = RESULT_DIR / "failed_parsing_files.csv"

VALID_SCENARIOS = {
    "Scenario0",
    "ScenarioA",
    "ScenarioE",
    "ScenarioF",
    "ScenarioF_5pct",
}

SCENARIO_LABEL_MAP = {
    "Scenario0": "Scenario 0",
    "ScenarioA": "Scenario 1",
    "ScenarioE": "Scenario 2",
    "ScenarioF": "Scenario 3",
    "ScenarioF_5pct": "Scenario 4",
}

# Files that contain the demand fed into the optimization model
SCENARIO_DEMAND_FILES = {
    "Scenario0": Path("/Users/porsa/Desktop/result/scenario0_avg_demands_from_nextplanning.xlsx"),
    "ScenarioA": Path("/Users/porsa/Desktop/result/XGBoost_pred_weekend_filled_old (1).xlsx"),
    "ScenarioE": Path("/Users/porsa/Desktop/result/scenario_E_fixed_3day_reopt_forecasts_XGBoost (2).xlsx"),
    "ScenarioF": Path("/Users/porsa/Desktop/result/scenario_E_daily_reopt_forecasts_XGBoost (1).xlsx"),
    "ScenarioF_5pct": Path("/Users/porsa/Desktop/result/scenario_E_daily_reopt_forecasts_XGBoost (1).xlsx"),
}

SCENARIO_DEMAND_COLUMN = {
    "Scenario0": "avg_withdrawal",
    "ScenarioA": "Y_PRED_WITHDRWLS_ATM",
    "ScenarioE": "Y_PRED_WITHDRWLS_ATM",
    "ScenarioF": "Y_PRED_WITHDRWLS_ATM",
    "ScenarioF_5pct": "Y_PRED_WITHDRWLS_ATM",
}

# likely date column names inside the forecast/demand Excel files
DATE_COLUMN_CANDIDATES = [
    "date",
    "Date",
    "DATE",
    "datum",
    "Datum",
    "DATUM",
    "day",
    "Day",
    "forecast_date",
    "Forecast_Date",
    "planning_date",
    "Planning_Date",
    "ds",
    "DS",
]

# =========================================================
# HELPERS
# =========================================================
def to_float(x: str):
    try:
        return float(x)
    except Exception:
        return None


def to_int(x: str):
    try:
        return int(x)
    except Exception:
        return None


def safe_div(a, b):
    try:
        if a is None or b is None or pd.isna(a) or pd.isna(b) or b == 0:
            return None
        return a / b
    except Exception:
        return None


def calc_service_level(shortage, demand):
    ratio = safe_div(shortage, demand)
    if ratio is None:
        return None
    return 1 - ratio


def extract_first(pattern: str, text: str, flags=0):
    m = re.search(pattern, text, flags)
    return m.group(1) if m else None


def normalize_scenario_name(raw_name: str):
    if raw_name is None:
        return None

    x = str(raw_name).strip()

    if x.lower().startswith("scenario 0"):
        return "Scenario0"
    if x.startswith("Scenario0"):
        return "Scenario0"
    if x.startswith("ScenarioF_5pct"):
        return "ScenarioF_5pct"
    if x.startswith("ScenarioF"):
        return "ScenarioF"
    if x.startswith("ScenarioE"):
        return "ScenarioE"
    if x.startswith("ScenarioA"):
        return "ScenarioA"
    if x.startswith("Scenario1"):
        return "ScenarioA"

    if x.startswith("ScenarioD"):
        return "ScenarioD"
    if x.startswith("Scenario2") or x.lower().startswith("scenario2"):
        return "Scenario2"

    return x


def detect_scenario_from_text(saved_path: str | None, file_name: str):
    scenario = None
    run_type = None
    start_ymd = None
    end_ymd = None

    saved_path = saved_path or ""

    m_peak = re.search(
        r"(Scenario[A-Za-z0-9_]+?)_peak_start_(\d{8})_end_(\d{8})",
        saved_path,
    )
    if m_peak:
        scenario = normalize_scenario_name(m_peak.group(1))
        run_type = "split_peak"
        start_ymd = m_peak.group(2)
        end_ymd = m_peak.group(3)
        return scenario, run_type, start_ymd, end_ymd

    m_saved_scn = re.search(r"(Scenario(?:0|A|D|E|F(?:_5pct)?|1|2))", saved_path)
    if m_saved_scn:
        scenario = normalize_scenario_name(m_saved_scn.group(1))

    m_saved_dates = re.search(r"start_(\d{8})_end_(\d{8})", saved_path)
    if m_saved_dates:
        start_ymd = m_saved_dates.group(1)
        end_ymd = m_saved_dates.group(2)

    if scenario is None:
        stem = Path(file_name).stem
        parts = stem.split("__")
        raw_prefix = parts[0].strip() if parts else stem
        scenario = normalize_scenario_name(raw_prefix)

    return scenario, run_type, start_ymd, end_ymd


def parse_filename_info(file_name: str, saved_path: str | None = None):
    stem = Path(file_name).stem
    scenario, run_type, start_ymd, end_ymd = detect_scenario_from_text(saved_path, file_name)

    if start_ymd is None or end_ymd is None:
        date_matches = re.findall(r"(\d{8})", stem)
        if len(date_matches) >= 2:
            start_ymd = date_matches[-2]
            end_ymd = date_matches[-1]

    if run_type is None:
        if scenario in {"Scenario0", "ScenarioA"}:
            run_type = "weekly"
        elif scenario == "ScenarioE":
            run_type = "three_day"
        elif scenario in {"ScenarioF", "ScenarioF_5pct"}:
            run_type = "daily"
        else:
            run_type = "standard"

    return {
        "scenario": scenario,
        "run_type": run_type,
        "start_ymd": start_ymd,
        "end_ymd": end_ymd,
    }


def find_date_column(df: pd.DataFrame):
    exact_hits = [c for c in DATE_COLUMN_CANDIDATES if c in df.columns]
    if exact_hits:
        return exact_hits[0]

    lowered = {str(c).strip().lower(): c for c in df.columns}
    for c in DATE_COLUMN_CANDIDATES:
        hit = lowered.get(c.lower())
        if hit is not None:
            return hit

    for c in df.columns:
        c_low = str(c).strip().lower()
        if "date" in c_low or "datum" in c_low or c_low == "ds":
            return c

    return None


def load_scenario_demand_tables():
    demand_tables = {}

    for scenario, path in SCENARIO_DEMAND_FILES.items():
        if not path.exists():
            print(f"Warning: demand file not found for {scenario}: {path}")
            continue

        demand_col = SCENARIO_DEMAND_COLUMN[scenario]

        try:
            df = pd.read_excel(path)
        except Exception as e:
            print(f"Warning: could not read demand file for {scenario}: {path} | {e}")
            continue

        if demand_col not in df.columns:
            print(f"Warning: demand column '{demand_col}' not found in {path.name}")
            continue

        # -------------------------------------------------
        # Scenario 0:
        # avg_withdrawal is a DAILY average repeated across the week.
        # Therefore total run demand = sum(avg_withdrawal * days_repeated)
        # across all ATMs in the same run window.
        # -------------------------------------------------
        if scenario == "Scenario0":
            source_cols = [c for c in ["scenario_folder", "json_file", "date_range"] if c in df.columns]
            repeat_col = "days_repeated" if "days_repeated" in df.columns else None

            if not source_cols:
                print(f"Warning: no run-identifying columns found in {path.name}")
                continue

            tmp = df[[demand_col] + source_cols].copy()
            tmp["demand_value"] = pd.to_numeric(tmp[demand_col], errors="coerce")
            tmp["days_repeated"] = pd.to_numeric(df[repeat_col], errors="coerce") if repeat_col else 7

            tmp["start_ymd"] = None
            tmp["end_ymd"] = None

            for col in source_cols:
                s = tmp[col].astype(str)

                # pattern like Scenario0_20071001_20071007
                m1 = s.str.extract(r"(\d{8})_(\d{8})")
                fill_mask = tmp["start_ymd"].isna() & m1[0].notna() & m1[1].notna()
                tmp.loc[fill_mask, "start_ymd"] = m1.loc[fill_mask, 0]
                tmp.loc[fill_mask, "end_ymd"] = m1.loc[fill_mask, 1]

                # pattern like 01.10.2007 - 07.10.2007
                m2 = s.str.extract(r"(\d{2}\.\d{2}\.\d{4})\s*-\s*(\d{2}\.\d{2}\.\d{4})")
                if not m2.empty:
                    start2 = pd.to_datetime(m2[0], format="%d.%m.%Y", errors="coerce")
                    end2 = pd.to_datetime(m2[1], format="%d.%m.%Y", errors="coerce")
                    fill_mask = tmp["start_ymd"].isna() & start2.notna() & end2.notna()
                    tmp.loc[fill_mask, "start_ymd"] = start2[fill_mask].dt.strftime("%Y%m%d")
                    tmp.loc[fill_mask, "end_ymd"] = end2[fill_mask].dt.strftime("%Y%m%d")

            tmp = tmp.dropna(subset=["start_ymd", "end_ymd", "demand_value", "days_repeated"]).copy()
            tmp["run_total_demand"] = tmp["demand_value"] * tmp["days_repeated"]

            run_level = (
                tmp.groupby(["start_ymd", "end_ymd"], as_index=False)["run_total_demand"]
                .sum()
                .rename(columns={"run_total_demand": "total_input_demand"})
            )
            run_level["scenario"] = scenario
            demand_tables[scenario] = run_level
            continue

        # -------------------------------------------------
        # Other scenarios:
        # demand is daily, so we sum over all days in the run window.
        # -------------------------------------------------
        date_col = find_date_column(df)
        if date_col is None:
            print(f"Warning: no usable date column found in {path.name}")
            continue

        tmp = df[[date_col, demand_col]].copy()
        tmp = tmp.rename(columns={date_col: "date_raw", demand_col: "demand_value"})
        tmp["date"] = pd.to_datetime(tmp["date_raw"], errors="coerce", dayfirst=True)
        tmp["demand_value"] = pd.to_numeric(tmp["demand_value"], errors="coerce")
        tmp = tmp.dropna(subset=["date", "demand_value"]).copy()
        tmp["date"] = tmp["date"].dt.normalize()

        daily = tmp.groupby("date", as_index=False)["demand_value"].sum()
        daily["scenario"] = scenario
        demand_tables[scenario] = daily

    return demand_tables


def get_total_input_demand_for_run(scenario: str, start_ymd: str | None, end_ymd: str | None, demand_tables: dict):
    table = demand_tables.get(scenario)
    if table is None or table.empty:
        return None

    if not start_ymd or not end_ymd:
        return None

    # Scenario 0 is already stored at run level
    if scenario == "Scenario0":
        match = table[
            (table["start_ymd"].astype(str) == str(start_ymd)) &
            (table["end_ymd"].astype(str) == str(end_ymd))
        ]
        if match.empty:
            return None
        return pd.to_numeric(match["total_input_demand"], errors="coerce").sum()

    # Other scenarios are stored at daily level
    try:
        start_dt = pd.to_datetime(str(start_ymd), format="%Y%m%d").normalize()
        end_dt = pd.to_datetime(str(end_ymd), format="%Y%m%d").normalize()
    except Exception:
        return None

    mask = (table["date"] >= start_dt) & (table["date"] <= end_dt)
    return pd.to_numeric(table.loc[mask, "demand_value"], errors="coerce").sum()


# =========================================================
# SERVICE SUMMARY / DAILY ROUTES PARSERS
# =========================================================
SERVICE_LINE_RE = re.compile(
    r"t=(\d+)\s*\|\s*date=([0-9.]+)\s*\|\s*ATM=([^\s|]+)\s*\|\s*k=(\d+)\s*\|\s*y=(\d+)\s*"
    r"\|\s*q=([0-9.]+)\s*\|\s*p=([0-9.]+)\s*\|\s*w=([0-9.]+)\s*\|\s*gamma=([0-9.]+)"
)

DAY_HEADER_RE = re.compile(r"Day t=(\d+)\s*\|\s*date=([0-9.]+),\s*k=(\d+):")
NO_ROUTE_DAY_RE = re.compile(r"Day t=(\d+)\s*\|\s*date=([0-9.]+),\s*k=(\d+):\s*\(no route\)")
DAY_TOTAL_KM_RE = re.compile(r"total_km\s*=\s*([0-9.]+)\s*km")
DAY_ROUTE_COST_RE = re.compile(r"route_cost\s*=\s*([0-9.]+)")


def parse_service_summary(text: str):
    service_matches = SERVICE_LINE_RE.findall(text)

    daily_visit_counts = {}
    for t_str, _date_str, _atm, _k_str, _y_str, _q_str, _p_str, _w_str, _gamma_str in service_matches:
        t = int(t_str)
        daily_visit_counts[t] = daily_visit_counts.get(t, 0) + 1

    return {
        "daily_visit_counts": daily_visit_counts,
    }


def parse_daily_routes(text: str):
    lines = text.splitlines()
    day_data = {}
    current_t = None

    for raw_line in lines:
        line = raw_line.strip()

        m_no_route = NO_ROUTE_DAY_RE.match(line)
        if m_no_route:
            t = int(m_no_route.group(1))
            date_str = m_no_route.group(2)
            day_data[t] = {
                "date": date_str,
                "has_route": False,
                "total_km": 0.0,
                "route_cost": 0.0,
            }
            current_t = None
            continue

        m_day = DAY_HEADER_RE.match(line)
        if m_day:
            t = int(m_day.group(1))
            date_str = m_day.group(2)
            day_data[t] = {
                "date": date_str,
                "has_route": True,
                "total_km": None,
                "route_cost": None,
            }
            current_t = t
            continue

        if current_t is None:
            continue

        m_km = DAY_TOTAL_KM_RE.search(line)
        if m_km:
            day_data[current_t]["total_km"] = float(m_km.group(1))
            continue

        m_rc = DAY_ROUTE_COST_RE.search(line)
        if m_rc:
            day_data[current_t]["route_cost"] = float(m_rc.group(1))
            continue

    return day_data


def detect_horizon_days(start_ymd, end_ymd, friendly_date_start, friendly_date_end, day_data):
    if start_ymd and end_ymd:
        try:
            start_dt = pd.to_datetime(str(start_ymd), format="%Y%m%d")
            end_dt = pd.to_datetime(str(end_ymd), format="%Y%m%d")
            return int((end_dt - start_dt).days) + 1
        except Exception:
            pass

    if friendly_date_start and friendly_date_end:
        try:
            start_dt = pd.to_datetime(friendly_date_start, format="%d.%m.%Y")
            end_dt = pd.to_datetime(friendly_date_end, format="%d.%m.%Y")
            return int((end_dt - start_dt).days) + 1
        except Exception:
            pass

    if day_data:
        try:
            return int(max(day_data.keys()))
        except Exception:
            pass

    return None


# =========================================================
# CORE PARSER
# =========================================================
def parse_one_file(path: Path):
    text = path.read_text(encoding="utf-8", errors="ignore")

    result_filename = extract_first(r"Result filename:\s*(.+)", text)
    saved_path = extract_first(r"Results will also be saved to:\s*(.+)", text)

    file_info = parse_filename_info(path.name, saved_path)

    final_status_code = to_int(extract_first(r"FINAL SUMMARY.*?Status:\s*(\d+)", text, flags=re.S))
    final_status_text = extract_first(
        r"FINAL SUMMARY.*?Status:\s*\d+\s*\(([^)]+)\)",
        text,
        flags=re.S,
    )
    objective_final = to_float(extract_first(r"FINAL SUMMARY.*?Objective:\s*([0-9.]+)", text, flags=re.S))

    mip_gap = to_float(extract_first(r"MIPGap\s*:\s*([0-9.]+)", text))
    abs_gap = to_float(extract_first(r"AbsGap\s*:\s*([0-9.]+)", text))

    total_visits = to_int(extract_first(r"Summary:\s*total visits\s*=\s*(\d+)", text))
    total_shortage_sum = to_float(extract_first(r"Total shortage sum\(S\):\s*([0-9.]+)", text))

    inv_cost = to_float(extract_first(r"inv_cost\s*=\s*([0-9.]+)", text))
    stockout_cost = to_float(extract_first(r"stockout_cost\s*=\s*([0-9.]+)", text))
    route_cost_matches = re.findall(r"route_cost\s*=\s*([0-9.]+)", text)
    route_cost = float(route_cost_matches[-1]) if route_cost_matches else None

    friendly_date_start = extract_first(
        r"optimization result\s+(\d{2}\.\d{2}\.\d{4})\s+to\s+\d{2}\.\d{2}\.\d{4}",
        text,
    )
    friendly_date_end = extract_first(
        r"optimization result\s+\d{2}\.\d{2}\.\d{4}\s+to\s+(\d{2}\.\d{2}\.\d{4})",
        text,
    )

    parse_service_summary(text)  # currently not used further, kept for extensibility
    day_data = parse_daily_routes(text)

    scenario = file_info["scenario"]
    run_type = file_info["run_type"]
    start_ymd = file_info["start_ymd"]
    end_ymd = file_info["end_ymd"]

    horizon_days = detect_horizon_days(
        start_ymd=start_ymd,
        end_ymd=end_ymd,
        friendly_date_start=friendly_date_start,
        friendly_date_end=friendly_date_end,
        day_data=day_data,
    )

    total_km = None
    if day_data:
        total_km = sum((info.get("total_km") or 0.0) for info in day_data.values())

    return {
        "file_name": path.name,
        "scenario": scenario,
        "run_type": run_type,
        "start_ymd": start_ymd,
        "end_ymd": end_ymd,
        "friendly_date_start": friendly_date_start,
        "friendly_date_end": friendly_date_end,
        "horizon_days": horizon_days,
        "result_filename": result_filename,
        "saved_path": saved_path,
        "final_status_code": final_status_code,
        "final_status_text": final_status_text,
        "objective_final": objective_final,
        "mip_gap": mip_gap,
        "abs_gap": abs_gap,
        "total_visits": total_visits,
        "total_shortage_sum": total_shortage_sum,
        "inv_cost": inv_cost,
        "stockout_cost": stockout_cost,
        "route_cost": route_cost,
        "total_km": total_km,
        "avg_daily_visits": safe_div(total_visits, horizon_days),
    }


def build_scenario_summary(run_df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for scenario, g in run_df.groupby("scenario", dropna=False):
        total_runs = len(g)
        optimal_runs = int(g["is_optimal"].fillna(False).sum())

        row = {
            "scenario": scenario,
            "runs": total_runs,
            "optimal runs": optimal_runs,
            "optimal runs (%)": safe_div(optimal_runs * 100, total_runs),
            "Average gap": g["Average gap"].mean(),
            "Total cost": g["Total cost"].sum(),
            "Service level": calc_service_level(g["Total shortage"].sum(), g["total_input_demand"].sum()),
            "Total shortage": g["Total shortage"].sum(),
            "Inventory cost": g["Inventory cost"].sum(),
            "Stockout cost": g["Stockout cost"].sum(),
            "Routing cost": g["Routing cost"].sum(),
            "Average daily cost": safe_div(g["Total cost"].sum(), g["horizon_days"].sum()),
            "Average daily shortage": safe_div(g["Total shortage"].sum(), g["horizon_days"].sum()),
            "Average visits per day": safe_div(g["total_visits"].sum(), g["horizon_days"].sum()),
            "routing cost per visit": safe_div(g["Routing cost"].sum(), g["total_visits"].sum()),
            "Kilometers per visit": safe_div(g["total_km"].sum(), g["total_visits"].sum()),
        }
        rows.append(row)

    summary = pd.DataFrame(rows)

    scenario_order = {
        "Scenario 0": 0,
        "Scenario 1": 1,
        "Scenario 2": 2,
        "Scenario 3": 3,
        "Scenario 4": 4,
    }

    if not summary.empty:
        summary["_order"] = summary["scenario"].map(scenario_order).fillna(999)
        summary = summary.sort_values(["_order", "scenario"]).drop(columns=["_order"])

    return summary


# =========================================================
# MAIN
# =========================================================
def main():
    if not RESULT_DIR.exists():
        raise FileNotFoundError(f"Folder not found: {RESULT_DIR}")

    txt_files = sorted(RESULT_DIR.glob("*.txt"))
    if not txt_files:
        raise FileNotFoundError(f"No .txt files found in: {RESULT_DIR}")

    rows = []
    failed_files = []

    for file_path in txt_files:
        try:
            rows.append(parse_one_file(file_path))
        except Exception as e:
            failed_files.append({
                "file_name": file_path.name,
                "error": str(e),
            })

    if not rows:
        raise RuntimeError("No files could be parsed successfully.")

    run_df = pd.DataFrame(rows)
    run_df = run_df[run_df["scenario"].isin(VALID_SCENARIOS)].copy()

    if run_df.empty:
        raise RuntimeError("No valid scenario rows remain after filtering.")

    demand_tables = load_scenario_demand_tables()
    run_df["total_input_demand"] = run_df.apply(
        lambda r: get_total_input_demand_for_run(r["scenario"], r["start_ymd"], r["end_ymd"], demand_tables),
        axis=1,
    )

    run_df["is_optimal"] = run_df["final_status_code"] == 2

    run_df["Total cost"] = run_df["objective_final"]
    run_df["Service level"] = run_df.apply(
        lambda r: calc_service_level(r["total_shortage_sum"], r["total_input_demand"]),
        axis=1,
    )
    run_df["Total shortage"] = run_df["total_shortage_sum"]
    run_df["Inventory cost"] = run_df["inv_cost"]
    run_df["Stockout cost"] = run_df["stockout_cost"]
    run_df["Routing cost"] = run_df["route_cost"]
    run_df["Average daily cost"] = run_df.apply(lambda r: safe_div(r["objective_final"], r["horizon_days"]), axis=1)
    run_df["Average daily shortage"] = run_df.apply(lambda r: safe_div(r["total_shortage_sum"], r["horizon_days"]), axis=1)
    run_df["Average visits per day"] = run_df.apply(lambda r: safe_div(r["total_visits"], r["horizon_days"]), axis=1)
    run_df["routing cost per visit"] = run_df.apply(lambda r: safe_div(r["route_cost"], r["total_visits"]), axis=1)
    run_df["Kilometers per visit"] = run_df.apply(lambda r: safe_div(r["total_km"], r["total_visits"]), axis=1)
    run_df["Average gap"] = run_df["mip_gap"]
    run_df["optimal runs"] = run_df["is_optimal"].astype(int)

    run_df["scenario"] = run_df["scenario"].map(SCENARIO_LABEL_MAP).fillna(run_df["scenario"])

    summary = build_scenario_summary(run_df)

    run_df = run_df[[
        "scenario",
        "run_type",
        "file_name",
        "start_ymd",
        "end_ymd",
        "horizon_days",
        "final_status_code",
        "final_status_text",
        "Total cost",
        "Service level",
        "Total shortage",
        "Inventory cost",
        "Stockout cost",
        "Routing cost",
        "Average daily cost",
        "Average daily shortage",
        "Average visits per day",
        "routing cost per visit",
        "Kilometers per visit",
        "Average gap",
        "optimal runs",
        "total_input_demand",
    ]].copy()

    scenario_order = {
        "Scenario 0": 0,
        "Scenario 1": 1,
        "Scenario 2": 2,
        "Scenario 3": 3,
        "Scenario 4": 4,
    }
    run_df["_order"] = run_df["scenario"].map(scenario_order).fillna(999)
    run_df = run_df.sort_values(["_order", "start_ymd", "end_ymd", "file_name"]).drop(columns=["_order"])

    run_df.to_csv(RUN_LEVEL_CSV, index=False)
    summary.to_csv(SCENARIO_SUMMARY_CSV, index=False)

    if failed_files:
        pd.DataFrame(failed_files).to_csv(FAILED_PARSING_CSV, index=False)
        print(f"Some files could not be parsed. See: {FAILED_PARSING_CSV}")

    print("Done.")
    print(f"Run-level KPI file saved to: {RUN_LEVEL_CSV}")
    print(f"Scenario summary file saved to: {SCENARIO_SUMMARY_CSV}")


if __name__ == "__main__":
    main()