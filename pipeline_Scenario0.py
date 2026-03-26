from __future__ import annotations

import os
import json
from dataclasses import dataclass
from typing import Dict, Tuple, Any

import pandas as pd


# =========================================================
# CONFIG
# =========================================================

@dataclass
class DataConfig:
    # Fixed path: main ATM data file
    excel_path: str

    # Fixed path: scenario control file
    scenario_file_path: str

    # Column names in ATM data file
    col_atm: str = "CASHP_ID_ATM"
    col_date: str = "DATE"
    col_withdraw: str = "WITHDRWLS_ATM"


    # Column names in scenario control file
    col_reopt: str = "reoptimization_date"
    col_plan_start: str = "planning_start_date"
    col_plan_end: str = "planning_end_date"

    # Base output folder
    out_base_dir: str = "outputs"


# =========================================================
# UTILITIES
# =========================================================

def ensure_out_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def parse_date_flexible(value: Any) -> pd.Timestamp:
    """
    Accepts:
    - Excel datetime cells
    - pandas Timestamp
    - strings like:
        25.02.2007
        2007-02-25
        2007-02-25 00:00:00
    """
    if value is None or pd.isna(value):
        raise ValueError("Date input is empty.")

    if isinstance(value, pd.Timestamp):
        return pd.to_datetime(value).normalize()

    s = str(value).strip()
    if s == "":
        raise ValueError("Date input is empty.")

    dt = pd.to_datetime(s, errors="raise", dayfirst=True)
    return dt.normalize()


def date_to_ddmmyyyy(dt: pd.Timestamp) -> str:
    return pd.to_datetime(dt).strftime("%d.%m.%Y")


def load_excel(cfg: DataConfig) -> pd.DataFrame:
    df = pd.read_excel(cfg.excel_path)

    required = [cfg.col_date, cfg.col_atm, cfg.col_withdraw]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"Missing column(s) in ATM data file: {missing}")

    df[cfg.col_date] = pd.to_datetime(df[cfg.col_date], errors="coerce")
    df[cfg.col_atm] = df[cfg.col_atm].astype(str).str.strip()
    df[cfg.col_withdraw] = pd.to_numeric(df[cfg.col_withdraw], errors="coerce")

    return df


def load_scenarios(cfg: DataConfig) -> pd.DataFrame:
    df = pd.read_excel(cfg.scenario_file_path)

    required = [cfg.col_reopt, cfg.col_plan_start, cfg.col_plan_end]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"Missing column(s) in scenario file: {missing}")

    return df


def build_average_window_from_reopt(
    df: pd.DataFrame,
    cfg: DataConfig,
    reopt_date: pd.Timestamp
) -> Tuple[pd.Timestamp, pd.Timestamp]:
    """
    Averaging window:
    from the first available date in the dataset
    up to reoptimization date (inclusive)
    """
    first_date = df[cfg.col_date].min()

    if pd.isna(first_date):
        raise ValueError("DATE column has no valid parsed dates.")

    avg_start = pd.to_datetime(first_date).normalize()
    avg_end = reopt_date
    return avg_start, avg_end


def build_t_to_date(
    planning_start: pd.Timestamp,
    planning_end: pd.Timestamp,
    t_index_start: int = 1
) -> Dict[str, str]:
    if planning_end < planning_start:
        raise ValueError("Planning end date cannot be earlier than planning start date.")

    n_days = (planning_end - planning_start).days + 1
    mapping: Dict[str, str] = {}

    for i in range(n_days):
        t = t_index_start + i
        d = planning_start + pd.Timedelta(days=i)
        mapping[str(t)] = date_to_ddmmyyyy(d)

    return mapping


# =========================================================
# MAIN CALCULATIONS
# =========================================================

def average_withdrawals_over_window(
    df: pd.DataFrame,
    cfg: DataConfig,
    avg_start: pd.Timestamp,
    avg_end: pd.Timestamp,
) -> pd.DataFrame:
    """
    Compute one overall average per ATM over the averaging window.
    """
    mask = (df[cfg.col_date] >= avg_start) & (df[cfg.col_date] <= avg_end)
    wk = df.loc[mask, [cfg.col_atm, cfg.col_withdraw]].copy()
    wk = wk.dropna(subset=[cfg.col_atm, cfg.col_withdraw])

    avg = (
        wk.groupby(cfg.col_atm)[cfg.col_withdraw]
        .mean()
        .reset_index()
        .rename(columns={
            cfg.col_atm: "atm_id",
            cfg.col_withdraw: "avg_withdrawal"
        })
        .sort_values("atm_id")
        .reset_index(drop=True)
    )

    return avg


def build_r_for_planning_horizon(
    avg_demands: pd.DataFrame,
    planning_start: pd.Timestamp,
    planning_end: pd.Timestamp,
    t_index_start: int = 1,
) -> Tuple[Dict[Tuple[str, int], float], Dict[str, str]]:
    t_to_date = build_t_to_date(
        planning_start=planning_start,
        planning_end=planning_end,
        t_index_start=t_index_start
    )

    r: Dict[Tuple[str, int], float] = {}
    for row in avg_demands.itertuples(index=False):
        atm = str(row.atm_id)
        val = float(row.avg_withdrawal)

        for t_str in t_to_date.keys():
            t = int(t_str)
            r[(atm, t)] = val

    return r, t_to_date


# =========================================================
# SAVE OUTPUTS
# =========================================================

def save_outputs(
    *,
    r: Dict[Tuple[str, int], float],
    meta: Dict[str, Any],
    scenario_folder: str,
    scenario_prefix: str,
) -> None:
    ensure_out_dir(scenario_folder)

    r_path = os.path.join(scenario_folder, f"{scenario_prefix}_r_nextplanning.json")
    meta_path = os.path.join(scenario_folder, f"{scenario_prefix}_meta.json")

    r_json = {f"{atm}|{t}": float(val) for (atm, t), val in r.items()}
    with open(r_path, "w", encoding="utf-8") as f:
        json.dump(r_json, f, indent=2, ensure_ascii=False)

    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    print("Saved:")
    print(" ", r_path)
    print(" ", meta_path)


# =========================================================
# ONE SCENARIO RUN
# =========================================================

def run_one_scenario(
    *,
    df_main: pd.DataFrame,
    cfg: DataConfig,
    reopt_date: pd.Timestamp,
    planning_start: pd.Timestamp,
    planning_end: pd.Timestamp,
) -> None:
    if planning_end < planning_start:
        raise ValueError("Planning end date must be on or after planning start date.")

    avg_start, avg_end = build_average_window_from_reopt(df_main, cfg, reopt_date)

    print("\n=== WINDOW CHECK ===")
    print("Reoptimization date      :", date_to_ddmmyyyy(reopt_date))
    print("AVG_START USED FOR MEAN :", date_to_ddmmyyyy(avg_start))
    print("AVG_END USED FOR MEAN   :", date_to_ddmmyyyy(avg_end))
    print("PLAN_START              :", date_to_ddmmyyyy(planning_start))
    print("PLAN_END                :", date_to_ddmmyyyy(planning_end))
    print("=========================")

    date_min = df_main[cfg.col_date].min()
    date_max = df_main[cfg.col_date].max()

    print("DATE range in main file:", date_min, "to", date_max)

    if pd.isna(date_min) or pd.isna(date_max):
        raise ValueError("DATE column has no valid parsed dates.")

    if reopt_date < pd.to_datetime(date_min).normalize():
        raise ValueError("Reoptimization date is earlier than the first available date in the dataset.")

    mask_avg = (df_main[cfg.col_date] >= avg_start) & (df_main[cfg.col_date] <= avg_end)
    rows_in_avg = int(mask_avg.sum())
    atms_in_avg = df_main.loc[mask_avg, cfg.col_atm].astype(str).nunique()

    print("Rows total:", len(df_main))
    print("Distinct ATMs total:", df_main[cfg.col_atm].astype(str).nunique())
    print("Rows in averaging window:", rows_in_avg)
    print("Distinct ATMs in averaging window:", atms_in_avg)

    if rows_in_avg == 0:
        print("WARNING: No rows found in the averaging window.")

    avg_demands = average_withdrawals_over_window(df_main, cfg, avg_start, avg_end)
    print("ATMs in avg_demands:", len(avg_demands))

    r, t_to_date = build_r_for_planning_horizon(
        avg_demands=avg_demands,
        planning_start=planning_start,
        planning_end=planning_end,
        t_index_start=1,
    )
    print("r entries:", len(r))

    parent_scenario_folder = os.path.join(cfg.out_base_dir, "scenario 0")
    ensure_out_dir(parent_scenario_folder)

    scenario_prefix = (
        f"Scenario0_"
        f"{planning_start.strftime('%Y%m%d')}_"
        f"{planning_end.strftime('%Y%m%d')}"
    )

    scenario_folder = os.path.join(parent_scenario_folder, scenario_prefix)

    if os.path.exists(scenario_folder):
        print(f"WARNING: Output folder already exists and files may be overwritten: {scenario_folder}")

    ensure_out_dir(scenario_folder)

    horizon_days = (planning_end - planning_start).days + 1

    meta = {
        "scenario_prefix": scenario_prefix,
        "planning_start_date": date_to_ddmmyyyy(planning_start),
        "planning_end_date": date_to_ddmmyyyy(planning_end),
        "reoptimization_date": date_to_ddmmyyyy(reopt_date),
        "avg_start_used_for_mean": date_to_ddmmyyyy(avg_start),
        "avg_end_used_for_mean": date_to_ddmmyyyy(avg_end),
        "horizon_days": horizon_days,
        "t_index_start": 1,
        "t_to_date": t_to_date,
        "source_file": os.path.basename(cfg.excel_path),
        "source_file_full_path": os.path.abspath(cfg.excel_path),
        "scenario_control_file": os.path.basename(cfg.scenario_file_path),
        "scenario_control_file_full_path": os.path.abspath(cfg.scenario_file_path),
        "output_folder": os.path.abspath(scenario_folder),
    }

    save_outputs(
        r=r,
        meta=meta,
        scenario_folder=scenario_folder,
        scenario_prefix=scenario_prefix,
    )


# =========================================================
# MAIN
# =========================================================

def main() -> None:
    cfg = DataConfig(
        excel_path="/Users/porsa/Library/Mobile Documents/com~apple~CloudDocs/OVGU/Scientifc project/Data/ATM_Branch_Data_Final_filled.xlsx",
        scenario_file_path="/Users/porsa/Library/Mobile Documents/com~apple~CloudDocs/OVGU/Scientifc project/Optimization/Code/outputs/scenario 0/week info for scenario 0.xlsx",
        out_base_dir="outputs",
    )

    print("\n=== Loading main ATM data file ===")
    df_main = load_excel(cfg)

    print("\n=== Loading scenario control file ===")
    df_scen = load_scenarios(cfg)

    print("\nScenario rows found:", len(df_scen))

    for idx, row in df_scen.iterrows():
        print("\n" + "=" * 70)
        print(f"RUNNING SCENARIO ROW {idx + 1}")
        print("=" * 70)

        try:
            reopt_date = parse_date_flexible(row[cfg.col_reopt])
            planning_start = parse_date_flexible(row[cfg.col_plan_start])
            planning_end = parse_date_flexible(row[cfg.col_plan_end])

            run_one_scenario(
                df_main=df_main,
                cfg=cfg,
                reopt_date=reopt_date,
                planning_start=planning_start,
                planning_end=planning_end,
            )

        except Exception as e:
            print(f"ERROR in scenario row {idx + 1}: {e}")

    print("\nDONE.")


if __name__ == "__main__":
    main()