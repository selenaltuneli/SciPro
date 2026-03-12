import os
import json
import pandas as pd

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))

# =========================
# INPUT / OUTPUT SETTINGS
# =========================
EXCEL_PATH = os.path.join(PROJECT_DIR, "weekly_20070226_20070304.xlsx")

# SAME folder as your previous builder code:
# (it used cfg.out_dir = "outputs")
OUT_DIR = os.path.join(PROJECT_DIR, "outputs")

PREFIX = "weeklyavg_Scenario1_end_20070304"                # <-- name it however you want

# Expected columns in your Excel
COL_DATE = "FORECAST_DATE"
COL_ATM  = "CASHP_ID_ATM"
COL_VAL  = "Y_PRED_WITHDRWLS_ATM"


def build_r_json_from_excel(
    excel_path: str,
    out_dir: str,
    prefix: str,
    col_date: str,
    col_atm: str,
    col_val: str,
) -> None:
    os.makedirs(out_dir, exist_ok=True)

    df = pd.read_excel(excel_path)

    # Basic typing
    df[col_date] = pd.to_datetime(df[col_date], errors="coerce")
    df[col_atm]  = df[col_atm].astype(str)
    df[col_val]  = pd.to_numeric(df[col_val], errors="coerce")

    # Drop unusable rows
    df = df.dropna(subset=[col_date, col_atm, col_val]).copy()

    # Build t-mapping from dates (sorted), t = 1..7
    unique_dates = sorted(df[col_date].dt.normalize().unique())
    if len(unique_dates) != 7:
        raise ValueError(
            f"Expected exactly 7 distinct dates, but found {len(unique_dates)}: "
            f"{[pd.to_datetime(d).strftime('%Y-%m-%d') for d in unique_dates]}"
        )

    date_to_t = {pd.to_datetime(d): (i + 1) for i, d in enumerate(unique_dates)}
    t_to_date = {str(i + 1): pd.to_datetime(d).strftime("%Y-%m-%d") for i, d in enumerate(unique_dates)}

    df["_date_norm"] = df[col_date].dt.normalize()
    df["_t"] = df["_date_norm"].map(date_to_t)

    # Detect duplicates (same ATM, same day)
    dup = df.duplicated(subset=[col_atm, "_t"], keep=False)
    if dup.any():
        bad = df.loc[dup, [col_atm, col_date, "_t", col_val]].sort_values([col_atm, "_t"])
        raise ValueError(
            "Duplicate rows detected for the same (ATM, t). "
            "Need exactly one value per ATM per day.\n"
            f"Examples:\n{bad.head(20).to_string(index=False)}"
        )

    # EXACT same structure as your builder:  { "atm|t": value }
    r_json = {
        f"{row[col_atm]}|{int(row['_t'])}": float(row[col_val])
        for _, row in df.iterrows()
    }

    r_path = os.path.join(out_dir, f"{prefix}_r_nextweek.json")
    with open(r_path, "w", encoding="utf-8") as f:
        json.dump(r_json, f, indent=2)

    # Optional meta.json (same idea as your builder; t_to_date stored as JSON string)
    meta = {
        "horizon_days": "7",
        "t_index_start": "1",
        "t_to_date": json.dumps(t_to_date),
        "source_file": os.path.basename(excel_path),
    }
    meta_path = os.path.join(out_dir, f"{prefix}_meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print("Saved:")
    print(" ", os.path.abspath(r_path))
    print(" ", os.path.abspath(meta_path))
    print(f"r entries: {len(r_json)}  |  ATMs: {df[col_atm].nunique()}  |  days: {len(unique_dates)}")


if __name__ == "__main__":
    build_r_json_from_excel(
        excel_path=EXCEL_PATH,
        out_dir=OUT_DIR,
        prefix=PREFIX,
        col_date=COL_DATE,
        col_atm=COL_ATM,
        col_val=COL_VAL,
    )
