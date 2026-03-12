import argparse
from pathlib import Path

import pandas as pd

FORECAST_SHEET = "Forecasts"
REQUIRED_COLS = [
    "WEEK_START",
    "TRAIN_END",
    "FORECAST_DATE",
    "CASHP_ID_ATM",
    "Y_PRED_WITHDRWLS_ATM",
]
OPTIONAL_COLS = ["Y_TRUE_WITHDRWLS_ATM", "ABS_ERROR", "APE"]


def add_missing_weekend_rows(df_forecasts: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """
    Do not modify existing rows.
    Add missing Saturday/Sunday rows per (WEEK_START, TRAIN_END, CASHP_ID_ATM) with Y_PRED=0.
    """
    missing = set(REQUIRED_COLS) - set(df_forecasts.columns)
    if missing:
        raise ValueError(f"Forecast sheet is missing required columns: {sorted(missing)}")

    df = df_forecasts.copy()
    df["WEEK_START"] = pd.to_datetime(df["WEEK_START"], errors="coerce")
    df["TRAIN_END"] = pd.to_datetime(df["TRAIN_END"], errors="coerce")
    df["FORECAST_DATE"] = pd.to_datetime(df["FORECAST_DATE"], errors="coerce")
    df["CASHP_ID_ATM"] = df["CASHP_ID_ATM"].astype(str)
    df = df.dropna(subset=["WEEK_START", "TRAIN_END", "FORECAST_DATE"])

    existing = set(
        zip(
            df["WEEK_START"].dt.date.astype(str),
            df["TRAIN_END"].dt.date.astype(str),
            df["CASHP_ID_ATM"],
            df["FORECAST_DATE"].dt.date.astype(str),
        )
    )

    


    new_rows = []
    group_cols = ["WEEK_START", "TRAIN_END", "CASHP_ID_ATM"]
    for (week_start, train_end, atm_id), _g in df.groupby(group_cols, dropna=False):
        sat = week_start + pd.Timedelta(days=5)
        sun = week_start + pd.Timedelta(days=6)
        for day in (sat, sun):
            key = (
                str(week_start.date()),
                str(train_end.date()),
                str(atm_id),
                str(day.date()),
            )
            if key in existing:
                continue

            row = {
                "WEEK_START": week_start,
                "TRAIN_END": train_end,
                "FORECAST_DATE": day,
                "CASHP_ID_ATM": str(atm_id),
                "Y_PRED_WITHDRWLS_ATM": 0.0,
            }
            for col in OPTIONAL_COLS:
                if col in df.columns:
                    row[col] = pd.NA
            new_rows.append(row)

    if not new_rows:
        return df_forecasts.copy(), 0

    new_df = pd.DataFrame(new_rows, columns=df.columns)
    out = pd.concat([df, new_df], ignore_index=True)
    out = out.sort_values(["WEEK_START", "CASHP_ID_ATM", "FORECAST_DATE"]).reset_index(drop=True)
    return out, len(new_rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Append missing weekend rows (Saturday/Sunday) per ATM-week in Forecasts sheet "
            "without changing existing rows."
        )
    )
    parser.add_argument("--input-file", default="XGBoost_pred.xlsx")
    parser.add_argument("--output-file", default="XGBoost_pred_weekend_filled.xlsx")
    args = parser.parse_args()

    input_path = Path(args.input_file)
    output_path = Path(args.output_file)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    xls = pd.ExcelFile(input_path)
    sheets = {name: pd.read_excel(input_path, sheet_name=name) for name in xls.sheet_names}
    if FORECAST_SHEET not in sheets:
        raise ValueError(f"Sheet '{FORECAST_SHEET}' not found in {input_path}")

    fixed_forecasts, added_count = add_missing_weekend_rows(sheets[FORECAST_SHEET])
    sheets[FORECAST_SHEET] = fixed_forecasts

    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        for sheet_name, sheet_df in sheets.items():
            sheet_df.to_excel(writer, sheet_name=sheet_name, index=False)

    print(f"Input : {input_path}")
    print(f"Output: {output_path}")
    print(f"Added weekend rows: {added_count}")


if __name__ == "__main__":
    main()
