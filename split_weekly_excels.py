import argparse
from pathlib import Path

import pandas as pd

FORECAST_SHEET = "Forecasts"
WEEK_START_COL = "WEEK_START"


def split_weekly_excels(input_file: Path, output_dir: Path) -> int:
    if not input_file.exists():
        raise FileNotFoundError(f"Input file not found: {input_file}")

    output_dir.mkdir(parents=True, exist_ok=True)

    xls = pd.ExcelFile(input_file)
    sheet_name = FORECAST_SHEET if FORECAST_SHEET in xls.sheet_names else xls.sheet_names[0]
    df = pd.read_excel(input_file, sheet_name=sheet_name)

    if WEEK_START_COL not in df.columns:
        raise ValueError(f"'{WEEK_START_COL}' column not found in sheet '{sheet_name}'")

    df[WEEK_START_COL] = pd.to_datetime(df[WEEK_START_COL], errors="coerce")
    df = df.dropna(subset=[WEEK_START_COL]).copy()

    created = 0
    for week_start, g in df.groupby(WEEK_START_COL):
        week_end = week_start + pd.Timedelta(days=6)
        fname = f"weekly_{week_start.strftime('%Y%m%d')}_{week_end.strftime('%Y%m%d')}.xlsx"
        out_path = output_dir / fname
        g_sorted = g.sort_values(["CASHP_ID_ATM", "FORECAST_DATE"]).reset_index(drop=True)
        g_sorted.to_excel(out_path, index=False)
        created += 1

    return created


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Split forecast Excel into one weekly Excel per WEEK_START."
    )
    parser.add_argument("input_file", help="Input Excel file (e.g. XGBoost_pred_weekend_filled.xlsx)")
    parser.add_argument(
        "--output-dir",
        default="weekly_excels",
        help="Output directory for weekly files (default: weekly_excels)",
    )
    args = parser.parse_args()

    input_path = Path(args.input_file)
    output_path = Path(args.output_dir)

    created = split_weekly_excels(input_path, output_path)
    print(f"Input : {input_path}")
    print(f"Output dir: {output_path}")
    print(f"Weekly files created: {created}")


if __name__ == "__main__":
    main()
