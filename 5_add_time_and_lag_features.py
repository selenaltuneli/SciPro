"""Add calendar and lag-based features to the ATM dataset."""

from pathlib import Path

import pandas as pd

INPUT_PATH = Path(r"C:\Users\Selen\Desktop\ORBA\Scientific Project\SciPro\ATM_Branch_With_FunctionalZone_Postcode.xlsx")
OUTPUT_PATH = Path(r"C:\Users\Selen\Desktop\ORBA\Scientific Project\SciPro\ATM_Branch_With_OtherFeatures.xlsx")

ATM_KEY = "CASHP_ID_ATM"
BRANCH_KEY = "BRANCH_KEY"
DATE_COL = "DATE"
TARGET_COL = "WITHDRWLS_ATM"


def validate_columns(df: pd.DataFrame) -> None:
    required_columns = [ATM_KEY, BRANCH_KEY, DATE_COL, TARGET_COL]
    missing_columns = [column for column in required_columns if column not in df.columns]
    if missing_columns:
        raise KeyError(f"Missing required columns: {missing_columns}. Available columns: {df.columns.tolist()}")


def prepare_dates(df: pd.DataFrame) -> pd.DataFrame:
    prepared_df = df.copy()
    prepared_df[DATE_COL] = pd.to_datetime(prepared_df[DATE_COL], errors="coerce")

    if prepared_df[DATE_COL].isna().any():
        invalid_rows = prepared_df[prepared_df[DATE_COL].isna()].head(10)
        raise ValueError(f"Some rows contain invalid {DATE_COL} values:\n{invalid_rows}")

    return prepared_df


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    output_df = df.copy()

    output_df["DAY_OF_WEEK"] = output_df[DATE_COL].dt.weekday
    output_df["IS_WEEKDAY"] = (output_df["DAY_OF_WEEK"] < 5).astype(int)
    output_df["DAY_OF_MONTH"] = output_df[DATE_COL].dt.day
    output_df["WEEK_OF_YEAR"] = output_df[DATE_COL].dt.isocalendar().week.astype(int)
    output_df["MONTH"] = output_df[DATE_COL].dt.month
    output_df["IS_MONTH_START"] = output_df[DATE_COL].dt.is_month_start.astype(int)
    output_df["IS_MONTH_END"] = output_df[DATE_COL].dt.is_month_end.astype(int)
    output_df["IS_QUARTER_END"] = output_df[DATE_COL].dt.is_quarter_end.astype(int)

    return output_df


def season_features(month: int) -> tuple[str, int]:
    if month in (12, 1, 2):
        return "Winter", 1
    if month in (3, 4, 5):
        return "Spring", 2
    if month in (6, 7, 8):
        return "Summer", 3
    return "Autumn", 4


def add_season_features(df: pd.DataFrame) -> pd.DataFrame:
    output_df = df.copy()
    output_df[["SEASON", "SEASON_NUM"]] = output_df["MONTH"].apply(
        lambda month: pd.Series(season_features(int(month)))
    )
    return output_df


def add_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    output_df = df.copy()

    output_df["ATM_WITHDRWLS_LAG_1"] = output_df.groupby(ATM_KEY)[TARGET_COL].shift(1)
    output_df["ATM_WITHDRWLS_LAG_7"] = output_df.groupby(ATM_KEY)[TARGET_COL].shift(7)

    output_df["ATM_WITHDRWLS_MA_7"] = (
        output_df.groupby(ATM_KEY)[TARGET_COL]
        .rolling(window=7, min_periods=1)
        .mean()
        .reset_index(level=0, drop=True)
    )

    return output_df


def main() -> None:
    df = pd.read_excel(INPUT_PATH)
    validate_columns(df)

    df = prepare_dates(df)
    df = df.sort_values([ATM_KEY, DATE_COL]).reset_index(drop=True)

    df = add_time_features(df)
    df = add_season_features(df)
    df = add_lag_features(df)

    df.to_excel(OUTPUT_PATH, index=False)

    print("Additional features were created successfully.")
    print(f"Output file: {OUTPUT_PATH}")
    print(f"ATM key column: {ATM_KEY}")
    print(f"Branch key column: {BRANCH_KEY}")
    print(f"Target column: {TARGET_COL}")


if __name__ == "__main__":
    main()
