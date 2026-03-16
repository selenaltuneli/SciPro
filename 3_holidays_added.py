"""Add holiday-related features to the merged ATM-branch dataset."""

import datetime as dt
import re
from pathlib import Path

import numpy as np
import pandas as pd

BASE_PATH = Path(r"C:\Users\Selen\Desktop\ORBA\Scientific Project\SciPro")

MERGED_PATH = BASE_PATH / "ATM_Branch_Merged_With_Location.xlsx"
ATM_BRANCH_PATH = BASE_PATH / "2024-12-09_ATM_Branch_Data.xlsx"
OUTPUT_PATH = BASE_PATH / "ATM_Branch_Merged_With_Location_HolidayFeatures_FINAL.xlsx"

MERGED_SHEET_NAME = 0
EVENTS_SHEET_NAME = "Events_Holidays"

GERMAN_MONTH_MAP = {
    "januar": 1,
    "jan": 1,
    "februar": 2,
    "feb": 2,
    "märz": 3,
    "maerz": 3,
    "marz": 3,
    "april": 4,
    "apr": 4,
    "mai": 5,
    "juni": 6,
    "jun": 6,
    "juli": 7,
    "jul": 7,
    "august": 8,
    "aug": 8,
    "september": 9,
    "sep": 9,
    "oktober": 10,
    "okt": 10,
    "november": 11,
    "nov": 11,
    "dezember": 12,
    "dez": 12,
}


def normalize_date(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce").dt.normalize()


def parse_german_date(date_text: str):
    text = (date_text or "").strip()
    match = re.match(r"(\d{1,2})\s+([A-Za-zÄÖÜäöüß\.]+)\s+(\d{4})", text)
    if not match:
        return None

    day = int(match.group(1))
    month_name = match.group(2).strip(".").lower()
    year = int(match.group(3))

    month = GERMAN_MONTH_MAP.get(month_name)
    if month is None:
        normalized_month_name = (
            month_name.replace("ä", "ae")
            .replace("ö", "oe")
            .replace("ü", "ue")
        )
        month = GERMAN_MONTH_MAP.get(normalized_month_name)

    if month is None:
        return None

    try:
        return pd.to_datetime(dt.date(year, month, day))
    except Exception:
        return None


def expand_school_holiday_range(text_line: str) -> pd.DataFrame:
    text = (text_line or "").strip()
    if "Schulferien" not in text:
        return pd.DataFrame()

    match = re.search(
        r"(\d{1,2}\s+[A-Za-zÄÖÜäöüß\.]+\s+\d{4})\s+bis\s+(\d{1,2}\s+[A-Za-zÄÖÜäöüß\.]+\s+\d{4})",
        text,
    )
    if not match:
        return pd.DataFrame()

    start_date = parse_german_date(match.group(1))
    end_date = parse_german_date(match.group(2))
    if start_date is None or end_date is None:
        return pd.DataFrame()

    expanded_dates = pd.date_range(start=start_date, end=end_date, freq="D").normalize()
    school_holiday_df = pd.DataFrame({"DATE": expanded_dates})
    school_holiday_df["IS_SCHOOL_HOLIDAY"] = 1
    return school_holiday_df


def parse_official_holidays(events_df: pd.DataFrame) -> pd.DataFrame:
    first_col, second_col, third_col = events_df.columns[:3]
    holiday_rows = []

    for _, row in events_df.iterrows():
        raw_date_value = row[first_col]
        if pd.isna(raw_date_value):
            continue

        if isinstance(raw_date_value, (int, np.integer)) or (
            isinstance(raw_date_value, float) and float(raw_date_value).is_integer()
        ):
            continue

        raw_date_text = str(raw_date_value).strip()
        if re.fullmatch(r"\d{4}", raw_date_text):
            continue

        if "Schulferien" in raw_date_text:
            continue

        holiday_name = None if pd.isna(row[second_col]) else str(row[second_col]).strip()
        description = None if pd.isna(row[third_col]) else str(row[third_col]).strip()
        if not holiday_name:
            continue

        parsed_date = parse_german_date(raw_date_text)
        if parsed_date is None:
            continue

        holiday_rows.append(
            {
                "DATE": parsed_date.normalize(),
                "HOLIDAY_NAME": holiday_name,
                "DESCRIPTION": description,
            }
        )

    official_holidays = pd.DataFrame(holiday_rows)
    if official_holidays.empty:
        return official_holidays

    official_holidays["DATE"] = normalize_date(official_holidays["DATE"])
    return official_holidays.dropna(subset=["DATE", "HOLIDAY_NAME"])


def classify_holiday_type(holiday_name: str) -> str:
    name = (holiday_name or "").lower()

    if any(keyword in name for keyword in ["opferfest", "ramazan", "arife"]):
        return "RELIGIOUS"

    if any(keyword in name for keyword in ["nationaler", "atat", "sieges", "tag der republik", "republik"]):
        return "NATIONAL"

    return "PUBLIC"


def holiday_duration_score(description: str) -> int:
    description_text = (description or "").lower()
    if "halb tag offen" in description_text:
        return 1
    return 2


def holiday_importance_score(duration_score: int) -> int:
    if duration_score == 2:
        return 3
    if duration_score == 1:
        return 2
    return 0


def build_official_holiday_features(official_holidays: pd.DataFrame) -> pd.DataFrame:
    if official_holidays.empty:
        return pd.DataFrame(
            columns=[
                "DATE",
                "HOLIDAY_NAME",
                "DESCRIPTION",
                "IS_HOLIDAY",
                "HOLIDAY_DURATION_SCORE",
                "HOLIDAY_IMPORTANCE_SCORE",
                "IS_RELIGIOUS_HOLIDAY",
                "IS_NATIONAL_HOLIDAY",
                "IS_PUBLIC_HOLIDAY",
            ]
        )

    official_holidays = official_holidays.copy()
    official_holidays["HOLIDAY_TYPE"] = official_holidays["HOLIDAY_NAME"].apply(classify_holiday_type)
    official_holidays["HOLIDAY_DURATION_SCORE"] = official_holidays["DESCRIPTION"].apply(holiday_duration_score)
    official_holidays["HOLIDAY_IMPORTANCE_SCORE"] = official_holidays["HOLIDAY_DURATION_SCORE"].apply(
        holiday_importance_score
    )
    official_holidays["IS_HOLIDAY"] = 1

    grouped_holidays = official_holidays.groupby("DATE", as_index=False).agg(
        HOLIDAY_NAME=("HOLIDAY_NAME", lambda x: " | ".join(pd.unique(x.dropna().astype(str)))),
        DESCRIPTION=("DESCRIPTION", lambda x: " | ".join(pd.unique(x.dropna().astype(str)))),
        HOLIDAY_DURATION_SCORE=("HOLIDAY_DURATION_SCORE", "max"),
        HOLIDAY_IMPORTANCE_SCORE=("HOLIDAY_IMPORTANCE_SCORE", "max"),
        IS_HOLIDAY=("IS_HOLIDAY", "max"),
    )

    type_flags = (
        official_holidays.assign(
            IS_RELIGIOUS_HOLIDAY=(official_holidays["HOLIDAY_TYPE"] == "RELIGIOUS").astype(int),
            IS_NATIONAL_HOLIDAY=(official_holidays["HOLIDAY_TYPE"] == "NATIONAL").astype(int),
            IS_PUBLIC_HOLIDAY=(official_holidays["HOLIDAY_TYPE"] == "PUBLIC").astype(int),
        )
        .groupby("DATE", as_index=False)[["IS_RELIGIOUS_HOLIDAY", "IS_NATIONAL_HOLIDAY", "IS_PUBLIC_HOLIDAY"]]
        .max()
    )

    return grouped_holidays.merge(type_flags, on="DATE", how="left")


def build_school_holiday_features(events_df: pd.DataFrame) -> pd.DataFrame:
    school_holiday_parts = []
    first_column = events_df.columns[0]

    for value in events_df[first_column].dropna().astype(str):
        if "Schulferien" in value:
            expanded = expand_school_holiday_range(value)
            if not expanded.empty:
                school_holiday_parts.append(expanded)

    if not school_holiday_parts:
        return pd.DataFrame(columns=["DATE", "IS_SCHOOL_HOLIDAY"])

    return pd.concat(school_holiday_parts, ignore_index=True).drop_duplicates(subset=["DATE"])


def add_pre_post_holiday_flags(df: pd.DataFrame, official_dates: set) -> pd.DataFrame:
    output_df = df.copy()
    date_values = pd.to_datetime(output_df["DATE"], errors="coerce").dt.normalize().values.astype("datetime64[D]")

    official_days = pd.to_datetime(pd.Series(list(official_dates)), errors="coerce").dt.normalize()
    official_day_set = set(official_days.dropna().values.astype("datetime64[D]"))

    def any_shift(shifts: list[int]) -> np.ndarray:
        result = np.zeros(len(date_values), dtype=int)
        for index, current_date in enumerate(date_values):
            result[index] = int(any((current_date + np.timedelta64(shift, "D")) in official_day_set for shift in shifts))
        return result

    output_df["IS_PRE_HOLIDAY_1"] = any_shift([1])
    output_df["IS_PRE_HOLIDAY_1_2"] = any_shift([1, 2])
    output_df["IS_PRE_HOLIDAY_1_2_3"] = any_shift([1, 2, 3])

    output_df["IS_POST_HOLIDAY_1"] = any_shift([-1])
    output_df["IS_POST_HOLIDAY_1_2"] = any_shift([-1, -2])
    output_df["IS_POST_HOLIDAY_1_2_3"] = any_shift([-1, -2, -3])

    return output_df


def fill_holiday_columns(df: pd.DataFrame) -> pd.DataFrame:
    output_df = df.copy()
    output_df["IS_HOLIDAY"] = output_df["IS_HOLIDAY"].fillna(0).astype(int)

    integer_columns = [
        "HOLIDAY_DURATION_SCORE",
        "HOLIDAY_IMPORTANCE_SCORE",
        "IS_RELIGIOUS_HOLIDAY",
        "IS_NATIONAL_HOLIDAY",
        "IS_PUBLIC_HOLIDAY",
    ]
    for column_name in integer_columns:
        output_df[column_name] = output_df[column_name].fillna(0).astype(int)

    if "HOLIDAY_NAME" not in output_df.columns:
        output_df["HOLIDAY_NAME"] = np.nan
    if "DESCRIPTION" not in output_df.columns:
        output_df["DESCRIPTION"] = np.nan

    return output_df


def main() -> None:
    merged_df = pd.read_excel(MERGED_PATH, sheet_name=MERGED_SHEET_NAME)
    if "DATE" not in merged_df.columns:
        raise ValueError("The merged dataset does not contain a DATE column.")

    merged_df["DATE"] = normalize_date(merged_df["DATE"])
    merged_df = merged_df.dropna(subset=["DATE"])

    events_df = pd.read_excel(ATM_BRANCH_PATH, sheet_name=EVENTS_SHEET_NAME)

    official_holidays = parse_official_holidays(events_df)
    official_holiday_features = build_official_holiday_features(official_holidays)
    school_holiday_features = build_school_holiday_features(events_df)

    output_df = merged_df.merge(official_holiday_features, on="DATE", how="left")
    output_df = fill_holiday_columns(output_df)

    output_df = output_df.merge(school_holiday_features, on="DATE", how="left")
    output_df["IS_SCHOOL_HOLIDAY"] = output_df["IS_SCHOOL_HOLIDAY"].fillna(0).astype(int)

    # When a date is an official holiday, that label takes priority over school holiday.
    output_df.loc[output_df["IS_HOLIDAY"] == 1, "IS_SCHOOL_HOLIDAY"] = 0

    official_date_set = set(official_holiday_features["DATE"].dropna().tolist())
    output_df = add_pre_post_holiday_flags(output_df, official_date_set)

    output_df.to_excel(OUTPUT_PATH, index=False)
    print(f"Holiday features were added successfully: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
