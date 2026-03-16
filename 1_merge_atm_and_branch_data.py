"""Combine ATM and branch sheets into a single master table."""

from pathlib import Path
import re

import pandas as pd

INPUT_FILE = Path(r"C:\Users\Selen\Desktop\ORBA\Scientific Project\SciPro\2024-12-09_ATM_Branch_Data.xlsx")
OUTPUT_FILE = Path(r"C:\Users\Selen\Desktop\ORBA\Scientific Project\SciPro\ATM_Branch_Merged.xlsx")

ATM_SHEET_NAME = "ATM"
BRANCH_SHEET_NAME = "Branches"
EVENTS_SHEET_NAME = "Events_Holidays"
OUTPUT_SHEET_NAME = "BRANCH_MASTER_WITH_ATM"


def extract_digits(value: object) -> str:
    if pd.isna(value):
        return ""
    return re.sub(r"\D+", "", str(value))


def build_branch_key_from_atm(atm_cashpoint_id: object) -> str:
    digit_string = extract_digits(atm_cashpoint_id)
    return digit_string[:4].zfill(4) if digit_string else ""


def build_branch_key_from_branch(branch_cashpoint_id: object) -> str:
    digit_string = extract_digits(branch_cashpoint_id)
    if not digit_string:
        return ""
    return digit_string[-4:] if len(digit_string) > 4 else digit_string.zfill(4)


def normalize_dates(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce", dayfirst=True).dt.normalize()


def format_branch_cashpoint_id(branch_key: object) -> str:
    return f"Z00{str(branch_key).zfill(4)}"


def load_input_tables(input_file: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    atm_df = pd.read_excel(input_file, sheet_name=ATM_SHEET_NAME)
    branch_df = pd.read_excel(input_file, sheet_name=BRANCH_SHEET_NAME)
    events_df = pd.read_excel(input_file, sheet_name=EVENTS_SHEET_NAME)
    return atm_df, branch_df, events_df


def prepare_keys_and_dates(atm_df: pd.DataFrame, branch_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    atm_prepared = atm_df.copy()
    branch_prepared = branch_df.copy()

    atm_prepared["BRANCH_KEY"] = atm_prepared["CASHP_ID"].apply(build_branch_key_from_atm)
    branch_prepared["BRANCH_KEY"] = branch_prepared["CASHP_ID"].apply(build_branch_key_from_branch)

    atm_prepared["DATE_N"] = normalize_dates(atm_prepared["DATE"])
    branch_prepared["DATE_N"] = normalize_dates(branch_prepared["DATE"])

    return atm_prepared, branch_prepared


def build_branch_master(branch_df: pd.DataFrame) -> pd.DataFrame:
    branch_master = branch_df.copy()
    branch_master["CASHP_ID_BRANCH"] = branch_master["BRANCH_KEY"].apply(format_branch_cashpoint_id)
    return branch_master


def merge_branch_and_atm(branch_master_df: pd.DataFrame, atm_df: pd.DataFrame) -> pd.DataFrame:
    merged_df = branch_master_df.merge(
        atm_df,
        on=["BRANCH_KEY", "DATE_N"],
        how="left",
        suffixes=("_BR", "_ATM"),
    )

    # Keep one DATE column in the final table.
    merged_df["DATE"] = merged_df["DATE_BR"] if "DATE_BR" in merged_df.columns else merged_df["DATE"]

    columns_to_drop = ["DATE_BR", "DATE_ATM", "DATE_N", "CASHP_ID_BR"]
    merged_df = merged_df.drop(columns=[col for col in columns_to_drop if col in merged_df.columns])

    atm_id_column = "CASHP_ID_ATM" if "CASHP_ID_ATM" in merged_df.columns else None
    if atm_id_column is None:
        raise RuntimeError("Expected column 'CASHP_ID_ATM' was not found after the merge.")

    merged_df["ROW_SOURCE"] = "BRANCH_ONLY"
    merged_df.loc[merged_df[atm_id_column].notna(), "ROW_SOURCE"] = "MATCHED"
    return merged_df


def append_unmatched_atm_rows(
    merged_df: pd.DataFrame,
    atm_df: pd.DataFrame,
    branch_df: pd.DataFrame,
) -> pd.DataFrame:
    # Add ATM rows that do not appear in the branch-based merge.
    unmatched_atm_df = atm_df.merge(
        branch_df[["BRANCH_KEY", "DATE_N"]],
        on=["BRANCH_KEY", "DATE_N"],
        how="left",
        indicator=True,
    )
    unmatched_atm_df = unmatched_atm_df[unmatched_atm_df["_merge"] == "left_only"].drop(columns=["_merge"])

    unmatched_atm_df["CASHP_ID_BRANCH"] = unmatched_atm_df["BRANCH_KEY"].apply(format_branch_cashpoint_id)
    unmatched_atm_df["ROW_SOURCE"] = "UNMATCHED_ATM"

    rename_map: dict[str, str] = {}
    for column_name in list(unmatched_atm_df.columns):
        atm_style_name = f"{column_name}_ATM"
        if atm_style_name in merged_df.columns:
            rename_map[column_name] = atm_style_name
    unmatched_atm_df = unmatched_atm_df.rename(columns=rename_map)

    for column_name in merged_df.columns:
        if column_name not in unmatched_atm_df.columns:
            unmatched_atm_df[column_name] = pd.NA

    unmatched_atm_df = unmatched_atm_df[merged_df.columns].copy()

    cleaned_merged_df = merged_df.dropna(axis=1, how="all")
    cleaned_unmatched_df = unmatched_atm_df.dropna(axis=1, how="all")

    return pd.concat([cleaned_merged_df, cleaned_unmatched_df], ignore_index=True)


def sort_final_output(final_df: pd.DataFrame) -> pd.DataFrame:
    sorted_df = final_df.copy()
    sorted_df["DATE"] = pd.to_datetime(sorted_df["DATE"], errors="coerce", dayfirst=True)
    return sorted_df.sort_values(["CASHP_ID_BRANCH", "DATE"]).reset_index(drop=True)


def save_output(final_df: pd.DataFrame, events_df: pd.DataFrame, output_file: Path) -> None:
    with pd.ExcelWriter(output_file, engine="openpyxl") as writer:
        final_df.to_excel(writer, sheet_name=OUTPUT_SHEET_NAME, index=False)
        events_df.to_excel(writer, sheet_name=EVENTS_SHEET_NAME, index=False)


def print_summary(final_df: pd.DataFrame, output_file: Path) -> None:
    print("Merge completed successfully.")
    print("MATCHED:", (final_df["ROW_SOURCE"] == "MATCHED").sum())
    print("BRANCH_ONLY:", (final_df["ROW_SOURCE"] == "BRANCH_ONLY").sum())
    print("UNMATCHED_ATM:", (final_df["ROW_SOURCE"] == "UNMATCHED_ATM").sum())
    print("Unique branch count:", final_df["CASHP_ID_BRANCH"].nunique())
    print("Output:", output_file)


def main() -> None:
    atm_df, branch_df, events_df = load_input_tables(INPUT_FILE)
    atm_df, branch_df = prepare_keys_and_dates(atm_df, branch_df)
    branch_master_df = build_branch_master(branch_df)
    merged_df = merge_branch_and_atm(branch_master_df, atm_df)
    final_df = append_unmatched_atm_rows(merged_df, atm_df, branch_df)
    final_df = sort_final_output(final_df)
    save_output(final_df, events_df, OUTPUT_FILE)
    print_summary(final_df, OUTPUT_FILE)


if __name__ == "__main__":
    main()
