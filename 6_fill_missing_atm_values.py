"""Fill missing ATM information for selected branch-only records."""

from pathlib import Path

import numpy as np
import pandas as pd

INPUT_PATH = Path(r"C:\Users\Selen\Desktop\ORBA\Scientific Project\SciPro\ATM_Branch_Data_Final.xlsx")
OUTPUT_PATH = Path(r"C:\Users\Selen\Desktop\ORBA\Scientific Project\SciPro\ATM_Branch_Data_Final_filled.xlsx")

TARGET_BRANCH_KEYS = {283, 284, 899, 908, 972, 1169, 1367, 1665, 1769, 2011, 2136}


def build_synthetic_atm_id(branch_key: int) -> str:
    return f"Z{branch_key:04d}001"


def add_missing_atm_ids(df: pd.DataFrame, branch_key_num: pd.Series) -> tuple[pd.DataFrame, pd.Series, dict[int, str]]:
    output_df = df.copy()
    target_mask = branch_key_num.isin(TARGET_BRANCH_KEYS)
    fill_atm_id_mask = target_mask & output_df["CASHP_ID_ATM"].isna() & branch_key_num.notna()

    unique_branches = sorted(branch_key_num[fill_atm_id_mask].dropna().unique().tolist())
    branch_to_atm_id = {int(branch): build_synthetic_atm_id(int(branch)) for branch in unique_branches}

    output_df.loc[fill_atm_id_mask, "CASHP_ID_ATM"] = (
        branch_key_num[fill_atm_id_mask].astype(int).map(branch_to_atm_id)
    )

    return output_df, fill_atm_id_mask, branch_to_atm_id


def compute_daily_share_table(df: pd.DataFrame) -> tuple[pd.DataFrame, float]:
    reference_df = df[
        df["WITHDRWLS_ATM"].notna()
        & df["WITHDRWLS_BR"].notna()
        & (df["WITHDRWLS_BR"] > 0)
    ].copy()

    reference_daily = (
        reference_df.groupby(["DATE", "BRANCH_KEY"], as_index=False)
        .agg(
            atm_sum=("WITHDRWLS_ATM", "sum"),
            branch_withdrawals=("WITHDRWLS_BR", "first"),
        )
    )

    reference_daily["share"] = (
        reference_daily["atm_sum"] / reference_daily["branch_withdrawals"]
    ).replace([np.inf, -np.inf], np.nan)
    reference_daily = reference_daily.dropna(subset=["share"])
    reference_daily["share"] = reference_daily["share"].clip(0, 1)

    daily_share = (
        reference_daily.groupby("DATE", as_index=False)["share"]
        .median()
        .rename(columns={"share": "share_median_day"})
    )

    global_share = float(reference_daily["share"].median()) if len(reference_daily) else 0.10
    return daily_share, global_share


def fill_missing_atm_withdrawals(
    df: pd.DataFrame,
    branch_key_num: pd.Series,
    daily_share: pd.DataFrame,
    global_share: float,
) -> tuple[pd.DataFrame, pd.Series]:
    output_df = df.copy()

    target_mask = branch_key_num.isin(TARGET_BRANCH_KEYS)
    fill_withdrawal_mask = (
        target_mask
        & output_df["WITHDRWLS_ATM"].isna()
        & output_df["WITHDRWLS_BR"].notna()
        & (output_df["WITHDRWLS_BR"] >= 0)
    )

    temp_df = output_df.loc[fill_withdrawal_mask, ["DATE", "WITHDRWLS_BR"]].merge(
        daily_share,
        on="DATE",
        how="left",
    )
    share_used = temp_df["share_median_day"].fillna(global_share).clip(0, 1)

    output_df.loc[fill_withdrawal_mask, "WITHDRWLS_ATM"] = (
        temp_df["WITHDRWLS_BR"] * share_used
    ).clip(lower=0).round(0).values

    return output_df, fill_withdrawal_mask


def main() -> None:
    df = pd.read_excel(INPUT_PATH)
    df["DATE"] = pd.to_datetime(df["DATE"], errors="coerce")

    # Use a numeric version of BRANCH_KEY for filtering without changing the original column.
    branch_key_num = pd.to_numeric(df["BRANCH_KEY"], errors="coerce")

    df, fill_atm_id_mask, branch_to_atm_id = add_missing_atm_ids(df, branch_key_num)
    daily_share, global_share = compute_daily_share_table(df)
    df, fill_withdrawal_mask = fill_missing_atm_withdrawals(df, branch_key_num, daily_share, global_share)

    df.to_excel(OUTPUT_PATH, index=False)

    print("Missing branch-only ATM values were filled successfully.")
    print(f"Output file: {OUTPUT_PATH}")
    print(f"Filled CASHP_ID_ATM rows: {int(fill_atm_id_mask.sum())}")
    print(f"Filled WITHDRWLS_ATM rows: {int(fill_withdrawal_mask.sum())}")
    print(f"Daily-share days: {int(daily_share['DATE'].nunique())}")
    print(f"Global share fallback: {global_share}")
    print(f"Sample synthetic ATM IDs: {list(branch_to_atm_id.items())[:5]}")


if __name__ == "__main__":
    main()
