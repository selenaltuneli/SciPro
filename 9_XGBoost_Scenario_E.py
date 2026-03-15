import warnings

warnings.filterwarnings("ignore")

import os
from glob import glob
from pathlib import Path

import numpy as np
import pandas as pd

# Try to help XGBoost find libomp on macOS Homebrew setups.
if "DYLD_LIBRARY_PATH" not in os.environ:
    candidates = sorted(glob("/opt/homebrew/Cellar/llvm/*/lib"))
    if candidates:
        os.environ["DYLD_LIBRARY_PATH"] = candidates[-1]

import xgboost as xgb


BASE_DIR = Path(__file__).resolve().parent
INPUT_PATH = str(BASE_DIR / "ATM_Branch_Data_Final_filled.xlsx")
OUTPUT_PATH = str(BASE_DIR / "scenario_E_fixed_3day_reopt_forecasts_XGBoost.xlsx")

DATE_COL = "DATE"
TARGET_COL = "WITHDRWLS_ATM"
ATM_COL = "CASHP_ID_ATM"

REOPT_FREQ_DAYS = 3
HORIZON_DAYS = 3
VALIDATION_DAYS = 60
RANDOM_SEED = 42
USE_LOG_TARGET = True
TRAIN_WINDOW_DAYS = 365
APE_MIN_TRUE = 1000


def load_data(path: str) -> pd.DataFrame:
    df = pd.read_excel(path)
    df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors="coerce")
    df = df.dropna(subset=[DATE_COL, TARGET_COL]).copy()

    min_date = df[DATE_COL].min()
    df["TIME_INDEX"] = (df[DATE_COL] - min_date).dt.days
    df["DAY_OF_WEEK"] = df[DATE_COL].dt.dayofweek
    df["IS_WEEKEND"] = (df["DAY_OF_WEEK"] >= 5).astype(int)
    df["DAY_OF_MONTH"] = df[DATE_COL].dt.day
    df["MONTH"] = df[DATE_COL].dt.month
    df["WEEK_OF_YEAR"] = df[DATE_COL].dt.isocalendar().week.astype(int)
    df["IS_MONTH_START"] = df[DATE_COL].dt.is_month_start.astype(int)
    df["IS_MONTH_END"] = df[DATE_COL].dt.is_month_end.astype(int)

    if ATM_COL in df.columns:
        df[ATM_COL] = df[ATM_COL].astype(str)
        df = df.sort_values([ATM_COL, DATE_COL]).reset_index(drop=True)
    else:
        df = df.sort_values([DATE_COL]).reset_index(drop=True)

    return df


def reopt_schedule(df: pd.DataFrame) -> pd.DatetimeIndex:
    max_date = df[DATE_COL].max().normalize()
    min_date = df[DATE_COL].min().normalize()
    start = max_date - pd.Timedelta(days=365)
    schedule_start = max(min_date, start)
    return pd.date_range(start=schedule_start, end=max_date, freq=f"{REOPT_FREQ_DAYS}D")


def feature_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in {DATE_COL, TARGET_COL}]


def one_hot_with_reference(train_df: pd.DataFrame, other_df: pd.DataFrame, feats: list[str]):
    x_train = pd.get_dummies(train_df[feats], drop_first=False)
    x_other = pd.get_dummies(other_df[feats], drop_first=False)
    x_other = x_other.reindex(columns=x_train.columns, fill_value=0)
    return x_train, x_other


def train_xgb_booster(train_df: pd.DataFrame, feats: list[str]):
    max_day = train_df[DATE_COL].max()
    valid_start = max_day - pd.Timedelta(days=VALIDATION_DAYS)

    tr = train_df[train_df[DATE_COL] < valid_start].copy()
    va = train_df[train_df[DATE_COL] >= valid_start].copy()
    use_valid = (len(tr) > 1000) and (len(va) > 200)

    x_tr, _ = one_hot_with_reference(tr, tr, feats)
    _, x_va = one_hot_with_reference(tr, va, feats)

    y_tr = tr[TARGET_COL].to_numpy()
    y_va = va[TARGET_COL].to_numpy()
    if USE_LOG_TARGET:
        y_tr = np.log1p(y_tr)
        y_va = np.log1p(y_va)

    dtrain = xgb.DMatrix(x_tr, label=y_tr, feature_names=list(x_tr.columns))
    dvalid = xgb.DMatrix(x_va, label=y_va, feature_names=list(x_tr.columns))

    params = {
        "objective": "reg:squarederror",
        "eval_metric": "mae",
        "learning_rate": 0.03,
        "max_depth": 6,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
        "lambda": 1.0,
        "seed": RANDOM_SEED,
        "tree_method": "hist",
    }

    if use_valid:
        booster = xgb.train(
            params=params,
            dtrain=dtrain,
            num_boost_round=5000,
            evals=[(dvalid, "valid")],
            callbacks=[xgb.callback.EarlyStopping(rounds=200, save_best=True)],
            verbose_eval=False,
        )

        best_iter = getattr(booster, "best_iteration", None)
        if best_iter is not None:
            y_va_pred = booster.predict(dvalid, iteration_range=(0, best_iter + 1))
        else:
            y_va_pred = booster.predict(dvalid)

        if USE_LOG_TARGET:
            y_va_pred = np.expm1(y_va_pred)
            y_va_true = np.expm1(y_va)
        else:
            y_va_true = y_va

        y_va_pred = np.clip(y_va_pred, 0, None)
        ratio = (y_va_true + 1.0) / (y_va_pred + 1.0)
        ratio = ratio[np.isfinite(ratio)]
        calib = float(np.clip(np.median(ratio), 0.85, 1.20)) if len(ratio) else 1.0
        return booster, x_tr.columns, calib

    x_all, _ = one_hot_with_reference(train_df, train_df, feats)
    y_all = train_df[TARGET_COL].to_numpy()
    if USE_LOG_TARGET:
        y_all = np.log1p(y_all)

    dtrain_all = xgb.DMatrix(x_all, label=y_all, feature_names=list(x_all.columns))
    booster = xgb.train(params=params, dtrain=dtrain_all, num_boost_round=1000, verbose_eval=False)
    return booster, x_all.columns, 1.0


def predict_with_booster(booster: xgb.Booster, dmatrix: xgb.DMatrix) -> np.ndarray:
    best_iter = getattr(booster, "best_iteration", None)
    if best_iter is not None:
        return booster.predict(dmatrix, iteration_range=(0, best_iter + 1))
    return booster.predict(dmatrix)


def compute_metrics(df_forecasts: pd.DataFrame) -> dict:
    y_true = df_forecasts["Y_TRUE_WITHDRWLS_ATM"].to_numpy()
    y_pred = df_forecasts["Y_PRED_WITHDRWLS_ATM"].to_numpy()
    abs_err = np.abs(y_true - y_pred)

    mask = y_true >= APE_MIN_TRUE if APE_MIN_TRUE > 0 else y_true > 0
    ape = np.full_like(y_true, np.nan, dtype=float)
    ape[mask] = abs_err[mask] / y_true[mask]

    mae = float(np.mean(abs_err))
    mean_ape = float(np.nanmean(ape))
    median_ape = float(np.nanmedian(ape))
    pos = y_true > 0
    weighted_mape = float(np.sum(abs_err[pos]) / np.sum(y_true[pos])) if np.sum(y_true[pos]) > 0 else np.nan

    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else np.nan

    return {
        "MAE": mae,
        "Mean_APE": mean_ape,
        "Median_APE": median_ape,
        "Weighted_MAPE": weighted_mape,
        "R2": r2,
    }


def run_scenario_e(df: pd.DataFrame) -> pd.DataFrame:
    feats = feature_cols(df)
    schedule = reopt_schedule(df)
    outputs = []

    for reopt_start in schedule:
        # Keep Scenario E aligned with the original optimization logic:
        # reoptimization happens on the last training night, and forecasting
        # starts on the following day.
        train_end = reopt_start
        forecast_start = reopt_start + pd.Timedelta(days=1)
        forecast_end = forecast_start + pd.Timedelta(days=HORIZON_DAYS - 1)

        train_df = df[df[DATE_COL] <= train_end].copy()
        if TRAIN_WINDOW_DAYS is not None:
            window_start = train_end - pd.Timedelta(days=TRAIN_WINDOW_DAYS - 1)
            train_df = train_df[train_df[DATE_COL] >= window_start].copy()

        if len(train_df) < 2000:
            continue

        pred_df = df[(df[DATE_COL] >= forecast_start) & (df[DATE_COL] <= forecast_end)].copy()
        if pred_df.empty:
            continue

        booster, ref_cols, calib = train_xgb_booster(train_df, feats)
        x_pred = pd.get_dummies(pred_df[feats], drop_first=False)
        x_pred = x_pred.reindex(columns=ref_cols, fill_value=0)
        dpred = xgb.DMatrix(x_pred, feature_names=list(ref_cols))

        y_pred = predict_with_booster(booster, dpred)
        if USE_LOG_TARGET:
            y_pred = np.expm1(y_pred)

        y_pred = np.clip(y_pred * calib, 0, None)

        tmp = pd.DataFrame(
            {
                "WEEK_START": forecast_start,
                "TRAIN_END": train_end,
                "FORECAST_DATE": pred_df[DATE_COL].values,
                ATM_COL: pred_df[ATM_COL].astype(str).values if ATM_COL in pred_df.columns else None,
                "Y_PRED_WITHDRWLS_ATM": y_pred,
                "Y_TRUE_WITHDRWLS_ATM": pred_df[TARGET_COL].values,
                "REOPT_CYCLE_DAYS": REOPT_FREQ_DAYS,
                "FORECAST_HORIZON_DAYS": HORIZON_DAYS,
            }
        )

        tmp["ABS_ERROR"] = np.abs(tmp["Y_TRUE_WITHDRWLS_ATM"] - tmp["Y_PRED_WITHDRWLS_ATM"])
        mask = tmp["Y_TRUE_WITHDRWLS_ATM"] >= APE_MIN_TRUE if APE_MIN_TRUE > 0 else tmp["Y_TRUE_WITHDRWLS_ATM"] > 0
        tmp["APE"] = np.nan
        tmp.loc[mask, "APE"] = tmp.loc[mask, "ABS_ERROR"] / tmp.loc[mask, "Y_TRUE_WITHDRWLS_ATM"]

        outputs.append(tmp)
        print(f"[OK] {reopt_start.date()} | rows={len(tmp)} | calib={calib:.3f}")

    if not outputs:
        return pd.DataFrame()

    return pd.concat(outputs, ignore_index=True)


def save_excel(df_forecasts: pd.DataFrame, path: str) -> None:
    metrics = compute_metrics(df_forecasts)

    cycle_summary = (
        df_forecasts.groupby(["WEEK_START", "TRAIN_END"], dropna=False)
        .agg(
            n=("Y_TRUE_WITHDRWLS_ATM", "size"),
            mae=("ABS_ERROR", "mean"),
            mean_ape=("APE", "mean"),
            median_ape=("APE", "median"),
            bias=("Y_PRED_WITHDRWLS_ATM", lambda s: float(np.mean(s - df_forecasts.loc[s.index, "Y_TRUE_WITHDRWLS_ATM"]))),
        )
        .reset_index()
    )

    metrics_df = pd.DataFrame([metrics])

    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        df_forecasts.to_excel(writer, index=False, sheet_name="Forecasts")
        cycle_summary.to_excel(writer, index=False, sheet_name="Reopt_Summary")
        metrics_df.to_excel(writer, index=False, sheet_name="Overall_Metrics")


def main() -> None:
    df = load_data(INPUT_PATH)
    forecasts = run_scenario_e(df)

    if forecasts.empty:
        raise RuntimeError("The output is empty. Please verify date coverage and column names.")

    save_excel(forecasts, OUTPUT_PATH)


    metrics = compute_metrics(forecasts)
    print("\nOverall Metrics (Scenario E XGBoost)")
    print(f"MAE           : {metrics['MAE']:.2f}")
    print(f"Mean APE      : {metrics['Mean_APE'] * 100:.2f}%")
    print(f"Median APE    : {metrics['Median_APE'] * 100:.2f}%")
    print(f"Weighted MAPE : {metrics['Weighted_MAPE'] * 100:.2f}%")
    print(f"R2            : {metrics['R2']:.4f}")
    print(f"\nSaved: {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
