"""Train a weekly re-optimization forecast model with XGBoost for Scenario A."""

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import r2_score

warnings.filterwarnings("ignore")

INPUT_PATH = Path(r"C:\Users\Selen\Desktop\ORBA\Scientific Project\SciPro\ATM_Branch_Data_Final_filled.xlsx")
OUTPUT_PATH = Path(r"C:\Users\Selen\Desktop\ORBA\Scientific Project\SciPro\scenario_A_weekly_reopt_forecasts_XGBoost.xlsx")

DATE_COL = "DATE"
TARGET_COL = "WITHDRWLS_ATM"

# Scenario A is re-trained every Monday and produces a 7-day ahead forecast horizon.
WEEK_FREQ = "W-MON"
HORIZON_DAYS = 7

# The most recent part of the training data is held out for early stopping.
VALIDATION_DAYS = 60
RANDOM_SEED = 42

# Using log1p on the target helps stabilize variance and reduces the impact of large values.
USE_LOG_TARGET = True

# Restricting training to recent history helps reduce concept drift.
TRAIN_WINDOW_DAYS = 365

# Very small actual values can make APE unstable, so they can be excluded if needed.
APE_MIN_TRUE = 0
MIN_TRAIN_ROWS = 2000


def load_data(path: Path) -> pd.DataFrame:
    df = pd.read_excel(path)
    df[DATE_COL] = pd.to_datetime(df[DATE_COL])

    # TIME_INDEX gives the model a simple numeric representation of long-term trend.
    min_date = df[DATE_COL].min()
    df["TIME_INDEX"] = (df[DATE_COL] - min_date).dt.days

    if "CASHP_ID_ATM" in df.columns:
        df = df.sort_values(["CASHP_ID_ATM", DATE_COL]).reset_index(drop=True)
    else:
        df = df.sort_values(DATE_COL).reset_index(drop=True)

    return df


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    # All columns except date and target are used as model inputs.
    return [column_name for column_name in df.columns if column_name not in {DATE_COL, TARGET_COL}]


def build_weekly_schedule(df: pd.DataFrame) -> pd.DatetimeIndex:
    max_date = df[DATE_COL].max()
    start_date = max_date - pd.Timedelta(days=365)
    end_date = max_date

    schedule = pd.date_range(start=start_date, end=end_date, freq=WEEK_FREQ)
    return schedule[(schedule >= df[DATE_COL].min()) & (schedule <= df[DATE_COL].max())]


def prepare_target(values: np.ndarray) -> np.ndarray:
    if USE_LOG_TARGET:
        return np.log1p(values)
    return values


def invert_target(values: np.ndarray) -> np.ndarray:
    if USE_LOG_TARGET:
        return np.expm1(values)
    return values


def one_hot_with_reference(
    train_df: pd.DataFrame,
    other_df: pd.DataFrame,
    feature_columns: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    # XGBoost does not handle categorical variables natively in this setup,
    # so both train and prediction matrices are aligned through one-hot encoding.
    x_train = pd.get_dummies(train_df[feature_columns], drop_first=False)
    x_other = pd.get_dummies(other_df[feature_columns], drop_first=False)
    x_other = x_other.reindex(columns=x_train.columns, fill_value=0)
    return x_train, x_other


def train_xgboost_model(train_df: pd.DataFrame, feature_columns: list[str]) -> tuple[xgb.Booster, pd.Index]:
    max_train_day = train_df[DATE_COL].max()
    validation_start = max_train_day - pd.Timedelta(days=VALIDATION_DAYS)

    # The newest part of the training window is used as a validation split.
    train_split = train_df[train_df[DATE_COL] < validation_start].copy()
    validation_split = train_df[train_df[DATE_COL] >= validation_start].copy()
    use_validation = len(train_split) > 1000 and len(validation_split) > 200

    x_train, _ = one_hot_with_reference(train_split, train_split, feature_columns)
    _, x_valid = one_hot_with_reference(train_split, validation_split, feature_columns)

    y_train = prepare_target(train_split[TARGET_COL].values)
    y_valid = prepare_target(validation_split[TARGET_COL].values)

    dtrain = xgb.DMatrix(x_train, label=y_train, feature_names=list(x_train.columns))
    dvalid = xgb.DMatrix(x_valid, label=y_valid, feature_names=list(x_train.columns))

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

    if use_validation:
        # Early stopping is used when the train/validation split is large enough to be reliable.
        booster = xgb.train(
            params=params,
            dtrain=dtrain,
            num_boost_round=5000,
            evals=[(dvalid, "valid")],
            early_stopping_rounds=200,
            verbose_eval=False,
        )
        return booster, x_train.columns

    # If the validation split is too small, the model is trained on the full available window.
    x_all, _ = one_hot_with_reference(train_df, train_df, feature_columns)
    y_all = prepare_target(train_df[TARGET_COL].values)
    dtrain_all = xgb.DMatrix(x_all, label=y_all, feature_names=list(x_all.columns))

    booster = xgb.train(
        params=params,
        dtrain=dtrain_all,
        num_boost_round=1000,
        verbose_eval=False,
    )
    return booster, x_all.columns


def compute_metrics(forecast_df: pd.DataFrame) -> dict[str, float]:
    y_true = forecast_df["Y_TRUE_WITHDRWLS_ATM"].to_numpy()
    y_pred = forecast_df["Y_PRED_WITHDRWLS_ATM"].to_numpy()
    abs_error = np.abs(y_true - y_pred)

    if APE_MIN_TRUE > 0:
        valid_mask = y_true >= APE_MIN_TRUE
    else:
        valid_mask = y_true > 0

    ape = np.full_like(y_true, np.nan, dtype=float)
    ape[valid_mask] = abs_error[valid_mask] / y_true[valid_mask]

    positive_mask = y_true > 0
    # Weighted MAPE gives more importance to high-volume days than standard MAPE.
    weighted_mape = (
        float(np.sum(abs_error[positive_mask]) / np.sum(y_true[positive_mask]))
        if np.sum(y_true[positive_mask]) > 0
        else np.nan
    )

    return {
        "MAE": float(np.mean(abs_error)),
        "Mean_APE": float(np.nanmean(ape)),
        "Median_APE": float(np.nanmedian(ape)),
        "Weighted_MAPE": weighted_mape,
        "R2": float(r2_score(y_true, y_pred)),
    }


def build_forecast_rows(
    week_start: pd.Timestamp,
    train_end: pd.Timestamp,
    prediction_df: pd.DataFrame,
    predictions: np.ndarray,
) -> pd.DataFrame:
    output_df = pd.DataFrame(
        {
            "MODEL": "XGBoost",
            "SCENARIO": "A",
            "WEEK_START": week_start,
            "TRAIN_END": train_end,
            "FORECAST_DATE": prediction_df[DATE_COL].values,
            "CASHP_ID_ATM": (
                prediction_df["CASHP_ID_ATM"].astype(str).values
                if "CASHP_ID_ATM" in prediction_df.columns
                else None
            ),
            "Y_PRED_WITHDRWLS_ATM": predictions,
            "Y_TRUE_WITHDRWLS_ATM": prediction_df[TARGET_COL].values,
        }
    )

    output_df["ABS_ERROR"] = np.abs(output_df["Y_TRUE_WITHDRWLS_ATM"] - output_df["Y_PRED_WITHDRWLS_ATM"])

    if APE_MIN_TRUE > 0:
        valid_mask = output_df["Y_TRUE_WITHDRWLS_ATM"] >= APE_MIN_TRUE
    else:
        valid_mask = output_df["Y_TRUE_WITHDRWLS_ATM"] > 0

    output_df["APE"] = np.nan
    output_df.loc[valid_mask, "APE"] = (
        output_df.loc[valid_mask, "ABS_ERROR"] / output_df.loc[valid_mask, "Y_TRUE_WITHDRWLS_ATM"]
    )

    return output_df


def run_scenario_a(df: pd.DataFrame) -> pd.DataFrame:
    feature_columns = get_feature_columns(df)
    weekly_schedule = build_weekly_schedule(df)
    forecast_parts = []

    for week_start in weekly_schedule:
        train_end = week_start - pd.Timedelta(days=1)
        forecast_end = week_start + pd.Timedelta(days=HORIZON_DAYS - 1)

        # Only past data is used for training to avoid leakage.
        train_df = df[df[DATE_COL] <= train_end].copy()

        if TRAIN_WINDOW_DAYS is not None:
            # A rolling training window avoids learning from data that may be too old.
            window_start = train_end - pd.Timedelta(days=TRAIN_WINDOW_DAYS - 1)
            train_df = train_df[train_df[DATE_COL] >= window_start].copy()

        # Skip forecast weeks that do not have enough training observations.
        if len(train_df) < MIN_TRAIN_ROWS:
            continue

        prediction_df = df[(df[DATE_COL] >= week_start) & (df[DATE_COL] <= forecast_end)].copy()
        if prediction_df.empty:
            continue

        booster, reference_columns = train_xgboost_model(train_df, feature_columns)

        # The prediction matrix must match the training feature space exactly.
        x_pred = pd.get_dummies(prediction_df[feature_columns], drop_first=False)
        x_pred = x_pred.reindex(columns=reference_columns, fill_value=0)

        dpred = xgb.DMatrix(x_pred, feature_names=list(reference_columns))
        predictions = invert_target(booster.predict(dpred))

        # Negative cash withdrawal forecasts are not meaningful, so they are clipped at zero.
        predictions = np.clip(predictions, 0, None)

        forecast_parts.append(build_forecast_rows(week_start, train_end, prediction_df, predictions))
        print(f"[OK] {week_start.date()} | rows={len(prediction_df)}")

    if not forecast_parts:
        return pd.DataFrame()

    return pd.concat(forecast_parts, ignore_index=True)


def save_excel(forecast_df: pd.DataFrame, output_path: Path) -> None:
    metrics = compute_metrics(forecast_df)

    weekly_summary = (
        forecast_df.groupby("WEEK_START")
        .agg(
            n=("Y_TRUE_WITHDRWLS_ATM", "size"),
            mae=("ABS_ERROR", "mean"),
            mean_ape=("APE", "mean"),
            median_ape=("APE", "median"),
        )
        .reset_index()
    )

    metrics_df = pd.DataFrame([metrics])

    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        forecast_df.to_excel(writer, index=False, sheet_name="Scenario_A_Forecasts")
        weekly_summary.to_excel(writer, index=False, sheet_name="Weekly_Summary")
        metrics_df.to_excel(writer, index=False, sheet_name="Overall_Metrics")


def print_overall_metrics(forecast_df: pd.DataFrame) -> None:
    metrics = compute_metrics(forecast_df)
    print("\nOverall Metrics (XGBoost)")
    print(f"MAE           : {metrics['MAE']:.2f}")
    print(f"Mean APE      : {metrics['Mean_APE'] * 100:.2f}%")
    print(f"Median APE    : {metrics['Median_APE'] * 100:.2f}%")
    print(f"Weighted MAPE : {metrics['Weighted_MAPE'] * 100:.2f}%")
    print(f"R2            : {metrics['R2']:.4f}")


def main() -> None:
    df = load_data(INPUT_PATH)
    forecast_df = run_scenario_a(df)

    if forecast_df.empty:
        raise RuntimeError("The forecast output is empty. Please verify date coverage and column names.")

    save_excel(forecast_df, OUTPUT_PATH)
    print_overall_metrics(forecast_df)
    print(f"\nSaved: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
