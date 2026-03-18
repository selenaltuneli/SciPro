"""Train a weekly re-optimization forecast model with LightGBM for Scenario A."""

import warnings
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score

warnings.filterwarnings("ignore")

INPUT_PATH = Path(r"C:\Users\Selen\Desktop\ORBA\Scientific Project\SciPro\ATM_Branch_Data_Final_filled.xlsx")
OUTPUT_PATH = Path(r"C:\Users\Selen\Desktop\ORBA\Scientific Project\SciPro\scenario_A_weekly_reopt_forecasts.xlsx")

DATE_COL = "DATE"
TARGET_COL = "WITHDRWLS_ATM"

WEEK_FREQ = "W-MON"
HORIZON_DAYS = 7
VALIDATION_DAYS = 60
RANDOM_SEED = 42
# Using log1p on the target helps stabilize variance and reduces the impact of very large values.
USE_LOG_TARGET = True
# Restricting training to the recent history helps reduce concept drift.
TRAIN_WINDOW_DAYS = 365
# Very small actual values can distort percentage-based error metrics.
MIN_DEMAND_FOR_APE = 0
MIN_TRAIN_ROWS = 2000


def load_data(path: Path) -> pd.DataFrame:
    df = pd.read_excel(path)
    df[DATE_COL] = pd.to_datetime(df[DATE_COL])

    # TIME_INDEX gives the model a simple numeric representation of trend over time.
    min_date = df[DATE_COL].min()
    df["TIME_INDEX"] = (df[DATE_COL] - min_date).dt.days

    if "CASHP_ID_ATM" in df.columns:
        df = df.sort_values(["CASHP_ID_ATM", DATE_COL]).reset_index(drop=True)
    else:
        df = df.sort_values(DATE_COL).reset_index(drop=True)

    # LightGBM can work directly with pandas category dtype, so one-hot encoding is not required here.
    for column_name in ["CASHP_ID_ATM", "CASHP_ID_BRANCH", "functional_zone"]:
        if column_name in df.columns:
            df[column_name] = df[column_name].astype("category")

    return df


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    # All columns except date and target are used as model inputs.
    return [column_name for column_name in df.columns if column_name not in {DATE_COL, TARGET_COL}]


def build_weekly_schedule(df: pd.DataFrame) -> pd.DatetimeIndex:
    max_date = df[DATE_COL].max()
    start_date = max_date - pd.Timedelta(days=365)
    end_date = max_date

    # Scenario A is evaluated on weekly re-training points, starting every Monday.
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


def train_lightgbm_model(train_df: pd.DataFrame, feature_columns: list[str]) -> lgb.LGBMRegressor:
    max_train_day = train_df[DATE_COL].max()
    validation_start = max_train_day - pd.Timedelta(days=VALIDATION_DAYS)

    # The most recent part of the training window is held out for early stopping.
    train_split = train_df[train_df[DATE_COL] < validation_start]
    validation_split = train_df[train_df[DATE_COL] >= validation_start]

    model = lgb.LGBMRegressor(
        objective="regression",
        n_estimators=5000,
        learning_rate=0.03,
        num_leaves=63,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=RANDOM_SEED,
        n_jobs=-1,
    )

    y_train = prepare_target(train_split[TARGET_COL].values)
    y_valid = prepare_target(validation_split[TARGET_COL].values)

    if len(train_split) > 1000 and len(validation_split) > 200:
        # Early stopping is used when the train/validation split is large enough to be reliable.
        model.fit(
            train_split[feature_columns],
            y_train,
            eval_set=[(validation_split[feature_columns], y_valid)],
            eval_metric="l1",
            callbacks=[lgb.early_stopping(stopping_rounds=200, verbose=False)],
        )
    else:
        # If the split is too small, the model is trained on the full available window.
        y_all = prepare_target(train_df[TARGET_COL].values)
        model.fit(train_df[feature_columns], y_all)

    return model


def compute_metrics(forecast_df: pd.DataFrame) -> dict[str, float]:
    y_true = forecast_df["Y_TRUE_WITHDRWLS_ATM"].to_numpy()
    y_pred = forecast_df["Y_PRED_WITHDRWLS_ATM"].to_numpy()
    abs_error = np.abs(y_true - y_pred)

    if MIN_DEMAND_FOR_APE > 0:
        valid_mask = y_true >= MIN_DEMAND_FOR_APE
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
    return pd.DataFrame(
        {
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


def run_scenario_a(df: pd.DataFrame) -> pd.DataFrame:
    feature_columns = get_feature_columns(df)
    weekly_schedule = build_weekly_schedule(df)
    forecast_parts = []

    for week_start in weekly_schedule:
        train_end = week_start - pd.Timedelta(days=1)
        forecast_end = week_start + pd.Timedelta(days=HORIZON_DAYS - 1)

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

        model = train_lightgbm_model(train_df, feature_columns)
        predictions = invert_target(model.predict(prediction_df[feature_columns]))
        # Negative cash withdrawal forecasts are not meaningful, so they are clipped at zero.
        predictions = np.clip(predictions, 0, None)

        forecast_parts.append(build_forecast_rows(week_start, train_end, prediction_df, predictions))
        print(f"[OK] {week_start.date()} | rows={len(prediction_df)}")

    if not forecast_parts:
        return pd.DataFrame()

    forecast_df = pd.concat(forecast_parts, ignore_index=True)
    forecast_df["ABS_ERROR"] = np.abs(
        forecast_df["Y_TRUE_WITHDRWLS_ATM"] - forecast_df["Y_PRED_WITHDRWLS_ATM"]
    )

    if MIN_DEMAND_FOR_APE > 0:
        valid_mask = forecast_df["Y_TRUE_WITHDRWLS_ATM"] >= MIN_DEMAND_FOR_APE
    else:
        valid_mask = forecast_df["Y_TRUE_WITHDRWLS_ATM"] > 0

    forecast_df["APE"] = np.nan
    forecast_df.loc[valid_mask, "APE"] = (
        forecast_df.loc[valid_mask, "ABS_ERROR"] / forecast_df.loc[valid_mask, "Y_TRUE_WITHDRWLS_ATM"]
    )

    return forecast_df


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

    # Repeating global metrics on each weekly row makes Excel comparison easier.
    for metric_name, metric_value in metrics.items():
        weekly_summary[metric_name] = metric_value

    metrics_df = pd.DataFrame([metrics])

    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        forecast_df.to_excel(writer, index=False, sheet_name="Scenario_A_Forecasts")
        weekly_summary.to_excel(writer, index=False, sheet_name="Weekly_Summary")
        metrics_df.to_excel(writer, index=False, sheet_name="Overall_Metrics")


def print_overall_metrics(forecast_df: pd.DataFrame) -> None:
    metrics = compute_metrics(forecast_df)
    print("\nOverall Metrics")
    print(f"MAE           : {metrics['MAE']:.2f}")
    print(f"Mean APE      : {metrics['Mean_APE'] * 100:.2f}%")
    print(f"Median APE    : {metrics['Median_APE'] * 100:.2f}%")
    print(f"Weighted MAPE : {metrics['Weighted_MAPE'] * 100:.2f}%")
    print(f"R2            : {metrics['R2']:.4f}")


def main() -> None:
    df = load_data(INPUT_PATH)
    forecast_df = run_scenario_a(df)

    if forecast_df.empty:
        raise RuntimeError("Forecast output is empty. Check date coverage and input columns.")

    save_excel(forecast_df, OUTPUT_PATH)
    print_overall_metrics(forecast_df)
    print(f"\nSaved: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
