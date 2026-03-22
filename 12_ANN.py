"""Train a weekly ANN forecasting model for ATM withdrawals."""

import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

DATA_PATH = Path("ATM_Branch_Data_Final_filled.xlsx")
OUTPUT_EXCEL = Path("ANN_pred.xlsx")
ACTUAL_PRED_PLOT = Path("ann_actual_vs_predicted.png")
RESIDUAL_PLOT = Path("ann_residuals.png")
WEEKLY_MAE_PLOT = Path("ann_weekly_mae.png")

DATE_COL = "DATE"
ATM_COL = "CASHP_ID_ATM"
TARGET_COL = "WITHDRWLS_ATM"

# The ANN uses the same weekly walk-forward setup as the tree-based models.
WEEK_FREQ = "W-MON"
HORIZON_DAYS = 7
MIN_TRAIN_ROWS = 100

FEATURE_COLUMNS = [
    "ATM_WITHDRWLS_LAG_1",
    "ATM_WITHDRWLS_LAG_7",
    "ATM_WITHDRWLS_MA_7",
    "DAY_OF_WEEK",
    "IS_WEEKDAY",
    "DAY_OF_MONTH",
    "WEEK_OF_YEAR",
    "MONTH",
    "SEASON_NUM",
    "IS_MONTH_START",
    "IS_MONTH_END",
    "IS_QUARTER_END",
    "IS_HOLIDAY",
    "IS_RELIGIOUS_HOLIDAY",
    "IS_NATIONAL_HOLIDAY",
    "IS_PUBLIC_HOLIDAY",
    "IS_SCHOOL_HOLIDAY",
    "IS_PRE_HOLIDAY_1",
    "IS_PRE_HOLIDAY_1_2",
    "IS_PRE_HOLIDAY_1_2_3",
    "IS_POST_HOLIDAY_1",
    "IS_POST_HOLIDAY_1_2",
    "IS_POST_HOLIDAY_1_2_3",
    "HOLIDAY_DURATION_SCORE",
    "HOLIDAY_IMPORTANCE_SCORE",
    "LATITUDE",
    "LONGITUDE",
    "functional_zone",
    "IS_BRANCH_OPEN",
    "WITHDRWLS_BR",
]


def load_data(path: Path) -> pd.DataFrame:
    print("Loading data ...")
    df = pd.read_excel(path, parse_dates=[DATE_COL])
    df = df.sort_values([ATM_COL, DATE_COL]).reset_index(drop=True)

    # Rows without lag information are excluded because the ANN relies on lag-based predictors.
    df = df[df["ATM_WITHDRWLS_LAG_7"] > 0].copy()

    print(
        f"  Rows: {len(df):,}  |  ATMs: {df[ATM_COL].nunique()}  "
        f"|  Date range: {df[DATE_COL].min().date()} -> {df[DATE_COL].max().date()}"
    )
    return df


def build_weekly_schedule(df: pd.DataFrame) -> pd.DatetimeIndex:
    schedule = pd.date_range(start="2007-02-26", end=df[DATE_COL].max(), freq=WEEK_FREQ)
    print(f"\nRolling windows : {len(schedule)}")
    print(f"First window    : {schedule[0].date()}")
    print(f"Last window     : {schedule[-1].date()}")
    return schedule


def train_ann_model(x_train: np.ndarray, y_train: np.ndarray) -> tuple[MLPRegressor, StandardScaler, StandardScaler]:
    scaler_x = StandardScaler()
    scaler_y = StandardScaler()

    x_train_scaled = scaler_x.fit_transform(x_train)
    y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1)).ravel()

    model = MLPRegressor(
        hidden_layer_sizes=(128, 64),
        activation="relu",
        solver="adam",
        learning_rate_init=0.001,
        max_iter=300,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=15,
        random_state=42,
        verbose=False,
    )
    model.fit(x_train_scaled, y_train_scaled)
    return model, scaler_x, scaler_y


def build_forecast_rows(
    week_start: pd.Timestamp,
    train_end: pd.Timestamp,
    test_df: pd.DataFrame,
    predictions: np.ndarray,
) -> list[dict]:
    rows = []
    test_df = test_df.reset_index(drop=True)

    for index, row in test_df.iterrows():
        y_true = row[TARGET_COL]
        y_pred = predictions[index]
        abs_error = abs(y_pred - y_true)
        ape = abs_error / y_true if y_true != 0 else np.nan

        rows.append(
            {
                "WEEK_START": week_start,
                "TRAIN_END": train_end,
                "FORECAST_DATE": row[DATE_COL],
                ATM_COL: row[ATM_COL],
                "Y_PRED_WITHDRWLS_ATM": y_pred,
                "Y_TRUE_WITHDRWLS_ATM": y_true,
                "ABS_ERROR": abs_error,
                "APE": ape,
            }
        )

    return rows


def run_weekly_ann_forecast(df: pd.DataFrame) -> pd.DataFrame:
    weekly_schedule = build_weekly_schedule(df)
    all_rows: list[dict] = []

    print("\nTraining ANN - rolling walk-forward ...\n")

    for window_index, week_start in enumerate(weekly_schedule):
        train_end = week_start - pd.Timedelta(days=1)
        week_end = week_start + pd.Timedelta(days=HORIZON_DAYS - 1)

        train_df = df[df[DATE_COL] <= train_end].copy()
        test_df = df[(df[DATE_COL] >= week_start) & (df[DATE_COL] <= week_end)].copy()

        if len(train_df) < MIN_TRAIN_ROWS or test_df.empty:
            continue

        x_train = train_df[FEATURE_COLUMNS].values
        y_train = train_df[TARGET_COL].values
        x_test = test_df[FEATURE_COLUMNS].values
        y_test = test_df[TARGET_COL].values

        model, scaler_x, scaler_y = train_ann_model(x_train, y_train)

        x_test_scaled = scaler_x.transform(x_test)
        predictions = scaler_y.inverse_transform(model.predict(x_test_scaled).reshape(-1, 1)).ravel()

        # Negative cash withdrawal forecasts are not meaningful, so they are clipped at zero.
        predictions = np.maximum(predictions, 0)

        all_rows.extend(build_forecast_rows(week_start, train_end, test_df, predictions))

        window_mae = mean_absolute_error(y_test, predictions)
        print(
            f"  Window {window_index + 1:02d}/{len(weekly_schedule)}  "
            f"{week_start.date()} -> {min(week_end, df[DATE_COL].max()).date()}  "
            f"MAE = {window_mae:>10,.0f}"
        )

    results = pd.DataFrame(all_rows)
    results["WEEK_START"] = pd.to_datetime(results["WEEK_START"])
    results["TRAIN_END"] = pd.to_datetime(results["TRAIN_END"])
    results["FORECAST_DATE"] = pd.to_datetime(results["FORECAST_DATE"])
    return results


def compute_metrics(results: pd.DataFrame) -> dict[str, float]:
    mae = mean_absolute_error(results["Y_TRUE_WITHDRWLS_ATM"], results["Y_PRED_WITHDRWLS_ATM"])
    rmse = np.sqrt(mean_squared_error(results["Y_TRUE_WITHDRWLS_ATM"], results["Y_PRED_WITHDRWLS_ATM"]))
    mape = results["APE"].dropna().mean() * 100
    return {"MAE": mae, "RMSE": rmse, "MAPE": mape}


def save_results(results: pd.DataFrame) -> None:
    results.to_excel(OUTPUT_EXCEL, index=False)
    print(f"\nSaved: {OUTPUT_EXCEL}")


def plot_actual_vs_predicted(results: pd.DataFrame) -> None:
    aggregated = (
        results.groupby("FORECAST_DATE")[["Y_TRUE_WITHDRWLS_ATM", "Y_PRED_WITHDRWLS_ATM"]]
        .sum()
        .reset_index()
    )

    fig, ax = plt.subplots(figsize=(13, 4))
    ax.plot(
        aggregated["FORECAST_DATE"],
        aggregated["Y_TRUE_WITHDRWLS_ATM"],
        label="Actual",
        lw=1.5,
        color="#1f77b4",
    )
    ax.plot(
        aggregated["FORECAST_DATE"],
        aggregated["Y_PRED_WITHDRWLS_ATM"],
        label="ANN Predicted",
        lw=1.2,
        color="#ff7f0e",
        alpha=0.85,
    )
    ax.set_title("ANN - Actual vs Predicted (All ATMs, Daily Aggregated)", fontsize=13)
    ax.set_xlabel("Date")
    ax.set_ylabel("Total ATM Withdrawals")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(ACTUAL_PRED_PLOT, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {ACTUAL_PRED_PLOT}")


def plot_residuals(results: pd.DataFrame, metrics: dict[str, float]) -> None:
    residuals = results["Y_TRUE_WITHDRWLS_ATM"] - results["Y_PRED_WITHDRWLS_ATM"]

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.hist(residuals, bins=60, color="#4C72B0", edgecolor="white", alpha=0.85)
    ax.axvline(0, color="red", lw=1.5, ls="--", label="Zero error")
    ax.set_title(
        f"ANN - Residual Distribution  (MAE={metrics['MAE']:,.0f}  RMSE={metrics['RMSE']:,.0f})",
        fontsize=12,
    )
    ax.set_xlabel("Actual - Predicted")
    ax.set_ylabel("Count")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(RESIDUAL_PLOT, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {RESIDUAL_PLOT}")


def plot_weekly_mae(results: pd.DataFrame, overall_mae: float) -> None:
    weekly_mae = (
        results.groupby("WEEK_START")
        .apply(lambda group: mean_absolute_error(group["Y_TRUE_WITHDRWLS_ATM"], group["Y_PRED_WITHDRWLS_ATM"]))
        .reset_index(name="MAE")
    )

    fig, ax = plt.subplots(figsize=(13, 4))
    ax.bar(
        weekly_mae["WEEK_START"],
        weekly_mae["MAE"],
        color="#4C72B0",
        edgecolor="none",
        alpha=0.85,
        width=5,
    )
    ax.axhline(overall_mae, color="red", lw=1.5, ls="--", label=f"Overall MAE = {overall_mae:,.0f}")
    ax.set_title("ANN - MAE per Weekly Window", fontsize=13)
    ax.set_xlabel("Week Start")
    ax.set_ylabel("MAE")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(WEEKLY_MAE_PLOT, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {WEEKLY_MAE_PLOT}")


def print_metrics(metrics: dict[str, float], row_count: int) -> None:
    print(f"\n{'=' * 45}")
    print("  ANN Results  (all windows, all ATMs)")
    print(f"{'=' * 45}")
    print(f"  MAE  : {metrics['MAE']:>12,.2f}")
    print(f"  RMSE : {metrics['RMSE']:>12,.2f}")
    print(f"  MAPE : {metrics['MAPE']:>11.2f}%")
    print(f"  Rows : {row_count:>12,}")
    print(f"{'=' * 45}")


def main() -> None:
    df = load_data(DATA_PATH)
    results = run_weekly_ann_forecast(df)

    if results.empty:
        raise RuntimeError("The ANN forecast output is empty. Please verify the input data and feature columns.")

    metrics = compute_metrics(results)
    print_metrics(metrics, len(results))
    save_results(results)
    plot_actual_vs_predicted(results)
    plot_residuals(results, metrics)
    plot_weekly_mae(results, metrics["MAE"])
    print("\nANN pipeline complete.")


if __name__ == "__main__":
    main()
