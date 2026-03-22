"""Train an improved weekly SARIMAX forecasting pipeline for ATM withdrawals."""

import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score
from statsmodels.tsa.statespace.sarimax import SARIMAX

warnings.filterwarnings("ignore")

DATA_PATH = Path("ATM_Branch_Data_Final_filled.xlsx")
OUTPUT_EXCEL = Path("SARIMAX_improved_pred.xlsx")
ACTUAL_PREDICTED_PLOT = Path("sarimax_v2_actual_vs_predicted.png")
WEEKLY_MAE_PLOT = Path("sarimax_v2_weekly_mae.png")
RESIDUALS_PLOT = Path("sarimax_v2_residuals.png")
ATM_R2_PLOT = Path("sarimax_v2_atm_r2_dist.png")

DATE_COL = "DATE"
ATM_COL = "CASHP_ID_ATM"
TARGET_COL = "WITHDRWLS_ATM"

EXOG_COLUMNS = [
    "IS_HOLIDAY",
    "IS_RELIGIOUS_HOLIDAY",
    "IS_NATIONAL_HOLIDAY",
    "IS_PUBLIC_HOLIDAY",
    "IS_SCHOOL_HOLIDAY",
    "IS_PRE_HOLIDAY_1",
    "IS_POST_HOLIDAY_1",
    "HOLIDAY_DURATION_SCORE",
    "HOLIDAY_IMPORTANCE_SCORE",
    "IS_BRANCH_OPEN",
    "IS_MONTH_START",
    "IS_MONTH_END",
    "DAY_OF_WEEK",
    "IS_WEEKDAY",
]

# Several order combinations are tested to reduce phase shift and pick a more stable specification.
CANDIDATE_ORDERS = [
    ((1, 1, 1), (1, 1, 1, 7)),
    ((1, 0, 1), (1, 1, 1, 7)),
    ((1, 1, 1), (1, 0, 1, 7)),
    ((2, 1, 1), (1, 1, 1, 7)),
]

WEEK_FREQ = "W-MON"
HORIZON_DAYS = 7
MIN_TRAIN_ROWS = 21


def load_data(path: Path) -> pd.DataFrame:
    print("Loading data ...")
    df = pd.read_excel(path, parse_dates=[DATE_COL])
    df = df.sort_values([ATM_COL, DATE_COL]).reset_index(drop=True)

    # Rows without lag information are excluded because the comparison setup expects lag-ready observations.
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
    print("\nImproved SARIMAX pipeline is starting ...\n")
    return schedule


def winsorize_series(series: pd.Series, k: float = 3.0) -> pd.Series:
    # Large spikes are clipped with an IQR rule so the model does not overfit outliers.
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    upper_bound = q3 + k * iqr
    return series.clip(upper=upper_bound)


def select_best_order(
    y_train: pd.Series,
    exog_train: pd.DataFrame,
    candidates: list[tuple[tuple[int, int, int], tuple[int, int, int, int]]],
) -> tuple[tuple[int, int, int], tuple[int, int, int, int]]:
    best_aic = np.inf
    best_order, best_seasonal_order = candidates[0]

    for order, seasonal_order in candidates:
        try:
            model = SARIMAX(
                y_train,
                exog=exog_train,
                order=order,
                seasonal_order=seasonal_order,
                enforce_stationarity=False,
                enforce_invertibility=False,
                freq="D",
            )
            result = model.fit(disp=False, maxiter=100)
            if result.aic < best_aic:
                best_aic = result.aic
                best_order = order
                best_seasonal_order = seasonal_order
        except Exception:
            continue

    return best_order, best_seasonal_order


def seasonal_naive_median(y_train: pd.Series, steps: int, period: int = 7) -> np.ndarray:
    # The fallback forecast uses the median of the same weekday from the last four weeks.
    predictions = []
    for step_index in range(steps):
        day_offset = -(period - (step_index % period))
        candidate_values = []
        for week_index in range(1, 5):
            lookup_index = day_offset - (week_index - 1) * period
            if abs(lookup_index) <= len(y_train):
                candidate_values.append(float(y_train.iloc[lookup_index]))
        value = float(np.median(candidate_values)) if candidate_values else 0.0
        predictions.append(max(value, 0.0))
    return np.array(predictions)


def apply_holiday_correction(
    predictions: np.ndarray,
    test_df: pd.DataFrame,
    raw_train_series: pd.Series,
) -> np.ndarray:
    # Weekend days are forced to zero, and holiday dates are blended with historical holiday behavior.
    corrected_predictions = predictions.copy()

    for index, date_value in enumerate(test_df.index):
        if "IS_WEEKDAY" in test_df.columns and test_df.loc[date_value, "IS_WEEKDAY"] == 0:
            corrected_predictions[index] = 0.0
            continue

        if "IS_HOLIDAY" in test_df.columns and test_df.loc[date_value, "IS_HOLIDAY"] == 1:
            holiday_values = []
            for year_offset in range(1, 5):
                target_date = date_value - pd.DateOffset(years=year_offset)
                window = raw_train_series[
                    (raw_train_series.index >= target_date - pd.Timedelta(days=3))
                    & (raw_train_series.index <= target_date + pd.Timedelta(days=3))
                ]
                if len(window) > 0:
                    holiday_values.append(window.mean())

            if holiday_values:
                historical_mean = float(np.mean(holiday_values))
                corrected_predictions[index] = 0.4 * predictions[index] + 0.6 * historical_mean

    return corrected_predictions


def postprocess_clip(predictions: np.ndarray, raw_train_series: pd.Series, multiplier: float = 3.0) -> np.ndarray:
    # Post-processing prevents implausibly large spikes relative to the recent level.
    if len(raw_train_series) < 30:
        return predictions
    recent_mean = float(raw_train_series.iloc[-30:].mean())
    cap = recent_mean * multiplier
    return np.clip(predictions, 0, cap)


def forecast_one_atm(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> np.ndarray:
    raw_train_series = train_df[TARGET_COL]
    clean_train_series = winsorize_series(raw_train_series)
    exog_train = train_df[EXOG_COLUMNS]
    exog_test = test_df[EXOG_COLUMNS]

    try:
        best_order, best_seasonal_order = select_best_order(clean_train_series, exog_train, CANDIDATE_ORDERS)
        model = SARIMAX(
            clean_train_series,
            exog=exog_train,
            order=best_order,
            seasonal_order=best_seasonal_order,
            enforce_stationarity=False,
            enforce_invertibility=False,
            freq="D",
        )
        fitted_model = model.fit(disp=False, maxiter=200)
        predictions = fitted_model.forecast(steps=len(test_df), exog=exog_test).values
        predictions = np.maximum(predictions, 0)
    except Exception:
        predictions = seasonal_naive_median(clean_train_series, len(test_df))

    predictions = apply_holiday_correction(predictions, test_df, raw_train_series)
    predictions = postprocess_clip(predictions, raw_train_series)
    return predictions


def build_forecast_rows(
    week_start: pd.Timestamp,
    train_end: pd.Timestamp,
    atm_id: str,
    test_df: pd.DataFrame,
    predictions: np.ndarray,
) -> list[dict]:
    rows = []
    y_true = test_df[TARGET_COL].values

    for index, date_value in enumerate(test_df.index):
        abs_error = abs(predictions[index] - y_true[index])
        ape = abs_error / y_true[index] if y_true[index] != 0 else np.nan
        rows.append(
            {
                "WEEK_START": week_start,
                "TRAIN_END": train_end,
                "FORECAST_DATE": date_value,
                ATM_COL: atm_id,
                "Y_PRED_WITHDRWLS_ATM": predictions[index],
                "Y_TRUE_WITHDRWLS_ATM": y_true[index],
                "ABS_ERROR": abs_error,
                "APE": ape,
            }
        )

    return rows


def run_sarimax_pipeline(df: pd.DataFrame) -> pd.DataFrame:
    weekly_schedule = build_weekly_schedule(df)
    atm_ids = df[ATM_COL].unique()
    all_rows: list[dict] = []

    for window_index, week_start in enumerate(weekly_schedule):
        train_end = week_start - pd.Timedelta(days=1)
        week_end = week_start + pd.Timedelta(days=HORIZON_DAYS - 1)
        window_rows: list[dict] = []

        for atm_id in atm_ids:
            atm_df = df[df[ATM_COL] == atm_id].set_index(DATE_COL).sort_index()
            train_df = atm_df[atm_df.index <= train_end]
            test_df = atm_df[(atm_df.index >= week_start) & (atm_df.index <= week_end)]

            if len(train_df) < MIN_TRAIN_ROWS or test_df.empty:
                continue

            predictions = forecast_one_atm(train_df, test_df)
            window_rows.extend(build_forecast_rows(week_start, train_end, atm_id, test_df, predictions))

        all_rows.extend(window_rows)

        if window_rows:
            window_df = pd.DataFrame(window_rows)
            window_mae = mean_absolute_error(
                window_df["Y_TRUE_WITHDRWLS_ATM"],
                window_df["Y_PRED_WITHDRWLS_ATM"],
            )
            actual_week_end = min(week_end, df[DATE_COL].max())
            print(
                f"  Window {window_index + 1:02d}/{len(weekly_schedule)}  "
                f"{week_start.date()} -> {actual_week_end.date()}  MAE = {window_mae:>10,.0f}"
            )

    forecasts = pd.DataFrame(all_rows)
    forecasts["WEEK_START"] = pd.to_datetime(forecasts["WEEK_START"])
    forecasts["TRAIN_END"] = pd.to_datetime(forecasts["TRAIN_END"])
    forecasts["FORECAST_DATE"] = pd.to_datetime(forecasts["FORECAST_DATE"])
    return forecasts


def build_weekly_summary(forecasts: pd.DataFrame) -> pd.DataFrame:
    return (
        forecasts.groupby("WEEK_START")
        .agg(
            n=("FORECAST_DATE", "count"),
            mae=("ABS_ERROR", "mean"),
            mean_ape=("APE", "mean"),
            median_ape=("APE", "median"),
        )
        .reset_index()
    )


def safe_r2(group: pd.DataFrame) -> float:
    if len(group) < 5:
        return np.nan
    try:
        return r2_score(group["Y_TRUE_WITHDRWLS_ATM"], group["Y_PRED_WITHDRWLS_ATM"])
    except Exception:
        return np.nan


def build_atm_metrics(forecasts: pd.DataFrame) -> pd.DataFrame:
    return (
        forecasts.groupby(ATM_COL)
        .apply(
            lambda group: pd.Series(
                {
                    "n": len(group),
                    "MAE": mean_absolute_error(
                        group["Y_TRUE_WITHDRWLS_ATM"],
                        group["Y_PRED_WITHDRWLS_ATM"],
                    ),
                    "Median_APE": group["APE"].median(),
                    "Weighted_MAPE": group["ABS_ERROR"].sum() / group["Y_TRUE_WITHDRWLS_ATM"].sum(),
                    "R2": safe_r2(group),
                }
            )
        )
        .reset_index()
    )


def build_overall_metrics(forecasts: pd.DataFrame) -> dict[str, float]:
    overall_mae = mean_absolute_error(
        forecasts["Y_TRUE_WITHDRWLS_ATM"],
        forecasts["Y_PRED_WITHDRWLS_ATM"],
    )
    overall_mean_ape = forecasts["APE"].dropna().mean()
    overall_median_ape = forecasts["APE"].dropna().median()
    weighted_mape = forecasts["ABS_ERROR"].sum() / forecasts["Y_TRUE_WITHDRWLS_ATM"].sum()
    overall_r2 = r2_score(
        forecasts["Y_TRUE_WITHDRWLS_ATM"],
        forecasts["Y_PRED_WITHDRWLS_ATM"],
    )

    return {
        "MAE": overall_mae,
        "Mean_APE": overall_mean_ape,
        "Median_APE": overall_median_ape,
        "Weighted_MAPE": weighted_mape,
        "R2": overall_r2,
    }


def save_excel(
    forecasts: pd.DataFrame,
    weekly_summary: pd.DataFrame,
    overall_metrics: dict[str, float],
    atm_metrics: pd.DataFrame,
) -> None:
    overall_metrics_df = pd.DataFrame([overall_metrics])

    with pd.ExcelWriter(OUTPUT_EXCEL, engine="openpyxl") as writer:
        forecasts.to_excel(writer, sheet_name="Forecasts", index=False)
        weekly_summary.to_excel(writer, sheet_name="Weekly_Summary", index=False)
        overall_metrics_df.to_excel(writer, sheet_name="Overall_Metrics", index=False)
        atm_metrics.to_excel(writer, sheet_name="ATM_Metrics", index=False)

    print(f"\nSaved: {OUTPUT_EXCEL}")
    print("  Sheet 1 -> Forecasts")
    print("  Sheet 2 -> Weekly_Summary")
    print("  Sheet 3 -> Overall_Metrics")
    print("  Sheet 4 -> ATM_Metrics")


def plot_actual_vs_predicted(forecasts: pd.DataFrame) -> None:
    aggregated = (
        forecasts.groupby("FORECAST_DATE")[["Y_TRUE_WITHDRWLS_ATM", "Y_PRED_WITHDRWLS_ATM"]]
        .sum()
        .reset_index()
    )

    fig, ax = plt.subplots(figsize=(13, 4))
    ax.plot(aggregated["FORECAST_DATE"], aggregated["Y_TRUE_WITHDRWLS_ATM"], label="Actual", lw=1.5, color="#1f77b4")
    ax.plot(
        aggregated["FORECAST_DATE"],
        aggregated["Y_PRED_WITHDRWLS_ATM"],
        label="SARIMAX Predicted",
        lw=1.2,
        color="#d62728",
        alpha=0.85,
    )
    ax.set_title("SARIMAX - Actual vs Predicted (All ATMs, Daily Aggregated)", fontsize=13)
    ax.set_xlabel("Date")
    ax.set_ylabel("Total ATM Withdrawals")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(ACTUAL_PREDICTED_PLOT, dpi=150, bbox_inches="tight")
    plt.close()


def plot_weekly_mae(weekly_summary: pd.DataFrame, overall_mae: float) -> None:
    fig, ax = plt.subplots(figsize=(13, 4))
    ax.bar(weekly_summary["WEEK_START"], weekly_summary["mae"], color="#d62728", edgecolor="none", alpha=0.85, width=5)
    ax.axhline(overall_mae, color="navy", lw=1.5, ls="--", label=f"Overall MAE = {overall_mae:,.0f}")
    ax.set_title("SARIMAX - Weekly MAE", fontsize=13)
    ax.set_xlabel("Week Start")
    ax.set_ylabel("MAE")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(WEEKLY_MAE_PLOT, dpi=150, bbox_inches="tight")
    plt.close()


def plot_residuals(forecasts: pd.DataFrame, overall_mae: float, overall_r2: float) -> None:
    residuals = forecasts["Y_TRUE_WITHDRWLS_ATM"] - forecasts["Y_PRED_WITHDRWLS_ATM"]

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.hist(residuals, bins=60, color="#d62728", edgecolor="white", alpha=0.85)
    ax.axvline(0, color="navy", lw=1.5, ls="--")
    ax.set_title(
        f"SARIMAX - Residual Distribution  (MAE={overall_mae:,.0f}  R2={overall_r2:.4f})",
        fontsize=12,
    )
    ax.set_xlabel("Actual - Predicted")
    ax.set_ylabel("Count")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(RESIDUALS_PLOT, dpi=150, bbox_inches="tight")
    plt.close()


def plot_atm_r2_distribution(atm_metrics: pd.DataFrame) -> None:
    r2_values = atm_metrics["R2"].dropna()

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.hist(r2_values, bins=30, color="#2ca02c", edgecolor="white", alpha=0.85)
    ax.axvline(0, color="red", lw=1.5, ls="--", label="R2 = 0 threshold")
    ax.axvline(r2_values.mean(), color="navy", lw=1.5, ls="-.", label=f"Mean R2 = {r2_values.mean():.3f}")
    ax.set_title("SARIMAX - ATM-level R2 Distribution", fontsize=13)
    ax.set_xlabel("R2")
    ax.set_ylabel("ATM Count")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(ATM_R2_PLOT, dpi=150, bbox_inches="tight")
    plt.close()


def print_metrics(overall_metrics: dict[str, float], row_count: int, atm_metrics: pd.DataFrame) -> None:
    negative_r2_count = int((atm_metrics["R2"] < 0).sum())

    print(f"\n{'=' * 55}")
    print("  SARIMAX Results")
    print(f"{'=' * 55}")
    print(f"  MAE           : {overall_metrics['MAE']:>12,.2f}")
    print(f"  Mean APE      : {overall_metrics['Mean_APE']:>12.4f}")
    print(f"  Median APE    : {overall_metrics['Median_APE']:>12.4f}")
    print(f"  Weighted MAPE : {overall_metrics['Weighted_MAPE']:>12.4f}")
    print(f"  R2            : {overall_metrics['R2']:>12.6f}")
    print(f"  Total rows    : {row_count:>12,}")
    print(f"{'=' * 55}")
    print(f"\n  ATMs with negative R2 : {negative_r2_count} / {len(atm_metrics)}")
    print(f"  Mean ATM-level R2     : {atm_metrics['R2'].mean():.4f}")


def main() -> None:
    df = load_data(DATA_PATH)
    forecasts = run_sarimax_pipeline(df)

    if forecasts.empty:
        raise RuntimeError("The SARIMAX forecast output is empty. Please verify the input data and feature columns.")

    weekly_summary = build_weekly_summary(forecasts)
    overall_metrics = build_overall_metrics(forecasts)
    atm_metrics = build_atm_metrics(forecasts)

    print_metrics(overall_metrics, len(forecasts), atm_metrics)
    save_excel(forecasts, weekly_summary, overall_metrics, atm_metrics)
    plot_actual_vs_predicted(forecasts)
    plot_weekly_mae(weekly_summary, overall_metrics["MAE"])
    plot_residuals(forecasts, overall_metrics["MAE"], overall_metrics["R2"])
    plot_atm_r2_distribution(atm_metrics)

    print("\nSaved plots:")
    print(f"  {ACTUAL_PREDICTED_PLOT.name}")
    print(f"  {WEEKLY_MAE_PLOT.name}")
    print(f"  {RESIDUALS_PLOT.name}")
    print(f"  {ATM_R2_PLOT.name}")
    print("\nSARIMAX pipeline complete.")


if __name__ == "__main__":
    main()
