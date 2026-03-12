import warnings
warnings.filterwarnings("ignore")

import os
from glob import glob
import numpy as np
import pandas as pd
from pathlib import Path

# Try to help XGBoost find libomp on macOS Homebrew setups.
if "DYLD_LIBRARY_PATH" not in os.environ:
    candidates = sorted(glob("/opt/homebrew/Cellar/llvm/*/lib"))
    if candidates:
        os.environ["DYLD_LIBRARY_PATH"] = candidates[-1]

import xgboost as xgb


BASE_DIR = Path(__file__).resolve().parent
INPUT_PATH = str(BASE_DIR / "ATM_Branch_Data_Final_filled.xlsx")
OUTPUT_PATH = str(BASE_DIR / "scenario_A_weekly_reopt_forecasts_XGBoost.xlsx")

DATE_COL = "DATE"
TARGET_COL = "WITHDRWLS_ATM"


WEEK_FREQ = "W-MON"   # retraining starts every Monday / her Pazartesi yeniden eğitim başlar
HORIZON_DAYS = 7      # 7-day ahead forecast / 7 günlük tahmin ufku

VALIDATION_DAYS = 60  # last 60 days held out for early stopping / erken durdurma için son 60 gün ayrılır
RANDOM_SEED = 42

# log1p(y) transform stabilizes variance in the target / hedef değişkendeki varyansı stabilize eder,
# prevents large values from dominating the model / büyük değerlerin modeli baskılamasını engeller
USE_LOG_TARGET = True

# set None to use all available history / None yapılırsa tüm geçmiş kullanılır
TRAIN_WINDOW_DAYS = 365

# values close to zero produce infinite APE and distort averages /
# 0'a yakın gerçek değerler sonsuz APE üretir ve ortalamaları bozar
APE_MIN_TRUE = 1000


def load_data(path: str) -> pd.DataFrame:
    df = pd.read_excel(path)
    df[DATE_COL] = pd.to_datetime(df[DATE_COL])

    # numeric proxy for linear trend / doğrusal trendi yakalamak için sayısal bir vekil değişken
    # counts days since the earliest date in the dataset / veri setindeki en erken tarihten itibaren gün sayısı
    min_date = df[DATE_COL].min()
    df["TIME_INDEX"] = (df[DATE_COL] - min_date).dt.days
    df["DAY_OF_WEEK"] = df[DATE_COL].dt.dayofweek
    df["IS_WEEKEND"] = (df["DAY_OF_WEEK"] >= 5).astype(int)
    df["DAY_OF_MONTH"] = df[DATE_COL].dt.day
    df["MONTH"] = df[DATE_COL].dt.month
    df["WEEK_OF_YEAR"] = df[DATE_COL].dt.isocalendar().week.astype(int)
    df["IS_MONTH_START"] = df[DATE_COL].dt.is_month_start.astype(int)
    df["IS_MONTH_END"] = df[DATE_COL].dt.is_month_end.astype(int)

    if "CASHP_ID_ATM" in df.columns:
        df = df.sort_values(["CASHP_ID_ATM", DATE_COL]).reset_index(drop=True)
    else:
        df = df.sort_values([DATE_COL]).reset_index(drop=True)

    return df


def weekly_schedule(df: pd.DataFrame) -> pd.DatetimeIndex:
    # all Mondays in the last year of data, clipped to actual date range /
    # verinin son 1 yılındaki tüm Pazartesiler, veri aralığı dışına çıkanlar filtrelenir
    max_d = df[DATE_COL].max()
    start = max_d - pd.Timedelta(days=365)
    end = max_d

    sch = pd.date_range(start=start, end=end, freq=WEEK_FREQ)
    return sch[(sch >= df[DATE_COL].min()) & (sch <= df[DATE_COL].max())]


def feature_cols(df: pd.DataFrame) -> list[str]:
    # everything except date and target is used as a feature /
    # tarih ve hedef dışındaki her sütun özellik olarak kullanılır
    return [c for c in df.columns if c not in {DATE_COL, TARGET_COL}]


def one_hot_with_reference(train_df: pd.DataFrame, other_df: pd.DataFrame, feats: list[str]):
    # XGBoost does not handle categoricals natively, so we one-hot encode /
    # XGBoost kategorik değişkenleri doğrudan işleyemez, bu yüzden one-hot encoding uygulanır

    X_train = pd.get_dummies(train_df[feats], drop_first=False)

    # reindex prediction/validation matrix to match training columns exactly /
    # tahmin matrisini eğitim sütunlarıyla tam eşleştir: eksik sütunlar 0, fazlalar atılır
    X_other = pd.get_dummies(other_df[feats], drop_first=False)
    X_other = X_other.reindex(columns=X_train.columns, fill_value=0)

    return X_train, X_other


def train_xgb_booster(train_df: pd.DataFrame, feats: list[str]) -> tuple[xgb.Booster, pd.Index, float]:
    max_day = train_df[DATE_COL].max()
    valid_start = max_day - pd.Timedelta(days=VALIDATION_DAYS)

    # hold out the last 60 days as validation / son 60 günü validasyon olarak ayır
    tr = train_df[train_df[DATE_COL] < valid_start].copy()
    va = train_df[train_df[DATE_COL] >= valid_start].copy()

    use_valid = (len(tr) > 1000) and (len(va) > 200)

    # build design matrices, enforcing training reference columns /
    # eğitim sütun uzayını referans alarak tasarım matrislerini oluştur
    X_tr, _ = one_hot_with_reference(tr, tr, feats)
    _, X_va = one_hot_with_reference(tr, va, feats)

    y_tr = tr[TARGET_COL].values
    y_va = va[TARGET_COL].values

    if USE_LOG_TARGET:
        y_tr = np.log1p(y_tr)
        y_va = np.log1p(y_va)

    dtrain = xgb.DMatrix(X_tr, label=y_tr, feature_names=list(X_tr.columns))
    dvalid = xgb.DMatrix(X_va, label=y_va, feature_names=list(X_tr.columns))

    params = {
        "objective": "reg:squarederror",
        "eval_metric": "mae",
        "learning_rate": 0.03,
        "max_depth": 6,
        "subsample": 0.9,        # use 90% of rows per iteration to reduce overfitting / aşırı öğrenmeyi azaltmak için her iterasyonda verinin %90'ı
        "colsample_bytree": 0.9, # use 90% of features per iteration / her iterasyonda özelliklerin %90'ı
        "lambda": 1.0,           # L2 regularization / L2 düzenlileştirme
        "seed": RANDOM_SEED,
        "tree_method": "hist",   # faster histogram-based tree building / daha hızlı histogram tabanlı ağaç kurma
    }

    if use_valid:
        # stop early if validation loss does not improve for 200 rounds /
        # validasyon kaybı 200 iterasyon iyileşmezse erken durdur
        booster = xgb.train(
            params=params,
            dtrain=dtrain,
            num_boost_round=5000,
            evals=[(dvalid, "valid")],
            callbacks=[xgb.callback.EarlyStopping(rounds=200, save_best=True)],
            verbose_eval=False
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
        return booster, X_tr.columns, calib
    else:
        # not enough data for early stopping, fit on everything with fewer rounds /
        # erken durdurma için yeterli veri yok, daha az iterasyonla tüm veriyle eğit
        X_all, _ = one_hot_with_reference(train_df, train_df, feats)
        y_all = train_df[TARGET_COL].values
        if USE_LOG_TARGET:
            y_all = np.log1p(y_all)

        dtrain_all = xgb.DMatrix(X_all, label=y_all, feature_names=list(X_all.columns))
        booster = xgb.train(
            params=params,
            dtrain=dtrain_all,
            num_boost_round=1000,
            verbose_eval=False
        )

        return booster, X_all.columns, 1.0


def predict_with_booster(booster: xgb.Booster, dmatrix: xgb.DMatrix) -> np.ndarray:
    best_iter = getattr(booster, "best_iteration", None)
    if best_iter is not None:
        return booster.predict(dmatrix, iteration_range=(0, best_iter + 1))
    return booster.predict(dmatrix)


def compute_metrics(df_forecasts: pd.DataFrame) -> dict:
    y_true = df_forecasts["Y_TRUE_WITHDRWLS_ATM"].to_numpy()
    y_pred = df_forecasts["Y_PRED_WITHDRWLS_ATM"].to_numpy()
    abs_err = np.abs(y_true - y_pred)

    if APE_MIN_TRUE > 0:
        mask = (y_true >= APE_MIN_TRUE)
    else:
        mask = (y_true > 0)

    ape = np.full_like(y_true, np.nan, dtype=float)
    ape[mask] = abs_err[mask] / y_true[mask]

    mae = float(np.mean(abs_err))
    mean_ape = float(np.nanmean(ape))
    median_ape = float(np.nanmedian(ape))

    # unlike classic MAPE, weights each day by its volume /
    # klasik MAPE'den farkı: her günü eşit değil hacmiyle orantılı ağırlar
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
        "R2": r2
    }


def run_scenario_A(df: pd.DataFrame) -> pd.DataFrame:
    feats = feature_cols(df)
    sch = weekly_schedule(df)

    outputs = []

    for week_start in sch:
        train_end = week_start - pd.Timedelta(days=1)
        forecast_end = week_start + pd.Timedelta(days=HORIZON_DAYS - 1)

        # past-only training data, no leakage / sadece geçmiş veri, veri sızıntısı yok
        train_df = df[df[DATE_COL] <= train_end].copy()

        # rolling window: prevents the model from learning on very old data (concept drift) /
        # kayan pencere: modelin çok eski veriden öğrenmesini engeller (konsept kayması)
        if TRAIN_WINDOW_DAYS is not None:
            window_start = train_end - pd.Timedelta(days=TRAIN_WINDOW_DAYS - 1)
            train_df = train_df[train_df[DATE_COL] >= window_start].copy()

        # skip this week if there is not enough training data /
        # yeterli eğitim verisi yoksa bu haftayı atla
        if len(train_df) < 2000:
            continue

        pred_df = df[(df[DATE_COL] >= week_start) & (df[DATE_COL] <= forecast_end)].copy()
        if pred_df.empty:
            continue

        booster, ref_cols, calib = train_xgb_booster(train_df, feats)

        # prediction matrix must match the training feature space exactly /
        # tahmin matrisi eğitim sütun uzayıyla tam eşleşmeli
        X_pred = pd.get_dummies(pred_df[feats], drop_first=False)
        X_pred = X_pred.reindex(columns=ref_cols, fill_value=0)

        dpred = xgb.DMatrix(X_pred, feature_names=list(ref_cols))
        y_pred = predict_with_booster(booster, dpred)

        if USE_LOG_TARGET:
            y_pred = np.expm1(y_pred)     # inverse of log1p / log1p'nin tersi

        y_pred = y_pred * calib
        y_pred = np.clip(y_pred, 0, None) # withdrawals cannot be negative / para çekimi negatif olamaz

        tmp = pd.DataFrame({
            "WEEK_START": week_start,
            "TRAIN_END": train_end,
            "FORECAST_DATE": pred_df[DATE_COL].values,
            "CASHP_ID_ATM": pred_df["CASHP_ID_ATM"].astype(str).values if "CASHP_ID_ATM" in pred_df.columns else None,
            "Y_PRED_WITHDRWLS_ATM": y_pred,
            "Y_TRUE_WITHDRWLS_ATM": pred_df[TARGET_COL].values,
        })

        tmp["ABS_ERROR"] = np.abs(tmp["Y_TRUE_WITHDRWLS_ATM"] - tmp["Y_PRED_WITHDRWLS_ATM"])

        if APE_MIN_TRUE > 0:
            m = tmp["Y_TRUE_WITHDRWLS_ATM"] >= APE_MIN_TRUE
        else:
            m = tmp["Y_TRUE_WITHDRWLS_ATM"] > 0

        tmp["APE"] = np.nan
        tmp.loc[m, "APE"] = tmp.loc[m, "ABS_ERROR"] / tmp.loc[m, "Y_TRUE_WITHDRWLS_ATM"]

        outputs.append(tmp)
        print(f"[OK] {week_start.date()} | rows={len(tmp)} | calib={calib:.3f}")

    if not outputs:
        return pd.DataFrame()

    return pd.concat(outputs, ignore_index=True)


def save_excel(df_forecasts: pd.DataFrame, path: str) -> None:
    metrics = compute_metrics(df_forecasts)

    weekly = (
        df_forecasts
        .groupby("WEEK_START")
        .agg(
            n=("Y_TRUE_WITHDRWLS_ATM", "size"),
            mae=("ABS_ERROR", "mean"),
            mean_ape=("APE", "mean"),
            median_ape=("APE", "median"),
            bias=("Y_PRED_WITHDRWLS_ATM", lambda s: float(np.mean(s - df_forecasts.loc[s.index, "Y_TRUE_WITHDRWLS_ATM"])))
        )
        .reset_index()
    )

    metrics_df = pd.DataFrame([metrics])

    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        df_forecasts.to_excel(writer, index=False, sheet_name="Forecasts")
        weekly.to_excel(writer, index=False, sheet_name="Weekly_Summary")
        metrics_df.to_excel(writer, index=False, sheet_name="Overall_Metrics")


def main():
    df = load_data(INPUT_PATH)
    forecasts = run_scenario_A(df)

    if forecasts.empty:
        raise RuntimeError("The output is empty. Please verify date coverage and column names.")

    save_excel(forecasts, OUTPUT_PATH)

    m = compute_metrics(forecasts)
    print("\nOverall Metrics (XGBoost)")
    print(f"MAE           : {m['MAE']:.2f}")
    print(f"Mean APE      : {m['Mean_APE']*100:.2f}%")
    print(f"Median APE    : {m['Median_APE']*100:.2f}%")
    print(f"Weighted MAPE : {m['Weighted_MAPE']*100:.2f}%")
    print(f"R2            : {m['R2']:.4f}")
    print(f"\nSaved: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
