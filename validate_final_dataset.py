import pandas as pd

# -------------------------------
# 0) Final dosyayı yükle
# -------------------------------
path = r"C:\Users\Selen\Desktop\ORBA\Scientific Project\SciPro\ATM_Branch_Holiday_Final.xlsx"
df = pd.read_excel(path)

print("\n=== İlk 5 Satır ===")
print(df.head())

print("\n=== Info ===")
print(df.info())


# ===========================================================
# 1) TEST – ATM_ID → BRANCH_NUM eşleşmesi doğru mu?
# ===========================================================
df["TEST_BRANCH_NUM"] = df["ATM_ID"].astype(str).str[1:5].astype(int)

wrong_branch = df[df["TEST_BRANCH_NUM"] != df["BRANCH_NUM"]]

print("\n=== TEST 1: Yanlış BRANCH_NUM eşleşmeleri ===")
print("Hata sayısı:", len(wrong_branch))
if len(wrong_branch) > 0:
    print(wrong_branch.head())


# ===========================================================
# 2) TEST – Lokasyonlar eksik mi? (tek lokasyon kolonları)
# ===========================================================
missing_loc = df[df["LATITUDE"].isna() | df["LONGITUDE"].isna()]

print("\n=== TEST 2: Eksik Lat/Lon ===")
print("Eksik lokasyon satırı:", len(missing_loc))
if len(missing_loc) > 0:
    print(missing_loc.head())


# ===========================================================
# 3) TEST – IS_WEEKDAY doğru hesaplanmış mı?
# ===========================================================
df["DATE_DT"] = pd.to_datetime(df["DATE"])
df["CALC_IS_WEEKDAY"] = df["DATE_DT"].dt.weekday.apply(lambda x: 1 if x < 5 else 0)

wrong_day = df[df["CALC_IS_WEEKDAY"] != df["IS_WEEKDAY"]]

print("\n=== TEST 3: IS_WEEKDAY yanlış olan satırlar ===")
print("Hata sayısı:", len(wrong_day))
if len(wrong_day) > 0:
    print(wrong_day.head())


# ===========================================================
# 4) TEST – Hafta sonlarında branch daily values boş olmalı
# ===========================================================
weekend = df[df["IS_WEEKDAY"] == 0]
wrong_weekend = weekend[weekend["BRANCH_WITHDRWLS"].notna()]

print("\n=== TEST 4: Hafta sonu için hata (branch value dolu olmamalı) ===")
print("Hata sayısı:", len(wrong_weekend))
if len(wrong_weekend) > 0:
    print(wrong_weekend.head())


# ===========================================================
# 5) TEST – Hafta içi branch datası boş kalmış mı? (opsiyonel)
# ===========================================================
weekday = df[df["IS_WEEKDAY"] == 1]
empty_weekday = weekday[weekday["BRANCH_WITHDRWLS"].isna()]

print("\n=== TEST 5: Hafta içi branch datası eksik satır ===")
print("Eksik satır sayısı:", len(empty_weekday))
if len(empty_weekday) > 0:
    print(empty_weekday.head())


# ===========================================================
# 6) TEST – Tatil günlerini göster
# ===========================================================
holidays = df[df["IS_HOLIDAY"] == 1]

print("\n=== TEST 6: Tatil günleri ===")
print("Tatil satır sayısı:", len(holidays))
if len(holidays) > 0:
    print(holidays.head())


# ===========================================================
# 7) TEST – Veri boyutu kontrolü
# ===========================================================
num_atms = df["ATM_ID"].nunique()
num_days = df["DATE"].nunique()

print("\n=== TEST 7: Veri boyutu kontrolü ===")
print("Toplam ATM:", num_atms)
print("Toplam gün:", num_days)
print("Beklenen satır (ATM * Gün):", num_atms * num_days)
print("Gerçek satır:", len(df))

# Temizlik
df = df.drop(columns=["DATE_DT", "CALC_IS_WEEKDAY", "TEST_BRANCH_NUM"], errors="ignore")

print("\n=== TÜM TESTLER ÇALIŞTI ===")
