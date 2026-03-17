"""Create location-based features such as postcode and functional zone."""

import hashlib
import json
import time
from pathlib import Path

import pandas as pd
import requests

BASE_PATH = Path(r"C:\Users\Selen\Desktop\ORBA\Scientific Project\SciPro")
INPUT_FILE = BASE_PATH / "ATM_Branch_Merged_With_Location_HolidayFeatures_FINAL.xlsx"
OUTPUT_FILE = BASE_PATH / "ATM_Branch_With_FunctionalZone_Postcode.xlsx"

CACHE_DIR = BASE_PATH / "_geo_cache"
CACHE_DIR.mkdir(exist_ok=True)

HEADERS = {"User-Agent": "SciPro-Istanbul-ATM/1.0"}
RADIUS_M = 1000

TOURISTIC_SCORE_THRESHOLD = 15
CBD_OFFICE_THRESHOLD = 10
CBD_BANK_THRESHOLD = 5


def build_cache_key(*parts: object) -> str:
    joined_text = "|".join(map(str, parts))
    return hashlib.md5(joined_text.encode("utf-8")).hexdigest()


def load_cached_response(cache_key: str):
    cache_file = CACHE_DIR / f"{cache_key}.json"
    if not cache_file.exists():
        return None

    try:
        return json.loads(cache_file.read_text(encoding="utf-8"))
    except Exception:
        return None


def save_cached_response(cache_key: str, value: dict) -> None:
    cache_file = CACHE_DIR / f"{cache_key}.json"
    cache_file.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")


def get_json_with_retry(url: str, params=None, sleep_seconds: float = 1.0, retries: int = 3):
    for attempt in range(retries):
        try:
            response = requests.get(url, params=params, headers=HEADERS, timeout=60)
            if response.status_code == 429:
                time.sleep(2.0 + attempt)
                continue

            response.raise_for_status()
            time.sleep(sleep_seconds)
            return response.json()
        except Exception:
            if attempt == retries - 1:
                return None
            time.sleep(1.0 + attempt)

    return None


def post_overpass_query(query_text: str, sleep_seconds: float = 1.0, retries: int = 3):
    url = "https://overpass-api.de/api/interpreter"

    for attempt in range(retries):
        try:
            response = requests.post(url, data=query_text.encode("utf-8"), headers=HEADERS, timeout=120)
            if response.status_code == 429:
                time.sleep(2.0 + attempt)
                continue

            response.raise_for_status()
            time.sleep(sleep_seconds)
            return response.json()
        except Exception:
            if attempt == retries - 1:
                return None
            time.sleep(1.0 + attempt)

    return None


def find_lat_lon_columns(df: pd.DataFrame):
    lower_case_columns = {column.lower(): column for column in df.columns}

    lat_candidates = ["latitude", "lat", "atm_latitude", "branch_latitude"]
    lon_candidates = ["longitude", "lon", "lng", "atm_longitude", "branch_longitude"]

    lat_column = next((lower_case_columns[name] for name in lat_candidates if name in lower_case_columns), None)
    lon_column = next((lower_case_columns[name] for name in lon_candidates if name in lower_case_columns), None)

    if lat_column is None:
        lat_column = next((column for column in df.columns if column.lower().startswith("lat")), None)
    if lon_column is None:
        lon_column = next((column for column in df.columns if column.lower().startswith(("lon", "lng"))), None)

    return lat_column, lon_column


def reverse_geocode_postcode(latitude: float, longitude: float) -> dict[str, object]:
    cache_key = build_cache_key("rev_postcode", round(latitude, 6), round(longitude, 6))
    cached_value = load_cached_response(cache_key)
    if cached_value is not None:
        return cached_value

    url = "https://nominatim.openstreetmap.org/reverse"
    params = {
        "format": "jsonv2",
        "lat": latitude,
        "lon": longitude,
        "zoom": 18,
        "addressdetails": 1,
    }

    response_json = get_json_with_retry(url, params=params, sleep_seconds=1.0)
    postcode = None

    if response_json and isinstance(response_json, dict):
        address = response_json.get("address", {}) or {}
        postcode = address.get("postcode")

    result = {"postcode": postcode}
    save_cached_response(cache_key, result)
    return result


def get_overpass_counts(latitude: float, longitude: float, radius_m: int) -> dict[str, int]:
    cache_key = build_cache_key("poi_counts", round(latitude, 6), round(longitude, 6), radius_m)
    cached_value = load_cached_response(cache_key)
    if cached_value is not None:
        return cached_value

    query = f"""
    [out:json][timeout:60];
    (
      node(around:{radius_m},{latitude},{longitude})[tourism];
      way(around:{radius_m},{latitude},{longitude})[tourism];
      rel(around:{radius_m},{latitude},{longitude})[tourism];

      node(around:{radius_m},{latitude},{longitude})[historic];
      way(around:{radius_m},{latitude},{longitude})[historic];
      rel(around:{radius_m},{latitude},{longitude})[historic];

      node(around:{radius_m},{latitude},{longitude})[amenity="bank"];
      way(around:{radius_m},{latitude},{longitude})[amenity="bank"];
      rel(around:{radius_m},{latitude},{longitude})[amenity="bank"];

      node(around:{radius_m},{latitude},{longitude})[office];
      way(around:{radius_m},{latitude},{longitude})[office];
      rel(around:{radius_m},{latitude},{longitude})[office];
    );
    out tags;
    """

    response_json = post_overpass_query(query, sleep_seconds=1.0)
    if not response_json:
        empty_result = {"tourism": 0, "historic": 0, "bank": 0, "office": 0}
        save_cached_response(cache_key, empty_result)
        return empty_result

    tourism_count = 0
    historic_count = 0
    bank_count = 0
    office_count = 0

    for element in response_json.get("elements", []) or []:
        tags = element.get("tags", {}) or {}
        if "tourism" in tags:
            tourism_count += 1
        if "historic" in tags:
            historic_count += 1
        if tags.get("amenity") == "bank":
            bank_count += 1
        if "office" in tags:
            office_count += 1

    result = {
        "tourism": tourism_count,
        "historic": historic_count,
        "bank": bank_count,
        "office": office_count,
    }
    save_cached_response(cache_key, result)
    return result


def compute_location_features(latitude: float, longitude: float) -> dict[str, object]:
    postcode_info = reverse_geocode_postcode(latitude, longitude)
    poi_counts = get_overpass_counts(latitude, longitude, RADIUS_M)

    tourist_score = 2 * poi_counts["tourism"] + 2 * poi_counts["historic"]

    if tourist_score >= TOURISTIC_SCORE_THRESHOLD:
        functional_zone = 2
    elif poi_counts["office"] >= CBD_OFFICE_THRESHOLD or poi_counts["bank"] >= CBD_BANK_THRESHOLD:
        functional_zone = 1
    else:
        functional_zone = 0

    return {
        "postcode": postcode_info.get("postcode"),
        "functional_zone": functional_zone,
    }


def build_location_feature_table(df: pd.DataFrame, lat_column: str, lon_column: str) -> pd.DataFrame:
    unique_coordinates = (
        df[[lat_column, lon_column]]
        .dropna()
        .drop_duplicates()
        .values
        .tolist()
    )

    rows = []
    for latitude, longitude in unique_coordinates:
        try:
            features = compute_location_features(float(latitude), float(longitude))
        except Exception:
            features = {"postcode": None, "functional_zone": None}

        rows.append({lat_column: latitude, lon_column: longitude, **features})

    return pd.DataFrame(rows)


def main() -> None:
    df = pd.read_excel(INPUT_FILE)

    lat_column, lon_column = find_lat_lon_columns(df)
    if not lat_column or not lon_column:
        raise ValueError(f"Latitude and longitude columns were not found. Columns: {list(df.columns)}")

    location_features = build_location_feature_table(df, lat_column, lon_column)
    final_df = df.merge(location_features, on=[lat_column, lon_column], how="left")

    final_df.to_excel(OUTPUT_FILE, index=False)
    print("Location features were created successfully.")
    print(f"Output file: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
