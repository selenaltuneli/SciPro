"""Add branch location coordinates to the merged ATM-branch dataset."""

import json
from pathlib import Path

import pandas as pd

BASE_PATH = Path(r"C:\Users\Selen\Desktop\ORBA\Scientific Project\SciPro")

EXCEL_PATH = BASE_PATH / "ATM_Branch_Merged.xlsx"
GEOJSON_PATH = BASE_PATH / "istanbul_ziraat_21.geojson"
OUTPUT_PATH = BASE_PATH / "ATM_Branch_Merged_With_Location.xlsx"

BRANCH_COLUMN = "BRANCH_KEY"
EXPECTED_BRANCH_COUNT = 21


def load_geojson_coordinates(geojson_path: Path) -> list[tuple[float, float]]:
    with geojson_path.open("r", encoding="utf-8") as file:
        geojson_data = json.load(file)

    coordinates: list[tuple[float, float]] = []
    for feature in geojson_data["features"]:
        longitude, latitude = feature["geometry"]["coordinates"]
        coordinates.append((latitude, longitude))

    return coordinates


def build_branch_coordinate_map(branches: list[str], coordinates: list[tuple[float, float]]) -> dict[str, dict[str, float]]:
    branch_to_coordinate: dict[str, dict[str, float]] = {}

    for index, branch in enumerate(branches):
        latitude, longitude = coordinates[index]
        branch_to_coordinate[branch] = {
            "LATITUDE": latitude,
            "LONGITUDE": longitude,
        }

    return branch_to_coordinate


def main() -> None:
    df = pd.read_excel(EXCEL_PATH)
    coordinates = load_geojson_coordinates(GEOJSON_PATH)

    if BRANCH_COLUMN not in df.columns:
        raise KeyError(f"Column '{BRANCH_COLUMN}' was not found. Available columns: {list(df.columns)}")

    unique_branches = sorted(df[BRANCH_COLUMN].dropna().unique())

    if len(unique_branches) != EXPECTED_BRANCH_COUNT:
        raise ValueError(
            f"Expected {EXPECTED_BRANCH_COUNT} unique branches in the Excel file, found {len(unique_branches)}."
        )

    if len(coordinates) != EXPECTED_BRANCH_COUNT:
        raise ValueError(
            f"Expected {EXPECTED_BRANCH_COUNT} coordinates in the GeoJSON file, found {len(coordinates)}."
        )

    # The branch list and the GeoJSON file are assumed to be in the same order.
    branch_to_coordinate = build_branch_coordinate_map(unique_branches, coordinates)

    df["LATITUDE"] = df[BRANCH_COLUMN].map(lambda branch: branch_to_coordinate[branch]["LATITUDE"])
    df["LONGITUDE"] = df[BRANCH_COLUMN].map(lambda branch: branch_to_coordinate[branch]["LONGITUDE"])

    if df["LATITUDE"].isna().any() or df["LONGITUDE"].isna().any():
        raise ValueError("Some rows could not be assigned location coordinates.")

    df.to_excel(OUTPUT_PATH, index=False)

    print("Location data was added successfully.")
    print(f"Output file: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
