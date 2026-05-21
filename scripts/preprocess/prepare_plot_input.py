from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import yaml


REQUIRED_CONFIG_COLUMNS = [
    "plot_id",
    "year",
    "latitude",
    "longitude",
    "plot_area_ha",
]


def load_config(config_path: Path) -> dict:
    """Load YAML config file."""
    with config_path.open("r") as f:
        return yaml.safe_load(f)


def validate_columns(df: pd.DataFrame, column_map: dict) -> None:
    """Check that all source columns listed in the config exist in the raw CSV."""
    missing = []

    for key in REQUIRED_CONFIG_COLUMNS:
        source_column = column_map.get(key)

        if source_column is None:
            missing.append(f"{key} missing from config")
            continue

        if source_column not in df.columns:
            missing.append(source_column)

    if missing:
        raise ValueError(f"Missing required input columns: {missing}")


def prepare_input(config_path: Path) -> None:
    """Prepare raw G2F phenotype data into site-year rows for AlphaEarth extraction."""
    config = load_config(config_path)

    raw_path = Path(config["input"]["raw_csv_path"])
    processed_path = Path(config["input"]["processed_csv_path"])
    processed_path.parent.mkdir(parents=True, exist_ok=True)

    column_map = config["columns"]

    df = pd.read_csv(raw_path)

    print(f"Read {len(df)} rows from {raw_path}")

    validate_columns(df, column_map)

    # ------------------------------------------------------------------
    # 1. Standardize raw columns into pipeline column names.
    # ------------------------------------------------------------------
    out = pd.DataFrame()
    out["plot_id"] = df[column_map["plot_id"]]
    out["year"] = df[column_map["year"]]
    out["latitude"] = df[column_map["latitude"]]
    out["longitude"] = df[column_map["longitude"]]

    # G2F Plot Area appears to be in hectares.
    # Convert hectares to square meters:
    # 1 hectare = 10,000 square meters.
    out["plot_area_ha"] = df[column_map["plot_area_ha"]].astype(float)
    out["plot_area_m2"] = out["plot_area_ha"] * 10_000

    before = len(out)

    # ------------------------------------------------------------------
    # 2. Standardize data types.
    # ------------------------------------------------------------------
    out["plot_id"] = out["plot_id"].astype(str)
    out["year"] = out["year"].astype(int)
    out["latitude"] = out["latitude"].astype(float)
    out["longitude"] = out["longitude"].astype(float)
    out["plot_area_m2"] = out["plot_area_m2"].astype(float)

    # ------------------------------------------------------------------
    # 3. Filter to AlphaEarth-supported years.
    # ------------------------------------------------------------------
    min_year = int(config.get("preprocess", {}).get("min_year", 2017))
    max_year = int(config.get("preprocess", {}).get("max_year", 2024))

    out = out[out["year"].between(min_year, max_year)]

    # ------------------------------------------------------------------
    # 4. Drop invalid rows.
    # ------------------------------------------------------------------
    out = out.dropna(
        subset=[
            "plot_id",
            "year",
            "latitude",
            "longitude",
            "plot_area_m2",
        ]
    )

    out = out[
        (out["latitude"].between(-90, 90))
        & (out["longitude"].between(-180, 180))
        & (out["plot_area_m2"] > 0)
    ]

    after_validation = len(out)

    # ------------------------------------------------------------------
    # 5. Aggregate tiny G2F plot rows to site-year/environment level.
    #
    # Individual G2F plot areas are smaller than one 10m AlphaEarth pixel.
    # So we generate embeddings at Field Location + Year level instead of
    # pretending each tiny plot has a separate satellite embedding.
    # ------------------------------------------------------------------
    out = (
        out.groupby(["plot_id", "year"], as_index=False)
        .agg(
            latitude=("latitude", "first"),
            longitude=("longitude", "first"),
            plot_area_m2=("plot_area_m2", "sum"),
            plot_area_ha=("plot_area_ha", "sum"),
            plot_count=("plot_area_m2", "count"),
        )
    )

    # AlphaEarth pixel is approximately 10m x 10m = 100m2.
    out["expected_pixel_count"] = out["plot_area_m2"] / 100.0

    out.to_csv(processed_path, index=False)

    print(f"Saved {len(out)} cleaned site-year rows to {processed_path}")
    print(f"Dropped {before - after_validation} invalid or unsupported rows")
    print(out.head().to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="configs/plot_area.yaml",
        help="Path to config file",
    )

    args = parser.parse_args()
    prepare_input(Path(args.config))


if __name__ == "__main__":
    main()