from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import yaml


REQUIRED_CONFIG_COLUMNS = [
    "site_id",
    "plot_number",
    "year",
    "latitude",
    "longitude",
    "plot_area_km2",
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
    """Prepare raw G2F phenotype data into plot-level rows for AlphaEarth extraction."""
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

    out["site_id"] = df[column_map["site_id"]]
    out["plot_number"] = df[column_map["plot_number"]]
    out["year"] = df[column_map["year"]]
    out["latitude"] = df[column_map["latitude"]]
    out["longitude"] = df[column_map["longitude"]]

    # G2F Plot Area is treated as km2.
    # Convert km2 to square meters:
    # 1 km2 = 1,000,000 m2.
    out["plot_area_km2"] = df[column_map["plot_area_km2"]].astype(float)
    out["plot_area_m2"] = out["plot_area_km2"] * 1_000_000

    before = len(out)

    # ------------------------------------------------------------------
    # 2. Standardize data types.
    # ------------------------------------------------------------------
    out["site_id"] = out["site_id"].astype(str)
    out["plot_number"] = out["plot_number"].astype(str)
    out["year"] = out["year"].astype(int)
    out["latitude"] = out["latitude"].astype(float)
    out["longitude"] = out["longitude"].astype(float)
    out["plot_area_km2"] = out["plot_area_km2"].astype(float)
    out["plot_area_m2"] = out["plot_area_m2"].astype(float)

    # ------------------------------------------------------------------
    # 3. Create a unique plot ID.
    #
    # Plot numbers can repeat across sites and years, so we combine:
    # site_id + year + plot_number
    # ------------------------------------------------------------------
    out["plot_id"] = (
        out["site_id"]
        + "_"
        + out["year"].astype(str)
        + "_"
        + out["plot_number"]
    )

    # ------------------------------------------------------------------
    # 4. Filter to AlphaEarth-supported years.
    # ------------------------------------------------------------------
    min_year = int(config.get("preprocess", {}).get("min_year", 2017))
    max_year = int(config.get("preprocess", {}).get("max_year", 2024))

    out = out[out["year"].between(min_year, max_year)]

    # ------------------------------------------------------------------
    # 5. Drop invalid rows.
    # ------------------------------------------------------------------
    out = out.dropna(
        subset=[
            "plot_id",
            "site_id",
            "plot_number",
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

    # AlphaEarth pixel is approximately 10m x 10m = 100m2.
    out["expected_pixel_count"] = out["plot_area_m2"] / 100.0

    # Keep columns in a clean order.
    out = out[
        [
            "plot_id",
            "site_id",
            "plot_number",
            "year",
            "latitude",
            "longitude",
            "plot_area_km2",
            "plot_area_m2",
            "expected_pixel_count",
        ]
    ]

    out.to_csv(processed_path, index=False)

    print(f"Saved {len(out)} cleaned plot-level rows to {processed_path}")
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