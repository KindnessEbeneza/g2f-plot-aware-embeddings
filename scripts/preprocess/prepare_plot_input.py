"""Prepare coordinate-indexed plot records for AlphaEarth extraction.

This module is the pipeline's flexible input adapter. It takes a dataset-specific
CSV plus a YAML column map and writes a standardized plot registry that the
Earth Engine extraction step can consume.

Key features:
- Accepts lat/lon/year with either plot_id or site_id as the minimum schema.
- Treats plot area as optional and normalizes any supported area unit to m2.
- Preserves configured modeling metadata without requiring those columns.
- Keeps backward compatibility with G2F-style site/year/plot identifiers.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pandas as pd
import yaml


REQUIRED_SPATIAL_COLUMNS = [
    "year",
    "latitude",
    "longitude",
]
ID_COLUMNS = ["plot_id", "site_id"]
AREA_COLUMNS_M2 = ["plot_area_m2", "area_m2"]
AREA_COLUMNS_KM2 = ["plot_area_km2", "area_km2"]


def load_config(config_path: Path) -> dict[str, Any]:
    """Load the YAML pipeline configuration.

    The config provides dataset-specific input paths, column mappings,
    Earth Engine scale, and optional metadata columns to preserve.
    """
    with config_path.open("r") as f:
        config = yaml.safe_load(f)

    if not isinstance(config, dict):
        raise ValueError(f"Config file must contain a YAML mapping: {config_path}")

    return config


def configured_source_column(
    df: pd.DataFrame,
    column_map: Mapping[str, str | None],
    keys: list[str],
) -> tuple[str | None, str | None]:
    """Resolve the first available source column for one or more standard keys.

    The config can map a standard key to a dataset-specific column name. As a
    fallback, the standard key itself is accepted when it appears in the CSV.
    This makes the pipeline usable both with G2F-native column names and already
    standardized input files.
    """
    for key in keys:
        source_column = column_map.get(key)
        if source_column and source_column in df.columns:
            return key, source_column

        if key in df.columns:
            return key, key

    return None, None


def validate_columns(df: pd.DataFrame, column_map: Mapping[str, str | None]) -> None:
    """Validate the minimum schema needed to extract satellite embeddings.

    Required fields are year, latitude, longitude, and at least one identifier:
    plot_id for true plot-level data or site_id for site-level datasets. Plot
    area and modeling metadata are optional by design.
    """
    missing = []

    for key in REQUIRED_SPATIAL_COLUMNS:
        _, source_column = configured_source_column(df, column_map, [key])
        if source_column is None:
            configured = column_map.get(key)
            missing.append(configured or key)

    _, id_column = configured_source_column(df, column_map, ID_COLUMNS)
    if id_column is None:
        missing.append("plot_id or site_id")

    if missing:
        raise ValueError(f"Missing required input columns: {missing}")


def area_column(
    df: pd.DataFrame,
    column_map: Mapping[str, str | None],
) -> tuple[str | None, str | None]:
    """Resolve an optional plot-area column and report its unit.

    Supported standardized keys are plot_area_m2, area_m2, plot_area_km2, and
    area_km2. Returning the unit separately lets preprocessing normalize every
    area input to plot_area_m2 while preserving the km2 companion column.
    """
    _, source_column = configured_source_column(df, column_map, AREA_COLUMNS_M2)
    if source_column is not None:
        return "m2", source_column

    _, source_column = configured_source_column(df, column_map, AREA_COLUMNS_KM2)
    if source_column is not None:
        return "km2", source_column

    return None, None


def add_optional_column(
    out: pd.DataFrame,
    df: pd.DataFrame,
    column_map: Mapping[str, str | None],
    key: str,
) -> None:
    """Add an optional standardized column when it exists in the raw dataframe.

    This is used for IDs such as site_id and plot_number: helpful when present,
    but not mandatory for lat/lon-only datasets.
    """
    _, source_column = configured_source_column(df, column_map, [key])
    if source_column is not None:
        out[key] = df[source_column]


def create_plot_ids(out: pd.DataFrame) -> pd.Series:
    """Create a stable plot_id when the source data only has site-level IDs.

    If a plot_id is already present, it is preserved. Otherwise, the function
    reproduces the historical site-year-plot pattern when plot_number exists,
    and falls back to a site-year-row identifier for site-only inputs.
    """
    if "plot_id" in out.columns:
        return out["plot_id"].astype(str)

    row_numbers = pd.Series(range(1, len(out) + 1), index=out.index).astype(str)

    if "plot_number" in out.columns:
        return (
            out["site_id"].astype(str)
            + "_"
            + out["year"].astype("Int64").astype(str)
            + "_"
            + out["plot_number"].astype(str)
        )

    return (
        out["site_id"].astype(str)
        + "_"
        + out["year"].astype("Int64").astype(str)
        + "_row"
        + row_numbers
    )


def prepare_input(config_path: Path) -> None:
    """Standardize raw plot or site records into the processed plot registry.

    The output CSV always contains year, latitude, longitude, plot_id, optional
    site/plot metadata, normalized area columns, expected_pixel_count when area
    exists, and any configured modeling metadata found in the raw input.
    """
    config = load_config(config_path)

    raw_path = Path(config["input"]["raw_csv_path"])
    processed_path = Path(config["input"]["processed_csv_path"])
    processed_path.parent.mkdir(parents=True, exist_ok=True)

    column_map = config.get("columns", {})
    if not isinstance(column_map, dict):
        raise ValueError("Config field 'columns' must be a mapping.")

    df = pd.read_csv(raw_path)

    print(f"Read {len(df)} rows from {raw_path}")

    validate_columns(df, column_map)

    # ------------------------------------------------------------------
    # 1. Standardize raw columns into pipeline column names.
    # ------------------------------------------------------------------
    out = pd.DataFrame()

    for key in REQUIRED_SPATIAL_COLUMNS:
        _, source_column = configured_source_column(df, column_map, [key])
        if source_column is None:
            raise ValueError(f"Missing required input column: {key}")
        out[key] = df[source_column]

    add_optional_column(out, df, column_map, "plot_id")
    add_optional_column(out, df, column_map, "site_id")
    add_optional_column(out, df, column_map, "plot_number")

    area_unit, source_column = area_column(df, column_map)
    if area_unit == "m2":
        if source_column is None:
            raise ValueError("Area unit resolved without an area source column.")
        out["plot_area_m2"] = pd.to_numeric(df[source_column], errors="coerce")
        out["plot_area_km2"] = out["plot_area_m2"] / 1_000_000
    elif area_unit == "km2":
        if source_column is None:
            raise ValueError("Area unit resolved without an area source column.")
        out["plot_area_km2"] = pd.to_numeric(df[source_column], errors="coerce")
        out["plot_area_m2"] = out["plot_area_km2"] * 1_000_000
    else:
        out["plot_area_km2"] = pd.NA
        out["plot_area_m2"] = pd.NA

    before = len(out)

    # ------------------------------------------------------------------
    # 2. Standardize data types.
    # ------------------------------------------------------------------
    out["year"] = pd.to_numeric(out["year"], errors="coerce")
    out["latitude"] = pd.to_numeric(out["latitude"], errors="coerce")
    out["longitude"] = pd.to_numeric(out["longitude"], errors="coerce")
    out["plot_area_km2"] = pd.to_numeric(out["plot_area_km2"], errors="coerce")
    out["plot_area_m2"] = pd.to_numeric(out["plot_area_m2"], errors="coerce")

    if "site_id" in out.columns:
        out["site_id"] = out["site_id"].astype(str)
    if "plot_number" in out.columns:
        out["plot_number"] = out["plot_number"].astype(str)

    # ------------------------------------------------------------------
    # 3. Create a unique plot ID.
    #
    # If the dataset does not provide plot_id, derive one from site_id.
    # When plot_number is present, preserve the previous site-year-plot pattern.
    # ------------------------------------------------------------------
    out["plot_id"] = create_plot_ids(out)

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
            "year",
            "latitude",
            "longitude",
        ]
    )

    out = out[
        (out["latitude"].between(-90, 90))
        & (out["longitude"].between(-180, 180))
    ]

    out.loc[out["plot_area_m2"] <= 0, ["plot_area_km2", "plot_area_m2"]] = pd.NA
    out["year"] = out["year"].astype(int)
    out["latitude"] = out["latitude"].astype(float)
    out["longitude"] = out["longitude"].astype(float)

    after_validation = len(out)

    scale_meters = float(config.get("earth_engine", {}).get("scale_meters", 10))
    pixel_area_m2 = scale_meters * scale_meters
    out["expected_pixel_count"] = out["plot_area_m2"] / pixel_area_m2

    preserve_columns = config.get("metadata", {}).get("preserve_columns", [])
    preserved = []
    for column in preserve_columns:
        if column in df.columns and column not in out.columns:
            out[column] = df.loc[out.index, column]
            preserved.append(column)

    # Keep columns in a clean order.
    core_columns = [
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
    ordered_columns = [column for column in core_columns if column in out.columns]
    ordered_columns.extend(column for column in preserved if column in out.columns)
    out = out[ordered_columns]

    out.to_csv(processed_path, index=False)

    print(f"Saved {len(out)} cleaned plot-level rows to {processed_path}")
    print(f"Dropped {before - after_validation} invalid or unsupported rows")
    if preserved:
        print(f"Preserved optional metadata columns: {preserved}")
    print(out.head().to_string(index=False))


def main() -> None:
    """Parse CLI arguments and run preprocessing."""
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
