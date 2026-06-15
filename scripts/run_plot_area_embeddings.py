"""Extract plot-aware AlphaEarth embeddings from Google Earth Engine.

This module turns standardized plot records into Earth Engine geometries,
extracts annual AlphaEarth embedding means, and writes long-format outputs with
geometry and pixel-count quality-control metadata.

Key features:
- Supports area-derived squares, fixed square windows, fixed circular buffers,
  and multi-scale extraction from the same coordinate.
- Treats polygon mode as an explicit future extension with a clear error.
- Preserves configured modeling metadata in the embedding output.
- Combines mean and count reducers so downstream users can inspect pixel
  coverage for every plot/year/geometry request.
"""

from __future__ import annotations

import argparse
import importlib
import json
import math
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

_EE_MODULE: Any | None = None


CORE_OUTPUT_COLUMNS = [
    "plot_id",
    "site_id",
    "plot_number",
    "year",
    "latitude",
    "longitude",
    "geometry_method",
    "geometry_mode",
    "geometry_shape",
    "window_m",
    "radius_m",
    "geometry_area_m2",
    "plot_area_m2",
    "expected_pixel_count",
    "actual_pixel_count",
    "pixel_coverage_ratio",
    "aggregation_method",
]


def require_earth_engine() -> Any:
    """Return the Earth Engine module or fail with a helpful install message.

    The import is lazy-friendly so unit tests can exercise geometry and parsing
    logic without a local Earth Engine installation or credentials.
    """
    global _EE_MODULE

    if _EE_MODULE is not None:
        return _EE_MODULE

    try:
        _EE_MODULE = importlib.import_module("ee")
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "earthengine-api is not installed. Install dependencies before running "
            "embedding extraction."
        ) from exc

    return _EE_MODULE


def initialize_earth_engine(project_env: str) -> None:
    """Initialize Google Earth Engine using a project ID environment variable.

    The project_env config value is usually EE_PROJECT. Keeping it configurable
    lets the same pipeline run across machines or cloud projects without code
    changes.
    """
    project = os.getenv(project_env, "").strip()

    if not project:
        raise RuntimeError(
            f"{project_env} is empty. Run: export {project_env}=your-google-cloud-project-id"
        )

    require_earth_engine().Initialize(project=project)


def alphaearth_band_names() -> list[str]:
    """Return the 64 AlphaEarth annual embedding band names.

    Google Earth Engine exposes these dimensions as A00 through A63. The final
    CSV renames them to embedding_00 through embedding_63 for model-friendly
    tabular use.
    """
    return [f"A{i:02d}" for i in range(64)]


def as_float(value: Any) -> float | None:
    """Convert a scalar to float, treating pandas/None missing values as None."""
    if value is None:
        return None

    try:
        if pd.isna(value):
            return None
    except TypeError:
        return None

    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def has_positive_number(value: Any) -> bool:
    """Return True only when a value can be read as a positive float."""
    number = as_float(value)
    return number is not None and number > 0


def clean_property(value: Any) -> Any:
    """Convert pandas/numpy values into Earth Engine feature-safe properties.

    Earth Engine properties cannot carry pandas NA values or numpy scalar
    wrappers reliably, so this helper normalizes them before feature creation.
    """
    try:
        if pd.isna(value):
            return None
    except TypeError:
        return value

    if hasattr(value, "item"):
        return value.item()

    return value


def normalized_geometry_config(config: Mapping[str, Any] | None) -> dict[str, Any]:
    """Return geometry config with backward-compatible defaults filled in.

    A missing geometry section behaves like the old pipeline for rows with plot
    area: auto mode chooses area_square. Rows without area use a default fixed
    square window instead of failing.
    """
    geometry_config = dict(config or {})
    geometry_config.setdefault("mode", "auto")
    geometry_config.setdefault("shape", "square")
    geometry_config.setdefault("fixed_windows_m", [30, 50, 100])
    geometry_config.setdefault("default_window_m", 50)
    geometry_config.setdefault("polygon_column", None)
    return geometry_config


def square_from_lat_lon_area(
    lon: float,
    lat: float,
    area_m2: float,
    ee_module: Any | None = None,
) -> Any:
    """Convert plot center plus area into an approximate square geometry.

    This is the MVP geometry for datasets that know plot area but not exact
    boundaries: same area, centered on the coordinate, not rotated.
    """
    side_m = math.sqrt(area_m2)
    return square_from_lat_lon_window(lon, lat, side_m, ee_module=ee_module)


def square_from_lat_lon_window(
    lon: float,
    lat: float,
    window_m: float,
    ee_module: Any | None = None,
) -> Any:
    """Create a coordinate-centered square window with side length in meters.

    This is the main fallback for lat/lon-only datasets. The chosen window size
    should be treated as a sampling assumption and recorded in output metadata.
    """
    _ee = ee_module or require_earth_engine()
    center = _ee.Geometry.Point([lon, lat])
    return center.buffer(window_m / 2).bounds()


def circle_from_lat_lon_radius(
    lon: float,
    lat: float,
    radius_m: float,
    ee_module: Any | None = None,
) -> Any:
    """Create a coordinate-centered circular buffer with radius in meters.

    Circular buffers are useful when the analysis wants distance-from-point
    context instead of a square pixel window.
    """
    _ee = ee_module or require_earth_engine()
    center = _ee.Geometry.Point([lon, lat])
    return center.buffer(radius_m)


def resolve_geometry_mode(row: pd.Series, geometry_config: Mapping[str, Any]) -> str:
    """Resolve the concrete geometry mode for one input row.

    In auto mode, the priority is future polygon support first, then area-based
    square when plot_area_m2 is present, and finally fixed_square for coordinate
    only records.
    """
    requested_mode = geometry_config.get("mode", "auto")

    if requested_mode == "auto":
        polygon_column = geometry_config.get("polygon_column")
        if polygon_column and polygon_column in row.index and not pd.isna(row[polygon_column]):
            return "polygon"

        if has_positive_number(row.get("plot_area_m2")):
            return "area_square"

        return "fixed_square"

    if requested_mode in {
        "area_square",
        "fixed_square",
        "fixed_circle",
        "multi_scale",
        "polygon",
    }:
        return requested_mode

    raise ValueError(f"Unsupported geometry mode: {requested_mode}")


def build_geometry(
    row: pd.Series,
    geometry_config: Mapping[str, Any],
    scale_meters: float = 10,
    ee_module: Any | None = None,
) -> tuple[Any, dict[str, Any]]:
    """Build one Earth Engine geometry and its output metadata for one row.

    This wrapper is for single-geometry modes. Multi-scale mode is handled by
    build_geometry_requests because it intentionally emits several geometries
    for the same input record.
    """
    geometry_config = normalized_geometry_config(geometry_config)
    mode = resolve_geometry_mode(row, geometry_config)

    if mode == "multi_scale":
        raise ValueError("Use build_geometry_requests for multi_scale geometry.")

    return build_single_geometry(
        row=row,
        mode=mode,
        geometry_config=geometry_config,
        scale_meters=scale_meters,
        ee_module=ee_module,
    )


def build_single_geometry(
    row: pd.Series,
    mode: str,
    geometry_config: Mapping[str, Any],
    scale_meters: float,
    ee_module: Any | None = None,
    window_override_m: float | None = None,
) -> tuple[Any, dict[str, Any]]:
    """Build one concrete geometry request and its audit metadata.

    The returned metadata is written to each output row, making the extraction
    reproducible: consumers can see the mode, shape, window/radius, geometry
    area, source plot area if available, and expected pixel count.
    """
    lon = as_float(row.get("longitude"))
    lat = as_float(row.get("latitude"))

    if lon is None or lat is None:
        raise ValueError("Rows must have valid latitude and longitude values.")

    plot_area_m2 = as_float(row.get("plot_area_m2"))
    window_m = None
    radius_m = None

    if mode == "polygon":
        raise NotImplementedError(
            "polygon geometry mode is declared for future support but is not implemented yet."
        )

    if mode == "area_square":
        if plot_area_m2 is None or plot_area_m2 <= 0:
            raise ValueError("area_square geometry requires a positive plot_area_m2 value.")

        geometry = square_from_lat_lon_area(lon, lat, plot_area_m2, ee_module=ee_module)
        geometry_shape = "square"
        geometry_area_m2 = plot_area_m2

    elif mode == "fixed_square":
        window_m = float(window_override_m or geometry_config["default_window_m"])
        geometry = square_from_lat_lon_window(lon, lat, window_m, ee_module=ee_module)
        geometry_shape = "square"
        geometry_area_m2 = window_m * window_m

    elif mode == "fixed_circle":
        radius_m = float(
            window_override_m
            or geometry_config.get("radius_m")
            or geometry_config["default_window_m"]
        )
        geometry = circle_from_lat_lon_radius(lon, lat, radius_m, ee_module=ee_module)
        geometry_shape = "circle"
        geometry_area_m2 = math.pi * radius_m * radius_m

    else:
        raise ValueError(f"Unsupported concrete geometry mode: {mode}")

    pixel_area_m2 = scale_meters * scale_meters
    expected_pixel_count = geometry_area_m2 / pixel_area_m2 if pixel_area_m2 > 0 else None

    metadata = {
        "geometry_method": mode,
        "geometry_mode": mode,
        "geometry_shape": geometry_shape,
        "window_m": window_m,
        "radius_m": radius_m,
        "geometry_area_m2": geometry_area_m2,
        "plot_area_m2": plot_area_m2,
        "expected_pixel_count": expected_pixel_count,
    }

    return geometry, metadata


def build_geometry_requests(
    row: pd.Series,
    geometry_config: Mapping[str, Any],
    scale_meters: float = 10,
    ee_module: Any | None = None,
) -> list[tuple[Any, dict[str, Any]]]:
    """Build one or more geometry requests for a dataframe row.

    Single-window modes return one request. multi_scale expands one input row
    into one request per configured window, producing long-format output that is
    easy to compare in downstream modeling.
    """
    geometry_config = normalized_geometry_config(geometry_config)
    mode = resolve_geometry_mode(row, geometry_config)

    if mode != "multi_scale":
        return [
            build_single_geometry(
                row=row,
                mode=mode,
                geometry_config=geometry_config,
                scale_meters=scale_meters,
                ee_module=ee_module,
            )
        ]

    shape = geometry_config.get("shape", "square")
    if shape not in {"square", "circle"}:
        raise ValueError(f"Unsupported multi_scale shape: {shape}")

    concrete_mode = "fixed_circle" if shape == "circle" else "fixed_square"
    windows = geometry_config.get("fixed_windows_m") or [geometry_config["default_window_m"]]

    requests = []
    for window_m in windows:
        requests.append(
            build_single_geometry(
                row=row,
                mode=concrete_mode,
                geometry_config=geometry_config,
                scale_meters=scale_meters,
                ee_module=ee_module,
                window_override_m=float(window_m),
            )
        )

    return requests


def present_metadata_columns(
    df: pd.DataFrame,
    config: Mapping[str, Any],
) -> list[str]:
    """Return configured metadata columns that are available in the processed CSV.

    Missing optional metadata is ignored here and later reported in the QC JSON
    rather than treated as a validation failure.
    """
    preserve_columns = config.get("metadata", {}).get("preserve_columns", [])
    return [column for column in preserve_columns if column in df.columns]


def build_feature_collection(
    df: pd.DataFrame,
    geometry_config: Mapping[str, Any],
    scale_meters: float,
    metadata_columns: Sequence[str] | None = None,
    ee_module: Any | None = None,
) -> Any:
    """Convert input rows into an Earth Engine FeatureCollection.

    Each feature is one plot-year-geometry request, which allows multi-scale
    output to stay in a simple long format. Original identifiers, geometry
    metadata, and optional modeling metadata are stored as feature properties so
    they return with the reducer results.
    """
    _ee = ee_module or require_earth_engine()
    metadata_columns = metadata_columns or []
    features = []

    for _, row in df.iterrows():
        for geom, geometry_metadata in build_geometry_requests(
            row=row,
            geometry_config=geometry_config,
            scale_meters=scale_meters,
            ee_module=_ee,
        ):
            properties = {
                "plot_id": clean_property(row.get("plot_id")),
                "site_id": clean_property(row.get("site_id")),
                "plot_number": clean_property(row.get("plot_number")),
                "year": clean_property(int(row["year"])),
                "latitude": clean_property(float(row["latitude"])),
                "longitude": clean_property(float(row["longitude"])),
                "aggregation_method": "mean_pooling",
            }
            properties.update(
                {key: clean_property(value) for key, value in geometry_metadata.items()}
            )

            for column in metadata_columns:
                properties[column] = clean_property(row.get(column))

            features.append(_ee.Feature(geom, properties))

    return _ee.FeatureCollection(features)


def extract_embeddings_for_year(
    year: int,
    feature_collection: Any,
    collection_name: str,
    scale_meters: int,
    ee_module: Any | None = None,
) -> Any:
    """Extract AlphaEarth mean embeddings and pixel counts for one year.

    The function filters the annual AlphaEarth image collection to a calendar
    year, selects A00-A63, and applies a combined mean/count reducer to every
    plot geometry. Counts are used later for conservative pixel-coverage QC.
    """
    _ee = ee_module or require_earth_engine()
    start_date = _ee.Date.fromYMD(year, 1, 1)
    end_date = start_date.advance(1, "year")

    image = (
        _ee.ImageCollection(collection_name)
        .filterDate(start_date, end_date)
        .mosaic()
        .select(alphaearth_band_names())
    )

    reducer = _ee.Reducer.mean().combine(
        reducer2=_ee.Reducer.count(),
        sharedInputs=True,
    )

    reduced = image.reduceRegions(
        collection=feature_collection,
        reducer=reducer,
        scale=scale_meters,
    )

    return reduced


def output_property(props: Mapping[str, Any], key: str) -> Any:
    """Read one Earth Engine feature property by key."""
    return props.get(key)


def parse_ee_features(
    features: list[dict[str, Any]],
    metadata_columns: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Convert Earth Engine getInfo features into the final output dataframe.

    The parser renames Axx_mean values to embedding_xx, derives one
    actual_pixel_count from the minimum band count, and computes
    pixel_coverage_ratio against expected_pixel_count.
    """
    rows = []
    bands = alphaearth_band_names()
    metadata_columns = metadata_columns or []

    for feature in features:
        props = feature["properties"]
        row = {column: output_property(props, column) for column in CORE_OUTPUT_COLUMNS}

        counts = []
        for i, band in enumerate(bands):
            row[f"embedding_{i:02d}"] = props.get(f"{band}_mean", props.get(band))
            count = as_float(props.get(f"{band}_count"))
            if count is not None:
                counts.append(count)

        actual_pixel_count = min(counts) if counts else None
        row["actual_pixel_count"] = actual_pixel_count

        expected_pixel_count = as_float(row.get("expected_pixel_count"))
        if expected_pixel_count and actual_pixel_count is not None:
            row["pixel_coverage_ratio"] = actual_pixel_count / expected_pixel_count
        else:
            row["pixel_coverage_ratio"] = None

        for column in metadata_columns:
            row[column] = props.get(column)

        rows.append(row)

    output_columns = list(CORE_OUTPUT_COLUMNS)
    output_columns.extend(column for column in metadata_columns if column not in output_columns)
    output_columns.extend(f"embedding_{i:02d}" for i in range(64))

    return pd.DataFrame(rows, columns=output_columns)


def numeric_summary(series: pd.Series) -> dict[str, float | None]:
    """Return min, median, and mean for a numeric series with null-safe output."""
    numeric = pd.to_numeric(series, errors="coerce").dropna()
    if numeric.empty:
        return {"min": None, "median": None, "mean": None}

    return {
        "min": float(numeric.min()),
        "median": float(numeric.median()),
        "mean": float(numeric.mean()),
    }


def build_qc_summary(
    input_df: pd.DataFrame,
    result: pd.DataFrame,
    metadata_columns: Sequence[str],
    missing_optional_metadata_columns: Sequence[str],
    low_coverage_threshold: float = 0.5,
) -> dict:
    """Build a compact JSON-serializable QC summary.

    The summary captures extraction shape, output size, embedding completeness,
    low pixel coverage, and which optional metadata columns were preserved or
    missing.
    """
    embedding_cols = [column for column in result.columns if column.startswith("embedding_")]

    if result.empty:
        rows_missing_embeddings = 0
        rows_with_low_pixel_coverage = 0
        pixel_coverage_stats = {"min": None, "median": None, "mean": None}
    else:
        rows_missing_embeddings = int(result[embedding_cols].isna().any(axis=1).sum())
        pixel_coverage = pd.to_numeric(result["pixel_coverage_ratio"], errors="coerce")
        rows_with_low_pixel_coverage = int((pixel_coverage < low_coverage_threshold).sum())
        pixel_coverage_stats = numeric_summary(pixel_coverage)

    windows = []
    if "window_m" in result.columns:
        window_values = pd.to_numeric(result["window_m"], errors="coerce")
        windows.extend(window_values.dropna().unique().tolist())
    if "radius_m" in result.columns:
        radius_values = pd.to_numeric(result["radius_m"], errors="coerce")
        windows.extend(radius_values.dropna().unique().tolist())

    return {
        "total_input_rows": int(len(input_df)),
        "total_output_rows": int(len(result)),
        "geometry_modes_used": sorted(
            str(value)
            for value in result.get("geometry_mode", pd.Series(dtype=object)).dropna().unique()
        ),
        "windows_used": sorted(float(value) for value in set(windows)),
        "rows_missing_embeddings": rows_missing_embeddings,
        "rows_with_low_pixel_coverage": rows_with_low_pixel_coverage,
        "min_pixel_coverage_ratio": pixel_coverage_stats["min"],
        "median_pixel_coverage_ratio": pixel_coverage_stats["median"],
        "mean_pixel_coverage_ratio": pixel_coverage_stats["mean"],
        "preserved_metadata_columns": metadata_columns,
        "missing_optional_metadata_columns": missing_optional_metadata_columns,
    }


def write_qc_summary(
    input_df: pd.DataFrame,
    result: pd.DataFrame,
    config: Mapping[str, Any],
    metadata_columns: Sequence[str],
    output_path: Path,
) -> Path:
    """Write the JSON QC summary next to the embedding output or configured path."""
    requested_columns = config.get("metadata", {}).get("preserve_columns", [])
    missing_optional = [column for column in requested_columns if column not in metadata_columns]
    threshold = float(config.get("output", {}).get("low_pixel_coverage_threshold", 0.5))

    summary = build_qc_summary(
        input_df=input_df,
        result=result,
        metadata_columns=metadata_columns,
        missing_optional_metadata_columns=missing_optional,
        low_coverage_threshold=threshold,
    )

    qc_path = Path(
        config.get("output", {}).get(
            "qc_summary_path",
            output_path.with_name(f"{output_path.stem}_qc_summary.json"),
        )
    )
    qc_path.parent.mkdir(parents=True, exist_ok=True)
    qc_path.write_text(json.dumps(summary, indent=2) + "\n")
    return qc_path


def write_tabular_report(result: pd.DataFrame, report_path: Path) -> None:
    """Write the legacy CSV report with geometry-aware summary fields.

    This preserves the old report_path contract while adding actual pixel counts
    and geometry-area statistics introduced by the upgraded extraction layer.
    """
    report_path.parent.mkdir(parents=True, exist_ok=True)
    embedding_cols = [column for column in result.columns if column.startswith("embedding_")]

    if result.empty:
        summary = pd.DataFrame(
            [
                {
                    "rows": 0,
                    "embedding_dimensions": len(embedding_cols),
                }
            ]
        )
    else:
        summary = pd.DataFrame(
            [
                {
                    "rows": len(result),
                    "embedding_dimensions": len(embedding_cols),
                    "min_geometry_area_m2": result["geometry_area_m2"].min(),
                    "max_geometry_area_m2": result["geometry_area_m2"].max(),
                    "mean_geometry_area_m2": result["geometry_area_m2"].mean(),
                    "min_expected_pixel_count": result["expected_pixel_count"].min(),
                    "max_expected_pixel_count": result["expected_pixel_count"].max(),
                    "mean_expected_pixel_count": result["expected_pixel_count"].mean(),
                    "min_actual_pixel_count": result["actual_pixel_count"].min(),
                    "max_actual_pixel_count": result["actual_pixel_count"].max(),
                    "mean_actual_pixel_count": result["actual_pixel_count"].mean(),
                }
            ]
        )

    summary.to_csv(report_path, index=False)
    print(f"Saved summary report to {report_path}")
    print(summary.to_string(index=False))


def run(config_path: Path) -> None:
    """Run the full Earth Engine extraction workflow from a YAML config."""
    with config_path.open("r") as f:
        config = yaml.safe_load(f)

    initialize_earth_engine(config["earth_engine"]["project_env"])

    input_path = Path(config["input"]["processed_csv_path"])
    output_path = Path(config["output"]["embeddings_path"])
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_path)
    metadata_columns = present_metadata_columns(df, config)
    geometry_config = normalized_geometry_config(config.get("geometry"))
    scale_meters = float(config["earth_engine"].get("scale_meters", 10))

    print(f"Read {len(df)} plots from {input_path}")

    all_outputs = []

    # Process by year so each group uses the correct annual AlphaEarth image.
    for year, year_df in df.groupby("year"):
        year = int(year)

        print(f"Processing year {year}: {len(year_df)} input rows")

        fc = build_feature_collection(
            year_df,
            geometry_config=geometry_config,
            scale_meters=scale_meters,
            metadata_columns=metadata_columns,
        )

        reduced = extract_embeddings_for_year(
            year=year,
            feature_collection=fc,
            collection_name=config["earth_engine"]["collection"],
            scale_meters=int(scale_meters),
        )

        info = reduced.getInfo()
        out_df = parse_ee_features(info["features"], metadata_columns=metadata_columns)
        all_outputs.append(out_df)

    if all_outputs:
        result = pd.concat(all_outputs, ignore_index=True)
    else:
        result = parse_ee_features([], metadata_columns=metadata_columns)

    result.to_csv(output_path, index=False)
    print(f"Saved {len(result)} embedding rows to {output_path}")

    report_path = Path(config["output"]["report_path"])
    write_tabular_report(result, report_path)

    qc_path = write_qc_summary(
        input_df=df,
        result=result,
        config=config,
        metadata_columns=metadata_columns,
        output_path=output_path,
    )
    print(f"Saved QC summary to {qc_path}")


def main() -> None:
    """Parse CLI arguments and run embedding extraction."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="configs/plot_area.yaml",
        help="Path to YAML config file",
    )

    args = parser.parse_args()
    run(Path(args.config))


if __name__ == "__main__":
    main()
