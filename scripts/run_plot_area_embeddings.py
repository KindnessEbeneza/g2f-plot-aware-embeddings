from __future__ import annotations

import argparse
import math
import os
from pathlib import Path

import ee
import pandas as pd
import yaml


def initialize_earth_engine(project_env: str) -> None:
    """
    Initialize Google Earth Engine using a project ID stored
    in an environment variable such as EE_PROJECT.
    """
    project = os.getenv(project_env, "").strip()

    if not project:
        raise RuntimeError(
            f"{project_env} is empty. Run: export {project_env}=your-google-cloud-project-id"
        )

    ee.Initialize(project=project)


def alphaearth_band_names() -> list[str]:
    """
    AlphaEarth annual embeddings expose 64 embedding bands.
    In our pipeline we use A00 through A63.
    """
    return [f"A{i:02d}" for i in range(64)]


def square_from_lat_lon_area(
    lon: float,
    lat: float,
    area_m2: float,
) -> ee.Geometry:
    """
    Convert plot center + plot area into an approximate square geometry.

    This is the cheap MVP geometry:
    - we do not know exact plot boundary
    - but we know plot area
    - so we approximate a square centered on the plot coordinate
    """
    side_m = math.sqrt(area_m2)
    half_side_m = side_m / 2

    center = ee.Geometry.Point([lon, lat])

    # Create square in meters using projection.
    # Buffer gives us meter-based geometry; bounds turns it into a square bbox.
    square = center.buffer(half_side_m).bounds()

    return square


def build_feature_collection(
    df: pd.DataFrame,
    columns: dict,
) -> ee.FeatureCollection:
    """
    Convert input dataframe into an Earth Engine FeatureCollection.
    Each feature is one plot-year with geometry and metadata.
    """
    features = []

    for _, row in df.iterrows():
        plot_id = str(row["plot_id"])
        year = int(row["year"])
        lat = float(row["latitude"])
        lon = float(row["longitude"])
        area_m2 = float(row["plot_area_m2"])

        geom = square_from_lat_lon_area(
            lon=lon,
            lat=lat,
            area_m2=area_m2,
        )

        expected_pixel_count = area_m2 / 100.0  # 10m x 10m pixel ~= 100m2

        feature = ee.Feature(
            geom,
            {
                "plot_id": plot_id,
                "year": year,
                "latitude": lat,
                "longitude": lon,
                "plot_area_m2": area_m2,
                "expected_pixel_count": expected_pixel_count,
                "geometry_method": "square_from_plot_area",
            },
        )

        features.append(feature)

    return ee.FeatureCollection(features)


def extract_embeddings_for_year(
    year: int,
    feature_collection: ee.FeatureCollection,
    collection_name: str,
    scale_meters: int,
) -> ee.FeatureCollection:
    """
    For one year:
    - load AlphaEarth annual image
    - select embedding bands
    - compute mean embedding inside every plot geometry
    """
    start_date = ee.Date.fromYMD(year, 1, 1)
    end_date = start_date.advance(1, "year")

    image = (
        ee.ImageCollection(collection_name)
        .filterDate(start_date, end_date)
        .mosaic()
        .select(alphaearth_band_names())
    )

    reduced = image.reduceRegions(
        collection=feature_collection,
        reducer=ee.Reducer.mean(),
        scale=scale_meters,
    )

    return reduced


def parse_ee_features(features: list[dict]) -> pd.DataFrame:
    """
    Convert Earth Engine FeatureCollection getInfo result
    into a normal pandas dataframe.
    """
    rows = []
    bands = alphaearth_band_names()

    for feature in features:
        props = feature["properties"]

        row = {
            "plot_id": props.get("plot_id"),
            "year": props.get("year"),
            "latitude": props.get("latitude"),
            "longitude": props.get("longitude"),
            "plot_area_m2": props.get("plot_area_m2"),
            "expected_pixel_count": props.get("expected_pixel_count"),
            "geometry_method": props.get("geometry_method"),
            "aggregation_method": "mean_pooling",
        }

        # Convert A00-A63 into embedding_00-embedding_63 columns
        for i, band in enumerate(bands):
            row[f"embedding_{i:02d}"] = props.get(band)

        rows.append(row)

    return pd.DataFrame(rows)


def run(config_path: Path) -> None:
    """
    Main pipeline runner.
    """
    with config_path.open("r") as f:
        config = yaml.safe_load(f)

    initialize_earth_engine(config["earth_engine"]["project_env"])

    input_path = Path(config["input"]["processed_csv_path"])
    output_path = Path(config["output"]["embeddings_path"])
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_path)

    print(f"Read {len(df)} plots from {input_path}")

    all_outputs = []

    # Process by year so each group uses the correct annual AlphaEarth image
    for year, year_df in df.groupby("year"):
        year = int(year)

        print(f"Processing year {year}: {len(year_df)} plots")

        fc = build_feature_collection(
            year_df,
            config["columns"],
        )

        reduced = extract_embeddings_for_year(
            year=year,
            feature_collection=fc,
            collection_name=config["earth_engine"]["collection"],
            scale_meters=int(config["earth_engine"]["scale_meters"]),
        )

        info = reduced.getInfo()
        out_df = parse_ee_features(info["features"])
        all_outputs.append(out_df)

    if all_outputs:
        result = pd.concat(all_outputs, ignore_index=True)
    else:
        result = pd.DataFrame()

    result.to_csv(output_path, index=False)
    report_path = Path(config["output"]["report_path"])
    report_path.parent.mkdir(parents=True, exist_ok=True)

    embedding_cols = [c for c in result.columns if c.startswith("embedding_")]

    summary = pd.DataFrame([
    {
        "rows": len(result),
        "embedding_dimensions": len(embedding_cols),
        "min_plot_area_m2": result["plot_area_m2"].min(),
        "max_plot_area_m2": result["plot_area_m2"].max(),
        "mean_plot_area_m2": result["plot_area_m2"].mean(),
        "min_expected_pixel_count": result["expected_pixel_count"].min(),
        "max_expected_pixel_count": result["expected_pixel_count"].max(),
        "mean_expected_pixel_count": result["expected_pixel_count"].mean(),
    }
])

    summary.to_csv(report_path, index=False)
    print(f"Saved summary report to {report_path}")
    print(summary.to_string(index=False))





def main() -> None:
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
