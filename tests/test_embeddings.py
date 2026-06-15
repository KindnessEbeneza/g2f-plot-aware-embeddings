"""Unit tests for geometry construction, output parsing, and QC summary logic.

The fake Earth Engine classes below provide just enough behavior to test local
geometry metadata and long-format output logic without authenticating against
Google Earth Engine.
"""

from __future__ import annotations

import json
import math
import tempfile
import unittest
from pathlib import Path
from typing import Any

import pandas as pd

from scripts.run_plot_area_embeddings import (
    build_feature_collection,
    build_geometry,
    parse_ee_features,
    write_qc_summary,
)


class FakeGeometry:
    """Tiny stand-in for ee.Geometry used by geometry unit tests."""

    def __init__(
        self,
        coords: list[float],
        buffers: list[float] | None = None,
        bounded: bool = False,
    ) -> None:
        self.coords = coords
        self.buffers = buffers or []
        self.bounded = bounded

    def buffer(self, distance: float) -> "FakeGeometry":
        """Return a new fake geometry recording the requested buffer distance."""
        return FakeGeometry(self.coords, self.buffers + [distance], self.bounded)

    def bounds(self) -> "FakeGeometry":
        """Return a fake square/bounding geometry."""
        return FakeGeometry(self.coords, self.buffers, True)


class FakeGeometryFactory:
    """Factory matching the ee.Geometry.Point call shape used in production."""

    @staticmethod
    def Point(coords: list[float]) -> FakeGeometry:
        """Create a fake point geometry."""
        return FakeGeometry(coords)


class FakeFeature:
    """Minimal feature object that stores geometry and properties."""

    def __init__(self, geometry: FakeGeometry, properties: dict[str, Any]) -> None:
        self.geometry = geometry
        self.properties = properties


class FakeFeatureCollection:
    """Minimal feature collection that stores generated features."""

    def __init__(self, features: list[FakeFeature]) -> None:
        self.features = features


class FakeEE:
    """Namespace-shaped Earth Engine substitute for tests."""

    Geometry = FakeGeometryFactory
    Feature = FakeFeature
    FeatureCollection = FakeFeatureCollection


class EmbeddingGeometryTests(unittest.TestCase):
    """Tests for geometry modes and geometry metadata calculations."""

    def test_area_square_geometry_computes_expected_area(self) -> None:
        row = pd.Series(
            {
                "plot_id": "P1",
                "year": 2020,
                "latitude": 40.1,
                "longitude": -93.2,
                "plot_area_m2": 400.0,
            }
        )

        geometry, metadata = build_geometry(
            row,
            {"mode": "area_square"},
            scale_meters=10,
            ee_module=FakeEE,
        )

        self.assertTrue(geometry.bounded)
        self.assertEqual(metadata["geometry_mode"], "area_square")
        self.assertEqual(metadata["geometry_shape"], "square")
        self.assertTrue(math.isclose(metadata["geometry_area_m2"], 400.0))
        self.assertTrue(math.isclose(metadata["expected_pixel_count"], 4.0))

    def test_fixed_square_geometry_computes_expected_area(self) -> None:
        row = pd.Series(
            {
                "plot_id": "P1",
                "year": 2020,
                "latitude": 40.1,
                "longitude": -93.2,
            }
        )

        geometry, metadata = build_geometry(
            row,
            {"mode": "fixed_square", "default_window_m": 50},
            scale_meters=10,
            ee_module=FakeEE,
        )

        self.assertTrue(geometry.bounded)
        self.assertEqual(metadata["geometry_mode"], "fixed_square")
        self.assertEqual(metadata["window_m"], 50.0)
        self.assertTrue(math.isclose(metadata["geometry_area_m2"], 2500.0))
        self.assertTrue(math.isclose(metadata["expected_pixel_count"], 25.0))

    def test_fixed_circle_geometry_computes_expected_area(self) -> None:
        row = pd.Series(
            {
                "plot_id": "P1",
                "year": 2020,
                "latitude": 40.1,
                "longitude": -93.2,
            }
        )

        geometry, metadata = build_geometry(
            row,
            {"mode": "fixed_circle", "default_window_m": 50},
            scale_meters=10,
            ee_module=FakeEE,
        )

        self.assertFalse(geometry.bounded)
        self.assertEqual(metadata["geometry_mode"], "fixed_circle")
        self.assertEqual(metadata["radius_m"], 50.0)
        self.assertTrue(math.isclose(metadata["geometry_area_m2"], math.pi * 50 * 50))
        self.assertTrue(math.isclose(metadata["expected_pixel_count"], math.pi * 25))

    def test_multi_scale_creates_one_feature_per_window(self) -> None:
        df = pd.DataFrame(
            {
                "plot_id": ["P1"],
                "year": [2020],
                "latitude": [40.1],
                "longitude": [-93.2],
            }
        )

        collection = build_feature_collection(
            df,
            geometry_config={
                "mode": "multi_scale",
                "shape": "square",
                "fixed_windows_m": [30, 50, 100],
            },
            scale_meters=10,
            metadata_columns=[],
            ee_module=FakeEE,
        )

        windows = [feature.properties["window_m"] for feature in collection.features]
        modes = [feature.properties["geometry_mode"] for feature in collection.features]

        self.assertEqual(windows, [30.0, 50.0, 100.0])
        self.assertEqual(modes, ["fixed_square", "fixed_square", "fixed_square"])


class EmbeddingOutputTests(unittest.TestCase):
    """Tests for Earth Engine response parsing and QC summary generation."""

    def test_parse_features_derives_actual_pixel_count_and_coverage(self) -> None:
        props = {
            "plot_id": "P1",
            "year": 2020,
            "latitude": 40.1,
            "longitude": -93.2,
            "geometry_mode": "fixed_square",
            "geometry_shape": "square",
            "window_m": 50.0,
            "radius_m": None,
            "geometry_area_m2": 2500.0,
            "plot_area_m2": None,
            "expected_pixel_count": 25.0,
            "aggregation_method": "mean_pooling",
            "Yield": 181.5,
        }
        for i in range(64):
            props[f"A{i:02d}_mean"] = float(i)
            props[f"A{i:02d}_count"] = 20 + (i % 3)

        out = parse_ee_features([{"properties": props}], metadata_columns=["Yield"])

        self.assertIn("actual_pixel_count", out.columns)
        self.assertIn("pixel_coverage_ratio", out.columns)
        self.assertEqual(float(out.loc[0, "actual_pixel_count"]), 20.0)
        self.assertTrue(math.isclose(float(out.loc[0, "pixel_coverage_ratio"]), 0.8))
        self.assertEqual(float(out.loc[0, "embedding_00"]), 0.0)
        self.assertEqual(float(out.loc[0, "embedding_63"]), 63.0)
        self.assertEqual(float(out.loc[0, "Yield"]), 181.5)

    def test_qc_summary_file_is_written(self) -> None:
        props = {
            "plot_id": "P1",
            "year": 2020,
            "latitude": 40.1,
            "longitude": -93.2,
            "geometry_mode": "fixed_square",
            "geometry_shape": "square",
            "window_m": 50.0,
            "radius_m": None,
            "geometry_area_m2": 2500.0,
            "plot_area_m2": None,
            "expected_pixel_count": 25.0,
            "aggregation_method": "mean_pooling",
            "Yield": 181.5,
        }
        for i in range(64):
            props[f"A{i:02d}_mean"] = float(i)
            props[f"A{i:02d}_count"] = 20

        result = parse_ee_features([{"properties": props}], metadata_columns=["Yield"])

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            qc_path = tmp_path / "qc.json"
            written_path = write_qc_summary(
                input_df=pd.DataFrame({"plot_id": ["P1"]}),
                result=result,
                config={
                    "metadata": {"preserve_columns": ["Yield", "Genotype"]},
                    "output": {"qc_summary_path": str(qc_path)},
                },
                metadata_columns=["Yield"],
                output_path=tmp_path / "embeddings.csv",
            )

            summary = json.loads(written_path.read_text())

        self.assertEqual(written_path, qc_path)
        self.assertEqual(summary["total_input_rows"], 1)
        self.assertEqual(summary["total_output_rows"], 1)
        self.assertEqual(summary["geometry_modes_used"], ["fixed_square"])
        self.assertEqual(summary["windows_used"], [50.0])
        self.assertEqual(summary["preserved_metadata_columns"], ["Yield"])
        self.assertEqual(summary["missing_optional_metadata_columns"], ["Genotype"])


if __name__ == "__main__":
    unittest.main()
