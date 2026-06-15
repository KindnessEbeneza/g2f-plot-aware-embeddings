"""Unit tests for flexible input preprocessing.

These tests verify the upgraded input contract: plot area is optional, either
plot_id or site_id is enough to identify rows, and optional modeling metadata is
preserved only when present.
"""

from __future__ import annotations

import io
import math
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

import pandas as pd
import yaml

from scripts.preprocess.prepare_plot_input import prepare_input


def run_prepare_input(config_path: Path) -> None:
    """Run preprocessing without noisy progress output in unit tests."""
    with redirect_stdout(io.StringIO()):
        prepare_input(config_path)


def write_config(tmp_path: Path, raw_path: Path, processed_path: Path, columns: dict) -> Path:
    """Write a minimal preprocessing config for a temporary test dataset."""
    config = {
        "input": {
            "raw_csv_path": str(raw_path),
            "processed_csv_path": str(processed_path),
        },
        "columns": columns,
        "preprocess": {
            "min_year": 2017,
            "max_year": 2024,
        },
        "earth_engine": {
            "scale_meters": 10,
        },
        "metadata": {
            "preserve_columns": ["Yield", "Genotype", "Block"],
        },
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config))
    return config_path


class PreprocessTests(unittest.TestCase):
    """Regression tests for the standardized plot registry builder."""

    def test_lat_lon_year_plot_id_only_passes_without_plot_area(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            raw_path = tmp_path / "raw.csv"
            processed_path = tmp_path / "processed.csv"

            pd.DataFrame(
                {
                    "plot_id": ["P1"],
                    "Year": [2020],
                    "Latitude": [40.1],
                    "Longitude": [-93.2],
                    "Yield": [181.5],
                }
            ).to_csv(raw_path, index=False)

            config_path = write_config(
                tmp_path,
                raw_path,
                processed_path,
                {
                    "plot_id": "plot_id",
                    "year": "Year",
                    "latitude": "Latitude",
                    "longitude": "Longitude",
                },
            )

            run_prepare_input(config_path)
            out = pd.read_csv(processed_path)

            self.assertEqual(str(out.loc[0, "plot_id"]), "P1")
            self.assertTrue(pd.isna(out.loc[0, "plot_area_m2"]))
            self.assertIn("Yield", out.columns)
            self.assertEqual(float(out.loc[0, "Yield"]), 181.5)

    def test_site_id_only_derives_plot_id(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            raw_path = tmp_path / "raw.csv"
            processed_path = tmp_path / "processed.csv"

            pd.DataFrame(
                {
                    "Site": ["S1"],
                    "Year": [2020],
                    "Latitude": [40.1],
                    "Longitude": [-93.2],
                }
            ).to_csv(raw_path, index=False)

            config_path = write_config(
                tmp_path,
                raw_path,
                processed_path,
                {
                    "site_id": "Site",
                    "year": "Year",
                    "latitude": "Latitude",
                    "longitude": "Longitude",
                },
            )

            run_prepare_input(config_path)
            out = pd.read_csv(processed_path)

            self.assertEqual(str(out.loc[0, "site_id"]), "S1")
            self.assertEqual(str(out.loc[0, "plot_id"]), "S1_2020_row1")

    def test_plot_area_km2_converts_to_m2(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            raw_path = tmp_path / "raw.csv"
            processed_path = tmp_path / "processed.csv"

            pd.DataFrame(
                {
                    "Site": ["S1"],
                    "Plot": [7],
                    "Year": [2020],
                    "Latitude": [40.1],
                    "Longitude": [-93.2],
                    "Plot Area": [0.001],
                }
            ).to_csv(raw_path, index=False)

            config_path = write_config(
                tmp_path,
                raw_path,
                processed_path,
                {
                    "site_id": "Site",
                    "plot_number": "Plot",
                    "year": "Year",
                    "latitude": "Latitude",
                    "longitude": "Longitude",
                    "plot_area_km2": "Plot Area",
                },
            )

            run_prepare_input(config_path)
            out = pd.read_csv(processed_path)

            self.assertEqual(str(out.loc[0, "plot_id"]), "S1_2020_7")
            self.assertTrue(math.isclose(float(out.loc[0, "plot_area_m2"]), 1000.0))
            self.assertTrue(math.isclose(float(out.loc[0, "expected_pixel_count"]), 10.0))

    def test_missing_optional_metadata_columns_do_not_fail(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            raw_path = tmp_path / "raw.csv"
            processed_path = tmp_path / "processed.csv"

            pd.DataFrame(
                {
                    "plot_id": ["P1"],
                    "Year": [2020],
                    "Latitude": [40.1],
                    "Longitude": [-93.2],
                }
            ).to_csv(raw_path, index=False)

            config_path = write_config(
                tmp_path,
                raw_path,
                processed_path,
                {
                    "plot_id": "plot_id",
                    "year": "Year",
                    "latitude": "Latitude",
                    "longitude": "Longitude",
                },
            )

            run_prepare_input(config_path)
            out = pd.read_csv(processed_path)

            self.assertNotIn("Yield", out.columns)
            self.assertEqual(len(out), 1)


if __name__ == "__main__":
    unittest.main()
