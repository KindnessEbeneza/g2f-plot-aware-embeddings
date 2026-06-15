# G2F Plot-Aware AlphaEarth Embeddings

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-3.10+-green.svg)
![Earth Engine API](https://img.shields.io/badge/Earth%20Engine-v1-orange.svg)

Unified satellite embedding extraction pipeline for agricultural crop research. Combines coordinate-indexed plot records with Google Earth Engine's **AlphaEarth satellite embeddings** (64-dimensional learned representations) to enable ML-powered crop yield prediction and trait modeling across diverse datasets.

**Key capability**: Works with minimal input (plot coordinates + year), gracefully handles optional plot area data, and supports 4 geometry sampling strategies. Production-ready with comprehensive quality control metrics.

---

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Quick Start](#quick-start)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Configuration](#configuration)
- [Project Structure](#project-structure)
- [Architecture](#architecture)
- [Usage Examples](#usage-examples)
- [Output Schema](#output-schema)
- [Testing](#testing)
- [Troubleshooting](#troubleshooting)
- [Tech Stack](#tech-stack)
- [Contributing](#contributing)
- [AI-Agent Compatibility](#ai-agent-compatibility)
- [References](#references)
- [License](#license)

---

## Overview

### Problem

Agricultural yield prediction and crop trait modeling require environmental context. Satellite imagery provides this data at scale, but:

- Manual feature engineering (NDVI, soil indices) is labor-intensive and brittle
- Different datasets use different coordinate/geometry standards
- Existing pipelines are dataset-specific, not reusable
- Quality assurance metrics are often missing

### Solution

This pipeline extracts **pre-trained satellite embeddings** from Google's AlphaEarth collection:

- **64-dimensional learned representations** of Earth's surface (bands A00–A63)
- **Pre-trained by Google** on multi-year multispectral imagery (no manual feature engineering)
- **Captures relevant patterns**: vegetation, soil, infrastructure, crop growth signals
- **10m native resolution**: 100m² per pixel (configurable aggregation strategy)

### Use Cases

- **Yield prediction**: Correlate embeddings with phenotypic yields to identify environmental drivers
- **Trait modeling**: Link satellite context to genotype effects
- **Environmental impact**: Quantify soil/vegetation/water dynamics
- **Cross-dataset research**: Adapt pipeline to global wheat, rice, or legume trials

### Why It Works

Mean pooling aggregates pixel-level embeddings within plot geometry, producing a single 64-dimensional vector per plot-year. This captures spatial heterogeneity while remaining computationally efficient. See [PIPELINE_ANALYSIS_REPORT.md](PIPELINE_ANALYSIS_REPORT.md) for mathematical details and comparison to alternatives.

---

## Features

- **Flexible input**: Works with minimal schema (plot_id or site_id, year, latitude, longitude; plot area optional)
- **4 geometry modes**: Area-based squares, fixed windows, circular buffers, or multi-scale sampling
- **Multi-scale support**: Extract embeddings for multiple window sizes in a single pass
- **Backward compatible**: Supports existing G2F format out-of-the-box
- **Dataset-agnostic**: Column mapping via config; graceful fallbacks to standard names
- **Quality control**: Pixel coverage ratios, pixel count tracking, JSON QC summaries
- **Production-ready**: Configuration-driven behavior, comprehensive error handling, tested with multiple datasets

| Geometry Mode | Use Case | Example |
|---|---|---|
| `auto` | Smart default | Use plot area when available; fall back to fixed window |
| `area_square` | Exact plot geometry | Plot area = 1 hectare → 100m × 100m square |
| `fixed_square` | Uniform sampling | All plots sampled at 50m window |
| `fixed_circle` | Alternative geometry | Circular buffer (useful for some trial designs) |
| `multi_scale` | Research flexibility | Extract at 30m, 50m, 100m windows (3 output rows per input) |

---

## Quick Start

### 1. Clone and Install

```bash
git clone https://github.com/your-org/g2f-plot-aware-embeddings.git
cd g2f-plot-aware-embeddings
pip install -e .
```

### 2. Set Up Google Earth Engine Authentication

```bash
# Authenticate (opens browser)
gcloud auth application-default login

# Create Google Cloud Project and enable Earth Engine API
export EE_PROJECT=your-google-cloud-project-id
```

### 3. Prepare Your Data

Place your dataset at `data/raw/your_dataset.csv` with columns:

```csv
plot_id,year,latitude,longitude,plot_area_km2,Yield
P1,2020,40.1,-93.2,0.001,181.5
P2,2020,40.2,-93.3,,150.2
```

### 4. Configure and Run

```bash
# Copy config template (or customize existing)
cp configs/plot_area.yaml configs/my_dataset.yaml

# Edit config to map your columns (see Configuration section)
nano configs/my_dataset.yaml

# Run pipeline
make install
python scripts/preprocess/prepare_plot_input.py --config configs/my_dataset.yaml
python scripts/run_plot_area_embeddings.py --config configs/my_dataset.yaml

# Or use convenience target
make run-clean
```

### 5. Check Output

```bash
# View extracted embeddings
head -5 data/outputs/plot_area_embeddings.csv

# Inspect quality control summary
cat data/outputs/plot_area_embeddings_qc_summary.json
```

Expected output files:
- `plot_area_embeddings.csv`: One row per plot-year-geometry with 64 embeddings
- `plot_area_summary.csv`: Geometry and pixel count statistics
- `plot_area_embeddings_qc_summary.json`: Quality metrics (coverage ratios, mode distribution)

---

## Prerequisites

### System Requirements

- **Python**: 3.10 or later
- **OS**: Linux, macOS, or WSL (Windows Subsystem for Linux)
- **Disk**: ≥10 GB for intermediate data + outputs

### Google Cloud Setup

1. **Create a Google Cloud Project**:
   ```bash
   gcloud projects create your-project-id
   gcloud config set project your-project-id
   ```

2. **Enable Earth Engine API**:
   ```bash
   gcloud services enable earthengine.googleapis.com
   ```

3. **Authenticate**:
   ```bash
   gcloud auth application-default login
   export EE_PROJECT=your-project-id
   ```

4. **Verify**:
   ```bash
   python -c "import ee; ee.Authenticate(); print('✓ Earth Engine API ready')"
   ```

### Dataset Requirements

Minimum schema:
- `latitude` (float, range [-90, 90])
- `longitude` (float, range [-180, 180])
- `year` (integer, range [2017–2024])
- `plot_id` **OR** `site_id` (string; plot_id auto-derived if only site_id provided)

Optional:
- `plot_area_km2` or `plot_area_m2` (float, > 0; used for area-based geometry)
- Metadata columns (Yield, Genotype, Treatment, Block, etc.; preserved in output)

---

## Installation

### Via pip

```bash
pip install -e .
```

This installs dependencies from `pyproject.toml`:
- `pandas` — Data manipulation
- `pyyaml` — Configuration management
- `ee` — Google Earth Engine Python API

### Verify Installation

```bash
python scripts/preprocess/prepare_plot_input.py --help
python scripts/run_plot_area_embeddings.py --help
python -m unittest discover -s tests -v
```

All tests should pass without Earth Engine credentials.

---

## Configuration

All behavior is controlled via a single YAML config file (`configs/plot_area.yaml`).

### Configuration Sections

#### `input`
Specify raw and processed dataset paths.

```yaml
input:
  raw_csv_path: data/raw/merged_pheno.csv
  processed_csv_path: data/processed/plots.csv
```

#### `columns`
Map your dataset column names to standardized names. Fallback to auto-detection if not specified.

```yaml
columns:
  plot_id:                    # Your plot ID column name (leave blank to auto-detect)
  site_id: "Field Location"   # Site identifier (used if plot_id not available)
  plot_number: "Plot"         # Plot number within site (used for plot_id derivation)
  year: "Year"                # Year column
  latitude: "Latitude"        # Latitude (degrees, -90 to +90)
  longitude: "Longitude"      # Longitude (degrees, -180 to +180)
  plot_area_km2: "Plot Area"  # Plot area in km²
  plot_area_m2:               # Plot area in m² (alternative)
```

**Fallback strategy**: If a mapped column doesn't exist, the pipeline automatically looks for standard names (plot_id, site_id, year, latitude, longitude, etc.). This enables config-less usage with standard datasets.

#### `preprocess`
Data filtering and standardization options.

```yaml
preprocess:
  min_year: 2017              # Exclude records before this year
  max_year: 2024              # Exclude records after this year
  area_unit: km2              # Unit of plot_area columns (km2 or m2)
  aggregate_level: site_year  # [Reserved for future] Aggregation strategy
```

#### `geometry`
Sampling strategy and window configuration.

```yaml
geometry:
  mode: auto                  # One of: auto | area_square | fixed_square | fixed_circle | polygon (future)
  shape: square               # Geometry shape (square or circle; polygon reserved for future)
  fixed_windows_m: [30, 50, 100]   # For multi-scale: emit one row per window size
  default_window_m: 50        # Default sampling window (side length or radius in meters)
  polygon_column:             # [Reserved] Column containing WKT/GeoJSON geometries
```

**Geometry Mode Details**:

| Mode | Description | When to Use |
|------|---|---|
| `auto` | Smart selection: uses plot area if available; falls back to `fixed_square` | Most datasets (default) |
| `area_square` | Square centered on lat/lon with area = `plot_area_m2` | Exact plot geometry available |
| `fixed_square` | Coordinate-centered square with side = `default_window_m` | Uniform sampling or area unavailable |
| `fixed_circle` | Circular buffer; radius = `default_window_m` or `radius_m` | Alternative geometry preference |
| `multi_scale` | [Special] Emit multiple rows per input (one per window in `fixed_windows_m`) | Research/sensitivity analysis |
| `polygon` | [Future] Support WKT/GeoJSON geometries from `polygon_column` | Custom field boundaries (not yet implemented) |

#### `metadata`
Columns to preserve from input to output (e.g., Yield, Genotype, Treatment).

```yaml
metadata:
  preserve_columns:
    - Yield
    - Genotype
    - Hybrid
    - Treatment
    - Replicate
    - Block
```

#### `earth_engine`
Google Earth Engine API configuration.

```yaml
earth_engine:
  project_env: EE_PROJECT     # Environment variable containing your GCP project ID
  collection: GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL  # AlphaEarth collection ID
  scale_meters: 10            # Resolution of AlphaEarth embeddings (10m native)
```

#### `output`
Output file paths (relative to project root).

```yaml
output:
  embeddings_path: data/outputs/plot_area_embeddings.csv
  report_path: data/outputs/plot_area_summary.csv
  qc_summary_path: data/outputs/plot_area_embeddings_qc_summary.json
```

### Example Configurations

#### Using G2F Dataset

```yaml
columns:
  site_id: "Field Location"
  plot_number: "Plot"
  year: "Year"
  latitude: "Latitude"
  longitude: "Longitude"
  plot_area_km2: "Plot Area"

geometry:
  mode: auto
  fixed_windows_m: [50]
  default_window_m: 50

metadata:
  preserve_columns: [Yield, Genotype, Hybrid, Treatment, Replicate, Block]
```

#### Using Custom CSV (Minimal)

```yaml
columns:
  plot_id: "id"
  year: "yr"
  latitude: "lat"
  longitude: "lon"

geometry:
  mode: fixed_square
  default_window_m: 100

metadata:
  preserve_columns: []
```

---

## Project Structure

```
g2f-plot-aware-embeddings/
├── Makefile                          # Convenience targets (install, prepare-input, run-embeddings)
├── pyproject.toml                    # Package metadata and dependencies
├── README.md                         # This file
├── PIPELINE_ANALYSIS_REPORT.md       # Detailed analysis (mean pooling, research alignment)
├── configs/
│   └── plot_area.yaml                # Main configuration file
├── data/
│   ├── raw/                          # Input CSV files
│   ├── processed/                    # Standardized plots.csv (intermediate)
│   └── outputs/                      # Final embeddings + QC reports
├── scripts/
│   ├── preprocess/
│   │   └── prepare_plot_input.py     # Input adapter (CSV → standardized plots)
│   └── run_plot_area_embeddings.py   # Main pipeline (geometry + EE extraction)
├── src/
│   └── plot_embeddings/
│       └── __init__.py               # Package initialization
└── tests/
    └── test_preprocess.py            # Unit tests (input handling, area conversion, ID derivation)
```

### Key Modules

#### `prepare_plot_input.py`
**Purpose**: Standardize diverse CSV inputs to a common schema.

**Key functions**:
- `configured_source_column()` — Map config column names with fallback to standard names
- `validate_columns()` — Ensure minimum required schema
- `area_column()` — Detect and normalize plot area (km² → m²)
- `create_plot_ids()` — Derive plot_id from site_id + year + plot_number if needed
- `prepare_input()` — Main orchestrator (load config, validate, coerce types, filter years, normalize area, preserve metadata)

#### `run_plot_area_embeddings.py`
**Purpose**: Extract satellite embeddings from Google Earth Engine.

**Key functions**:
- `initialize_earth_engine()` — Authenticate using EE_PROJECT env var
- `resolve_geometry_mode()` — Map config mode (auto/area_square/fixed_square/fixed_circle) to concrete strategy
- `build_geometry_requests()` — Construct EE geometries (supports multi-scale)
- `build_feature_collection()` — Create EE FeatureCollection with plot properties
- `extract_embeddings_for_year()` — Query AlphaEarth collection, apply mean + count reducer
- `parse_ee_features()` — Convert EE output to DataFrame, rename bands (Axx_mean → embedding_xx), compute pixel coverage
- `run()` — Main orchestrator (load config, initialize EE, process by year, write outputs)

---

## Architecture

### Data Flow

```
Input CSV
   ↓
[Preprocess: prepare_plot_input.py]
   ├─ Validate columns & data types
   ├─ Normalize area (km² → m²)
   ├─ Derive plot IDs (if needed)
   └─ Output: standardized plots.csv
   ↓
[Geometry Setup: build_geometry_requests()]
   ├─ Resolve geometry mode (auto/area_square/fixed_square/fixed_circle)
   ├─ Build EE geometries (support multi-scale)
   └─ Create EE FeatureCollection
   ↓
[Earth Engine Extraction: extract_embeddings_for_year()]
   ├─ Query GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL (annual AlphaEarth mosaics)
   ├─ Filter by date (year-01-01 to year-12-31)
   ├─ Select bands A00–A63 (64-dimensional embeddings)
   ├─ Apply combined mean + count reducer
   └─ Return: {A00_mean, A01_mean, ..., A63_mean, count}
   ↓
[Parse & QC: parse_ee_features()]
   ├─ Rename bands (A00_mean → embedding_00, etc.)
   ├─ Compute pixel_coverage_ratio = actual_pixel_count / expected_pixel_count
   ├─ Preserve metadata (Yield, Genotype, etc.)
   └─ Output row: [plot_id, year, embedding_00, ..., embedding_63, pixel_coverage_ratio, metadata...]
   ↓
Output Files
   ├─ plot_area_embeddings.csv     (main output: one row per plot-year-geometry)
   ├─ plot_area_summary.csv        (geometry statistics)
   └─ plot_area_embeddings_qc_summary.json  (quality metrics)
```

### Mean Pooling Strategy

**Why mean pooling?**
- Captures spatial heterogeneity within plot (averages out local noise)
- Single vector per plot-year (efficient for downstream ML)
- Robust to pixel-level artifacts (cloud shadows, sensor noise)
- Empirically effective for yield prediction

**Formula**: For each embedding band b ∈ {0, 1, ..., 63}:

```
embedding[b] = mean( pixel_value[b, i] for all pixels i in plot geometry )
```

Coupled with count reduction to track pixel coverage:

```
pixel_coverage_ratio = actual_pixel_count / expected_pixel_count
```

This enables quality filtering (e.g., exclude rows with coverage < 0.5).

See [PIPELINE_ANALYSIS_REPORT.md](PIPELINE_ANALYSIS_REPORT.md) for comparison to alternatives (max, min, median pooling).

---

## Usage Examples

### Example 1: Use Default Configuration (G2F-Compatible)

```bash
# Prepare input (assumes columns match standard G2F names)
python scripts/preprocess/prepare_plot_input.py --config configs/plot_area.yaml

# Run embeddings extraction
python scripts/run_plot_area_embeddings.py --config configs/plot_area.yaml

# Check results
head -5 data/outputs/plot_area_embeddings.csv
cat data/outputs/plot_area_embeddings_qc_summary.json | python -m json.tool
```

### Example 2: Use Custom Dataset with YAML Override

Create `configs/my_wheat_dataset.yaml`:

```yaml
input:
  raw_csv_path: data/raw/wheat_trials.csv
  processed_csv_path: data/processed/wheat_plots.csv

columns:
  site_id: "trial_id"
  plot_number: "block_num"
  year: "season"
  latitude: "lat_dd"
  longitude: "lon_dd"
  plot_area_m2: "area_m2"

geometry:
  mode: area_square
  default_window_m: 50

metadata:
  preserve_columns: [grain_yield_kg_ha, variety, irrigation]

output:
  embeddings_path: data/outputs/wheat_embeddings.csv
  report_path: data/outputs/wheat_summary.csv
  qc_summary_path: data/outputs/wheat_qc_summary.json
```

Then run:

```bash
python scripts/preprocess/prepare_plot_input.py --config configs/my_wheat_dataset.yaml
python scripts/run_plot_area_embeddings.py --config configs/my_wheat_dataset.yaml
```

### Example 3: Multi-Scale Extraction

Configure in `configs/plot_area.yaml`:

```yaml
geometry:
  mode: fixed_square
  fixed_windows_m: [30, 50, 100]
  default_window_m: 50
```

This produces 3 rows per input plot:

```csv
plot_id,year,latitude,longitude,window_m,embedding_00,embedding_01,...,embedding_63,pixel_coverage_ratio
P1,2020,40.1,-93.2,30,0.123,-0.456,...,0.789,0.95
P1,2020,40.1,-93.2,50,0.120,-0.450,...,0.785,0.98
P1,2020,40.1,-93.2,100,0.118,-0.448,...,0.780,0.99
```

Use for sensitivity analysis or to capture multi-scale environmental features.

---

## Output Schema

### Main Output: `plot_area_embeddings.csv`

One row per plot-year-geometry combination.

| Column | Type | Description |
|--------|------|---|
| `plot_id` | string | Unique plot identifier |
| `site_id` | string | Site/trial identifier (if available) |
| `plot_number` | string | Plot number within site |
| `year` | integer | Year of observation |
| `latitude` | float | Geographic latitude (degrees, -90 to +90) |
| `longitude` | float | Geographic longitude (degrees, -180 to +180) |
| `geometry_method` | string | Method used to construct geometry (e.g., "area_square", "fixed_square") |
| `geometry_mode` | string | Config mode (e.g., "auto", "fixed_square") |
| `geometry_shape` | string | Shape of sampled geometry (square or circle) |
| `window_m` | float | Side length of square or radius of circle (meters) |
| `radius_m` | float | Radius for circular geometries (meters) |
| `geometry_area_m2` | float | Computed area of geometry in m² |
| `plot_area_m2` | float | Input plot area in m² (if provided) |
| `expected_pixel_count` | integer | Expected number of pixels (geometry_area / pixel_area) |
| `actual_pixel_count` | integer | Actual number of pixels with valid data |
| `pixel_coverage_ratio` | float | Ratio of actual to expected pixels (0.0–1.0) |
| `aggregation_method` | string | Aggregation strategy (e.g., "mean_pooling") |
| `embedding_00` to `embedding_63` | float | 64-dimensional satellite embeddings (AlphaEarth bands A00–A63) |
| `[Yield]`, `[Genotype]`, ... | float/string | Optional metadata columns (as configured) |

### Secondary Output: `plot_area_summary.csv`

Geometry and pixel statistics (one row per plot-year-geometry):

```csv
plot_id,year,geometry_method,window_m,geometry_area_m2,expected_pixel_count,actual_pixel_count,pixel_coverage_ratio
```

### QC Summary: `plot_area_embeddings_qc_summary.json`

Quality control metrics:

```json
{
  "total_input_rows": 1000,
  "total_output_rows": 1200,
  "rows_with_missing_embeddings": 5,
  "rows_with_low_pixel_coverage": 12,
  "geometry_mode_distribution": {
    "area_square": 800,
    "fixed_square": 400
  },
  "window_size_distribution": {
    "50": 1200
  },
  "embedded_metadata_columns": ["Yield", "Genotype", "Treatment"],
  "pixel_coverage_statistics": {
    "mean": 0.94,
    "min": 0.52,
    "max": 1.0
  }
}
```

### Understanding Pixel Coverage

- **expected_pixel_count**: Theoretical number of AlphaEarth pixels within geometry (area ÷ 100m²)
- **actual_pixel_count**: Pixels with valid data in AlphaEarth collection (may be less due to clouds, missing years, etc.)
- **pixel_coverage_ratio**: actual / expected
  - ✅ **> 0.5**: Good coverage, embedding is reliable
  - ⚠️ **0.2–0.5**: Moderate coverage, use caution
  - 🔴 **< 0.2**: Poor coverage, consider filtering out

---

## Testing

### Unit Tests (No Earth Engine Credentials Required)

```bash
python -m unittest discover -s tests -v
```

**Tests included**:

1. `test_lat_lon_year_plot_id_only_passes_without_plot_area` — Validates minimal input (lat/lon/year/plot_id; no area)
2. `test_site_id_only_derives_plot_id` — Auto-derivation of plot_id from site_id + year + plot_number
3. `test_plot_area_km2_converts_to_m2` — Area unit conversion (0.001 km² → 1000.0 m²)
4. `test_missing_optional_metadata_columns_do_not_fail` — Graceful handling when metadata columns absent

All tests pass without Earth Engine API access or credentials.

### Integration Testing (Requires EE API & GCP Project)

After installing Earth Engine credentials:

```bash
# Test with small sample dataset
python scripts/preprocess/prepare_plot_input.py --config configs/plot_area.yaml
python scripts/run_plot_area_embeddings.py --config configs/plot_area.yaml

# Inspect first few rows
head -5 data/outputs/plot_area_embeddings.csv

# Check QC metrics
python -c "import json; d = json.load(open('data/outputs/plot_area_embeddings_qc_summary.json')); print(f\"Coverage: {d['pixel_coverage_statistics']['mean']:.2%}\")"
```

---

## Troubleshooting

### Earth Engine Authentication Issues

**Error**: `"ee.EEException: Authorization failed"`

**Solution**:
```bash
gcloud auth application-default login
export EE_PROJECT=your-google-cloud-project-id
python -c "import ee; ee.Initialize(); print('✓ OK')"
```

### "Invalid coordinates" or "No data found"

**Causes**:
- Coordinates outside Earth Engine collection bounds (typically ocean/Antarctica)
- Plot centroid far from actual field boundaries
- Incorrect coordinate order (should be latitude, longitude)

**Solution**:
```bash
# Check coordinate ranges in your data
python -c "import pandas as pd; df = pd.read_csv('data/processed/plots.csv'); print(df[['latitude', 'longitude']].describe())"

# Ensure: -90 ≤ latitude ≤ 90, -180 ≤ longitude ≤ 180
```

### Low Pixel Coverage Ratio

**Cause**: Few valid pixels in AlphaEarth collection for your plot area/year.

**Common reasons**:
- Plot area very small (< 10 pixels expected)
- Year before 2015 (AlphaEarth available 2015+; configure `min_year`)
- Cloud cover or missing imagery for region/year

**Solution**:
```bash
# Filter by coverage ratio in downstream analysis
import pandas as pd
df = pd.read_csv('data/outputs/plot_area_embeddings.csv')
df_filtered = df[df['pixel_coverage_ratio'] >= 0.5]
print(f"Retained {len(df_filtered)}/{len(df)} rows (coverage ≥ 50%)")

# Or increase default_window_m to capture more pixels
```

### Memory Issues with Large Datasets

**Symptoms**: "MemoryError" or slow extraction

**Solutions**:
1. **Batch by year**: Modify `run()` to process 1–2 years at a time
2. **Reduce windows**: Fewer windows in `fixed_windows_m`
3. **Subsample**: Filter to subset of sites/plots before extraction

---

## Tech Stack

### Core Dependencies

| Component | Package | Version | Purpose |
|-----------|---------|---------|---------|
| **Data manipulation** | pandas | ≥1.0 | DataFrames, CSV I/O |
| **Configuration** | PyYAML | ≥5.1 | YAML config parsing |
| **Geospatial** | ee (Earth Engine API) | latest | Google Earth Engine access |
| **Runtime** | Python | ≥3.10 | Type hints, language features |

### Optional

- `docker` — Containerized deployment
- `pytest` — Alternative test runner

### Development

- `black` — Code formatting (recommended)
- `mypy` — Static type checking (recommended)

See `pyproject.toml` for complete dependency list.

---

## Contributing

We welcome contributions! Areas for collaboration:

- **Dataset adapters**: Add support for new crop trials (Wheat Initiative, Legume research, etc.)
- **Geometry modes**: Implement polygon support (WKT/GeoJSON)
- **Quality metrics**: Add anomaly detection, value range validation
- **Deployment**: Docker containerization, cloud infrastructure

### Getting Started

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/my-feature`
3. Make changes following PEP 8 guidelines
4. Add/update tests: `python -m unittest discover -s tests`
5. Open a pull request with description

### Before Submitting

- ✅ All tests pass
- ✅ Code follows PEP 8
- ✅ New functions include docstrings
- ✅ Configuration changes documented

See [CONTRIBUTING.md](CONTRIBUTING.md) (forthcoming) for detailed guidelines.

---

## AI-Agent Compatibility

This section helps AI coding assistants (Copilot, Claude, etc.) understand the codebase structure and conventions.

### Entry Points

- **Preprocessing**: `scripts/preprocess/prepare_plot_input.py` (input validation, column mapping, plot ID derivation)
- **Extraction**: `scripts/run_plot_area_embeddings.py` (geometry construction, Earth Engine queries, QC metrics)
- **Configuration**: `configs/plot_area.yaml` (single source of truth for all behavior)

### Key Concepts

1. **Configuration-Driven Design**: All behavior controlled via YAML. No hardcoded parameters in code.
2. **Geometry Abstraction**: Four geometry modes (auto, area_square, fixed_square, fixed_circle) provide flexibility without code changes.
3. **Lazy Earth Engine Imports**: `ee` module imported only when needed (enables unit testing without credentials).
4. **Column Mapping Strategy**: Fallback chain (explicit mapping → standard names → None) handles diverse CSV formats.
5. **Multi-Scale Support**: Single input can produce multiple output rows (one per window size).

### Code Organization

| Module | Responsibility |
|--------|---|
| `prepare_plot_input.py` | Input adapter: CSV → standardized plots |
| `run_plot_area_embeddings.py` | Pipeline orchestrator: geometry + EE queries |
| `test_preprocess.py` | Unit tests (input scenarios, area conversion, ID derivation) |
| `plot_area.yaml` | Configuration: all adjustable parameters |

### Common Tasks

**Add support for new dataset**:
1. Edit `configs/plot_area.yaml`: Add column mappings in `columns:` section
2. Run `prepare_plot_input.py` to standardize input
3. Run `run_plot_area_embeddings.py` to extract embeddings

**Add new geometry mode**:
1. Implement builder function in `run_plot_area_embeddings.py` (e.g., `ellipse_from_lat_lon()`)
2. Add case to `resolve_geometry_mode()` to route mode to builder
3. Add case to `build_feature_collection()` to handle new properties

**Debug QC metrics**:
1. Check `pixel_coverage_statistics` in `plot_area_embeddings_qc_summary.json`
2. For low coverage: inspect expected vs actual pixel counts
3. Verify year/region have AlphaEarth data (post-2015, not ocean/Antarctica)

### Testing Strategy

- **Unit tests** (`test_preprocess.py`): Input validation, column mapping, area conversion — **no EE credentials needed**
- **Integration tests**: Earth Engine queries — **requires EE_PROJECT env var**
- **Manual validation**: Compare output embeddings against Earth Engine console queries

---

## References

### Documentation

- **[PIPELINE_ANALYSIS_REPORT.md](PIPELINE_ANALYSIS_REPORT.md)** — Comprehensive technical analysis: mean pooling theory, research alignment, architecture details
- **Google Earth Engine API** — https://developers.google.com/earth-engine
- **AlphaEarth Collection** — https://developers.google.com/earth-engine/datasets/catalog/GOOGLE_SATELLITE_EMBEDDING_V1_ANNUAL

### Related Projects

- **[G2F Data Repository](https://www.genomes2fields.org/)** — Global corn phenotyping initiative
- **[CGIAR Wheat Initiative](https://www.cgiar.org/)** — International wheat research program
- **[Google Earth Engine](https://earthengine.google.com/)** — Planetary-scale geospatial analysis platform

### Key Papers

- AlphaEarth satellite embeddings (citation TBD)
- Crop yield prediction with remote sensing (citation TBD)

---

## License

This project is licensed under the **MIT License** — see [LICENSE](LICENSE) for details.

Permission to use, modify, and distribute this software is granted under the terms of the MIT License.

---

## Acknowledgments

- **Google Earth Engine** for AlphaEarth satellite embeddings and compute infrastructure
- **G2F Consortium** for field phenotype data and research context
- **Contributors**: [Your names here]

---

## Citation

If you use this pipeline in your research, please cite:

```bibtex
@software{g2f_embeddings_2024,
  title={G2F Plot-Aware AlphaEarth Embeddings},
  author={[Your names]},
  year={2024},
  url={https://github.com/your-org/g2f-plot-aware-embeddings}
}
```

---

**Last Updated**: June 2024  
**Maintainer**: [Your contact info]
