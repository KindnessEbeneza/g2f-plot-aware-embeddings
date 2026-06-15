# G2F Plot-Aware AlphaEarth Embeddings

This pipeline prepares coordinate-indexed agricultural plot records and extracts
annual AlphaEarth satellite embeddings from Google Earth Engine.

The upgraded pipeline supports datasets with only:

```text
plot_id or site_id
year
latitude
longitude
```

Plot area is optional. When exact plot area is unavailable, the pipeline uses a
configured coordinate-centered sampling window.

## Workflow

```bash
make install
make prepare-input
make run-embeddings
```

`make run-embeddings` requires:

```bash
export EE_PROJECT=your-google-cloud-project-id
```

## Geometry Modes

Configure geometry in `configs/plot_area.yaml`:

```yaml
geometry:
  mode: auto
  shape: square
  fixed_windows_m: [30, 50, 100]
  default_window_m: 50
  polygon_column:
```

Supported modes:

| Mode | Behavior |
| --- | --- |
| `auto` | Uses area-based square when `plot_area_m2` is available; otherwise uses `fixed_square`. |
| `area_square` | Builds a square centered on lat/lon with area equal to `plot_area_m2`. |
| `fixed_square` | Builds a square centered on lat/lon with side length `default_window_m`. |
| `fixed_circle` | Builds a circular buffer centered on lat/lon with radius `default_window_m` or `radius_m`. |
| `multi_scale` | Emits one output row per configured window in `fixed_windows_m`. |
| `polygon` | Reserved for future WKT/GeoJSON support; currently returns a clear not-implemented error. |

## Outputs

The embedding output is long format: one row per plot, year, and geometry
request. Multi-scale extraction therefore creates multiple rows per input row.

Key output columns include:

```text
plot_id
year
latitude
longitude
geometry_mode
geometry_method
geometry_shape
window_m
radius_m
geometry_area_m2
plot_area_m2
expected_pixel_count
actual_pixel_count
pixel_coverage_ratio
embedding_00 ... embedding_63
```

The pipeline also writes a JSON QC summary containing output row counts,
geometry modes, windows used, missing embedding counts, low coverage counts, and
preserved optional metadata columns.

## Tests

Run unit tests without Earth Engine credentials:

```bash
python -m unittest discover -s tests
```
