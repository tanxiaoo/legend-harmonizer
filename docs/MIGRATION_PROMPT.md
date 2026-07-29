# Migration task: land cover viewer → COG + catalog + dynamic tiles

## Context

This repo contains a web application for viewing local land cover raster datasets.
The data is heterogeneous: some products are single-file mosaics, others are
directories of tiles. Products differ in CRS, resolution, dtype, class codes, and
legend format. The data lives under `data`.

The current implementation `<FILL IN: describe it briefly — e.g. "pre-renders XYZ
tiles with gdal2tiles and serves them statically, with layer definitions hardcoded
in the frontend">`.

I want to migrate to an architecture where datasets are discovered automatically,
served as dynamic tiles, and styled at render time from legend documents rather
than baked-in colors.

## Before you write any code

1. Read the existing codebase and tell me what's there: how rasters are currently
   located, converted, served, and styled; where layer definitions live; what the
   frontend stack is.
2. Inspect a representative sample of the actual data files (do not guess). For each,
   report: driver, CRS, resolution, dtype, nodata, block size, whether internal
   overviews exist, whether a color table or raster attribute table is present, and
   what sidecar files sit next to it.
3. Produce a written migration plan with ordered, independently shippable stages and
   an estimate of what breaks at each stage. Stop and wait for my approval before
   implementing. Do not start coding until I approve the plan.

## Target architecture

Four components, in this order of implementation:

**1. COG conversion (`tools/to_cog.py`)**
- Convert single-file mosaics to COG: `-of COG`, DEFLATE + PREDICTOR=2, BLOCKSIZE=512,
  `OVERVIEW_RESAMPLING=MODE`.
- For tile sets: build a VRT and convert to one COG when the result is manageable;
  otherwise convert tiles individually and emit a MosaicJSON.
- Idempotent: skip files that are already valid COGs. Verify with `rio cogeo validate`.
- Never modify files in the source data directory. Write to a separate `cog/` tree
  and keep a mapping back to the original path.

**2. Indexer (`tools/index.py`)**
- Walk the data root, probe every raster with rasterio, and group files into datasets.
  Grouping rule: same parent directory + same CRS + same resolution + same dtype +
  non-overlapping bounds = one tile set. Everything else is standalone.
- For each dataset emit: id, title, source paths, COG/MosaicJSON path, CRS, resolution,
  dtype, nodata, bounds in EPSG:4326, native bounds, overview levels, and a class
  histogram computed from the coarsest overview.
- Extract the legend by trying, in this order, and recording which succeeded:
  embedded color table → raster attribute table (`.aux.xml`, `.vat.dbf`) →
  sidecar style files in the same folder (`.qml`, `.sld`, `.clr`, `.txt`) →
  fallback to `np.unique()` on the coarsest overview with null labels.
- Write the result as a catalog. Use a static STAC catalog via `pystac` if you judge
  the dataset count justifies it; otherwise a single `catalog.json`. Justify the choice
  in the plan.
- Support incremental reindexing — re-probing unchanged files on every run is not
  acceptable once the collection is large.

**3. Tile server (`server/`)**
- FastAPI app mounting `titiler.core.factory.TilerFactory` and
  `titiler.mosaic.factory.MosaicTilerFactory`.
- Additional endpoints:
  - `GET /catalog` — dataset list with bounds, status flags, and thumbnails.
  - `GET /legend/{dataset_id}` — the legend document, standalone so notebooks and
    QGIS can consume the same source of truth.
  - `GET /crosswalk/{dataset_id}` — native value → target value mapping.
  - `POST /reindex` — re-run the indexer.
  - `GET /inspect?lon=&lat=&datasets=` — pixel value from every requested dataset in
    one call, resolved to class labels in both native and target legends.
- Restrict readable paths to the configured data root. Reject arbitrary `url=` params
  pointing outside it.

**4. Frontend**
- Keep the existing framework unless there is a concrete reason to change it; tell me
  first if you think there is.
- Build the layer list from `/catalog` at runtime. No hardcoded dataset definitions
  anywhere in the frontend.
- Legend panel per dataset with a native / harmonized toggle. Class rows are clickable
  to isolate or hide a class — implement by rebuilding the discrete colormap with
  alpha 0 for hidden classes and re-requesting tiles, not by any server-side change.
- Opacity slider per layer and a swipe-compare control between any two layers.
- Click-to-inspect: calls `/inspect` and shows the class from every visible dataset in
  one panel.

## Data model

Legend document, one per dataset **version**:

```json
{
  "id": "esa_worldcover_v200_2021",
  "title": "ESA WorldCover 2021 (v200)",
  "nodata": 0,
  "source": "rat",
  "status": "complete",
  "classes": [
    {
      "value": 10,
      "label": "Tree cover",
      "color_official": "#006400",
      "color_display": "#006400",
      "definition": "Vegetation >5 m, canopy cover >10%",
      "pixel_count": 184203
    }
  ]
}
```

Crosswalk, keyed to a single pivot legend (not pairwise between products):

```json
{
  "dataset_id": "esa_worldcover_v200_2021",
  "target_legend": "pivot_v1",
  "mappings": [
    { "native_value": 10, "target_value": 1, "probability": 0.97, "entropy": 0.11 },
    { "native_value": 20, "target_value": 3, "probability": 0.52, "entropy": 0.94 }
  ],
  "entropy_threshold": 0.7
}
```

Every dataset maps to the pivot legend, never directly to another dataset. Adding a
new product must not require touching any existing crosswalk.

## Hard constraints

These are categorical rasters. Violating any of these silently corrupts the output:

- Never use bilinear, cubic, or average resampling on class codes. `NEAREST` for
  warping, `MODE` for overview generation. Check every code path, including any the
  frontend triggers.
- Never bake RGB into the raster. Class codes stay in the pixels; color is applied at
  render time from the legend document.
- `nodata` must be explicit in every COG, otherwise 0 renders as a real class.
- Legends are keyed by dataset **and version**. Do not let two versions of the same
  product share a legend id.
- Datasets whose class codes could not be labeled must be flagged in the catalog and
  visibly marked in the UI, not silently rendered with invented labels.
- Classes above the entropy threshold render in the harmonized view as a distinct
  "needs review" appearance, never collapsed into their most likely target class.

## Definition of done

- I can drop a new folder of tiles into the data root, call `POST /reindex`, reload
  the page, and the dataset appears with a working legend.
- Switching a layer between native and harmonized view changes only a query parameter.
- No dataset ids, class codes, colors, or file paths appear as literals in frontend code.
- `tools/to_cog.py` and `tools/index.py` are re-runnable without side effects.
- A short `MIGRATION.md` records what changed, what the old paths were, and how to
  roll back.

## Non-goals

Do not implement the GMM modeling, affinity matrix computation, or expert
adjudication workflow. This task only needs to *consume* a crosswalk file and render
it. Assume the crosswalk is produced elsewhere; use a hand-written one for testing.

## Working style

Work in stages, and stop after each for review. Do not refactor code unrelated to this
migration. If you find that something in this plan conflicts with what is actually in
the repo or the data, say so and propose an alternative rather than working around it
silently.
