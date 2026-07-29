# Migration: fast local-raster map tiles (COG + MosaicJSON)

Why the map was slow, what changed, and how to roll back.

## The problem

Switching between products on the harmonize page took tens of seconds, and some
products hung or crashed the server outright.

The cause was not the frontend, the network, or rendering. The local-raster
products were served through GDAL **VRTs** in `cache/vrt/`. A VRT is only an XML
index over source tiles -- it holds no pixel data and, critically, **no overview
pyramid**. Without overviews, a zoomed-out tile forces GDAL to read the *entire*
raster at native resolution and downsample it on the fly.

Measured on this data, one zoom-4 tile:

| Product | Files | Overviews | z=4 tile |
|---|---|---|---|
| `hrlc30` (plain GeoTIFF) | 1 | `[2..256]` | **0.01 s** |
| `worldcover_local` | 98 | `[2..32]` | 0.40 s |
| `wsf` | 199 | `[2..32]` | 6.6 s |
| `gsw_yearly` | 15 | none | 7.8 s |
| `gwl_fcs30d` | 41 | none | 24 s |
| `fnf4` | 7 | none | 25 s |
| `dynamicworld_local` | 25 | none | 48 s |
| `gfc` | 16 | none | **>60 s, OOM-killed** |
| `from_glc` | 199 | none | **>60 s** |

`gfc` is 360000x600000 px across 16 files, so one zoom-4 tile touched roughly
216 gigapixels. `hrlc30` -- the one file that already had a proper overview
pyramid -- was ~4000x faster than its neighbours. That is the whole story.

Two further multipliers: the VRTs used 128x128 internal blocks (16x more block
reads per tile than `hrlc30`'s 512x512, over a Lustre mount), and *all* tile
reads were serialised through a single global semaphore, so one 48 s tile froze
every other layer.

## What changed

### 1. `tools/to_cog.py` (new)

Converts every local-raster product to Cloud-Optimized GeoTIFFs with overviews.

- Output goes to a **separate tree**, `$HARMONIZER_COG_DIR`; files under `data/`
  are never modified.
- Per-source-tile COGs plus a `mosaic.json` (MosaicJSON) per multi-file product.
  Single-file products get one COG and no mosaic.
- DEFLATE + `PREDICTOR=2`, `BLOCKSIZE=512`, explicit `nodata`.
- **Overviews use `MODE`**, never average/bilinear/cubic -- these are categorical
  class codes, and interpolating them fabricates classes that exist nowhere in
  the legend. Verified: an overview read of the converted `gfc` returns only
  `[0, 1]`, the real legend codes.
- Idempotent: valid COGs are skipped unless the source is newer or `--force` is
  passed, so an interrupted run resumes. Writes go to `*.partial.tmp` /
  `*.staging.tmp` and are renamed only on success, so a killed run never leaves a
  half-written file that a later run would accept.

Written against rasterio directly rather than `rio_cogeo.cog_translate` because
the source tiles are **striped** (one scanline per block) and up to 120000 x
120000 px -- ~14 GB decompressed each. `cog_translate` materialises a full-size
intermediate and gets OOM-killed here even with `in_memory=False`. Copying
window-by-window holds peak memory to ~870 MB regardless of source size.

### 2. `tools/to_cog.slurm` (new)

Runs the conversion as a batch job. The login nodes sit at load ~100, where the
full ~600-file run makes almost no progress (2 files in 25 min vs 90 files in
19 min on a dcgp node).

Note the account: this project has one account per partition family --
`iscrc_hsigfm` (boost/GPU) and `iscrc_hsigfm_0` (dcgp/CPU). This job is CPU-only
and needs `_0`; the default account fails with a misleading "invalid account or
expired budget" even though the budget is fine.

### 3. `harmonizer/local_tiles.py`

- Prefers the converted COG/MosaicJSON when present, and **falls back to the
  original VRT path** when it is not -- so the app keeps working for products
  that have not been converted yet.
- The single global tile semaphore was split in two. Reads of the original
  Lustre-mounted sources stay fully serialised (concurrent reads there have twice
  segfaulted the process inside libtiff, correlated with a Lustre "Flock LR
  mismatch"); reads of the converted COG tree, which lives on ordinary project
  storage and has never shown that failure, now allow 4 in flight.
- Added a disk cache of rendered PNGs, keyed by product / z / x / y / visible
  class subset. Repeat views drop from ~0.6 s to **0.0005 s**.
- Added `max_native_zoom()`, derived from each product's actual ground
  resolution, so the browser upscales past the data's real detail instead of
  requesting tiles the raster cannot fill.

### 4. `harmonizer/api.py`

- Tile responses now carry `Cache-Control: public, max-age=604800` and an
  `ETag`; a matching `If-None-Match` returns `304` without rendering. The ETag
  covers the legend, so recolouring a legend does not serve stale tiles.
- `/api/tiles/{product}` also returns `max_native_zoom`.

### 5. `web/app.js`

- Layer switching no longer removes the old layer before fetching the new one.
  It adds the new layer first and drops the old one on `load`, so the map never
  goes blank mid-switch. **This alone accounted for much of the perceived
  slowness**, independent of tile speed.
- `keepBuffer: 4`, `updateWhenIdle: false`, `updateWhenZooming: false` so small
  pans re-use tiles instead of re-requesting them.
- `maxNativeZoom` comes from the API, not a hardcoded number.

### 6. `harmonizer/config.py`

Added `COG_DIR` / `Config.cog_dir`, overridable via `HARMONIZER_COG_DIR`.

It deliberately does **not** default to a path inside the repo: the home
filesystem here has a 50 GB quota that is already ~95% full, and a previous
attempt to build overviews in-repo produced a single **41 GB** uncompressed
`.ovr` sidecar (`cache/vrt/worldcover_local.vrt.ovr`, still present -- see
rollback).

## Results

| Product | Before | After |
|---|---|---|
| `gfc` z=4 | >60 s, OOM-killed | **0.076 s** |
| `dynamicworld_local` z=4 | ~48 s | **2.5 s** |
| `dynamicworld_local` z=8 | ~48 s | **0.084 s** |
| any cached repeat view | full re-render | **~0.0005 s** |

## How to run

```bash
export HARMONIZER_COG_DIR=/leonardo_work/IscrC_HSIGFM/$USER/legend-harmonizer-cog
sbatch tools/to_cog.slurm            # full run, resumable
python tools/to_cog.py --list        # what would be converted
python tools/to_cog.py --only gfc    # one product
```

The server picks converted products up automatically -- no restart needed for
new products, though an already-open reader is cached per process.

## Running locally (off the cluster)

Nothing here is Cineca-specific. `HARMONIZER_COG_DIR` defaults to `cache/cog`
inside the repo, so a local clone needs no environment variable at all -- put
the COGs there and it just works.

To copy a subset down from Leonardo (the full tree is ~23 GB, so start with one
or two products):

```bash
# on your laptop, from the repo root
mkdir -p cache/cog
rsync -avP --info=progress2 \
  xtan0001@login05-ext.leonardo.cineca.it:/leonardo_work/IscrC_HSIGFM/xtan0001/legend-harmonizer-cog/hrlc30 \
  cache/cog/
```

Add more products by repeating with a different directory name
(`worldcover_local`, `dynamicworld_local`, `gfc`, `wsf`, `from_glc`, `fnf4`,
`gsw_yearly`, `gwl_fcs30d`). Copy each product's **whole directory** -- a
multi-file product needs its `mosaic.json` alongside the `.tif`s.

Then:

```bash
pip install -e .
python run.py
```

Products whose COGs are not present simply fall back to their registry
`access.path`; if that source is not on the machine either, the product returns
404 for tiles and the rest of the app still works.

## Rollback

Nothing under `data/` was touched, and the original VRTs in `cache/vrt/` are
untouched and still work.

1. Unset `HARMONIZER_COG_DIR` and delete the COG tree, **or** just move it aside.
   With no COGs present, `local_tiles` falls back to the original VRT paths
   automatically and behaviour returns to exactly what it was (slow, but
   unchanged).
2. To revert the code: `git revert` the migration commit. The frontend and API
   changes are independent of the COG tree and can be reverted separately.
3. To drop only the rendered-tile cache: `local_tiles.clear_tile_cache()` or
   delete `$HARMONIZER_COG_DIR/_tilecache/`.

## Follow-ups not done here

- `cache/vrt/worldcover_local.vrt.ovr` is a **41 GB** uncompressed overview
  sidecar on the near-full home filesystem. It is superseded by the converted
  COGs and is the single biggest reclaimable item on that volume, but deleting it
  is a destructive act on data this migration did not create, so it is left for
  a human to confirm.
- The wider `MIGRATION_PROMPT.md` scope -- the automatic indexer, the STAC/
  `catalog.json` catalog, `/legend`, `/crosswalk`, `/inspect`, `POST /reindex`,
  and the native/harmonized legend toggle -- is **not** implemented. This work
  targeted the performance problem only. Layer definitions still come from the
  registry YAML, which already keeps dataset ids and colours out of the frontend.
