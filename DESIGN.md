# Design: fast local viewing and full-overlap sampling

This document specifies a performance and workflow redesign of the
legend-harmonizer app around two observed problems:

1. **Viewing.** Displaying and interacting with local map data is slow, whether
   a product is a single mosaic or a set of many tiles.
2. **Sampling.** Sampling points over the overlap area of two maps is too slow
   and cannot cover the whole overlap.

It is written to be read standalone: section 1 records what the code does today
and where the time actually goes (established by reading and tracing the current
tree); sections 2–4 specify the design; section 5 lists small standalone fixes;
section 6 gives the build order and per-stage verification. File/line references
are against the tree at the time of writing (HEAD `b9e098f` plus the uncommitted
indexer/tile work).

**Scope decision: local-first.** Locally downloaded rasters (the `data/` folder
products) are the first-class citizens of this design. GEE-backed maps remain
supported as a convenience (no download needed) but their performance work is
deferred to the appendix. The pipeline method itself (embedding + GMM +
Bures–Wasserstein, per `docs/PIPELINE.md`) is unchanged — this design touches
*how fast and how far* the existing method runs, not what it computes.

---

## 1. Current behavior and where the time goes

### 1.1 Viewing

The fast path already exists in the code — it has just never been activated.

- `tools/to_cog.py` converts each local product into tiled COGs (512-px blocks,
  DEFLATE) with **MODE-resampled overview pyramids** plus a MosaicJSON index for
  multi-file products. Its own benchmark notes (docstring, `tools/to_cog.py:10-18`)
  measure zoom-4 tiles at 7.8–60 s from a raw VRT vs ~0.01 s from an overviewed
  file — a 100–1000× difference.
- `harmonizer/local_tiles.py` prefers converted COGs when serving
  (`_cog_source`, `local_tiles.py:111-126`; `_open_reader`, `:158-192`), falling
  back to the registry's raw `access.path` when none exist.
- **On this machine `cache/cog/` contains no converted products**, so every tile
  request takes the fallback: e.g. `worldcover_2020` is a GDAL VRT over 96
  GeoTIFFs / 6.3 GB with zero overviews, so one zoomed-out tile reads all 96
  member files at native resolution.

Three code-level issues compound it:

- **Serialized rendering.** All non-COG tile reads share
  `_SOURCE_SEMAPHORE = threading.Semaphore(1)` (`local_tiles.py:107`), a guard
  added against libtiff crashes on a Lustre network mount. On a local-disk
  Windows checkout that rationale does not apply, but the effect does: with two
  synced map panes, every visible tile renders strictly one at a time.
- **Class toggles re-render everything.** The visible-class subset is part of
  the tile URL, the disk-cache key, and the ETag (`web/app.js:309-319`,
  `api.py:632-654`), so each distinct toggle state is a full server re-render of
  every visible tile with no reuse.
- **Minor overheads.** `_cog_source()` does an `exists()` + `glob("*.tif")`
  directory scan and is called 3–4× per tile request (`local_tiles.py:150`,
  `:313`, `:371`, plus via `api.py:646`). Leaflet is tuned for a fast backend
  (`keepBuffer: 4`, `updateWhenIdle: false`, `app.js:343-345`), maximizing
  in-flight requests during pans — backwards while tiles cost seconds. And if a
  `to_cog.py` run leaves ≥2 COGs but no `mosaic.json`, `_cog_source` silently
  falls back to the slow raw source with no warning.

The "many tiles vs single mosaic" distinction is already handled correctly
*after* conversion (MosaicBackend vs single Reader); the felt difference today
is that conversion never ran and the raw multi-file VRT is the pathological
case.

> **Correction, measured during Stage V1 (2026-08-04).** The paragraph above is
> wrong *for the data on this machine*, and the conversion step it motivates was
> therefore skipped. The local sources here **already carry overview pyramids**:
>
> | source | overviews | z4 tile |
> |---|---|---|
> | `worldcover_2020` VRT (96 files, 6.3 GB) | `[2,4,8,16,32,64]` | 0.12 s |
> | each WorldCover source tile (1024 px blocks) | `[2,4,8,16,32,64]` | — |
> | `hrlc30` single GeoTIFF (512 px blocks) | `[2,4,8,16,32,64,128,256]` | 0.06 s |
>
> Measured over 12 tiles per zoom on the raw path: z3 median 0.30 s, z4 0.12 s,
> z5 0.06 s, z6 0.03 s — not the 7.8–60 s predicted.
>
> The benchmark table cited above (`tools/to_cog.py:10-18`) is genuine but was
> measured on the **cluster's** copy of the data (`data/Africa/2019/`, the Lustre
> mount) against products that do not exist here — `python tools/to_cog.py
> --list` reports 8 of 11 local products as `unavailable (FileNotFoundError)`.
> The three present on this machine (`hrlc`, `hrlc30`, `worldcover_2020`) are a
> different, already-overviewed download. So "`cache/cog/` contains no converted
> products" is true, but the conclusion that the fallback is therefore
> pathologically slow does not follow for this data.
>
> Consequences: COG conversion is **not** the *speed* win for local viewing here,
> so Stage V1 kept its tile-path hardening (cached `_cog_source`, configurable
> concurrency, degraded-mode and mosaic-gap warnings, Leaflet tuning) and dropped
> the blanket conversion step and its ~100× expectation. The COG-preference
> serving order stays — it is still correct, and still the right path on the
> cluster, where the unconverted products genuinely are that slow. **The remaining
> viewing cost is the class-toggle full re-render below, which Stage V2
> addresses.**
>
> **But conversion is not optional in general** — see section 5 item 4. `hrlc30`
> ships `AVERAGE`-resampled overviews that fabricate class codes at z4–z11, and
> `tools/to_cog.py`'s `MODE` pyramid is the only fix. Conversion is therefore a
> *correctness* requirement for any product whose published overviews are not
> mode/nearest, independent of whether it is already fast.

### 1.2 Sampling — and why it is slow at all

A common intuition is that sampling is "just pick N points per class", which
should be instant. It is not, because **choosing the points is cheap but finding
where they are allowed to go is a whole-raster operation**. Per class, the
pipeline must (per `docs/PIPELINE.md` Stage 2):

1. build the class mask over the entire sampling region;
2. erode it 2 pixels inward so boundary/mixed pixels are excluded;
3. require a homogeneous 3×3 neighbourhood;
4. draw points spread across 0.25° grid cells with a minimum spacing;
5. fetch each surviving point's 64-dim AlphaEarth embedding **from Google Earth
   Engine over the network** (batched per class, but still a network call — the
   embeddings are not local).

Steps 1–3 touch every pixel of the region per class. Step 5 is the irreducible
network cost. Everything else about today's slowness is implementation:

**Local path** (`harmonizer/local_sampling.py`, used for `access.method:
local_raster` products):

- `read_label_window` (`local_sampling.py:80-115`) reads the **entire AOI
  window at native resolution into RAM in one `ds.read()`**. For a full
  local×local overlap (e.g. 36°×24° at 10 m ≈ 10¹¹ pixels) this is physically
  impossible — this is the direct reason sampling cannot cover the whole
  overlap.
- The `sample_scale_m` run parameter is ignored on this path — reads are always
  native resolution.
- `stratified_candidates` (`:193-233`) converts **every** masked pixel to
  lon/lat before subsampling, and `pixel_to_lonlat` (`:68-77`) constructs a new
  pyproj `Transformer` per call.
- Erosion uses a dense `(2r+1)²` structuring element
  (`_erode_square`, `:131-145`), which scipy cannot decompose.

**Shared machinery** (`harmonizer/sampling.py`):

- `_thin_by_spacing` (`sampling.py:351-375`) is an O(n²) pure-Python distance
  loop.
- The truncation `cands[:target]` (`sampling.py:448`) happens after thinning in
  arrival order, so for large candidate pools it is not spatially fair and
  partially undoes the per-cell spread.

**Overlap computation** (`harmonizer/overlap.py:35-95`): a plain bbox
intersection of the products' *declared* registry rectangles. For a tile-set
product the declared footprint is the union bbox of its tiles
(`worldcover_2020.yaml`), so no-data holes sit inside the "overlap"; the
source-derived footprint that `docs/PIPELINE.md` §2 prescribes is not
implemented (`registry/products.py:151` returns the YAML box verbatim).

**GEE path** (for completeness; see appendix): ~34 strictly sequential blocking
`getInfo()` round trips per two-map run; `_label_image_for(product_id)` is
called without a region (`sampling.py:495`) so Dynamic World's year-long modal
composite is never `filterBounds`-restricted; `sampling.py:272` omits the
`bbox=` argument to `_grid_cell_image`, costing one avoidable network round
trip per class.

**Outright bug:** `harmonizer/auxiliary.py:730` and `:766` still call
`sample_class` with its pre-refactor signature (the refactor in commit
`32ea765` updated `sampling.py`'s call sites only), so **every auxiliary-AOI
sampling job crashes with a `TypeError`.**

---

## 2. Design A — Viewing: COG-first serving

Goal: every local product is served from tiled, overviewed COGs; toggling
classes costs zero server work; nothing serializes needlessly.

### 2.1 Conversion becomes part of registration

`tools/to_cog.py` stays the conversion engine, but it is no longer a separate
manual step: the indexer (section 4) runs it as the final stage of registering
a dataset, so a product cannot reach the picker without its COG tree. The
serving code's preference order (`mosaic.json` → single COG → raw source) is
unchanged, but the raw-source fallback becomes an explicitly logged degraded
mode: when `tile_png` falls back, log a prominent warning naming the product
and the command that fixes it, and surface the state in the product's
ready-status (section 4.2).

### 2.2 Hardening the tile path

- **Cache the source lookup.** `_cog_source(product_id)` becomes cached
  (e.g. `functools.lru_cache`) with an explicit invalidation hook called when a
  conversion job finishes. This removes 3–4 directory scans per tile request.
- **Configurable concurrency.** Replace the hardcoded
  `_SOURCE_SEMAPHORE = Semaphore(1)` with a config value
  (e.g. `CONFIG.tiles.source_concurrency`, default 4 for local disk). The
  Lustre rationale documented at `local_tiles.py:86-94` is real but
  environment-specific; keep the comment, make the number a setting, and note
  that deployments on network filesystems should set it back to 1. The
  per-thread GDAL handle discipline (`_reader`, `:195-224`) is load-bearing and
  must be preserved.
- **Close the mosaic gap.** If a product's `cache/cog/` dir holds ≥2 `.tif`
  files but no `mosaic.json` (a conversion that died between COG writing and
  mosaic building), do not silently fall back to the raw source: log an error
  and report the product as `needs-conversion` so the indexer re-runs
  `build_mosaic`.

### 2.3 Instant class toggles: class-code tiles + client-side rendering

Today the server renders RGBA PNGs with the class subset baked in, so every
toggle is a new tile. Instead:

- **The server serves one tile per position, encoding the class *code* per
  pixel** — a single-channel (8-bit grayscale; 16-bit if any product's codes
  exceed 255) PNG, rendered with nearest resampling exactly as now but without
  applying the palette or the subset. The tile URL loses its `?classes=` query;
  the cache key and ETag drop the class subset (keeping product, z/x/y, band).
- **The frontend renders color and visibility in a canvas layer**
  (`L.GridLayer` with a per-tile `<canvas>`): decode the PNG, map each pixel's
  code through the legend palette (from the registry via the existing legend
  API), and set alpha 0 for hidden classes and for greyed-out absent classes.
  Toggling a class re-runs only the cheap per-tile canvas pass over
  already-fetched tiles — no network, no server work, instant.
- Every toggle state, and every band/year of a product, shares one cached tile
  per position across browser cache, server PNG disk cache, and ETag.

Fallback if code-tiles prove awkward (e.g. PNG grayscale handling in some
browser paths): keep serving full-legend RGBA tiles and have the client filter
by exact color→class lookup — colors are exact because all resampling is
nearest/mode — but the code-tile form is preferred (smaller tiles, no color
collisions, palette changes need no re-render).

### 2.4 Leaflet tuning

- `updateWhenIdle: true` and a smaller `keepBuffer` (e.g. 2), so pans don't
  queue dozens of stale-viewport requests; revisit after COG conversion if
  tiles are fast enough to afford eagerness again.
- Keep the recent uncommitted improvements: layer `bounds` from
  `/api/tiles/{id}` (kills out-of-footprint requests) and the
  `load`/`tileload`-driven layer swap that replaced the fixed 2 s timeout.

---

## 3. Design B — Sampling: chunked full-overlap streaming (local path)

Goal: sampling any AOI up to and including the **entire overlap** of two local
products completes in bounded memory and reasonable time, honoring
`sample_scale_m`, with the sampling *semantics* (erosion, homogeneity,
stratification, spacing, floors, absent-vs-buffered-away rule) unchanged.

### 3.1 Source-derived overlap

Implement the footprint rule `docs/PIPELINE.md` §2 already prescribes: a local
product's operational footprint comes from **rasterio bounds of the actual
source** (reprojected to EPSG:4326), not the declared YAML box. The overlap of
a run is the bbox intersection of the selected products' derived footprints,
AlphaEarth's, and the user AOI — same shape of computation as today
(`overlap.py`), different inputs. Registry `footprint:` stays as advisory
display data. No-data holes inside the bbox need no polygon machinery: they
fall out naturally because per-chunk class masks are empty there (3.2).

> **Verified, and one bug fixed (2026-08-05).** A real overlap is not a
> rectangle — products have coastlines, missing tiles and no-data corners — so
> the "holes fall out naturally" claim is what makes the bbox safe. It now
> genuinely does, but only after fixing a crash: `rasterio`'s
> `Window.intersection` **raises** `WindowError` for a cell wholly outside the
> raster instead of returning an empty window, which aborted the whole scan
> partway through rather than skipping the cell. Caught in
> `chunked_sampling._read_cell`.
>
> Measured after the fix — cells outside the data contribute nothing and cost
> only a skipped read:
>
> | region | cells | classes found | candidates |
> |---|---|---|---|
> | `hrlc30`, inside its footprint | 4 | 10 | 5655 |
> | `hrlc30`, straddling its west edge | 16 | 10 | 2680 |
> | `hrlc30`, wholly outside | 32 | 0 | **0** |
> | `worldcover_2020`, Gulf of Guinea (open sea) | 96 | 0 | **0** |
> | `worldcover_2020`, coastline | 48 | 9 | 4611 |
>
> For the `hrlc30 × worldcover_2020` pair the bbox is a good approximation
> anyway: 98.6% of it has data in *both* products, so only ~1.4% of cells are
> wasted effort. That figure is pair-specific — an L-shaped or archipelagic
> pairing would waste much more — but the cost of a wasted cell is one empty
> read, not a wrong sample. **The sampled points are always correct regardless
> of overlap shape; only scan time is affected.**

### 3.2 The chunked sampler

Replace `local_sampling.py`'s read-everything-once approach with a streaming
loop. Reuse the existing constants (`CONFIG.sampling.grid_cell_deg` = 0.25°,
`points_target`, `points_floor`, `min_spacing_m`, erosion/homogeneity settings)
— no new tunables beyond what §2 of PIPELINE.md already defines.

Per map, per run:

1. **Grid the overlap** into `grid_cell_deg` cells — the same grid the
   stratification already uses, now also the unit of I/O and memory.
2. **Per cell, one decimated windowed read from the COG tree** (the same
   converted files the viewer uses — `cache/cog/<pid>/`, via the mosaic for
   multi-file products) at the run's `sample_scale_m`: compute the window for
   the cell and pass `out_shape` scaled to the sample resolution, letting GDAL
   satisfy the read from the overview pyramid. A 0.25° cell at 100 m scale is
   ~280×280 px — trivial. Memory is bounded by one cell regardless of AOI size.
   - *Semantic note, to be recorded in code comments:* the overviews are
     MODE-resampled (`tools/to_cog.py:36-47`), so a coarse-scale pixel is the
     modal class of its footprint, whereas GEE's `scale=` sampling is
     nearest-at-scale. Mode is at least as defensible for label sampling
     (arguably better — it cannot fabricate classes and is less noisy), but it
     is a deliberate difference; document it, don't hide it.
   - Products not yet COG-converted fall back to a decimated read of the raw
     source for correctness, with the same degraded-mode warning as tiles.
3. **Per cell, per present class:** class mask → erosion (radius in *pixels at
   sample scale*, consistent with how the GEE path applies `erode_pixels` at
   `sample_scale_m`) → 3×3 homogeneity → accumulate the pre-erode and
   post-erode candidate counts into per-class run totals (so the
   absent-vs-buffered-away rule works exactly as before, now over the whole
   region) → draw up to the per-cell quota of candidate pixel centres and
   convert *only those* to lon/lat (one cached pyproj `Transformer` per CRS,
   not one per call).
4. **Across cells:** accumulate candidates per class; thin by `min_spacing_m`
   using `scipy.spatial.cKDTree` (or a grid-bucket equivalent) instead of the
   O(n²) loop; truncate to `points_target` **round-robin across cells** rather
   than list order, so the kept points stay spread over the overlap.
5. **Embeddings:** unchanged mechanism — batched AlphaEarth `sampleRegions`
   per class through the existing adapter — but fetch classes concurrently with
   a small thread pool (the calls are independent, network-bound, and the GEE
   client is used sequentially per thread). Label read-back for local products
   should come from the values already in hand from the windowed reads (the
   per-point `hrlc_local` adapter loop is only needed as a cross-check, and if
   kept must be batched — one windowed/`ds.sample` read, not one `ds.read()`
   per point).

The relaxed-buffer resample (the absent-vs-buffered-away rule's second pass)
re-runs steps 2–4 with `erode_pixels=1` for the affected classes only.

### 3.3 Full-overlap default and the cost model

- The **default AOI becomes the full derived overlap** (restoring PIPELINE.md
  §2's stated default for the local path; the existing hard block on *global*
  GEE×GEE overlaps in `api.py:765-774` stays, since that path still samples
  server-side on GEE).
- Cost is predictable: `pixels ≈ overlap_area_m² / sample_scale_m²`, cells =
  overlap / 0.25°². Extend the existing slow-combination estimator
  (`api.py`, `_SLOW_SCORE_THRESHOLD`) to the local path: given the overlap, the
  UI **auto-suggests the finest `sample_scale_m` that keeps the estimated run
  in the minutes range**, shows the estimate, and lets the user override with
  a finer scale or a narrower AOI (accepting the shown cost).
- **Progress reporting** becomes per-cell/per-class (completed cells ÷ total),
  replacing today's 2% → 50% → 100% jumps (`pipeline.py:285-294`), so a long
  full-overlap run visibly advances.

### 3.4 What does not change

Point floors and targets, erosion and homogeneity rules, declustering
constants, the absent-vs-buffered-away decision, per-class independent
sampling (no co-located points), caching and run signatures, and everything
downstream (Stage 3 GMM fitting onward) are untouched. A run over the same
inputs remains reproducible under the fixed seed; note that changing
`sample_scale_m` (including via the new auto-suggestion) is already a
signature input and correctly invalidates the cache.

---

## 4. Design C — Drop-in datasets and UI

Goal: *"download a land-cover map and its legend into `data/`, run the app,
and the map is choosable in the window"* — no CLI steps — plus a picker and
legend that tell the user what they are looking at.

### 4.1 Auto-registration on startup / refresh

`harmonizer/indexer.py` (currently a manual CLI:
`python -m harmonizer.indexer`) becomes the engine behind an automatic flow:

- **Detection.** On server startup and on a UI "refresh datasets" action, scan
  `data/`: a candidate dataset is a folder (or single file) of rasters with a
  matching legend CSV under `data/legend/` (convention: `<id>.csv` matching
  the folder name; `data/datasets.yaml` remains as an optional override for
  names, band selection, year, and irregular pairings — the indexer's existing
  manifest logic).
- **Registration job.** Each new/changed candidate runs as a background job
  through the existing job/progress machinery: index (validate CRS/bands,
  build the VRT for tile sets, reconcile legend codes against observed codes,
  write the registry YAML) → **COG conversion + mosaic** (`tools/to_cog.py`,
  per decision 2.1) → invalidate `_cog_source` cache → product appears in the
  picker. Progress is visible in the UI; failures surface as a product-level
  error state, not a silent absence.
- A candidate with rasters but no legend CSV still appears — greyed, marked
  `needs-legend`, naming the CSV path it expects — so the user learns the
  convention from the UI instead of from documentation.
- The indexer keeps its no-overwrite-without`--force` behavior for YAMLs a
  human has edited; auto-registration only creates missing entries and
  completes missing COG trees.

### 4.2 Map picker

Replace the current flat product list with a grouped, informative picker:

- **Grouped: Local datasets / GEE datasets**, local first.
- Each row: display name, year(s), native resolution, and a **ready-state
  badge**: `ready` / `indexing…` / `converting…` / `needs-legend` /
  `needs-conversion` / `error`. Only `ready` products are selectable for
  viewing and runs.
- Data comes from the registry plus the registration-job state — no hardcoded
  product knowledge in `app.js` (consistent with PIPELINE.md §2.5).

### 4.3 Legend chips: grey out classes the data does not have

The indexer already scans the codes actually present in a dataset and
reconciles them against the legend CSV. Record the observed-code list in the
product's registry YAML (or a sidecar the legend API merges in). In the map
legend, a class that is **declared in the legend but not observed in the
dataset** renders as a greyed, non-toggleable chip (with a tooltip: "not
present in this dataset"), so the user sees at a glance which classes the data
contains. This is dataset-level presence (cheap, computed once at
registration), not per-viewport presence — the latter is out of scope here.

---

## 5. Small standalone fixes

Independent of the designs above; do them early. **Items 1–3 landed with Stage
S1** (all three live in the sampling files S1 touched); item 4 was found and
resolved during V1/V2.

1. **Auxiliary-AOI crash.** *(FIXED.)* Update `harmonizer/auxiliary.py:730` and `:766` to
   `sample_class`'s current signature
   (`sample_class(class_value, label_adapter, embedding_adapter, *,
   count_candidates_both, draw_candidates)`), mirroring how
   `sampling.py:504-519` builds the two closures. Every auxiliary-AOI job is
   currently dead on a `TypeError`. Landed as a shared `_sample_gee_class`
   helper so the two call sites cannot drift from the signature again.
2. **Dead code.** *(FIXED.)* Remove the unused `default_overlap` import
   (`sampling.py:48`) and `_CELL_PROP` (`local_sampling.py:42`).
3. **Wasted GEE round trip.** *(FIXED.)* Pass `bbox=overlap.bbox` at `sampling.py:272`
   (as `explorer.py:494` already does) so `_grid_cell_image` doesn't call
   `getInfo` to recover a bbox the `Overlap` object already holds.

4. **`hrlc30` was broken two ways; both fixed 2026-08-04.** *(RESOLVED — kept
   because the second half is a standing hazard for any product whose publisher
   ships averaged overviews, and because it is the one case where COG conversion
   is load-bearing on this machine.)*

   **(a) The source file was an incomplete download.** Replaced with a complete
   one (789 MB vs the truncated 383 MB); `--check` now reports every window
   readable, and the z12 `RasterioIOError` is gone.

   **(b) The replacement's own overviews are still `AVERAGE`-resampled**, so
   re-downloading did *not* fix the fabricated codes — that was a separate
   defect in the published file, not a symptom of the truncation. Measured on
   the good file, tiles served from those overviews carried ~100-130 spurious
   codes at **every zoom from z4 to z11**; only z12+ (native resolution) was
   clean. For a 30 m product that is every zoom a user actually browses at.

   **Fix: `python tools/to_cog.py --only hrlc30`** (87 s). The COG's `MODE`
   pyramid replaces the averaged one, and the result is **0 spurious codes at
   every zoom**, tiles at 0.01-0.03 s. This is the one product on this machine
   where conversion is required for *correctness* rather than speed — the
   general finding in section 1.1 (sources already fast, conversion skipped)
   still holds for `worldcover_2020`, whose own overviews are correct.

   How it presented, for reference. The truncated file was 383 MB for a
   51031×68896 raster, and `python -m harmonizer.indexer --check --only HRLC30`
   reported `221/221 windows unreadable (100%)` — only the overview pyramid was
   intact. That produced two symptoms which initially looked like one cause:
   `RasterioIOError: Read failed` on any native-resolution (z12) tile, and a z8
   tile decoding to `9, 11, 12, 13, …, 139` — over 100 values, none in hrlc30's
   legend. Replacing the file fixed the first and left the second untouched,
   which is what separated them.

   The fabricated codes are precisely the failure mode `local_tiles.py`'s
   docstring and `hrlc30.yaml`'s legend note both warn about, and they are
   **not** a read-time resampling mistake: the tile path already forces
   `resampling_method="nearest"`, which cannot undo averaging baked into a
   stored overview. Only rebuilding the pyramid with `MODE` fixes it.

   **Standing rule this establishes:** a local product must be checked for
   spurious codes at low zoom before it is trusted — for viewing *or* for an S1
   run, since the chunked sampler reads the same decimated levels and would
   otherwise sample invented classes. Where a publisher ships averaged
   overviews, `tools/to_cog.py` is the fix and is mandatory, not optional.

---

## 6. Build order and verification

Staged one at a time, each with a checkable artifact, per `CLAUDE.md`.

**Stage V1 — COG activation + tile-path hardening (2.1 minus indexer wiring,
2.2, 2.4).** ~~Convert the existing products (running the current
`tools/to_cog.py` by hand is acceptable at this stage)~~, cache `_cog_source`,
make source concurrency configurable, add the degraded-mode and mosaic-gap
warnings, tune Leaflet.
*Verify:* `python scripts/verify_v1.py` — reports per-product source state, times
z4/z8/z12 tiles, checks that hiding a converted product's `mosaic.json` produces
a logged ERROR rather than silent slowness, and compares serial vs parallel
rendering across two products.

**Status: done, minus conversion.** The conversion step was dropped — see the
measured correction in section 1.1: the sources on this machine already carry
overview pyramids and render tiles in 0.03–0.30 s, so there was no ~100× to
gain and the expectation was removed rather than manufactured. The mosaic-gap
and degraded-mode checks are unverifiable here for the same reason (nothing is
converted, so the mosaic-gap check reports SKIP); both are exercised on the
cluster's data. Everything else landed.

**Stage V2 — class-code tiles + client-side rendering (2.3).**
*Verify:* `python scripts/verify_v2.py` covers the server and encoding half —
the tile URL carries no `classes`, code-tile bytes and ETag are identical across
subsets, codes round-trip with no scaling, and the emulated client paint is
**pixel-identical** to the server's RGBA render for both the full legend and a
subset. The browser half stays manual: with the network tab open, toggling
legend chips causes **zero** tile requests; switching band/year still re-fetches.

**Status: done.** 8-bit greyscale+alpha encoding (alpha carries nodata, since 255
is a real class code in `gwl_fcs30d`/`wsf`, so no grey-channel sentinel is safe);
codes fit a byte in every registry legend, so the 16-bit variant was not needed.
The RGBA renderer is retained at `/api/tiles/local/rgba/...` as the documented
fallback — a distinct path segment, because a `.rgba.png` suffix makes `{y}.png`
match first and fail to parse the tile row.

**Stage S1 — chunked sampler + source-derived overlap + full-overlap default
(3.1–3.3).**
*Verify:* `python scripts/verify_s1.py` — checks the derived footprints, the
exact-semantics match against the reference implementation, a full-overlap scan
with memory measured, the round-robin spread property, and the cost model.

**Status: done.** Measured results on `hrlc30 × worldcover_2020`:

| claim | result |
|---|---|
| semantics unchanged | per-class pre/post-erode counts **exactly equal** to `local_sampling`, all 10 classes |
| bounded memory | **0.02 GB** above baseline over a 3588-cell full-overlap scan (design allowed < 2 GB) |
| full overlap tractable | 17.2° × 12.8° scanned per map; previously impossible at any scale |
| throughput | ~150 ms/cell, **~190 s per gigapixel** per map |

Three things the build settled that the design left open:

- **`sample_scale_m` is relative to the source's *true* resolution, not its
  nominal one.** hrlc30 is catalogued as 30 m but its pixels are 27.83 m, so a
  run at `sample_scale_m=30` legitimately decimates ~1.08× and returns ~85% of
  the native candidate count. The exact-match check therefore runs at the
  measured native scale. Do not treat a small shortfall here as a bug without
  checking the raster's actual transform.
- **The cost budget is calibrated, not guessed.** At ~190 s/Gpx/map, the
  original `4.0e9` implied ~13 min/map (~25 min a run), which is not "the
  minutes range" §3.3 asks for. It is now `1.0e9` (~3 min/map, ~6 min a run),
  and the endpoint returns `estimated_seconds` so the UI can show time rather
  than a pixel count.
- **Per-cell erosion differs microscopically from whole-region erosion.** A
  class patch straddling a cell boundary is eroded from that seam as well as
  from its true edge. At 0.25° cells this touches a vanishing fraction of
  candidates and only ever *omits* points, never invents them. Recorded in
  `chunked_sampling._sampling_mask` rather than engineered away.

*Three UI/UX bugs found by running it (2026-08-05), all fixed:*

- **"Check overlap" appeared not to overwrite a filled-in AOI.** It sent the
  card's own box as the `aoi` constraint, so the server intersected the overlap
  *with* it and returned the box you already had. Only after "clear" did it look
  like it worked. It now asks the question the button actually poses (`aoi:
  null`): what is the overlap of the two selected maps? Relabelled **"use
  overlap"**, and it now adopts the result into the card instead of only printing
  it.
- **The auto-suggested scale was computed but never applied.** §3.3 asks the UI
  to "auto-suggest the finest `sample_scale_m` that keeps the estimated run in
  the minutes range"; it was only ever *displayed*. Adopting the full overlap at
  the 10 m default is ~27 Gpx/map ≈ **170 minutes** — which is what "sample
  points keeps running, no end" actually was. "use overlap" now raises the scale
  to the suggestion (only ever coarsens; a user's already-coarser choice wins)
  and says so.
- **Progress went backwards and stalled.** The scan owned the whole bar, so it
  filled to ~50% and then froze through every per-class Earth Engine round trip
  — with no per-class reporting at all — and each relaxed-buffer rescan reset the
  cell counter, dragging the bar back to ~12%. Now: the scan owns 0–50%, the
  per-class phase 50–100%, rescans are announced (`class 141: too few points
  after eroding -- rescanning with a relaxed buffer`) rather than silent, and the
  sequence is strictly monotonic to 100%. Verified: 20 callbacks, 0
  non-monotonic steps.

*On the round-robin truncation (§3.2 step 4):* it is correct and verified, but on
this data it is currently **latent rather than load-bearing**. The full overlap
is ~3588 cells against a 1500-point target, so the per-cell quota works out at
roughly one candidate per cell and arrival order already spreads points across
every cell — the two strategies agree class for class. The fix matters for the
case it was written for: a *small* AOI (few cells) where one cell holds more than
the whole target. `verify_s1.py` therefore checks the property on a synthetic
skew (one cell of 3000 points + 50 cells of 10 → round-robin uses **51 cells,
arrival order 1**) and separately reports, per class, whether truncation was
binding at all, so the real-data comparison is never read as a pass when it was
vacuous.

**Stage W1 — drop-in registration + picker + greyed legend (4).**
*Verify:* `python scripts/verify_w1.py` covers detection, legend discovery, the
picker payload and the greyed-class flag; `--register <folder>` runs a real
registration end to end. The browser half stays manual: the picker groups
Local/GEE and disables non-ready products, `↻ datasets` picks up a folder
dropped while the server is running, and an absent legend class is a greyed,
non-clickable chip.

**Status: done.** New `harmonizer/registration.py` wraps the indexer: scan
`data/` → index → COG-convert → invalidate caches → selectable, as background
jobs started on server startup (`HARMONIZER_NO_AUTOREGISTER=1` to skip) and by
`POST /api/datasets/refresh`. `GET /api/datasets` and the extended
`/api/products` carry the ready state.

Verified against real drop-ins: **JAXA_HRLULC_SEA_2023 (619 files) registered
from nothing to `ready`**, and `worldcover_2020` re-indexed to carry the new
flags — its class 100 (Moss and lichen), declared by the ESA legend but absent
from the Africa subset, is now `observed: false` and greys out.

Four things the build settled or corrected:

- **The legend convention was simplified to one file in one place.** The design
  specified a shared `data/legend/` directory with `<id>.csv` name matching.
  Real downloads do not comply — `JAXA_HRLULC_SEA_2023/` ships
  `HRLULC_legend.csv`, `Copernicus_LCFM_LCM-10_2020/` ships `LCM-10_legend.csv`
  — so the first attempt bolted a token matcher onto it (4+ chars, generic words
  and years excluded, must be unambiguous). That worked, but needed three rules,
  a length threshold, a tie-break, and a manifest override for the cases tokens
  could not reach — and it still refused Copernicus, whose only distinctive
  token is 3 characters.

  **Replaced with: the legend lives in the dataset's own folder, named
  `legend.csv`.**

  ```
  data/
  ├── WorldCover_2020/
  │   ├── *.tif
  │   └── legend.csv
  ├── JAXA_HRLULC_SEA_2023/
  │   ├── *.tif
  │   └── legend.csv
  ```

  Nothing is matched, ranked or guessed: the legend is the file next to the
  rasters it describes. The folder name is free (it becomes the map's display
  name), so only one name has to be remembered, and a wrong legend cannot be
  attached to a map even in principle. `needs-legend` collapses from a paragraph
  of options to one instruction naming the exact path. `data/datasets.yaml` is
  now purely optional metadata (display name, provider, year, band) — no entry
  carries a `legend:` line any more.

  **`data/legend/` is gone**, along with `LEGEND_DIR` and every fallback to it.
  Keeping the old location working "so existing checkouts keep running" would
  have left two valid answers to *where does the legend live?* — which is the
  ambiguity this change existed to remove. The five legends were migrated into
  their dataset folders; `index_one` now defaults to the dataset's own
  `legend.csv` instead of requiring a manifest entry, and the generated YAML
  header records the real path.
- **`gdalbuildvrt` was passed 619 paths on argv** and died on Windows with
  `[WinError 206] filename or extension too long` (~59 000 chars against a
  ~32 768 limit) — an error naming nothing useful. Now always via
  `-input_file_list`, so large tile sets stop being a special case.
- **Fill/no-data rows were being treated as land-cover classes — in the legend
  *and* in the sampler.** First seen as code `0` ("No data") rendering a
  permanently greyed chip; the general case appeared with Copernicus LCM-10,
  which declares **254 "Unclassifiable"** ("no Sentinel-1/2 observations or
  observations of insufficient quality") and **255 "No Data"** ("pixels not
  processed"). Fixed in two places, because there are two independent gates:

  1. **The legend — declared by the user, in an `IsClass` column.** The legend
     CSV gains an optional column after `Label`:

     ```csv
     Class Code,Color code,Label,IsClass,Description
     10,#006400,Tree cover,TRUE,Areas dominated by trees...
     255,#000000,No Data,FALSE,Pixels not processed.
     ```

     An earlier version inferred this from the label text ("no data",
     "unclassifiable", …). That worked on these five legends but was guesswork:
     it depends on the producer's wording, silently misses anything phrased
     differently (*Sin datos*, *Fill*, *Cloud*), and could in principle drop a
     real class whose name happens to contain a matching word. Whether a row is
     land cover is the **user's** knowledge, not something to reverse-engineer
     from prose — so they state it, in the file they already edit.

     Optional and fail-safe: a blank cell, an unrecognised value, or a legend
     with no such column all mean `TRUE`, so adding the column can never
     silently drop land cover. `TRUE/yes/1` and `FALSE/no/0` are accepted in any
     case, since these files get edited in Excel.

  2. **The sampler**, which is the one that actually matters for the crosswalk.
     Class discovery reads *pixels* (`np.unique` per cell on the local path,
     `frequencyHistogram` on the GEE path), and a fill value occupies pixels
     exactly like a class does — so filtering the legend alone would not have
     stopped it. Only code `0` was skipped. Left unfixed, a run over Copernicus
     would erode 254/255, draw points from them, spend network round trips
     fetching 64-dim embeddings for them, fit a GMM to *"places the producer
     could not classify"*, and emit crosswalk rows matching that non-class
     against real land cover in the other map — output that looks exactly as
     confident as a real row and means nothing.

     Now an **allowlist**: sample a value only if the product's registry legend
     names it as a class (`sampling.drawable_classes`,
     `chunked_sampling._class_codes`), applied on the local path, the GEE path
     and the auxiliary-AOI path. An allowlist rather than a blocklist because it
     also catches unmapped codes the legend never declared, and cannot be
     out-grown by a product whose codes exceed any assumed range. Dropped values
     are logged, not silently discarded. Verified by injecting 0/254/255 into a
     real Copernicus cell: all three present in the pixels, none reach sampling.

- **`--force` could destroy a hand-verified registry YAML — now it cannot.**
  Found the hard way: re-indexing every dataset with `force=True` (to refresh
  the new `observed:` flags) silently overwrote `hrlc30.yaml`, which is
  hand-written and carries a legend *verified against the raster in QGIS* plus
  notes recording which observed codes are resampling artifacts rather than
  classes. The regenerated file took its colours from whatever CSV was to hand,
  so the map still drew — in the wrong colours. `verify_v2` caught it as
  "client-painted == server-rendered: 65536 of 65536 px differ", because the
  legend API and the tile renderer had diverged.

  The file was restored from git, and `index_one` now refuses to overwrite any
  YAML lacking the `# Generated by ...` first-line marker, **even with
  `--force`** — `force` exists to refresh *generated* entries, not to discard
  human work. Deleting the file is still the way to demand a rebuild. The
  indexer's docstring and CLAUDE.md both already said not to regenerate
  `hrlc30.yaml`; a documented rule that only a human enforces is not enforced.

- **Deleting a dataset folder is detected, and its leftovers are cleanable.**
  Registration only ever *created*; nothing removed. Deleting
  `data/<Dataset>/` therefore stranded the whole derived tree — for
  `JAXA_HRLULC_SEA_2023` that is **7.9 GB** of COGs plus the VRT, registry entry
  and tile cache — with nothing in the UI referring to it.

  Worse, the product stayed *selectable*: `_local_raster_ready()` tests
  `access.path`, which for a **tile set** is the VRT under `cache/`, not a file
  under `data/`. The VRT survives the deletion, so the readiness check passed
  and the map looked fine while every tile read failed.

  Now `scan_datasets()` reports such a product as `missing`, `/api/products`
  lists it **disabled** with the leftover size in its tooltip, and
  `DELETE /api/datasets/{id}` removes exactly what `product_artifacts()` names.
  Three deliberate constraints:

  * **`data/` is never a target.** `product_artifacts()` lists only derived
    files, all regenerable from the source folder; `verify_w1` asserts nothing
    under `data/` can ever appear there, and `verify_data_readonly` covers the
    same guarantee from the other side.
  * **Deletion is explicit.** A rescan flags and offers; the browser confirms
    with the exact paths and size. Silently freeing gigabytes because a folder
    went missing is the kind of surprise that loses work — the folder may have
    been moved aside deliberately, and keeping the COGs means restoring it costs
    nothing.
  * **A live product is refused** (409) unless `force=true`: removing the
    derived files of a *working* dataset is not a cleanup, it is throwing away
    hours of conversion.

  One false positive found while building it: the repo shipped hand-written
  registry entries for nine cluster-only products (`hrlc`, `wsf`,
  `worldcover_local`, …) that have a YAML but no data on this machine. The first
  version flagged all nine as deleted datasets. **Those entries have now been
  deleted** — the registry holds exactly the datasets under `data/` plus the
  GEE-backed products, and `scripts/register_africa_products.py` regenerates
  them on the cluster where their rasters live. The *detection rule* still
  requires something derived (a COG tree, VRT or cache) rather than the registry
  YAML alone, so hand-writing an entry for data that has not arrived yet does
  not resurrect the false positive.

  **Ownership is recorded, not inferred (`harmonizer/manifest.py`).** The first
  implementation deduced ownership from names — glob `samples_<pid>*`, assume
  `cache/cog/<pid>/` belongs to `<pid>`. That fails in both directions, and both
  are bad: it over-claims (`worldcover` and `worldcover_2020` are different
  products whose names are prefixes of one another) and under-claims (auxiliary
  caches use a scoped id `<pid>__aux_<name>`; cross-label caches name *both*
  products of a pair; the next stage will invent another scheme, and until
  someone remembers to teach the pattern list, its files are silently orphaned).

  Now every step that writes a derived file **declares it** — one JSON per
  product under `cache/manifests/`, recording repo-relative path, `kind`, and
  the `stage` that produced it. Cleanup reads that list back. The difference is
  between *"these files look like they belong to X"* and *"X created these
  files"*.

  Four decisions worth keeping:

  * **A sidecar, not the registry YAML.** That file is hand-editable and
    sometimes hand-written (`hrlc30.yaml`); machine bookkeeping does not belong
    in it.
  * **`data/` is refused at record time**, not filtered at delete time. The
    manifest is the *input* to a delete, so a bad entry is the one bug that could
    destroy a download — `record()` raises rather than skipping quietly.
  * **Lazily-written caches are folded in at read time.** Tile PNGs and Stage 2/3
    caches are produced by browsing and running, not by registration, so there is
    no moment at which to record them; their names are built by this app's own
    cache-path helpers, so they are derived by construction rather than guessed.
  * **No manifest means no artifacts.** A hand-written registry entry for data
    that never arrived owns nothing and can never be offered for cleanup — the
    same protection the earlier `_is_derived` heuristic gave, now answered by a
    record instead of an inference.

  Delete order is deliberate: files, then directories, then the registry YAML
  **last**, and the manifest itself only once everything it listed is gone. An
  interrupted removal therefore leaves the product *more* cleanable, never a
  stranded orphan. Products registered before manifests existed are backfilled
  once, on the next scan.

- **The drop-in convention is now served to the UI, not buried in a YAML
  comment.** `registration.drop_in_rules()` returns the folder layout, the CSV
  columns, and the three-step legend-matching order; `/api/datasets` carries it,
  and the app prints it in the info panel when a dataset sits at
  `needs-legend` — the moment the user needs it. Also written up in the README
  under *Adding a dataset*. One source, so the help text cannot drift from the
  code that enforces it.
- **A corrupt-download failure now names the files.** `GLC_FCS30D_2019` fails to
  index — **16 of 35 files are truncated**. The raw `RasterioIOError: Read
  failed` is replaced by a message listing them and the re-fetch instruction,
  surfaced on the product as an `error` badge rather than a silent absence. See
  the root-cause note below.

### The damaged files: the app did not cause them, and cannot

*(2026-08-05. Two distinct questions were asked here; both are answered.)*

**Q1 — "is the app corrupting files that were fine on disk?" No, and it is now
structurally impossible.** This was the important question and it deserved
evidence, not reassurance:

| check | result |
|---|---|
| source rasters opened in mode… | **`r`** (read-only) |
| conversion output paths inside `data/` | **none** — all go to `cache/cog/` |
| `data/` files added/removed/modified by a full exercise run (tiles + footprints + chunked sampling over every local product) | **0 / 0 / 0**, across 939 files |
| mtimes of the damaged `GLC_FCS30D_2019` files | still **Jul 29 11:19**, unchanged through a week of app activity |

The one way GDAL could ever write into `data/` was its PAM sidecar mechanism
(a `.aux.xml` next to a raster, memoising statistics). That is now disabled at
package import (`harmonizer._protect_source_data`, `GDAL_PAM_ENABLED=NO`), so
the process has **no** write path into `data/` at all. The two `.aux.xml` files
present there are dated Jul 29 10:51 — written by QGIS or the download tool
before this app ever opened the folder. `scripts/verify_data_readonly.py`
re-proves all of this on demand and is the regression test for it.

**Q2 — "so where did the damage come from?"** From the transfer, before the
files ever reached this project. Every affected file is **truncated**: the
TIFF's own tile-offset table declares pixel data at byte offsets that run past
the end of the file. Nothing can repair that locally — the bytes were never
written.

The evidence is unambiguous:

| finding | value |
|---|---|
| files truncated in `GLC_FCS30D_2019` | 16 of 35 |
| **files sharing the exact byte length `219,414,528`** | **14** |
| mtime window of every truncated file | **11:19:01–11:19:07 (6 seconds)** |
| last intact file's mtime | 11:16:29 |
| truncated files in the other 891 rasters | **0** |

Independently LZW-compressed rasters of different regions never coincide on an
exact byte length; 14 doing so, all finishing within six seconds of each other,
is a single transfer batch cut off at a common point — a disk that filled, an
expired session/token, or a killed transfer — not a bad mirror. The old HRLC30
file carried the same 11:19 timestamp. `WorldCover_2020` (96), `JAXA` (619),
`Copernicus` (176) and the re-downloaded HRLC30 are **all complete**, which is
what rules out "these publishers ship broken files".

**`scripts/check_downloads.py`** makes this a first-class check. It compares each
TIFF's declared extent against its actual size, reading only headers and offset
tables — no pixel decompression — so it scans **927 files in 2.3 s**, against
~15 minutes for `indexer --check` (which decodes real windows and answers the
different question of *which regions* fail). It also flags the repeated-size
signature explicitly. `--delete` removes truncated files so a re-fetch replaces
them. Registration now uses it for the `error` badge.

**Practical rule:** run `python scripts/check_downloads.py` immediately after
any bulk download, before indexing. A truncated GeoTIFF opens fine and reports
correct metadata — the damage only surfaces later, and (per section 5 item 4) a
partially-readable file can render a *plausible but wrong* map from its
overviews.

**Division of responsibility, made explicit.** Downloading the data is the
user's job; this project neither fetches nor owns those files. What it owes the
user in return is (a) never to touch them — enforced above and tested by
`verify_data_readonly.py` — and (b) to *say clearly* when a file it was given is
incomplete, naming the files and the remedy, instead of failing with an opaque
`RasterioIOError` that reads like an app bug. Both halves are now in place.

**Fixes (5)** land alongside whichever stage touches the file first; the
auxiliary crash fix is verified by running an auxiliary-AOI job end to end.

---

## Appendix — deferred GEE-path speedups

Not part of the local-first effort; recorded so the findings aren't lost.

- Push the region into the label image: call `_label_image_for(product_id,
  region=...)` at `sampling.py:495` so Dynamic World's collection is
  `filterBounds`-restricted before the year-long modal composite — likely the
  single biggest GEE win.
- Parallelize the per-class loop in `_sample_map_gee` (`sampling.py:504`) with
  a small thread pool; the ~4 blocking round trips per class are independent
  I/O.
- Reuse the KD-tree thinning from 3.2 for the GEE path's candidates, and apply
  the same round-robin truncation fix at `sampling.py:448`.
- Revisit `bestEffort=True` on `present_classes` (`sampling.py:214`) and
  `_count_candidates_both` (`:337`): over a large region GEE silently coarsens
  the computation scale, which can make rare classes vanish from discovery
  with no warning — at minimum, log the effective scale.
