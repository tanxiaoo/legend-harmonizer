# Legend Harmonization — Pipeline Implementation Plan

A build specification for coding the tool one stage at a time. This document is
prose, not code: each stage describes its purpose, inputs, processing, output
artifact, and how to verify that artifact before moving on. Implement the stages
in order; each produces something runnable that the next depends on.

---

## 1. Overview and scope

The tool works out how the legends of two land-cover maps relate. For each class
in one map it finds the class (or classes) in the other map that it belongs to,
by describing each class as a probability distribution in a deep-learning
embedding space and comparing those distributions. The primary deliverable is a
**legend matching table** (a probabilistic crosswalk) plus the **affinity matrix**
behind it, both exported as CSV.

The method is embedding + Gaussian Mixture Model throughout. Each class is
characterised by what its pixels look like in the AlphaEarth embedding space; a
GMM is fitted per class; classes across the two maps are compared by
distribution distance. This answers questions of the form "does Dynamic World's
`Trees` correspond to CCI's `Forest` or `Cropland`" at the level of the
categories themselves — not by adjudicating individual pixels and not by deciding
which map is correct.

MVP scope: two maps, both reachable in one run — **CCI HRLC** as the reference
(basic) map, read from a local GeoTIFF, and **Dynamic World** as the compare map,
read from Google Earth Engine — over their overlapping region in the Eastern
Sahel of Africa, for the year 2019. The design keeps room to add more products
and more embedding models later, but the MVP is exactly this pair.

The harmonized-raster output (repainting one map into the other's legend) is out
of scope for the MVP and is not built here.

---

## 2. Decisions and default constants

All tunable values live here. Every stage refers back to this section rather than
restating numbers, so a change is made in one place. Values marked *(tune)* are
starting points to calibrate against real data, not fixed truths.

**Maps and embedding**

> **Per-map facts live in the product registry, not here.** The asset ids, bands,
> footprints, resolutions, and legends (class codes, names, colours) below are the
> MVP's values, but their **authoritative home is the per-map YAML files** in the
> product registry (section 2.5). Section 2 keeps only the *run-level* constants
> (sampling, buffering, GMM, affinity thresholds) plus the cross-map settings
> (working year, target CRS, GEE project). The lines below are a summary; treat
> the registry YAML as the source of truth when they disagree.

- Reference map: CCI HRLC, static 2019, read locally as GeoTIFF via rasterio.
- Compare map: Dynamic World, GEE asset `GOOGLE/DYNAMICWORLD/V1`, composited to an annual label by per-pixel modal class over the working year.
- Embedding: AlphaEarth Satellite Embedding V1, GEE asset `GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL`, 64 dimensions, kept at full 64 dimensions (no PCA by default).
- Working year: 2019. Target CRS for sample coordinates: EPSG:4326.

**Product footprints (derived from the source, never hardcoded).** A product's
footprint is a *property of the loaded source*, not a constant in config or the
registry:
- For a **local raster** (HRLC), read the raster's **actual bounds** with rasterio
  and reproject them to EPSG:4326. HRLC is *not* defined by any Sahel bounding
  box — that box is merely the extent of the particular tile currently in `data/`,
  and it changes when a different tile is loaded. Whatever tile is present defines
  the footprint; there is no fixed HRLC extent.
- For a **GEE dataset**, use the asset's coverage. WorldCover v200 and Dynamic
  World are **global**, so their footprint does not constrain the overlap (treated
  as unbounded / `None`).
- The **overlap region** for a run is the intersection of the *selected products'*
  derived footprints with the user's AOI, restricted to the working year. It is
  computed at run time from these live sources (see Stage 1 / Stage 5), never from
  a stored box.

> **Test swap (temporary).** To let Stage 1 be exercised without a local
> download, the reference map is currently ESA WorldCover v200 (GEE asset
> `ESA/WorldCover/v200`, an ImageCollection whose single global image is taken
> with `.first()`; sample the discrete `Map` band; class values 10, 20, 30, 40,
> 50, 60, 70, 80, 90, 95, 100). WorldCover v200 is a 2021 product, so the
> **working year is set to 2021** for the swap and the Dynamic World composite
> and AlphaEarth sampling follow suit. CCI HRLC (and its local rasterio adapter,
> left untouched) remains the eventual real reference; revert the working year to
> 2019 and repoint `default_registry()` back to HRLC when the local raster is in
> use.

**Sampling**
- Points per class: floor 300, target 1500 *(tune)*, both user-settable.
- Spatial declustering: minimum spacing between points within a class of 100 m *(tune)*, and points spread across the overlap by gridding it into cells of 0.25° *(tune)* and drawing per cell so one large patch cannot dominate a class's sample.
- Drop any point whose AlphaEarth vector is masked/no-data, or whose map label is a fill/no-data value.

**Run parameters (AOI and sample scale are first-class, with defaults).** A run is
configured by, at minimum: the two products (and which is reference), the working
year, the **AOI**, and the **sample scale** — the last two are proper run
parameters exposed in the UI, not fixed constants:
- **AOI** — the area of interest. Default: the **full overlap** of the selected
  products' derived footprints (section above). The user may narrow it by
  uploading a boundary or drawing on the map. AOI size is a real per-run choice
  (Stage 3 note): too small starves even common classes below the floor; too large
  is slow and blends distinct biomes into one class.
- **Sample scale** (`sample_scale_m`, default 10 m — the native label resolution).
  Exposed so a test run can trade spatial fidelity for speed by eroding/sampling at
  a coarser scale (e.g. 30–100 m) without touching the point floor/target.
- **Slow-combination warning.** A large AOI combined with a fine sample scale makes
  the Stage 2 server-side erosion and stratified sampling expensive and can hang
  the run. The UI must **estimate the cost of the chosen AOI × scale and warn**
  before launching (suggesting a coarser scale or a smaller AOI) rather than
  letting Stage 2 stall silently.

**Buffering**
- Erode each class mask inward by 2 pixels (~20 m at 10 m resolution) before sampling.
- Require a homogeneous 3×3 neighbourhood around each point (single class).
- Absent-vs-buffered-away rule (decides why a class is starved):
  - Below the floor **only after** eroding (its pre-erode count was adequate) → *buffered away*: relax its buffer to 1 pixel, resample, and flag it as sampled under looser purity.
  - Below the floor **even before** eroding → *genuinely rare*: flag the class `absent` and do not force-fit it; relaxing the buffer would not help.

**GMM**
- Components K per class: default 4, user-settable (suggested range 1–10).
- Covariance type: full. Under-floor classes fall back to diagonal covariance (see Stage 3).
- Auto-cap K so each component has at least ~50 points; if a class cannot support the requested K, reduce K and record that it was capped.
- Covariance regularisation `reg_covar`: 1e-6 *(tune)*, added to the diagonal of each covariance during fitting to keep it non-singular (especially for small classes).
- Random seed: 42 (sampling and GMM initialisation), so runs are reproducible.

**Affinity and decision**
- Distance: closed-form 2-Wasserstein (Bures–Wasserstein) between Gaussians, combined across GMM components (Mixture-Wasserstein / MW₂; see Stage 4).
- Raw similarity from distance: `s = 1 / (1 + d)`. This is the **absolute-match** signal and feeds the orphan floor; it is **not** what the row probabilities are built from.
- Row probabilities: a **temperature-scaled softmax over negative distances**, `P_ij = exp(−d_ij / T) / Σ_k exp(−d_ik / T)`, **not** a linear row-normalisation of the raw similarities. Rationale: `s = 1/(1+d)` compresses the distance range into a narrow band (~0.47–0.91 on the first real matrix), so linear normalisation flattens every row to near-uniform (entropy ≈ 0.99) and entropy can no longer separate `strong` from `mixed` — even a clear water→water best match reads `mixed`. The softmax sharpens the winner; smaller `T` is sharper.
- Softmax temperature `T`: 0.25 *(tune)*. Calibrated so the entropy of a focused row drops below the high-entropy threshold while genuinely spread rows stay above it.
- `strong` vs `mixed` split: the **top1−top2 margin** of the normalised (softmax) row — the gap between the best and second-best mapping probability — **not** entropy. `margin ≥ margin_threshold` → `strong`, else `mixed`. Rationale: entropy penalises tail-spread and **inverts on two-way ties**. A clean single winner with a long thin tail (e.g. Tree cover→Trees, peak 0.44) can score *higher* entropy than a genuine two-way split (e.g. Herbaceous wetland, entropy 0.771 vs Tree 0.818), so no entropy threshold separates "one clear match" from "two real matches." The margin reads a clean winner as large and a two-way tie as small, which is exactly the `strong`/`mixed` line.
- Margin threshold: 0.18 *(tune)* on the top1−top2 margin. Calibrated together with `T` after inspecting the first affinity matrix; revisit if `T` changes.
- Mapping entropy per row, normalised by `log(N)` so it lies in [0, 1], computed on the **softmax** probability distribution above. **Reported** in the matching table but **not** used to classify (see the margin split above).
- High-entropy threshold: 0.65 — retained as a reported diagnostic only; it is no longer the `strong`/`mixed` classifier.
- Absolute-affinity floor: on the row's best raw (pre-normalisation) similarity `s = 1/(1+d)`; below it the class is an orphan. **0.60** *(tune)* — read off the observed raw-similarity distribution (values bunch 0.47–0.91; 0.60 sits below every genuine best match and above the weakest, non-matching rows). Its **shipped default was `None`** (uncalibrated), and the guard remains: **while the floor is `None`, orphan classification must not run** — Stage 4 must fail or warn and leave orphan status unassigned rather than default the floor and mark every class an orphan. The two-pass flow (emit distribution → set floor → re-run) is how 0.60 was obtained.

**Semantic prior (Stage 8)**
- `semantic_prior_alpha`: **0.0** *(tune)* — exponent α on the semantic prior π in the fused logits `−d_ij / T + α · log π_ij`. 0 reproduces the AEF-only behaviour bit-for-bit; set after the Stage 8c sweep, with `T` held fixed.
- `semantic_veto_floor`: **0.10** *(tune)* — lower clip on each veto-attribute score (surface, cultivation, life form), so a mismatch is a strong penalty, never an impossibility (WorldCover grassland may contain uncultivated cropland; HRLC croplands include annual pastures).
- `semantic_orphan_floor`: **0.30** *(tune)* — a row is flagged `semantic_orphan` when `max_j π_ij` is below it. Reported alongside, never replacing, the observational orphan status.
- `semantic_prior_epsilon`: **1e-6** — lower clip on π before the logarithm (numerical only).
- Categorical correspondence (FAO cb5130en Appendix B, Table 8-1, scores ÷ 10; same value = 1.0):
  - life form: tree/shrub 0.6, tree/herbaceous 0.3, shrub/herbaceous 0.4, woody/lichen_moss 0.1, herbaceous/lichen_moss 0.4;
  - surface: vegetated/bare 0.3, water/snow 0.3, all other cross pairs 0.1;
  - cultivation: natural/cultivated 0.3;
  - leaf type and phenology: different 0.3.
- Interval attributes (cover %, height m, flooding months) use directed inclusion `|src ∩ tgt| / |src|`; unspecified on either side scores 1.0; open-ended bounds cap at cover 100, height 50, flooding 12.

**Multi-AOI absence handling (Stage 7)**
- Auxiliary AOI limit: **3** *(tune)* per run, so a run cannot fan out into an
  unbounded number of GEE sampling passes.
- **No absence-check scale, and no absence-check query.** Absence is read from the
  Stage 2/3 caches, never from a separate Earth Engine call: sampling *is* the
  observation (Stage 2 must call `present_classes` server-side to know what to
  sample), so once a run has sampled an AOI, what that AOI lacks is already on
  disk. A separate probe would re-pay for a known answer and give a worse one —
  it cannot separate `not_in_aoi` from `too_rare`, because rarity depends on the
  point floor after erosion, which only sampling establishes.
- Absence reasons (recorded per class, per side, distinct from status):
  - `not_in_aoi` — declared in the registry legend but no pixels observed in any of
    the run's AOIs. Registry-level signal (section 2.5 reconciliation check).
  - `too_rare` — observed in an AOI but below the point floor even before eroding
    (the Stage 2 absent-vs-buffered-away rule).

**Review (Stage 6 evidence explorer)**
- Representative locations sampled per class-pair: **10** *(tune)* (docs section 6.3 range 9–12) — few enough to review quickly, enough to be representative.
- Patch window: **256 px** *(tune)* on a side, in pixels of the label map, so the ground context shown is consistent regardless of location (≈ 2.5 km at 10 m). The window in metres is `patch_window_px × the label map's resolution`.
- Spatial declustering of the sampled locations **reuses the Stage 2 constants** (`min_spacing_m`, `grid_cell_deg`) rather than introducing new ones, so evidence points are spread exactly as sample points are.

**Output statuses** (assigned per class, on **both** sides — see Stage 7)
- `strong`: best match above the absolute floor and top1−top2 margin **at or above** the margin threshold (one clear winner).
- `mixed`: best match above the floor but margin **below** the threshold (genuine many-to-many / two-way tie).
- `orphan`: best match below the absolute floor (present but nothing corresponds).
- `absent`: could not be modelled from any of the run's AOIs, carrying an explicit
  **reason** (`not_in_aoi` or `too_rare`, above). Not force-fit and never silently
  dropped: an `absent` class still appears in the matching table with no candidates,
  and remains selectable in Stage 6 review so an expert may declare edges by hand.

These statuses are assigned **symmetrically**: every declared legend class of the
reference map *and* of the compare map carries one. A reference class's status is
read from its matrix row; a compare class's from its column (`absent` compare
classes have neither, and are reported as unmatched targets). See Stage 7.

---

## 2.5. Product registry (single source of truth for map metadata and legends)

Map metadata and legends are **not** hardcoded across the codebase. They live in a
**product registry**: one **YAML file per map** in a registry folder
(`harmonizer/registry/products/<map-id>.yaml`), each file the single source of
truth for that one map. Everything that needs to know a map's asset id, footprint,
resolution, or legend (names, colours) reads it from the registry — not from
constants in `config.py`, not from dicts in `affinity.py` / `tiles.py`, not from
the frontend. Section 2 keeps only the **tunable pipeline constants** (sampling,
buffering, GMM, affinity thresholds); **per-map facts move out of it into the
registry**.

A registry entry is a **property of the map**, distinct from the section-2 values,
which are properties of the *run*. Changing a threshold is editing section 2;
adding a map or fixing a class colour is editing that map's YAML.

### Two-layer schema (per YAML file)

Each file has two layers: **map-level fields** describing the source, and a
**legend list** describing its classes.

**Map-level fields.**
- `id` — stable short identifier used as the registry key and in filenames/URLs
  (e.g. `worldcover`, `dynamicworld`, `hrlc`).
- `display_name` — human-readable name shown in the UI.
- `provider` — the producing organisation / dataset family (e.g. `ESA`, `Google`).
- `access` — how the map is read, one of two shapes:
  - `method: gee` with `asset_id` (e.g. `ESA/WorldCover/v200`), and for an
    ImageCollection any compositing note (e.g. `.first()`, or annual per-pixel
    modal for Dynamic World); **or**
  - `method: local_raster` with `path` (e.g. a GeoTIFF under `data/`).
- `band` — the band holding the class values (e.g. `Map` for WorldCover,
  `label` for Dynamic World).
- `resolution_m` — native pixel resolution in metres.
- `available_years` — the years the map covers (a list or range), so a run's
  working year can be validated against the map.
- `crs` — the map's native CRS (e.g. `EPSG:4326`, or the raster's own CRS for a
  local GeoTIFF).
- `footprint` — the map's spatial extent as a `(min_lon, min_lat, max_lon,
  max_lat)` box in EPSG:4326, or `null` for a **global** map. See the note below
  on how this reconciles with section 2's "footprint derived from the source" rule.
- `licence` / `citation` — the map's licence and a citation string, so exports can
  attribute their sources.

**Legend list** — one entry per class:
- `code` — the raw integer class value in the raster/label band (e.g. `10`, or `0`
  for a Dynamic World class).
- `name` — human-readable class name (e.g. `Tree cover`).
- `color` — display colour as a hex string (`#RRGGBB`), the map's published
  palette colour.
- `description` — a short prose description of what the class means on the ground
  (optional but recommended; used in review, Stage 6).
- `shared_scheme` — **optional** mapping of this class onto a shared/common legend
  scheme (e.g. an IPCC/FAO-style super-class), for grouping classes across maps.
  Absent where no shared mapping is defined.

### Footprint: registry declaration vs. source-derived (reconciles with section 2)

Section 2 states a product's **operational footprint is derived from the loaded
source at run time** (rasterio bounds for a local raster; asset coverage for GEE)
and is never a stored box that constrains a run. The registry's `footprint` field
does **not** override that. It is the **declared/expected** extent used for
display and as a coarse pre-check (drawing the map's outline, offering it in the
UI, an early "these two can't overlap" refusal). The **authoritative** footprint
for sampling and overlap is still the source-derived one computed in Stage 1 /
Stage 5. For a **local raster the declared footprint should be left `null` or
treated as advisory**, because the actual extent is whatever tile is currently in
`data/` — exactly as section 2 says HRLC has no fixed extent. The reconciliation
check below compares declared vs. observed and flags drift.

### Registration flow (auto-detect, then confirm by hand)

Adding or refreshing a map runs a **registration flow** that pre-fills the YAML
from the source, so a human confirms rather than authors from scratch:

- **Auto-detected from the source** (filled in automatically):
  - `crs`, `footprint` bounds, and `resolution_m` — from the rasterio dataset
    (local) or the GEE asset's projection/geometry (GEE).
  - the **class codes actually present** — by scanning the raster / reducing the
    GEE image over a sample region, so the legend's `code` list starts from what
    the data really contains, not a guess.
  - `name` and `color` **where the source provides them** — from an embedded
    GeoTIFF colour table / category names, or from GEE asset properties
    (many GEE datasets publish class names and a palette in their metadata).
- **Left for a human to complete or confirm** (prompted, not invented):
  - `name`, `color`, and `description` for any class where the source carries no
    such metadata (e.g. a bare local GeoTIFF with codes but no colour table).
  - `display_name`, `provider`, `licence`/`citation`, `available_years`,
    `shared_scheme` — descriptive fields the source generally can't supply.

The flow is **automatic where the source allows and prompted where it doesn't**:
it writes a pre-filled YAML with the auto-detected fields populated and the
remaining fields left as clearly-marked TODO placeholders for a human to fill in.

### Reconciliation check (declared vs. observed)

A **reconciliation check** compares the legend declared in the YAML against the
codes actually observed in the source (over the run's AOI), and reports two
signals:
- **Undeclared codes** — a class code present in the raster/label band but
  **missing from the legend list**. This means the YAML is incomplete and should
  be updated (the class has no name/colour). Flagged loudly.
- **Absent-in-AOI classes** — a legend class **declared** in the YAML but **not
  observed** anywhere in the AOI. This is the **"absent in AOI"** signal: the
  class is legitimately part of the map's legend but has no pixels in this run's
  area, so it cannot be sampled or modelled. This is the registry-level source of
  the `absent` reasoning that Stage 2's floor/erosion rules then refine (a class
  absent from the AOI is genuinely rare here, not merely buffered away).

### The registry is the single source — what it replaces

The adapters, overlap logic, per-class toggles, matrix labels, and CSV export all
read map metadata and legend (names, colours) **from the YAML**, not from
hardcoded values in `config.py` or the frontend. Concretely, this registry
**replaces** the following existing hardcoded definitions:

- **`harmonizer/config.py` → `MapsConfig`**: the per-map facts
  `hrlc_bbox`, `worldcover_asset` / `worldcover_band`,
  `worldcover_class_values`, `dynamicworld_asset`, and `alphaearth_asset` move
  into the corresponding maps' YAML files (`footprint`, `access.asset_id`,
  `band`, and the legend's `code` list). `MapsConfig` retains only genuinely
  cross-map run settings (`gee_project`, `working_year`, `target_crs`,
  `embedding_dims`).
- **`harmonizer/registry/products.py` → `Product` / `default_registry()`**: the
  hardcoded `name`, `role`, `footprint`, and the per-product entries built in
  `default_registry()` are **loaded from the YAML files** instead. `Product`
  becomes a view over a registry file; the adapter factories stay, but their
  metadata (asset id, band, footprint) comes from the YAML.
- **`harmonizer/affinity.py` → `CLASS_NAMES`**: the class-name dictionary is
  replaced by the legend `name` fields read from the registry. `class_name(...)`
  resolves through the registry.
- **`harmonizer/tiles.py` → `_PALETTES`**: the per-class colour dictionaries are
  replaced by the legend `color` fields read from the registry, so tiles,
  toggles, and legends all colour from the one source.
- **Frontend (`web/`)**: any class names/colours currently emitted to or embedded
  in the page come from the registry via the API, not from constants in `app.js`.

**Verification (for the implementer, once built).** Register the two MVP maps
(WorldCover, Dynamic World) by running the registration flow and inspecting the
generated YAML: confirm the auto-detected `crs`, `footprint`, `resolution_m`, and
observed `code` list are correct, and that names/colours were pulled from the
source where available and left as TODO where not. Run the reconciliation check
over the AOI and confirm it flags any undeclared code and lists the absent-in-AOI
classes. Finally, confirm the adapters, `class_name(...)`, tile palettes, matrix
labels, and CSV export all read from the YAML by editing one class's `name`/`color`
in a YAML file and seeing the change propagate everywhere with no code edit.

---

## 3. Repository layout and stack

Stack: Python for the pipeline and backend, plain HTML/CSS/JS for the frontend.
Key libraries: `earthengine-api` (GEE access), `rasterio` and `pyproj` (local
raster and CRS handling), `geopandas`/`shapely` (footprints and overlap),
`numpy`/`scipy` (Bures–Wasserstein via `scipy.linalg.sqrtm`), `scikit-learn`
(GMM), `fastapi` and `uvicorn` (backend). Frontend uses Leaflet and Plotly from a
CDN — no build step.

```
legend-harmonizer/
├── docs/
│   └── PIPELINE.md            # this document
├── CLAUDE.md                  # standing instructions for Claude Code
├── pyproject.toml
├── run.py                     # starts uvicorn and opens the browser
├── data/                      # user drops the HRLC GeoTIFF here
├── cache/                     # sampled points, fitted GMMs, matrices, job store
├── harmonizer/
│   ├── config.py              # loads the constants from section 2
│   ├── registry/
│   │   ├── products.py        # loads the YAML registry + adapter interface
│   │   ├── products/          # one YAML per map (section 2.5) — single source of
│   │   │                      #   truth for map metadata and legends
│   │   └── adapters/          # hrlc_local, dynamicworld_gee, alphaearth_gee
│   ├── overlap.py             # Stage 1 / 5
│   ├── sampling.py            # Stage 2
│   ├── buffering.py           # Stage 2
│   ├── modeling.py            # Stage 3
│   ├── affinity.py            # Stage 4
│   ├── decision.py            # Stage 4
│   ├── review.py              # Stage 6
│   └── api.py                 # Stage 5
└── web/
    ├── index.html
    ├── app.js
    └── style.css
```

Each stage below maps to one or two of these modules.

---

## 4. Running the tool (once built)

The intended flow for an end user:

1. `git clone …` and install (`pip install -e .`).
2. Place the HRLC GeoTIFF in `data/`.
3. `earthengine authenticate` once, so GEE calls (Dynamic World, AlphaEarth) run under the user's own account. This is the only credential step; there is no login.
4. `python run.py`, which starts the local server and opens the one-page app in the browser.

There is no shared infrastructure: everything runs on the user's machine under
the user's GEE quota.

---

# Implementation stages

Build these in sequence. After each, run its verification and commit before
starting the next.

---

## Stage 1 — Data access

**Purpose.** Provide one uniform way to get, for any sample coordinate, the class
label from each map and the 64-dim embedding vector. Everything downstream reads
data only through this layer.

**Inputs.** The HRLC GeoTIFF path in `data/`; the GEE asset IDs for Dynamic World
and AlphaEarth from section 2; a set of coordinates in EPSG:4326.

**Processing.** Define a small adapter interface with a single operation: given a
list of coordinates, return values for those points. Implement three adapters
behind it.

The HRLC adapter reads locally with rasterio. Because the raster is not in GEE,
CRS alignment is this adapter's responsibility: sample coordinates arrive in
EPSG:4326, so transform each coordinate into the HRLC raster's own CRS before
reading the pixel value. Read the class value at the transformed location. Treat
fill/no-data values as "no label." Getting this transform wrong silently returns
neighbouring-pixel labels, so it must be explicit and tested.

The Dynamic World adapter reads from GEE. Because Dynamic World is per-scene, it
first composites the 2019 collection to a single annual label by taking the modal
class per pixel over the year, then samples the label at the given points.

The AlphaEarth adapter reads from GEE, sampling the 64-dim embedding at the given
points for 2019. Points whose embedding is masked/no-data are marked so they can
be dropped later.

Follow the sampling order that avoids moving rasters: choose points and read HRLC
labels locally first, then send only the point coordinates to GEE to fetch
Dynamic World labels and AlphaEarth vectors. Only coordinates and small point
tables cross the network; never whole rasters.

**Output artifact.** For a list of coordinates, a table with columns for each
map's class label and the 64-dim embedding vector, with masked/no-data points
flagged.

**Verification.** A short script that feeds five hardcoded coordinates inside the
Eastern Sahel box and prints, for each, the HRLC label, the Dynamic World label,
and the embedding vector. Confirm labels are plausible for those locations and
vectors have 64 values. This proves CRS handling and GEE access before anything
else is built.

---

## Stage 2 — Sampling and buffering

**Purpose.** Produce clean, per-class sample points for each map independently.
Because classes are modelled separately, the two maps' point sets are unrelated —
there is no need for co-located points across maps.

**Inputs.** The overlap region (from the overlap computation, shared with Stage
5); the sampling and buffering constants from section 2; the Stage 1 adapters.

**Cost control (server-side).** The annual label image is built **once per map and
clipped to the sampling region** before any reduction, so the expensive composite —
notably Dynamic World's year-long per-pixel modal label — is bounded to the AOI and
reused across every class rather than re-evaluated over the global asset on each
call. Per class the candidate-availability counts at the two buffers (pre- and
post-erode, for the absent-vs-buffered-away rule) are read in a **single**
`reduceRegion` (two mask bands) rather than two round-trips. These are performance
measures only; they do not change which points are sampled or how classes are
modelled. Even so, a large AOI at a fine sample scale can still be slow on GEE —
this is why the AOI and sample scale are run parameters and the UI warns on a
costly combination.

**Where the raster work happens.** Erosion and stratified sampling are raster
operations, but the Stage 1 adapters only sample *points*. So Stage 2 does the
raster work at the source and returns only candidate coordinates:

- *GEE maps (the MVP: WorldCover reference and Dynamic World compare).* Erode and
  stratified-sample **server-side in GEE**. Build the annual label image (as in
  Stage 1), then per class: erode the class mask inward by the configured pixels
  (e.g. a focal-min / morphological erosion on the binary mask), optionally
  confirm a homogeneous 3×3 neighbourhood, and draw stratified candidate points
  with the configured minimum spacing and per grid-cell quota. Only the resulting
  candidate coordinates come back over the network — never the rasters. Those
  coordinates then go through the Stage 1 adapters to fetch the map label (a
  cross-check) and the AlphaEarth embedding for each point.
- *Local raster (later HRLC implementation).* When HRLC is read locally, the same
  erosion and stratified sampling run against the rasterio dataset in-process, and
  the chosen coordinates go through the HRLC and AlphaEarth adapters. This path is
  built later alongside the real HRLC reference; the GEE path above is what Stage 2
  implements now.

**Processing.** For each map, and within it for each class, sample points from the
interior of that class's patches using the mechanism above. First buffer: take the
class mask, erode it inward by the configured pixels so boundary and mixed pixels
are removed, leaving homogeneous cores; optionally confirm a single-class 3×3
neighbourhood at each candidate point. Then stratify: draw up to the target number
of points per class, enforcing the minimum spacing and spreading points across the
overlap by gridding it and sampling per cell, so one large patch cannot dominate a
class's sample.

Drop points with masked embeddings or no-data labels. After sampling, check each
class's surviving count against the point floor, and decide *why* any starved
class is short using the absent-vs-buffered-away rule from section 2: if it was
below the floor **only after** eroding (its pre-erode count was adequate) it was
buffered away — relax its buffer to one pixel, resample, and flag the looser
purity; if it was below the floor **even before** eroding it is genuinely rare in
the AOI — flag it `absent` and do not force-fit it, since relaxing the buffer
cannot help. This requires keeping the pre-erode candidate count per class to
compare against.

Unequal point counts across classes are expected and acceptable; only counts
below the floor need action.

**Output artifact.** A per-map point table: coordinates, class label, embedding
vector, and per-class survival counts and buffer-relaxation flags. Saved to
`cache/`.

**Verification.** Print the per-class counts for both maps. Confirm every class is
at or above the floor or is flagged (relaxed or absent), and that counts look
reasonable relative to how common each class is in the region.

---

## Stage 3 — GMM modeling

**Purpose.** Turn each class's embedding vectors into a probability distribution,
so classes can be compared as distributions rather than point clouds.

**Inputs.** The per-map point tables cached by Stage 2 (`cache/samples_<product>.npz`
plus the JSON sidecar), read as-is. Stage 3 consumes the **full 64-dim** embedding
vectors already sampled and **does not resample** — no GEE access here. Also the
GMM constants from section 2, and the per-class Stage 2 flags (`status`,
`buffer_relaxed`, `absent`, point count).

**Processing.** Fit one Gaussian Mixture Model per class using that class's
embedding vectors, with full covariance in 64 dimensions and the configured
number of components. Before fitting, cap K for a class if its point count cannot
support the requested components (using the minimum points-per-component rule),
and record when a class was capped. Apply covariance regularisation — add
`reg_covar` (section 2, default 1e-6) to the diagonal of each component covariance
during fitting — to keep covariances non-singular, which matters most for small
classes. Use the fixed seed for initialisation so fits are reproducible. Keep the
fitted parameters (weights, means, covariances) for each class.

Classes flagged `absent` in Stage 2 are not force-fit and carry through with no
GMM (they are handled downstream, not modelled here).

**Under-floor classes (relaxed, and the permanent thin-class rule).** A class
whose point count is **below the floor** — whether flagged `relaxed` in Stage 2
(sampled under a looser 1-pixel buffer) or simply thin — is still fit, but with a
**simpler covariance to avoid overfitting** a full 64×64 covariance on few points:
force `covariance_type` to **diagonal** for any class under the floor,
regardless of the configured default. K is also capped as usual. The `relaxed`
flag (and, more generally, an "under-floor / low-confidence" marker) is
**propagated through Stage 3's output** so it reaches the matching table in Stage 4
as a lower-confidence marker on that class's matches.

The floor here is the **same** threshold used in Stage 2 — `CONFIG.sampling.points_floor`
(section 2, default 300) — read from config, not redefined in Stage 3, so "below
the floor" means exactly what it did during sampling.

This is a permanent rule, not a test-only workaround. Some classes are genuinely
rare and fall below the floor **regardless of AOI size** — e.g. mangroves, snow
and ice — so there will always be under-floor classes to model carefully even at
full scale. Relatedly, **AOI size is a real per-run parameter**, not just a test
knob: too small and even common classes are starved below the floor; too large and
the run is slow and mixes distinct biomes into one class's distribution. Choosing
the AOI is part of configuring a run.

Note for the implementer: full covariance in 64 dimensions is parameter-hungry,
which is exactly why the Stage 2 point floor exists. Do not silently drop
dimensions to compensate; instead rely on the floor, the K auto-cap, and the
simpler-covariance fallback for under-floor classes above.

**Output artifact.** A stored set of fitted GMMs, one per class per map, with
their component parameters, the point count, the capped-K flag, the covariance
type actually used, and the propagated under-floor / `relaxed` (low-confidence)
flag for each. Saved to `cache/`.

**Verification.** For each class, print the number of components actually used,
the point count, the covariance type used, and whether K was capped. Confirm that
every under-floor class was fit with the simpler covariance and carries the
low-confidence (`relaxed` / under-floor) flag through to the output, that `absent`
classes were skipped rather than force-fit, and that K-capping only happened where
expected.

---

## Stage 4 — Affinity matrix, decision, and matching table

**Purpose.** Compare every reference class to every compare class, turn the
comparisons into a probabilistic matrix, classify each class, and produce the
matching table — the core result.

**Inputs.** The fitted GMMs from Stage 3; the affinity and decision constants from
section 2.

**Processing.** Between each reference class and each compare class, compute the
distribution distance using the closed-form 2-Wasserstein (Bures–Wasserstein)
distance between Gaussians. For a single Gaussian pair `p = N(m₁, Σ₁)` and
`q = N(m₂, Σ₂)` the **squared** distance is

```
W₂²(p, q) = ‖m₁ − m₂‖² + Tr( Σ₁ + Σ₂ − 2 (Σ₁^½ Σ₂ Σ₁^½)^½ )
```

— the squared gap between the means plus a covariance term comparing the two
covariance matrices via a matrix square root. This per-Gaussian squared distance is
the building block for the mixture distance below.

**GMM-to-GMM distance.** The distance between two GMMs is computed with a
**Mixture Wasserstein (MW₂)** formulation: each mixture is treated as a discrete
distribution over its Gaussian components, the ground cost between a pair of
components is their (squared) Bures–Wasserstein distance, and the mixture distance
is the optimal transport between the two mixtures' component weight vectors under
that cost. So the inputs are the component-wise Bures–Wasserstein distances and the
mixture weights; the output is a single distance `d` that feeds the similarity
`s = 1 / (1 + d)`. This reduces to the plain Bures–Wasserstein distance when both
GMMs have a single component, which is the sanity check for the implementation.

The implementation must use **one, well-defined combination method applied
consistently** across every class pair — not an ad-hoc heuristic that varies with
the number of components. Do not use `scipy.stats.wasserstein_distance`, which is
one-dimensional only and cannot compare 64-dim distributions.

**Mixed covariance types.** After Stage 3, full-covariance and diagonal-covariance
GMMs coexist in the same run (under-floor classes are fit with diagonal covariance;
see Stage 3). The distance code must therefore **not assume a uniform covariance
shape**: reconstruct every component's covariance to a full D×D matrix before
computing the Bures–Wasserstein term — a diagonal covariance is expanded to a full
matrix (`np.diag`) and treated as full internally. The stored covariance shape is
`(K, D, D)` for full fits and `(K, D)` for diagonal fits (as sklearn stores them,
recorded per class in Stage 3's output), so branch on `covariance_type_used` when
loading. A diagonal-vs-full or diagonal-vs-diagonal pair is a valid comparison and
must compute correctly, not raise on the shape mismatch.

Convert each distance to a similarity with `s = 1 / (1 + d)`, forming an M×N raw
similarity matrix (reference classes by compare classes). Keep these raw
similarities — they carry the absolute-match information needed to detect orphans
(the floor is tested against them), and they are **not** what the row
probabilities are built from.

Build each reference row's probability distribution with a **temperature-scaled
softmax over the negative distances**, `P_ij = exp(−d_ij / T) / Σ_k exp(−d_ik / T)`
with temperature `T` (section 2), rather than linearly normalising the raw
similarities. A linear normalisation of `s = 1/(1+d)` flattens every row to
near-uniform, because `s` compresses the distance range into a narrow band, so
entropy cannot separate a focused match from a spread one — even an obvious
water→water pairing then reads `mixed`. The softmax operates on the distances
directly and `T` sharpens the winner. Compute the normalised mapping entropy of
each row on this softmax distribution.

**Emit the raw-similarity distribution for calibration.** The absolute-affinity
floor (section 2) starts uncalibrated (`None`) and can only be set from real
values. Stage 4 must therefore **emit the distribution of raw (pre-normalisation)
similarities** — at minimum the per-row best raw similarity across all reference
rows, and ideally summary statistics (min/median/max, or a small histogram) — so
the floor can be read off actual data rather than guessed. This is a required
output of the stage, not an optional diagnostic.

Classify each reference class into one of the four statuses using two signals
together: the row's best raw similarity against the absolute floor (is there a
real match at all), and the row's **top1−top2 margin** against the margin
threshold (is there one clear winner or a genuine tie). Map the combinations to
`strong`, `mixed`, `orphan`, and `absent` as defined in section 2. The margin, not
entropy, decides `strong` vs `mixed`: entropy penalises tail-spread and inverts on
two-way ties, so a clean single winner with a thin tail can score higher entropy
than a real two-way split — no entropy threshold separates them. Entropy is still
computed and reported per row, just not used as the classifier. **Guard the orphan
classification against an uncalibrated floor:** if the absolute-affinity floor is still `None`, do not run
the orphan test — fail or warn and leave orphan status unassigned, rather than
treating the missing floor as zero (which passes everything) or as some default
(which could mark every class an orphan). The intended flow is to run the stage
once to emit the raw-similarity distribution, set the floor from it, then re-run to
assign orphan status. Provide a direction toggle so the same matrix can be read
compare-to-reference as well as reference-to-compare.

**Output artifact.** Three CSVs. **Both** affinity matrices are saved, because they
serve different purposes and neither can be reconstructed from the other without
loss:

- `raw_similarity.csv` — the full M×N grid of **raw (pre-normalisation)**
  similarities `s = 1 / (1 + d)`. This is the matrix used to **calibrate the
  absolute-affinity floor** (its values are the raw-similarity distribution
  emitted for calibration above) and to test each row's best match against that
  floor for orphan detection.
- `normalized_affinity.csv` — the same grid turned into per-row probability
  distributions by the **temperature-scaled softmax over negative distances**
  (section 2), so each reference row sums to 1 over compare classes. This is the
  matrix used for **mapping entropy and for the matching probabilities** in the
  matching table. (It is a softmax of the distances, not a linear normalisation of
  `raw_similarity.csv`, so it cannot be reconstructed from that file.)

Plus the matching table CSV — one row per class pairing, with reference class,
compare class(es), mapping probability (from `normalized_affinity.csv`), and status
— which is the primary deliverable. All three saved to `cache/` and offered for
download in Stage 5.

**Confidence column.** The matching table must carry the per-class confidence flag
propagated from Stage 3 (the `low_confidence` marker — set when a class was
under-floor or `relaxed`, and reflected in its diagonal-covariance fit) as an
explicit column, so a mapping backed by a thin, low-confidence GMM is **visibly
distinguished** from one backed by a well-sampled class. A match can be `strong` on
distance yet rest on a diagonal-covariance fit of a few hundred points; the reader
must be able to see that. Carry the flag for the reference class (and, where the
table names compare classes, theirs too) rather than silently folding it into the
status.

**Verification.** Open the matching table and confirm obvious pairs read correctly
(water to water, forest to trees as `strong`), that at least some classes come back
`mixed` with a sensible split, and that entropy and status are consistent with the
probabilities. At this point a working crosswalk exists from the command line,
with no UI — the first real milestone.

---

## Stage 5 — API and UI

**Purpose.** Wrap Stages 1–4 behind a small local API and a one-page frontend, so
the pipeline is usable without the command line.

**Inputs.** The pipeline modules from Stages 1–4; the run flow from section 4.

Stage 5 is split into three verifiable sub-stages. **Build 5.1 first and verify it
before starting 5.2.** 5.2 builds on top of 5.1 — it adds the map-comparison
surface to the working app, it does not replace it. 5.3 is a **presentation-only**
pass over 5.2: it restyles and re-lays-out the same app without changing behaviour.

- **5.1 — API + minimal run UI.** All backend endpoints, plus a minimal HTML page
  to select two products, set an AOI, run the pipeline, view the affinity matrix
  as a heatmap, and download the CSVs. **No split map, no class toggles, no
  footprint/overlap drawing** yet.
- **5.2 — Comparison map UI.** The split-map view of the two land-cover maps with
  per-class show/hide toggles, footprints and overlap drawn on the map, and the
  upload/draw/full-overlap AOI modes — layered onto the 5.1 app.
- **5.3 — Workspace layout + theme.** A pure UI pass over 5.2: layout and styling
  only, no changes to endpoints, run logic, toggle behaviour, or the matrix
  computation.

---

### Stage 5.1 — API and minimal run UI

**Purpose.** Stand the pipeline up behind a local API and the smallest possible
page: pick two products, pick an AOI, run, see the affinity matrix as a heatmap,
download the CSVs. This is the end-to-end backbone; 5.2 decorates it.

**Endpoints.** Build a FastAPI backend whose endpoints mirror the flow:
- **product list** — the registry's products with their `id`, `name`, `kind`, and
  `role`, so the UI offers only valid reference × compare pairings (registry-driven;
  an impossible pairing is never offered).
- **overlap** — given two product ids and an AOI, **derive each product's footprint
  from its source** (rasterio bounds for HRLC, asset coverage for GEE — section 2),
  intersect the two footprints with the AOI, and return the overlap geometry/bbox;
  refuse early if the overlap is empty. Also **estimate the AOI × sample-scale
  cost** and return a **slow-combination warning** (section 2) when it risks
  hanging Stage 2. **A *global* (unbounded) sampling region is a hard blocker, not
  a warning**: when both selected maps are global and the AOI is blank, the full
  overlap is the entire globe, and Stage 2 sampling it server-side will time out on
  GEE (`EEException: Computation timed out`). The endpoint flags this as not
  runnable and `/api/run` refuses it (400) before backgrounding — the user must
  supply a bounding-box AOI to bound the run. (A cache-reuse run does no GEE
  sampling and is exempt.) (5.1 returns the overlap as data; drawing it on a map is
  5.2.)
- **run job** — launch a harmonization run (Stages 1–4) as a **background job** and
  return a job id. Because it is local and single-user, the job runner is FastAPI
  background tasks with a simple in-process/`cache/` job store — no external queue.
  **Cache reuse.** Stage 2 (server-side GEE sampling) is the slow, quota-consuming
  part, so a run first checks whether the cache already holds a result for the
  *same inputs*: the product pair (and which is reference), the year, the AOI, and
  the run parameters that affect sampling/fitting (sample scale, K, point
  floor/target) are hashed into a **run signature** stored beside the GMM cache.
  If that signature matches and both maps' `cache/gmm_<product>.json` are present,
  the run **skips Stage 2/3 and reuses the cached GMMs**, recomputing only the
  cheap Stage 4 (which also re-reads the live calibration constants). Any change to
  a signature input — or a **force-refresh** flag on the request — invalidates the
  reuse and re-samples from GEE. This makes "run again without changing anything"
  fast and offline, while never silently serving a stale result for changed inputs.
- **progress** — poll a job id for its state and progress (queued / running with a
  stage/percent / done / failed with a message).
- **results** — for a finished job, return the affinity matrix (for the heatmap)
  and the matching table with its full Stage-4 decision detail (status, margin,
  entropy, low-confidence — presented, not recomputed).
- **exports** — download `raw_similarity.csv`, `normalized_affinity.csv`, and the
  matching table CSV that the run wrote to `cache/`.

GEE calls run under the user's authenticated account; no credentials pass through
the API.

**Run parameters.** The request that launches a run carries the three real choices
(area, the two maps with which is reference, year) plus the adjustable inputs —
**sample scale**, GMM component count, and point floor/target — defaulting from
config where omitted. The **calibration constants** (softmax **temperature**,
absolute-affinity **floor**, **margin** threshold) are **read live from config**;
exposing them as request overrides is allowed but the point is that changing them
must not require a code change (they are the `(tune)` values Stage 4 already reads
from config).

**Minimal frontend.** A single HTML page (plus `app.js`, `style.css`) — plain
HTML/CSS/JS, Plotly from a CDN for the heatmap, no build step. It lets the user:
select the two products and which is reference; set an AOI (for 5.1, a bounding
box entered/typed or the full-overlap default returned by the overlap endpoint —
upload/draw comes in 5.2); adjust the run parameters; launch the run and watch its
progress; then see the **affinity matrix as a heatmap** and the **matching table**
with `strong` / `mixed` / `orphan` **visually distinguished** and low-confidence
rows marked; and **download the CSVs**. No split map, class toggles, or
footprint/overlap drawing in 5.1.

**Output artifact.** A running local app: `python run.py` starts the server and
opens the page; the user picks the two products and an AOI, runs, watches progress,
sees the matrix heatmap and the decorated matching table, and downloads the CSVs.

**Verification.** From a clean checkout, `python run.py` and in the browser choose
**WorldCover × Dynamic World** (the test-swap pairing) with the full-overlap
default for the working year, run it, and confirm: the run completes via the
background job with progress; the affinity-matrix heatmap and the matching table
appear; the matching table's statuses/margins/entropies/low-confidence and the
downloaded CSVs **match what the Stage 4 CLI (`scripts/verify_stage4.py`)
produced**. This proves the whole API + minimal UI path end-to-end before the
comparison map is added. Also confirm **cache reuse**: running the same inputs a
second time reports that it reused the cache and returns quickly without
re-sampling GEE, while changing an input (or setting force-refresh) re-samples.

---

### Stage 5.2 — Comparison map UI

**Purpose.** Turn the 5.1 run app into the full comparison UI, with the map as the
dominant element. Built **on top of** the working 5.1 app and its endpoints.

**Processing.** Add to the frontend a **split-map view** of the two selected
land-cover maps side by side (tiles served under the user's GEE account / the
Stage 1 sources), each with **per-class show/hide toggles**. Draw both products'
**derived footprints and their overlap outline** on the map before any run (from
the 5.1 overlap endpoint). Let the **AOI** be set by **uploading a boundary,
drawing on the map, or accepting the full-overlap default**, feeding whichever
back into the same run flow. The map uses Leaflet from a CDN — no build step. New
backend support is added only where the map needs it (e.g. tile/style endpoints
for the two label maps and any class legend), reusing the 5.1 run/results/export
path unchanged.

**Output artifact.** The full app: both maps visible side by side with working
per-class toggles, footprints and overlap drawn, AOI selectable by
upload/draw/default, and the 5.1 run → heatmap → matching table → CSV flow intact.

**Verification.** Load the app and confirm: both selected maps display side by side;
per-class show/hide toggles work independently per map; and the overlap outline
drawn on the map is correct (matches the overlap endpoint's geometry for the
selected products and AOI).

---

### Stage 5.3 — Workspace layout and theme

**Purpose.** Turn the 5.2 single scrolling page into a fixed, full-viewport
three-column workspace with a dark theme. This is a **layout and styling pass
only** — no changes to endpoints, run logic, toggle behaviour, or the matrix
computation, and no new functionality. Every 5.2 element `id` is preserved so
`web/app.js` is untouched; the work is in `web/index.html` (restructure) and
`web/style.css` (theme).

**Processing.** Restructure the page body into a fixed `100vh` three-column grid,
no page scrolling — each column scrolls internally if needed:

- **Left — run parameters.** Choose reference/compare maps, set the AOI (draw /
  upload / bounding box / full-overlap default), the run parameters, and the Run
  button.
- **Centre — comparison maps.** The two split maps side by side, sized to fill the
  column height, each with its per-class legend toggles below.
- **Right — output.** The affinity matrix (heatmap) and the matching table with
  the `strong` / `mixed` / `orphan` / `absent` statuses.

Apply a dark theme: dark panel backgrounds, light text, one accent colour for
buttons and active states. **Do not restyle the land-cover class colours** — those
are legend *data* (Tree cover green, Water blue, …), applied inline from the API to
the legend swatches and the map tiles, and must stay exactly as they are. Only the
panel / text / chrome colours change.

**Output artifact.** The same 5.2 app, re-laid-out as the fixed three-column dark
workspace, with all existing functionality intact.

**Verification.** Load the app and confirm: the three columns fill the viewport
with no page scroll (columns scroll internally); the maps, legends, run flow,
toggles, overlap drawing, and CSV downloads all still work exactly as in 5.2; and
the land-cover swatch/tile colours are unchanged (only the surrounding chrome is
dark-themed).

---

### Stage 5.4 — Explorer-style workspace redesign

**Purpose.** Re-lay-out the 5.3 workspace so the two comparison maps dominate the
viewport (like the Esri Sentinel-2 Land Cover Explorer) and replace the unhelpful
on-screen affinity heatmap + matching table with a **mapping-probability Sankey**.
Frontend only — endpoints, run logic, and the matrix computation are unchanged; the
`/api/jobs/{id}/results` payload already carries what the Sankey needs.

**Processing.** Split the viewport top-to-bottom: the **top** holds the two maps side
by side (each filling its half), with a floating map-picker + AOI-tool box over each
map and that map's **legend directly beneath it**. The **bottom band** holds the AOI
coordinate inputs + tools, the run parameters + Run, and the **ECharts Sankey**
(with the CSV download links in its header). Replace the checkbox/square legend
toggles with **color chips**: **left-click shows only that class; Ctrl/Cmd+left-click
adds/removes a class** from the visible set (faded chip = hidden). The **affinity
heatmap and matching table are removed from the screen** and kept only as CSV
downloads. Land-cover class colours stay as legend data (swatches, tiles, and Sankey
nodes); only the panel/chrome colours are themed.

**Output artifact.** The re-laid-out app: two dominant side-by-side maps with
floating pickers and chip legends, a bottom control band, a mapping-probability
Sankey, and CSV downloads — all existing run/toggle/overlap/AOI behaviour intact.

**Verification.** Load the app and confirm: the two maps fill the top of the viewport
side by side with their legends beneath them and no page scroll; left-click on a
legend chip shows only that class and Ctrl+click adds/removes classes, with tiles
updating; a run produces a Sankey whose links go from reference classes to compare
classes with widths matching the mapping probabilities; the three CSVs still download
and match the Stage 4 output; and land-cover colours are unchanged (only chrome is
themed).

---

## Stage 6 — Human-in-the-loop review

Replaces the earlier Stage 6 stub. Stage 6 is where an expert resolves the class
correspondences the algorithm could not settle, using ground evidence rather than
the algorithm's ranking, and where those decisions feed back to improve future
proposals. It is a build spec in the same prose style as the other stages: purpose,
inputs, processing (in parts), output, verification, and a suggested build order.

**Purpose.** Let an expert investigate the semantic relationship between legend
classes and confirm correspondences, basing the decision primarily on satellite
imagery and legend definitions, with the affinity ranking serving only as guidance.
Confirmed decisions become authoritative edges in the matching table and, separately,
become training signal that refines the model's future proposals — without ever
overwriting what the expert confirmed.

**Inputs.** The matching table with per-edge affinity and statuses (Stage 4); the
fitted GMMs and per-class sample points (Stages 2–3); the two selected products,
their legends (from the registry), the AOI and year; GEE access for imagery and for
co-located-pixel queries.

---

### 6.1 Review queue — what an expert looks at

Build the queue from the matching table by status, because the different statuses
need different questions:

- **`mixed`** — the primary case. Real correspondence exists but is split across
  several compare classes (e.g. Shrubland → Shrub / Trees / Built). The expert
  confirms which candidate edge(s) are correct.
- **`orphan`** — included, but the question differs: the best match fell below the
  affinity floor, so the expert confirms whether the class is *genuinely* unmatched
  (a real legend divergence, kept as orphan) or whether a weak-but-real correspondence
  should be promoted.
- **`absent`** — excluded. Too few samples in the AOI to model; there is nothing to
  adjudicate. It stays flagged in the output but never enters the queue.
- **`strong`** — excluded, with one exception: a `strong` edge on a
  **low-confidence** class (thin/relaxed GMM) is flagged as *optionally* reviewable,
  not forced into the queue.

Order the queue **low-confidence first**, since those are where the model is least
sure and expert input helps most.

---

### 6.2 Evidence explorer — a three-mode contingency browser

The explorer lets the expert walk the relationship between the two legends by
selecting classes on each side. It is a class-pair exploration tool, not only a
disagreement viewer: it surfaces both agreement and disagreement. There are exactly
three modes, corresponding to a cell, a row, and a column of the contingency
structure:

1. **Both classes selected** (a cell) — e.g. reference = Shrubland, compare = Grass.
   Retrieve co-located samples where both conditions hold. Both map panels show the
   same location, each displaying its own dataset's label. Use: evaluate a specific
   pairwise correspondence.
2. **Reference class only** (a row) — e.g. reference = Shrubland, compare = all.
   Retrieve representative samples of the reference class; the compare panel shows
   whatever it labels at each location. Use: see how one reference class distributes
   across the whole compare legend (one-to-many).
3. **Compare class only** (a column) — e.g. reference = all, compare = Grass.
   Retrieve representative samples of the compare class; the reference panel shows
   whatever it labels at each location. Use: see which reference classes feed into
   one compare class (many-to-one).

**Hard boundary — co-occurrence is evidence retrieval only, never scoring.** Every
mode is powered by a co-located-pixel query (find pixels where reference = X and/or
compare = Y at the same location). This locates representative places for a human to
inspect. It must never be used to derive or weight correspondences — that remains the
embedding + GMM + Bures–Wasserstein method. The contingency counts are a way to find
evidence, not a vote. Keep this boundary explicit in the code and comments so it does
not erode.

**The co-located query is the one new server-side operation — scope and mechanics.**
Unlike Stage 2, which samples each map's classes *independently* (there are no
co-located points across maps), mode 1 needs a **joint mask** evaluated pixelwise:
`(reference label == X) AND (compare label == Y)` at the same location. Building that
mask means **co-registering the two label images** to a common scale and projection
server-side before intersecting them. For the MVP test-swap both maps are GEE-native
(WorldCover reference, Dynamic World compare), so the join and the stratified draw
run in GEE exactly like Stage 2's mask sampling — reuse `_label_image_for`,
`_grid_cell_image`, `_stratified_candidates`, and `_thin_by_spacing` from
`sampling.py`, adding only the two-image intersection and the joint-scale choice
(default: the coarser of the two maps' resolutions). Modes 2 and 3 (row / column) are
just a **single-class Stage 2 sample** of the reference or compare class and should
call the existing `_stratified_candidates` path rather than a new mechanism.

**Local-raster (HRLC) join is deferred, like Stage 2.** When the reference is the
local HRLC raster, one side is not in GEE, so the joint mask cannot be built
server-side; the intersection must run in-process against the rasterio dataset. That
path is built later alongside the real HRLC reference — the two-GEE-product join above
is what Stage 6b implements now (mirroring how Stage 2 defers the local-raster path).

---

### 6.3 Evidence patches

The unit of evidence is a **fixed-size patch centered on a sampled representative
pixel**, not a single pixel (too little context to judge) and not the whole
intersection (a scatter of thousands of unrelated locations, with no single coherent
area to show).

For a selected mode, sample **N representative locations** from the qualifying pixels
and, at each, show a fixed patch:

- **N = `CONFIG.review.patches_per_pair`** locations per class-pair (default 10, range
  9–12; section 2) — few enough to review quickly, enough to be representative.
- **Spatially declustered** — spread the sampled locations across the AOI using the
  same gridding/min-spacing approach as Stage 2 (`CONFIG.sampling.min_spacing_m`,
  `grid_cell_deg`), so they are not all drawn from one field.
- Fixed **patch window** (`CONFIG.review.patch_window_px`, default 256 px ≈ 2.5 km at
  10 m) so context is consistent regardless of location, with the queried pixel's
  label footprint outlined so the expert knows exactly which pixel the label refers to.

**Backend vs. imagery split.** Stage 6b is the **backend engine** and returns, per
location, only the **center coordinate and each side's label** there — the data
needed to fly the maps and outline the queried pixel. It does **not** render or fetch
patch imagery: the actual patch (basemap tiles, §6.4) is drawn client-side by the
6c Leaflet layers at the returned coordinate and window size. So the patch window is a
number 6b emits (for the outline) and 6c consumes (for the view), not a raster 6b
produces.

Each patch is a **location**: selecting it flies **both** synchronized map panels to
that spot, where the expert inspects it across basemaps and years.

---

### 6.4 Multi-source basemaps

Because the decision rests on the expert seeing ground truth, and a single Sentinel-2
date often cannot resolve the ambiguous classes (seasonal cropland vs bare fallow, a
wetland that floods only part of the year), provide switchable basemap layers on
**both** synchronized panels:

- Google Satellite
- ESRI World Imagery — latest, plus Wayback year-end snapshots 2018–2025 (no API key)
- Bing
- Sentinel-2 cloudless — 2018–2024
- Planet PlanetScope — monthly mosaics 2016–2026

Seasonal and multi-year imagery is decision-critical for exactly the classes that end
up `mixed`, so the basemap/year switcher is a primary control on the maps, not a
buried setting.

---

### 6.5 Expert decision — edge-level confirmation

The expert decides based on the imagery and the legend definitions first; the affinity
ranking is shown only as guidance and never constrains the choice.

- **Multi-select candidates.** A reference class may genuinely correspond to more than
  one compare class. The expert can confirm several candidate edges, and the retained
  affinity probabilities among the chosen candidates are kept (the crosswalk stays
  quantitative, not collapsed to a flat set).
- **Ranked list is a hint, not a constraint.** The full compare legend is selectable.
  The expert can confirm a correspondence the algorithm ranked low or did not surface
  — this is the whole point of ground-truth review.
- **Confirmation is at the edge level, not the class row.** The atomic unit is a single
  edge (e.g. `Shrubland → Shrub`, confirmed). Within one reference class, some edges
  may be expert-confirmed while others remain algorithm-proposed and open, because
  other correspondences for that class may still be unreviewed. Every edge in the
  matching table therefore carries **provenance**: `expert-confirmed` (frozen) or
  `algorithm-proposed` (open).

---

### 6.6 Feedback loop — two separate effects

An expert confirmation does two distinct things, with different rules. This separation
is what lets the model learn from feedback while never undoing a human decision.

1. **Authoritative crosswalk edge.** The confirmed edge is written to the matching
   table as `expert-confirmed` and **frozen** — it is the trusted output, exported as
   such, and never overwritten by any later refit or recompute.
2. **Training signal.** The samples validated by that confirmation are relabeled and
   used to **warm-start-refit** the relevant GMM. The refit improves the model's
   **proposals** — the affinity it generates for edges the expert has *not* confirmed,
   and for future runs.

Rules that keep the two from colliding:

- The refit and any subsequent affinity recompute (the plan's "global re-balancing")
  apply to **unconfirmed edges only**. Confirmed edges are excluded from recompute and
  re-normalization.
- The model never overwrites a confirmed edge. It may re-propose or re-weight the
  open edges around it.

**Honest scope of the self-improvement.** Within a single small AOI with few classes,
refitting will not visibly "get smarter" — there are too few classes for it to matter.
The compounding benefit is a **cross-run / longitudinal** effect: a GMM refined by
feedback in one region proposes better on the next run, and the review queue shrinks
over time. This is measured by the ablation experiment (accuracy vs. number of expert
samples — the "S-curve"), not expected inside one test box.

---

### 6.7 UI integration — a Review mode

Stage 6 is an investigate-and-decide activity and does not fit the run-and-read
three-column workspace. Integrate it as a **mode switch within the same app** (Harmonize
/ Review) — not a fourth column, and not a separate page.

- **Entry.** In the matching table, a `mixed` or `orphan` edge is clickable; clicking
  it enters Review focused on that class pair. The output is the entry point, so the
  expert never hunts for what to review.
- **Layout (Review mode).** The two **synchronized maps go large in the center** — this
  is the inspector and needs the room (reading fine imagery texture is how shrub vs
  grass is called). A **left rail** holds the class-pair selection (two dropdowns
  driving the three modes) and the sampled patches as a scrollable thumbnail index
  (clicking a patch flies both maps to it). A **right rail** holds the decision: the
  candidate edges for the selected class (affinity ranking as guidance), each with a
  checkbox for multi-select, the confirm action, and per-edge provenance showing which
  are already confirmed-frozen vs still open. The basemap/year switcher sits on the
  maps themselves.
- **Shared live state.** Review shares state with Harmonize (the matching table, fitted
  models, AOI). Confirming an edge updates the matching table in place — the edge flips
  to `expert-confirmed`, and unconfirmed edges may re-propose after a refit. Switching
  back to Harmonize shows the decision landed in the crosswalk. The Sankey and matrix
  are overview artifacts and drop out of Review mode; they are one click away in
  Harmonize.

  *Mechanically* this "shared live state" is **file-backed, not in-memory** (matching
  the Stage 5 job/cache model): a confirm POST persists the feedback store to `cache/`,
  the reviewed matching table is recomputed from the on-disk GMM caches (6a's
  `recompute_reviewed_table`), and both modes read that same recomputed table back — so
  "updates in place" means the next fetch reflects the confirmation, not that a shared
  object was mutated in memory.

---

**Output artifact.** An updated matching table where each edge carries provenance
(`expert-confirmed` frozen / `algorithm-proposed` open) and confirmed edges keep their
retained probabilities; a record of reviewed class-pairs and decisions; refined GMMs;
and re-proposed affinity for the unconfirmed edges. All exportable as before.

**Verification.** Take a `mixed` class (e.g. Shrubland). Enter Review from the matching
table. Confirm the explorer works in all three modes (both/reference-only/compare-only)
and that patches fly both synced maps to the right location with each dataset's own
label, across at least two basemap sources. Multi-select and confirm an edge set; check
it appears in the matching table as `expert-confirmed` and frozen with its probabilities.
Then confirm that an *unconfirmed* edge can change after a refit while the
*confirmed* edges do not.

---

### Suggested build order for Stage 6

Stage 6 is large; build it in three verifiable sub-stages, backend first (same pattern
as Stage 5):

- **6a — Feedback data model + logic (backend).** Per-edge provenance in the matching
  table; the confirm/freeze operation; the warm-start refit on confirmed samples; the
  rule that recompute/re-balance touches unconfirmed edges only. Verify from the CLI:
  confirm an edge, see it freeze; refit; see unconfirmed edges change but confirmed ones
  unchanged. **Status: built as `harmonizer/review.py` and exercised by
  `scripts/verify_stage6a.py`, but CLI-only — its confirm/refit/reviewed-table
  functions are not yet exposed over the API. Wiring them into FastAPI is 6b's job (see
  below), so that 6c has endpoints to call.**
- **6b — Evidence explorer engine + review API (backend).** Two parts, both backend:
  (1) the three-mode co-located-pixel query and the spatially-declustered patch
  sampling (returning center coordinates + per-side labels, not imagery — §6.3); and
  (2) the **review/feedback API layer** that wraps *both* 6a's `review.py`
  (confirm-edge, warm-start-refit, reviewed matching table) *and* this stage's explorer
  as FastAPI endpoints on the existing `harmonizer/api.py`, so the whole review backend
  is reachable over HTTP before any UI is built. The endpoints follow the existing
  file-backed model: a confirm POST persists the feedback store to `cache/` and the
  reviewed table is recomputed from the on-disk GMM caches (there is no shared in-memory
  state — §6.7). Verify: for a class-pair, the explorer endpoint returns N spread
  locations with correct labels on both sides; and a confirm → refit → reviewed-table
  round-trip over the API reproduces the `verify_stage6a.py` invariants.
- **6c — Review UI (frontend).** The Harmonize/Review mode switch, the table-to-review
  handoff, the large synced maps, the two rails, the basemap/year switcher, and live
  shared state — consuming the 6b endpoints only, no new backend. The Review layout
  builds on the current shipped frontend stage (5.3/5.4); pin it to whichever is in
  `web/` at build time. Verify per 6.7.

## Stage 7 — Multi-AOI absence handling

**Purpose.** Make every declared legend class of **both** maps account for itself in
the output, and let a user cover classes their primary AOI cannot evidence by adding
targeted auxiliary AOIs. This closes two gaps: classes that silently vanish from the
deliverable, and the fact that an AOI-conditional method cannot, from one AOI, speak
about classes that do not occur there.

**The problem.** The pipeline is AOI-conditional by construction: a class is modelled
from what its pixels look like *here*. Two consequences the earlier stages do not
handle:

1. **Silent disappearance.** Stage 2 builds its class list from the codes *observed*
   in the AOI (`present_classes`), not from the registry legend. A legend class with
   no pixels in the AOI never enters sampling, never reaches Stages 3–4, and never
   appears in `matching_table.csv` — not even as `absent`. On the Sahel test run,
   WorldCover's Snow and ice / Mangroves / Moss and lichen and Dynamic World's Snow
   and ice are missing from the deliverable with no trace. A reader cannot tell a
   class was *never considered* from one that was considered and matched.
2. **Asymmetry.** The `absent` machinery exists only for reference classes (rows). An
   absent **compare** class simply loses its column, which also silently changes the
   softmax normalisation of every row.

Legend *harmonization* is a statement about the legends; a crosswalk from one AOI is
evidence about that AOI. Stage 7 keeps those honest and lets the user close the gap.

---

### 7.1 Every declared class accounts for itself (both sides)

Union the *observed* class list with the **registry-declared** legend, for the
reference map and the compare map alike. Every declared class of both maps appears in
the output with a status; where that status is `absent` it carries a **reason**
(`not_in_aoi` / `too_rare`, section 2) so incompleteness is visible rather than
inferred from a missing row.

The registry reconciliation check (section 2.5) already computes exactly this
declared-vs-observed signal but is a standalone utility; Stage 7 wires it into the run.
Its **undeclared-code** signal (a code in the data but not in the YAML) keeps its
existing meaning and is surfaced the same way — loudly, as a registry bug.

`absent` classes are never force-fit. A reference class that is `absent` gets a
candidate-less matching-table row; a compare class that is `absent` is reported as an
unmatched target rather than being dropped. Both stay selectable in Stage 6 review.

---

### 7.2 Absence check after sampling, and the popup

**Sampling is the observation — the check costs nothing.** A run must sample its AOI
regardless, and Stage 2 already calls `present_classes` server-side to decide what to
sample. So by the time a run lands, which classes the AOI lacks is established and
recorded in the sample cache. The check therefore runs **after Stage 2, off the
caches**, with **no Earth Engine query of its own**.

This is not merely cheaper than a pre-run probe, it is **more accurate**. A probe can
only ask "is this class here", so every answer it gives is `not_in_aoi`. Sampling
separates the two reasons that need *different remedies*: `not_in_aoi` (another AOI
would supply it) versus `too_rare` (it *is* here, but below the point floor after
erosion — a bigger AOI might help, a different one will not). Only sampling can tell
these apart, because rarity is defined by the floor.

Once the run has its result, if any declared class of either map went unmodelled the
UI raises a **dialog** naming them per map with each one's reason (e.g. *"WorldCover:
Snow and ice, Mangroves, Moss and lichen — Dynamic World: Snow and ice"*), offering:

- **Add another AOI** — the user draws or uploads an auxiliary AOI targeting the
  missing classes, which is sampled for **those classes only** (7.3). Up to
  `max_auxiliary_aois` may be added; the same reporting runs on each.
- **Keep them absent** — dismiss. The classes carry through as `absent` with their
  reason, on both sides, resolvable in review (7.5).

The dialog is **informational, not a gate**. The primary run's GMMs and crosswalk are
valid regardless and are never discarded: an auxiliary AOI *tops up* the result with
extra rows (7.4) rather than replacing it, so there is no half-run to strand and
nothing to re-do if the user simply dismisses.

---

### 7.3 Auxiliary AOIs — targeted, one GMM per class

A run's AOI becomes a **list**: one **primary** AOI (typically the largest, evidencing
most classes) plus zero or more **auxiliary** AOIs.

**Each class has exactly one home AOI, and exactly one GMM.** A class's points are
never pooled across AOIs, because AlphaEarth embedding distributions shift with biome:
pooling one class's points from two biomes yields a smeared, multi-modal distribution
that represents neither. A targeted auxiliary AOI only samples classes the primary
lacked, so the question of pooling one class across AOIs does not arise.

An auxiliary AOI samples, at its own locations:
- the **absent reference classes** it was added to cover, **and**
- the **compare-map classes co-present at those locations** (and symmetrically, when
  the missing class is on the compare side, the reference-map classes co-present
  there).

That second half is not optional and not scope creep — it is what makes the auxiliary
edges meaningful. An absent class must be compared against the other map's classes
**as they look in the same AOI**: Mangroves-on-the-coast against Dynamic World's
classes on the coast, never against Dynamic World's classes as fitted in the Sahel.
Every distance in the run is therefore computed **within one AOI**, which is the
invariant Stage 7 exists to preserve. It stays cheap: a handful of extra GMMs per
auxiliary AOI, not a second full run.

---

### 7.4 Merged table and AOI provenance

Each AOI yields its own self-consistent affinity sub-matrix over the classes modelled
in *that* AOI (both maps). The matching table is the **union of their rows**, each
tagged with an **`evidence_aoi`** column naming the AOI that produced it, so a reader
sees at a glance which correspondences rest on primary evidence and which on an
auxiliary. AOI provenance is recorded end to end — in the sample cache, the GMM cache,
and every export — so a result can always be traced to the ground it came from.

Because sub-matrices are per-AOI, row probabilities normalise **within** their AOI's
sub-matrix. A row's probabilities are comparable to other rows from the same AOI; the
`evidence_aoi` tag is what tells the reader when they are not.

**Caching.** Sample and GMM caches become **per-AOI**, and the run signature (Stage
5.1) covers the AOI *list*. Adding an auxiliary AOI must not invalidate the primary's
expensive GEE sampling — the primary's cache is reused and only the new auxiliary is
sampled.

---

### 7.5 Review fallback for still-absent classes

A class absent from *every* AOI the user supplied stays `absent` and is resolvable only
by an expert. Stage 6's engine already accepts a confirmed edge to a class outside the
affinity matrix ([`review.py`](../harmonizer/review.py) — the retained-probability
path), so this is mostly a UI obligation: **`absent` classes must remain selectable in
the review page's class pickers on both sides**, rather than being hidden for having no
matrix row or column. An edge so declared is `expert-confirmed` like any other, with
its `evidence_aoi` recorded as none — the expert, not an AOI, is its evidence.

---

**Output artifact.** A matching table in which every declared legend class of both maps
appears with a status, `absent` classes carry a reason (`not_in_aoi` / `too_rare`), and
every matched row carries the `evidence_aoi` that produced it; per-AOI sample and GMM
caches; and an AOI-absence report driving the UI dialog.

**Verification.** On the WorldCover × Dynamic World Sahel AOI, confirm the matching
table now lists **all 11** WorldCover classes (not 8), with Snow and ice / Mangroves /
Moss and lichen as `absent` + `not_in_aoi`, and that Dynamic World's Snow and ice is
reported as an unmatched compare-side target rather than silently missing. Confirm the
AOI dialog names exactly those classes per map on AOI entry, and that **Continue**
proceeds with them `absent`. Then add an auxiliary AOI over a mangrove coast: confirm
only Mangroves and its co-present compare classes are sampled there (the primary's
cache is reused, not re-sampled), that the Mangroves row appears in the merged table
tagged `evidence_aoi=<aux>`, and that its compare candidates were fitted in that same
auxiliary AOI. Finally, confirm a still-`absent` class is selectable in review and that
an expert-confirmed edge on it persists.

**Suggested build order.** Backend first, as with Stages 5–6:
- **7a — Symmetric absence in the output.** Union declared-vs-observed on both sides,
  absence reasons, both-side statuses, all classes in the table. Verify from the CLI on
  the existing cache: 11 rows, correct reasons, compare-side absent reported.
- **7b — Absence check + dialog.** The post-sampling absence report on the run's
  results payload (cache-driven, no GEE) and the UI dialog with Add-AOI / Keep-absent.
- **7c — Auxiliary AOIs.** Per-AOI sampling/caching, targeted class selection with
  co-present classes from the other map, per-AOI sub-matrices, the merged table with
  `evidence_aoi`, and the review-picker fallback.

---

## Stage 8 — Semantic prior from LCCS attributes

**Pair in scope:** `hrlc30_africa` ↔ `worldcover_2020`. Other products are untouched
(see "Out of scope" at the end of this stage).

### Why

The AEF + GMM pipeline measures whether pixels labelled X in map A *look like* pixels
labelled Y in map B. It cannot separate classes that are semantically distinct but
spectrally similar (cropland vs grassland, irrigated crops vs herbaceous wetland).
Stage 8 adds a **semantic prior** built from each legend's LCCS attribute encoding,
folded into the existing row-softmax as a power prior, plus a **disagreement report**
that keeps the observational (α = 0) and fused tables side by side.

### Design decisions (settled; do not re-litigate)

- Structured **LCCS v2 attributes**, not neural text embeddings. WorldCover's Product
  User Manual v2.0 Table 3 gives explicit LCCS codes; the CCI HRLC Product User Guide
  Table 5 gives LCCS-style prose with explicit thresholds, hand-transcribed.
- Categorical likeness uses the **FAO correspondence table** (FAO cb5130en, *Register
  implementation for land cover legends*, Appendix B, Table 8-1, scores 1–10, ÷ 10).
  Interval attributes use **directed interval inclusion** (fraction of the source
  interval lying inside the target), under a uniform-density assumption that is
  stated, not hidden.
- The prior is **asymmetric** (inclusion, source → target). Source-side OR
  alternatives take the **max**; target-side alternatives are **merged attribute-wise**
  (union) before scoring.
- **Veto attributes** (surface, cultivation, life form) combine multiplicatively and
  are clipped from below at `semantic_veto_floor`, never zero. Graded attributes
  (leaf type, phenology, cover, height, flooding) are averaged.
- Fusion: `logit_ij = −d_ij / T + α · log π_ij`, then the existing row-softmax.
  `T` stays at its section-2 value; `α` (`semantic_prior_alpha`) is swept in
  Stage 8c, never co-tuned with `T`.
- **Orphans stay observational** (the raw-similarity floor is untouched). A separate
  `semantic_orphan` flag marks rows whose best π is below `semantic_orphan_floor`.
- **Direction needs no pipeline change.** `compute_affinity(ref, cmp)` is already
  recomputed with swapped roles by the direction toggle (`decision.py` →
  `compute_affinity_directed`), so a prior built for the ordered pair (rows = first
  id, columns = second id) is oriented correctly in both directions for free.
- Products without a `semantics` block get a uniform prior (π ≡ 1) and a warning, so
  every other pair keeps today's behaviour exactly.
- The crosswalk table from an earlier thesis is an *illustration* of the cardinalities
  the tool must handle (one-to-many, many-to-one, zero), not a tuning target. The
  pipeline must produce its best table on its own; the reference is used only to read
  an α off a one-dimensional sweep and is reported as such.

All constants live in section 2 → "Semantic prior (Stage 8)" and are read through
`config.py` (`AffinityConfig`). Build strictly 8a → 8b → 8c, with the verification
script of each sub-stage run and committed before the next.

### Stage 8a — Attribute encoding + semantic similarity module

**Goal.** A per-class attribute block in the registry YAML, a parser, and a pure
function that returns the directed prior matrix π for any ordered product pair.

**8a.1 Schema** (`harmonizer/registry/schema.py`). Add a frozen dataclass
`ClassSemantics` and an optional `semantics` field on `LegendClass`; parse it in
`_parse_legend`. `ProductSpec` gains `has_semantics` (true when every legend class
carries a block).

```yaml
semantics:
  alternatives:              # one entry per OR-branch; single-branch classes have one
    - surface: vegetated     # vegetated | built | bare | water | snow
      cultivation: natural   # natural | cultivated | any
      life_form: tree        # tree | shrub | herbaceous | lichen_moss | any
      leaf_type: any         # broadleaf | needleleaf | any
      phenology: any         # evergreen | deciduous | any
      cover: [10, 100]       # % of dominant life form; omit = unspecified
      height: [5, null]      # metres; null = open-ended; omit = unspecified
      flooding: [0, 12]      # months/year; [0,0] = dry; omit = unspecified
```

`any` / omitted = unspecified. Unknown enum values raise at load time.

**8a.2 Encodings** (YAML edits).

`worldcover_2020.yaml`, from the WorldCover PUM v2.0 Table 3:

| code | alternatives (key attributes) |
|---|---|
| 10 Tree cover | tree, cultivation any, cover [10,100], height [5,∞], flooding [0,12] — 3 alternatives (A12A3 natural; A11A1 cultivated; A24A3 natural flooded) |
| 20 Shrubland | shrub, natural, cover [10,100], height [0,5], flooding [0,0] |
| 30 Grassland | herbaceous, natural, cover [10,100], flooding [0,0] |
| 40 Cropland | herbaceous, cultivated — 2 alternatives: dry `[0,0]` (A11A3) and aquatic `[1,12]` (A23) |
| 50 Built-up | built |
| 60 Bare / sparse | bare, cover [0,10] |
| 70 Snow and ice | snow, flooding unspecified (persistent) |
| 80 Permanent water | water, flooding [9,12] |
| 90 Herbaceous wetland | herbaceous, natural, cover [10,100], flooding [1,12] |
| 95 Mangroves | tree, natural, flooding [1,12] (intertidal) |
| 100 Moss and lichen | lichen_moss, natural |

`hrlc30_africa.yaml`, from the CCI HRLC PUG Table 5:

| code | attributes |
|---|---|
| 10/20/30/40 | tree, cultivation any, cover [50,100], height [5,∞], flooding [0,0]; leaf/phenology fixed per class (10 broad/evergreen, 20 needle/evergreen, 30 broad/deciduous, 40 needle/deciduous) |
| 50/60 | shrub, cultivation any, cover [50,100], height [0,5], phenology evergreen/deciduous |
| 70 | herbaceous, natural, cover [50,100], flooding [0,0] |
| 80 | herbaceous, cultivated, cover [50,100], flooding unspecified (includes aquatic crops) |
| 90 | 2 alternatives: tree and shrub, cover [50,100], flooding [4,12] |
| 100 | 2 alternatives: herbaceous and lichen_moss, cover [50,100], flooding [4,12] |
| 110 | lichen_moss, cover [50,100] |
| 120 | bare, cover [0,50] |
| 130 | built |
| 140 | water, flooding [5,12] (parent of 141/142; kept for legend completeness) |
| 141 | water, flooding [5,9] |
| 142 | water, flooding [9,12] |
| 150 | snow, flooding [9,12] |

The HRLC 50 % boilerplate ("snow/ice, water or built-up cover < 50 %") is ignored.

**8a.3 Module** `harmonizer/semantics.py` (new).

- Correspondence tables come from `config.py` (section 2), one per categorical
  attribute, values in [0, 1].
- `attribute_score(attr, src, tgt)`: categorical → table lookup, unspecified on either
  side → 1.0; interval → `|src ∩ tgt| / |src|`, unspecified → 1.0; an open-ended
  `null` bound is capped at the attribute's natural max (cover 100, height 50,
  flooding 12).
- `inclusion(src_alt, tgt_merged) = ∏_veto max(score, semantic_veto_floor) × mean_graded score`.
  Veto set: surface, cultivation, life_form. Graded set: leaf_type, phenology, cover,
  height, flooding.
- `merge_alternatives(tgt_alts)`: categorical → set union (score = max over members);
  interval → hull.
- `semantic_prior(reference_id, compare_id, ref_classes, cmp_classes) -> np.ndarray`
  (M×N, values in (0, 1]): `π_ij = max_k inclusion(alt_k(i), merged(j))`. If either
  product lacks semantics → `np.ones` and `warnings.warn`.
- `semantic_orphans(prior) -> np.ndarray[bool]` using `semantic_orphan_floor`.

**Verification** `scripts/verify_stage8a.py`. Prints π for both directions of the
pair as labelled tables and asserts:

- all π in (0, 1]; HRLC 10 → WC 10 = 1.0; WC 10 → HRLC 10 ≈ 0.56 (cover inclusion 50/90);
- asymmetry: `π[WC10→HRLC10] < π[HRLC10→WC10]`;
- WC 40 Cropland → HRLC 70 Grassland equals the veto-floor-driven value and is the
  smallest entry in that row among vegetated targets;
- WC 95 Mangroves → HRLC 90 is the row maximum; HRLC 141 seasonal water has
  `max π < semantic_orphan_floor` (semantic orphan) while HRLC 142 → WC 80 = 1.0;
- WC 10 → HRLC 10/20/30/40 are all equal (no leaf-type information on the source side);
- a product without semantics (e.g. `dynamicworld`) yields all-ones and a warning.

### Stage 8b — Fold the prior into the affinity, decision, and CSVs

**Goal.** With `α = 0` every existing output is byte-identical; with `α > 0` the
prior reshapes the softmax rows. Orphans unchanged.

**Edits.**

- `harmonizer/affinity.py`: `compute_affinity` accepts `alpha: float | None`
  (default from config); builds `prior` via `semantics.semantic_prior`;
  `logits = −distance/T + alpha · log(clip(prior, semantic_prior_epsilon))`.
  `raw_similarity` is unchanged. `AffinityResult` gains `semantic_prior` (M×N),
  `alpha`, `semantic_orphan` (M,), and an unfused `normalized_affinity_aef` (α = 0)
  so the disagreement report does not need a second run.
- `harmonizer/decision.py`: `ClassDecision` / `MatchingRow` gain
  `semantic_orphan: bool`, `best_semantic_value/name`, `aef_best_compare_value`
  (argmax of the α = 0 row), and `agreement` ∈ {`agree`, `semantic_overrides`,
  `aef_only`, `both_orphan`}. `classify_rows` is unchanged except that it reads the
  fused row. `save_matching_table_csv` writes the new columns; add
  `save_semantic_prior_csv` and `save_aef_affinity_csv` beside
  `save_normalized_affinity_csv`. `compute_affinity_directed` needs no change.
- `harmonizer/pipeline.py`: save the two extra CSVs in `run_pipeline`; expose
  `alpha` on `RunParams` (optional, default `None` → config).
- `harmonizer/config.py`: the section-2 constants and correspondence tables.
- No UI/API change in this stage; the fused table flows through the existing
  endpoints. An α control in the UI is a possible later 8d, outside this plan.

**Verification** `scripts/verify_stage8b.py`, on the cached GMMs for the pair (run
the pipeline once first if absent):

1. Regression: `compute_affinity(..., alpha=0)` → `normalized_affinity`, statuses and
   matching-table CSV identical to a pre-stage run (saved copy, or recompute with the
   prior forced to ones).
2. `alpha=1`: for WC 40 Cropland the probability on HRLC 70 drops and on HRLC 80
   rises; for WC 10 the split across HRLC 10–40 is unchanged in *ratio* (flat prior
   there); orphan statuses identical between α = 0 and α = 1.
3. `semantic_orphan` set for HRLC 141; `agreement` populated for every row; the
   three new CSVs exist with matching headers and shapes.
4. Direction toggle: `compute_affinity_directed(..., "compare_to_reference")` uses
   the transposed-orientation prior (check one asymmetric cell).

### Stage 8c — α sweep and disagreement report

**Goal.** A reproducible script that produces the calibration curve and the
two-dimensional disagreement view, so α can be chosen and justified.

- `scripts/semantic_sweep.py`: for α in {0, 0.25, 0.5, 0.75, 1.0}, both directions,
  write `matching_table_alpha{α}_{dir}.csv`; if `--reference path.csv` (optional,
  `reference_value,compare_values|...`) is given, print per-α set-overlap metrics —
  per-row Jaccard between listed candidates and the reference set, and top-1 hit
  rate — separately per direction. Rows with an empty reference set score as correct
  when the pipeline reports `orphan` or `semantic_orphan`.
- `scripts/disagreement_report.py`: per class pair, `s_aef = raw_similarity`,
  `s_sem = π`; classify into quadrants using the existing `absolute_affinity_floor`
  and `semantic_orphan_floor`: agree-match, agree-nonmatch, spectral-confusion (sem
  low, aef high), definition-drift (sem high, aef low). Emit `disagreement.csv` and a
  scatter PNG (matplotlib).
- **Verification** `scripts/verify_stage8c.py`: runs both scripts on the cached pair,
  checks the α = 0 sweep table equals the Stage 8b regression output, and that the
  quadrant counts sum to M×N.

After 8c the human picks α, sets `semantic_prior_alpha` in section 2 / `config.py`,
and the UI shows the fused table automatically.

### Files touched

- Modify: `harmonizer/registry/schema.py`, `harmonizer/registry/products/worldcover_2020.yaml`,
  `harmonizer/registry/products/hrlc30_africa.yaml`, `harmonizer/affinity.py`,
  `harmonizer/decision.py`, `harmonizer/pipeline.py`, `harmonizer/config.py`.
- New: `harmonizer/semantics.py`, `scripts/verify_stage8a.py`, `scripts/verify_stage8b.py`,
  `scripts/verify_stage8c.py`, `scripts/semantic_sweep.py`, `scripts/disagreement_report.py`.

### Out of scope

Encodings for GLC_FCS30D, LCM-10, Dynamic World, HRLC Siberia; any UI control for α;
neural text embeddings; changes to sampling, GMM fitting, or the orphan floor.

---

## Notes for the implementer

- The constants in section 2 are the single source of truth. Read values from
  there (via `config.py`), never hardcode them inside a stage.
- Build strictly one stage at a time and run its verification before the next.
  Stages 1–4 give a working crosswalk before any UI exists; that is the fastest
  path to knowing the method works.
- Values marked *(tune)* — the entropy threshold and the absolute-affinity floor
  especially — are placeholders to calibrate once real similarity values are in
  hand. Expect to revisit them after Stage 4.
