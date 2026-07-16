# Legend Harmonizer

**Legend Harmonizer** works out how the legends of two land-cover maps relate to
one another — automatically, and without ever deciding which map is "correct."

Different land-cover products use different legends: one map's `Forest` may span
another's `Trees` and part of its `Shrubland`; one map's single `Wetland` class
may be split three ways in another. Legend Harmonizer answers questions of the
form *"does Dynamic World's `Trees` correspond to CCI's `Forest` or `Cropland`?"*
at the level of the **categories themselves** — not by adjudicating individual
pixels, and not by assuming either legend is the reference truth.

## How it works

The method is **embedding + Gaussian Mixture Model + Bures–Wasserstein**
throughout. Each class becomes a probability distribution in embedding space, and
two classes are compared by the distance between their distributions.

### 1. Sample each class, embed it, and model it

**Sampling.** For each map and each class independently, points are drawn from the
**interior** of that class's patches — never at boundaries, where pixels are mixed:

- The class mask is **eroded inward** by `erode_pixels` (default 2 px ≈ 20 m at
  10 m resolution), so boundary and mixed pixels are removed and only homogeneous
  cores remain. A homogeneous `homogeneous_window`×`homogeneous_window` (default
  3×3) neighbourhood is required around each candidate.
- Points are then **spatially declustered** so one large patch cannot dominate a
  class's sample: a minimum spacing of `min_spacing_m` (default 100 m) is enforced
  between points of the same class, and the area of interest is gridded into
  `grid_cell_deg` (default 0.25°) cells with points drawn per cell.
- Up to `points_target` points per class are drawn (default 1500), with a hard
  minimum of `points_floor` (default 300). Points whose AlphaEarth vector is
  masked/no-data, or whose label is a fill value, are dropped.

The classes are modelled separately, so the two maps' point sets are unrelated —
there is no need for co-located points across maps. Erosion and stratified drawing
run **server-side** (in Earth Engine for GEE maps, in-process for a local raster);
only the resulting point coordinates cross the network, never the rasters.

**Starved classes** (below the floor) are handled by *why* they are short:

- Short **only after** eroding (the pre-erode count was adequate) → *buffered
  away*: the buffer is relaxed to `relaxed_erode_pixels` (default 1 px), the class
  is resampled, and it is flagged low-confidence.
- Short **even before** eroding → *genuinely rare* in this area: the class is
  flagged **`absent`** and not force-fit, since relaxing the buffer cannot help.

**Embed and model.** Each surviving point carries its
[AlphaEarth Satellite Embedding](https://developers.google.com/earth-engine/datasets/catalog/GOOGLE_SATELLITE_EMBEDDING_V1_ANNUAL)
vector (**64 dimensions**, kept in full — no PCA). A **Gaussian Mixture Model**
with $K$ components (default 4) is fitted to each class's vectors, so the class is
a distribution

$$p(x) = \sum_{k=1}^{K} w_k \, \mathcal{N}(x \mid m_k, \Sigma_k), \qquad \sum_k w_k = 1$$

rather than a point cloud. $K$ is auto-capped so each component keeps at least
`min_points_per_component` points (default 50); under-floor classes fall back to
**diagonal covariance** to avoid overfitting a full $64 \times 64$ matrix on few
points.

### 2. Distance between two Gaussians — Bures–Wasserstein

The ground distance between two single Gaussians $\mathcal{N}(m_1,\Sigma_1)$ and
$\mathcal{N}(m_2,\Sigma_2)$ is the closed-form squared **2-Wasserstein
(Bures–Wasserstein)** distance:

$$W_2^2 = \lVert m_1 - m_2 \rVert^2 \;+\; \operatorname{Tr}\!\left(\Sigma_1 + \Sigma_2 - 2\left(\Sigma_1^{1/2}\,\Sigma_2\,\Sigma_1^{1/2}\right)^{1/2}\right)$$

— the squared gap between the means plus a covariance term comparing shape and
orientation via a matrix square root. (Implemented in
[`harmonizer/affinity.py`](harmonizer/affinity.py) as `bures_wasserstein_sq`,
using a symmetric-PSD `sqrtm` so no complex round-off leaks through.
`scipy.stats.wasserstein_distance` is deliberately *not* used — it is
one-dimensional only.)

### 3. Distance between two GMMs — Mixture-Wasserstein (MW₂)

Because each class is a *mixture*, the two mixtures are compared with
**Mixture-Wasserstein**. Treat each GMM as a discrete distribution over its
components. The cost of moving mass from component $i$ of one class to component
$j$ of the other is their squared Bures–Wasserstein distance $C_{ij}$; the mixture
distance is the **optimal transport** between the two weight vectors $w^a$, $w^b$
under that cost:

$$\mathrm{MW}_2^2(a, b) = \min_{T \ge 0}\ \sum_{i,j} T_{ij}\, C_{ij} \quad\text{s.t.}\quad \sum_j T_{ij} = w^a_i,\ \ \sum_i T_{ij} = w^b_j$$

solved exactly as a min-cost transport problem, then $d = \sqrt{\mathrm{MW}_2^2}$.
This reduces to the plain Bures–Wasserstein distance when both GMMs have a single
component — the sanity check for the implementation.

### 4. Affinity, probabilities, and decision

Each distance becomes a **raw similarity** $s = \dfrac{1}{1 + d}$ — the
absolute-match signal used to detect orphans (a best match below the affinity
floor means nothing corresponds).

Each reference row is turned into a probability distribution over compare classes
with a **temperature-scaled softmax over negative distances**:

$$P_{ij} = \frac{\exp(-d_{ij} / T)}{\sum_k \exp(-d_{ik} / T)}$$

(rather than linearly normalising $s$, which compresses the range and flattens
every row to near-uniform). The temperature $T$ sharpens the winner. Each class is
then classified from two signals — its best raw similarity against the absolute
floor, and its **top1−top2 margin** against the margin threshold — into:

- **`strong`** — one clear winner (best above the floor, margin at or above threshold),
- **`mixed`** — a genuine many-to-many split (above the floor, margin below threshold),
- **`orphan`** — present but nothing corresponds (best match below the floor),
- **`absent`** — too rare to model in the area of interest.

### 5. Review

An expert resolves the `mixed` and `orphan` cases the algorithm could not settle,
using satellite imagery and legend definitions as ground evidence (a co-located
pixel query is used only to *find* representative locations, never to score the
correspondence). Confirmed decisions become authoritative edges in the matching
table and feed back to refine future proposals, without overwriting what the
expert confirmed.

The primary deliverables are the **legend matching table** (CSV) and the
**affinity matrix** (CSV) behind it. Repainting one map into the other's legend
(the harmonized raster) is out of scope for the MVP.

## Scope

The goal is **legend matching**: a probabilistic crosswalk between two maps'
legends, plus the affinity matrix behind it. Which map pair, which area, and which
year are all just the demonstration case — the method is not tied to any one
dataset, and repainting a map into another legend (a harmonized raster) is out of
scope.

The MVP demonstrates the method on **CCI HRLC** (reference, local GeoTIFF) ×
**Dynamic World** (compare, Google Earth Engine) over their overlap. A test-swap
configuration uses **ESA WorldCover v200** as the reference (fully on GEE) so the
pipeline can be exercised without a local download. The design leaves room to add
more products and more embedding models; per-map metadata and legends live in a
[product registry](harmonizer/registry/products/) (one YAML per map), never
hardcoded — adding a map is editing a YAML file, not the code.

Only coordinates and small point tables cross the network — never rasters. Local
rasters are read with rasterio; GEE datasets (Dynamic World, AlphaEarth) come from
Earth Engine under your own account.

## Setup

1. `pip install -e .`
2. Place the HRLC GeoTIFF in `data/` (not needed for the WorldCover test-swap).
3. `earthengine authenticate` — GEE runs under your own account; this is the only
   credential step, and there is no login.
4. `python run.py` — starts the local server and opens the one-page app in your
   browser.

Everything runs on your machine under your own GEE quota; there is no shared
infrastructure.

## Parameters and defaults

Every tunable value lives in [`harmonizer/config.py`](harmonizer/config.py) — the
single in-code home for these constants, so a change is made in one place. The run
parameters (maps, area of interest, year, sample scale, $K$, point floor/target)
are also settable per run from the UI; the calibration thresholds are read live
from config. Values marked *(tune)* are calibrated starting points, not fixed
truths.

**Sampling** (`SamplingConfig`)

| Parameter | Default | Meaning |
|---|---|---|
| `points_floor` | `300` | Minimum points per class; below this a class is starved (relaxed or flagged `absent`). |
| `points_target` | `1500` *(tune)* | Points drawn per class when available. |
| `min_spacing_m` | `100.0` m *(tune)* | Minimum spacing between two points of the same class (spatial declustering). |
| `grid_cell_deg` | `0.25°` *(tune)* | Grid-cell side the area is divided into; points are drawn per cell so no single patch dominates. |
| `sample_scale_m` | `10.0` m | Raster scale masks/erosion/sampling run at. 10 m is native; raise to 30–100 m to trade fidelity for speed. |

**Buffering** (`BufferingConfig`)

| Parameter | Default | Meaning |
|---|---|---|
| `erode_pixels` | `2` px | Inward erosion of the class mask before sampling (~20 m at 10 m), removing boundary/mixed pixels. |
| `relaxed_erode_pixels` | `1` px | Looser buffer used to resample a class that was buffered away below the floor. |
| `homogeneous_window` | `3` | Side of the homogeneous $N{\times}N$ neighbourhood required around each point. |

**GMM** (`GMMConfig`)

| Parameter | Default | Meaning |
|---|---|---|
| `n_components` | `4` | Default number of GMM components $K$ per class (user-settable, 1–10). |
| `covariance_type` | `"full"` | Covariance shape; under-floor classes fall back to `"diag"`. |
| `min_points_per_component` | `50` | Auto-cap on $K$ so every component keeps at least this many points. |
| `reg_covar` | `1e-6` *(tune)* | Added to covariance diagonals to keep them non-singular (matters for small classes). |
| `random_seed` | `42` | Seed for sampling and GMM initialisation, so runs are reproducible. |

**Affinity and decision** (`AffinityConfig`)

| Parameter | Default | Meaning |
|---|---|---|
| `softmax_temperature` $T$ | `0.25` *(tune)* | Temperature in the row softmax $P_{ij}=\exp(-d_{ij}/T)/\sum_k\exp(-d_{ik}/T)$; smaller $T$ sharpens the winner. |
| `margin_threshold` | `0.18` *(tune)* | Top1−top2 margin splitting `strong` (≥) from `mixed` (<). |
| `absolute_affinity_floor` | `0.60` *(tune)* | Floor on the best raw similarity $s=1/(1+d)$; below it the class is an `orphan`. Starts `None` (uncalibrated); while `None`, orphan classification is skipped rather than guessed. |
| `high_entropy_threshold` | `0.65` *(tune)* | Reported diagnostic only — mapping entropy is reported per row but no longer used to classify. |

**Review — evidence explorer** (`ReviewConfig`)

| Parameter | Default | Meaning |
|---|---|---|
| `patches_per_pair` | `10` *(tune)* | Representative locations sampled per class-pair (range 9–12: few enough to review quickly, enough to be representative). |
| `patch_window_px` | `256` px *(tune)* | Fixed patch window shown at each location (≈ 2.5 km at 10 m); metres = px × the label map's resolution. |
| `live_oversample` | `2.0` *(tune)* | Oversample factor when finding evidence, so ~$n$ survive after declustering/exactness filtering. |
| `draw_scale_m` | `100.0` m *(tune)* | Coarse scale locations are *found* at (labels are still read at native resolution). Dominant cost lever for "Find evidence". |

Spatial declustering of evidence points reuses the sampling `min_spacing_m` /
`grid_cell_deg`, so evidence is spread exactly as sample points are.

**Absence — multi-AOI handling** (`AbsenceConfig`)

A class is `absent` when it cannot be modelled in the area of interest, carrying an
explicit **reason**:

- **`too_rare`** — observed in an area of interest but below the point floor even
  before eroding. A larger area might help; relaxing the buffer would not.
- **`not_in_aoi`** — declared in the map's legend but with **no pixels observed in
  any** of the run's areas of interest. A different area covering the class could
  help.

To rescue a `not_in_aoi` class, a run may add **auxiliary areas of interest** that
cover it. Each class is still modelled from **one home area** (never pooled across
areas, which would blend distinct biomes). The number of auxiliary areas is capped:

| Parameter | Default | Meaning |
|---|---|---|
| `max_auxiliary_aois` | `3` *(tune)* | Maximum auxiliary areas of interest per run, so a run cannot fan out into unbounded GEE sampling passes. |

Absence is read straight from the sampling caches, never from a separate Earth
Engine query — sampling *is* the observation, so the check costs nothing.

## Documentation

- [`docs/PIPELINE.md`](docs/PIPELINE.md) — the full specification of what the tool
  does and the constants behind it.
- [`harmonizer/config.py`](harmonizer/config.py) — the tunable pipeline constants
  (sampling, buffering, GMM, affinity thresholds).

## Developers

- **Xiao Tan** — GEOlab, Politecnico di Milano, 2025–2026

## License

Released under the [MIT License](LICENSE). Copyright © 2026 Xiao Tan, GEOlab,
Politecnico di Milano.
