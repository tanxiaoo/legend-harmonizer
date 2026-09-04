"""Evidence explorer engine (Stage 6b).

The backend half of Stage 6's evidence explorer: the **three-mode co-located-pixel
query** and the **spatially-declustered patch sampling** that let an expert walk the
relationship between two legends by inspecting representative ground locations
(docs/PIPELINE.md, sections 6.2 and 6.3).

There are exactly three modes, corresponding to a cell, a row, and a column of the
contingency structure between the two legends:

  1. **Both classes** (a cell) -- reference = X *and* compare = Y at the same pixel.
     A joint mask ``(reference == X) & (compare == Y)`` is built server-side, which
     needs the two label images **co-registered** to a common scale (this is the one
     new server-side operation in Stage 6b; Stage 2 samples each map independently).
  2. **Reference class only** (a row) -- representative samples of reference = X; the
     compare label is read back at each so the expert sees how X distributes across
     the compare legend.
  3. **Compare class only** (a column) -- representative samples of compare = Y; the
     reference label is read back at each.

Whatever the mode, the engine returns, per location, only the **center coordinate
and each side's label** there -- the data the UI needs to fly both synced maps and
outline the queried pixel. It does **not** render or fetch patch imagery: the actual
patch (basemap tiles, section 6.4) is drawn client-side by Stage 6c at the returned
coordinate and window size (docs/PIPELINE.md, section 6.3 "Backend vs. imagery split").

**Hard boundary (docs/PIPELINE.md, section 6.2).** Every mode is powered by a
co-located-pixel query, but that query is **evidence retrieval only, never scoring**.
It locates representative places for a human to inspect; it must never be used to
derive or weight correspondences -- that stays the embedding + GMM + Bures-Wasserstein
method of Stages 3-4. Nothing here feeds a co-occurrence vote.

Scope. For the MVP test-swap both maps are GEE-native (WorldCover reference, Dynamic
World compare), so the join and the stratified draw run in GEE, reusing the Stage 2
sampling machinery (``_label_image_for``, ``_grid_cell_image``,
``_stratified_candidates``, ``_thin_by_spacing``). The local-raster (HRLC) join is
deferred, exactly as Stage 2 defers its local path.

**Cache-backed evidence (default).** When the drawn-from product has a Stage 2
sample cache covering the requested region, evidence points are taken from those
cached per-class points instead of a fresh server-side draw. This is both faster
(the expensive mask + ``stratifiedSample`` draw disappears; only one label
read-back on the *other* map remains) and more faithful: the expert then reviews
the exact points that trained the class's GMM, so a confirmation directly
validates the training signal the warm-start refit uses (docs/PIPELINE.md, Stage
6.6). The live server-side draw remains as the fallback whenever the cache is
missing, doesn't cover the AOI, or can't supply enough points.

Constants (patch count, window, declustering) come from ``CONFIG``; nothing tunable
is hardcoded here.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Literal

from harmonizer.buffering import class_mask, sampling_mask
from harmonizer.config import CONFIG
from harmonizer.overlap import Overlap, overlap_for_products
from harmonizer.registry.legends import class_name
from harmonizer.registry.legends import spec as _product_spec
from harmonizer.registry.products import Coord
from harmonizer.sampling import (
    _grid_cell_image,
    _label_image_for,
    _thin_by_spacing,
    region_bbox,
)

Mode = Literal["both", "reference", "compare"]

# Property name the grid-cell stratified sampling tags each candidate with.
_CELL_PROP = "cell"


# --------------------------------------------------------------------------- #
# Local-raster support
# --------------------------------------------------------------------------- #
# The explorer was written against the GEE label products, but Stage 2 has read
# local rasters since the Africa product set landed (``harmonizer.local_sampling``,
# dispatched from ``sampling.sample_map``). Only the *explorer* still assumed GEE,
# so Review failed with "Stage 2 sampling supports the GEE label products only"
# on exactly the pairs the rest of the pipeline handles fine. These helpers give
# it the same local path, so evidence works for any label product the registry
# can read.


def _is_local(product_id: str) -> bool:
    """True when this product is read from a local raster rather than GEE."""
    spec = _product_spec(product_id)
    return spec is not None and spec.access.method == "local_raster"


def _local_labels_at(product_id: str, coords) -> list[int | None]:
    """Read one local raster's label at each (lon, lat), in order.

    The GEE counterpart is ``_gee.sample_image``; this is the local twin, used
    both to read back the "other" map's label at cached points and to label a
    live draw. Coordinates are transformed into the raster's CRS explicitly --
    getting that wrong silently returns a neighbouring pixel's class
    (docs/PIPELINE.md, Stage 1).
    """
    import rasterio
    from pyproj import Transformer

    spec = _product_spec(product_id)
    if spec is None or not spec.access.path:
        raise ValueError(f"no local raster path registered for {product_id!r}")

    path = _resolve_raster_path(spec.access.path)
    band = int(spec.band) if spec.band is not None else 1
    pts = [(float(lon), float(lat)) for lon, lat in coords]
    if not pts:
        return []

    with rasterio.open(path) as ds:
        if str(ds.crs) not in ("EPSG:4326", "epsg:4326"):
            tf = Transformer.from_crs("EPSG:4326", ds.crs, always_xy=True)
            xs, ys = tf.transform([p[0] for p in pts], [p[1] for p in pts])
            sample_pts = list(zip(xs, ys))
        else:
            sample_pts = pts
        nodata = ds.nodata
        out: list[int | None] = []
        for vals in ds.sample(sample_pts, indexes=[band]):
            v = vals[0]
            # Outside the raster, rasterio yields the nodata (or 0) fill; a
            # no-label point must read as None, not as class 0.
            out.append(None if nodata is not None and v == nodata else int(v))
    return out


def _resolve_raster_path(path: str) -> str:
    """A registry ``access.path`` as an absolute path (they are repo-relative)."""
    from pathlib import Path

    from harmonizer.config import REPO_ROOT

    p = Path(path)
    return str(p if p.is_absolute() else (REPO_ROOT / p))


def _labels_at(product_id: str, coords, overlap: Overlap) -> list[int | None]:
    """One map's label at each coordinate, whichever source it comes from.

    The dispatcher the mixed case needs: a local x GEE pair reads one side with
    rasterio and the other from Earth Engine, and the caller should not care
    which is which.
    """
    if not coords:
        return []
    if _is_local(product_id):
        return _local_labels_at(product_id, coords)

    # Imported here, not at module scope, so a local-only run never needs the
    # GEE adapter (and its credentials) to be importable at all.
    from harmonizer.registry.adapters import _gee

    region = overlap.ee_geometry()
    region_key = tuple(round(float(x), 4) for x in overlap.bbox)
    img = _label_image_for(product_id, region, region_key).clip(region)
    rows = _gee.sample_image(img, coords, scale=_native_scale_m(product_id))
    return [
        None if props is None or props.get("label") is None else int(props["label"])
        for props in rows
    ]


def _local_candidates(
    product_id: str, class_value: int, overlap: Overlap, n: int
) -> list[Coord]:
    """Declustered candidate coordinates for one class of a local raster.

    Mirrors the GEE live draw: build the class's sampling mask (eroded, with the
    homogeneous-neighbourhood test), draw from it, then thin by the Stage 2
    spacing so evidence points are spread exactly as sample points are.
    """
    import numpy as np

    from harmonizer import local_sampling as _local

    spec = _product_spec(product_id)
    band = int(spec.band) if spec.band is not None else 1
    window = _local.read_label_window(
        _resolve_raster_path(spec.access.path), band, tuple(overlap.bbox)
    )
    mask = _local.sampling_mask(
        window, int(class_value), CONFIG.buffering.erode_pixels
    )
    rows, cols = np.nonzero(mask)
    if rows.size == 0:
        return []

    # Draw a bounded random subset before converting, so a class covering
    # millions of pixels does not build a huge coordinate array to throw away.
    rng = np.random.default_rng(CONFIG.gmm.random_seed)
    take = min(rows.size, max(int(n) * 200, 2000))
    if rows.size > take:
        pick = rng.choice(rows.size, size=take, replace=False)
        rows, cols = rows[pick], cols[pick]

    lons, lats = window.pixel_to_lonlat(rows, cols)
    coords = list(zip(lons.tolist(), lats.tolist()))
    rng.shuffle(coords)
    return _thin_by_spacing(coords, CONFIG.sampling.min_spacing_m)[: int(n)]


# --------------------------------------------------------------------------- #
# Result types
# --------------------------------------------------------------------------- #


@dataclass
class EvidenceLocation:
    """One representative location for a class-pair query.

    ``lon``/``lat`` is the sampled center pixel. ``reference_label`` and
    ``compare_label`` are the two maps' class values read back at that pixel (a label
    may be ``None`` if that side is masked / no-data there, which only happens in the
    row/column modes where the other side is unconstrained). ``patch_window_px`` is
    the fixed window the UI outlines/flies to (from config).
    """

    lon: float
    lat: float
    reference_label: int | None
    reference_label_name: str | None
    compare_label: int | None
    compare_label_name: str | None
    patch_window_px: int


@dataclass
class EvidenceResult:
    """The declustered evidence locations for one three-mode query.

    ``source`` records where the locations came from: ``"cache"`` (the Stage 2
    sample points that trained the GMMs -- the default when available) or
    ``"live"`` (a fresh server-side draw, the fallback).
    """

    mode: Mode
    reference_id: str
    compare_id: str
    reference_value: int | None
    compare_value: int | None
    patch_window_px: int
    patch_window_m: float
    #: Each map's native pixel size (metres) -- the side of the single-pixel box
    #: the UI outlines at a location (drawn per side, so the two boxes differ when
    #: the products' resolutions differ). Independent of ``patch_window_*`` (the
    #: fly-to context window).
    reference_pixel_m: float
    compare_pixel_m: float
    source: str = "live"
    locations: list[EvidenceLocation] = field(default_factory=list)

    @property
    def n(self) -> int:
        return len(self.locations)


# --------------------------------------------------------------------------- #
# Co-registration of the two label images (the one new server-side operation)
# --------------------------------------------------------------------------- #


def _native_scale_m(product_id: str) -> float:
    """A product's native label resolution (metres), for authoritative read-back.

    Candidates are *found* at the (possibly coarse) join/draw scale for speed, but a
    location's reported label is read at the map's own resolution so it is exactly
    what the map says at that pixel -- not a coarse-grid resampling. This keeps the
    two-sided labels authoritative and consistent with the mask.
    """
    s = _product_spec(product_id)
    if s is not None and s.resolution_m:
        return float(s.resolution_m)
    return float(CONFIG.sampling.sample_scale_m)


def _join_scale_m(reference_id: str, compare_id: str) -> float:
    """The common scale the two label images are compared at.

    Uses the **coarser** of the two products' native resolutions so neither map is
    upsampled beyond its real detail when the masks are intersected pixelwise, but
    never finer than the run's ``CONFIG.sampling.sample_scale_m``. That floor lets a
    test run trade spatial fidelity for speed (30-100 m) exactly as Stage 2's sample
    scale does -- the explorer's joint draw over a fine scale is as expensive as
    Stage 2 sampling. Falls back to the sample scale if a resolution is missing.
    """
    scales = [float(CONFIG.sampling.sample_scale_m)]
    for pid in (reference_id, compare_id):
        s = _product_spec(pid)
        if s is not None and s.resolution_m:
            scales.append(float(s.resolution_m))
    return max(scales)


def _joint_mask(reference_image, compare_image, reference_value, compare_value):
    """Binary mask where reference == X *and* compare == Y at the same pixel.

    This is the cell query of mode 1. The two class masks are intersected with
    ``.And`` (server-side); ``sampleRegions`` at the join scale then draws only from
    pixels satisfying both conditions -- the co-located-pixel query. Evidence
    retrieval only: this mask locates places to inspect, it never scores a
    correspondence (docs/PIPELINE.md, section 6.2).
    """
    ref_hit = class_mask(reference_image, int(reference_value))
    cmp_hit = class_mask(compare_image, int(compare_value))
    return ref_hit.And(cmp_hit).rename("mask").selfMask()


# --------------------------------------------------------------------------- #
# Cache-backed evidence: reuse the Stage 2 sample points (the training data)
# --------------------------------------------------------------------------- #


def _cached_class_coords(product_id: str, class_value: int) -> list[Coord] | None:
    """A class's Stage 2 sample coordinates from ``cache/samples_<product>.npz``.

    These are the points that trained the class's GMM: declustered, label
    cross-checked at native resolution, and embedding-valid. Returns ``None``
    when no sample cache exists for the product.
    """
    import numpy as np

    from harmonizer.sampling import cache_path

    path = cache_path(product_id)
    if not path.exists():
        return None
    with np.load(path) as data:
        coords = np.asarray(data["coords"], dtype=float)
        class_values = np.asarray(data["class_values"], dtype=int)
    sel = coords[class_values == int(class_value)]
    return [(float(lon), float(lat)) for lon, lat in sel]


def _aux_class_coords(
    product_id: str, class_value: int
) -> tuple[str, list[Coord]] | None:
    """A class's sample coordinates from the first active auxiliary AOI whose
    cache holds them (Stage 7c).

    Returns ``(scoped_cache_id, coords)`` -- the scoped id keys the cross-label
    cache to the auxiliary's own sample file -- or ``None`` when no auxiliary
    has points for the class. Used when the primary cache cannot supply the
    class: its training points live in the auxiliary's AOI, not the request's,
    and that is exactly where its evidence must be shown.
    """
    from harmonizer.auxiliary import active_auxiliaries, aux_scoped_id

    for entry in active_auxiliaries():
        scoped = aux_scoped_id(product_id, entry["name"])
        coords = _cached_class_coords(scoped, class_value)
        if coords:
            return scoped, coords
    return None


def _cross_label_cache_path(drawn_id: str, other_id: str):
    """On-disk cache of the OTHER map's labels at ``drawn_id``'s sample points."""
    return CONFIG.cache_dir / f"crosslabels_{drawn_id}__{other_id}.json"


def _cached_cross_labels(
    drawn_id: str, other_id: str, class_value: int, coords: list[Coord]
) -> list[int | None]:
    """The other map's label at each of a class's cached sample points.

    The answer never changes for a given sample cache, so it is computed **once
    per class** -- one GEE read-back over ALL the class's points (this is the
    only slow step, dominated by Dynamic World's annual modal composite) -- and
    persisted beside the sample cache
    (``cache/crosslabels_<drawn>__<other>.json``). Every later evidence query for
    the class, in any mode and against any compare class, is then fully local:
    zero GEE calls.

    Keyed to the samples ``.npz`` mtime so a re-run of Stage 2 invalidates it.
    The read-back region is the POINTS' own bounding box (padded), not the
    request AOI, so cached labels stay valid whatever AOI the query used.
    """
    import json

    from harmonizer.registry.adapters import _gee
    from harmonizer.sampling import cache_path

    samples_mtime = cache_path(drawn_id).stat().st_mtime_ns
    path = _cross_label_cache_path(drawn_id, other_id)
    payload: dict = {"samples_mtime_ns": samples_mtime, "classes": {}}
    if path.exists():
        try:
            existing = json.loads(path.read_text(encoding="utf-8"))
            if existing.get("samples_mtime_ns") == samples_mtime:
                payload = existing
        except (json.JSONDecodeError, OSError):
            pass  # unreadable cache: recompute below

    key = str(int(class_value))
    if key in payload["classes"]:
        return [None if v is None else int(v) for v in payload["classes"][key]]

    # Read the other map's label at these points: locally with rasterio when it
    # is a local raster, otherwise one GEE read-back bounded to the points' bbox.
    if _is_local(other_id):
        labels = _local_labels_at(other_id, coords)
    else:
        import ee

        lons = [c[0] for c in coords]
        lats = [c[1] for c in coords]
        pad = 0.01
        bbox = (min(lons) - pad, min(lats) - pad, max(lons) + pad, max(lats) + pad)
        region = ee.Geometry.Rectangle(
            list(bbox), proj=CONFIG.maps.target_crs, geodesic=False
        )
        region_key = tuple(round(float(x), 4) for x in bbox)
        other_img = _label_image_for(other_id, region, region_key).clip(region)
        rows = _gee.sample_image(other_img, coords, scale=_native_scale_m(other_id))
        labels = [
            None if props is None or props.get("label") is None else int(props["label"])
            for props in rows
        ]

    payload["classes"][key] = labels
    CONFIG.cache_dir.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return labels


def prewarm_cross_labels(reference_id: str, compare_id: str) -> dict[str, int]:
    """Build the cross-label caches for every class of a pair, both directions.

    Meant to run in the background right after a pipeline run finishes, while the
    expert is still reading the results -- so by the time they enter Review,
    every class's evidence query is fully local (zero GEE calls). All of one
    direction's missing classes are read back in a SINGLE batched GEE call (one
    point table for the whole map), then split per class into the same
    ``crosslabels_*.json`` files ``_cached_cross_labels`` reads. Classes already
    cached (mtime-current) are skipped, so re-running after a cache-reuse run is
    a no-op. Returns ``{"<drawn>-><other>": n_points_read}``.
    """
    import json

    import numpy as np

    from harmonizer.registry.adapters import _gee
    from harmonizer.registry.adapters._gee import ensure_initialized
    from harmonizer.sampling import cache_path

    ensure_initialized()
    out: dict[str, int] = {}
    for drawn_id, other_id in (
        (reference_id, compare_id),
        (compare_id, reference_id),
    ):
        npz_path = cache_path(drawn_id)
        if not npz_path.exists():
            out[f"{drawn_id}->{other_id}"] = 0
            continue
        samples_mtime = npz_path.stat().st_mtime_ns
        with np.load(npz_path) as data:
            coords = np.asarray(data["coords"], dtype=float)
            class_values = np.asarray(data["class_values"], dtype=int)

        path = _cross_label_cache_path(drawn_id, other_id)
        payload: dict = {"samples_mtime_ns": samples_mtime, "classes": {}}
        if path.exists():
            try:
                existing = json.loads(path.read_text(encoding="utf-8"))
                if existing.get("samples_mtime_ns") == samples_mtime:
                    payload = existing
            except (json.JSONDecodeError, OSError):
                pass

        missing = [
            int(cv)
            for cv in np.unique(class_values)
            if str(int(cv)) not in payload["classes"]
        ]
        if not missing:
            out[f"{drawn_id}->{other_id}"] = 0
            continue

        # One batched read-back over every missing class's points.
        sel = np.isin(class_values, missing)
        pts = [(float(lon), float(lat)) for lon, lat in coords[sel]]
        pts_cvs = class_values[sel]

        import ee

        lons = [c[0] for c in pts]
        lats = [c[1] for c in pts]
        pad = 0.01
        bbox = (min(lons) - pad, min(lats) - pad, max(lons) + pad, max(lats) + pad)
        region = ee.Geometry.Rectangle(
            list(bbox), proj=CONFIG.maps.target_crs, geodesic=False
        )
        region_key = tuple(round(float(x), 4) for x in bbox)
        other_img = _label_image_for(other_id, region, region_key).clip(region)
        # GEE aborts a collection getInfo past 5000 elements, so read the whole
        # map's points in chunks under that cap (still ~2-3 calls per direction).
        chunk = 4000
        scale = _native_scale_m(other_id)
        rows = []
        for start in range(0, len(pts), chunk):
            rows.extend(_gee.sample_image(other_img, pts[start:start + chunk], scale=scale))
        labels = [
            None
            if props is None or props.get("label") is None
            else int(props["label"])
            for props in rows
        ]

        for cv in missing:
            payload["classes"][str(cv)] = [
                lab for lab, pcv in zip(labels, pts_cvs) if int(pcv) == cv
            ]
        CONFIG.cache_dir.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload), encoding="utf-8")
        out[f"{drawn_id}->{other_id}"] = len(pts)
    return out


def _explore_from_cache(
    reference_id: str,
    compare_id: str,
    *,
    mode: Mode,
    reference_value: int | None,
    compare_value: int | None,
    overlap: Overlap,
    n: int,
) -> list[tuple[Coord, int | None, int | None]] | None:
    """Evidence locations from the Stage 2 sample cache, or ``None`` to go live.

    The drawn-from side's points come straight from the cache (its own label is
    already cross-checked there, so it is taken as exact); the OTHER map's label
    comes from the per-class cross-label cache -- one GEE call the first time a
    class is explored, none after. Mode ``"both"`` filters the reference class's
    cached points to those whose compare label equals the queried compare class
    -- the co-located cell, evaluated at the training points.

    Falls back to ``None`` (live server-side draw) when: the cache is missing,
    fewer than ``n`` cached points fall inside the requested region (e.g. the
    AOI changed since the run), or a ``"both"`` query finds no surviving cell
    points among the cores the cache covers.
    """
    import numpy as np

    if mode == "compare":
        drawn_id, drawn_value, other_id = compare_id, compare_value, reference_id
    else:  # "reference" and "both" both draw from the reference class's points
        drawn_id, drawn_value, other_id = reference_id, reference_value, compare_id

    cache_id = drawn_id
    coords = _cached_class_coords(drawn_id, int(drawn_value))

    min_lon, min_lat, max_lon, max_lat = overlap.bbox
    in_aoi = [
        i
        for i, (lon, lat) in enumerate(coords or [])
        if min_lon <= lon <= max_lon and min_lat <= lat <= max_lat
    ]
    if len(in_aoi) < n:
        # Auxiliary AOIs (Stage 7c): a class the primary cache cannot supply --
        # only an auxiliary evidences it -- has its training points in that
        # auxiliary's cache, and in ITS AOI, not the request's. Use them all,
        # wherever they are: the class not living in the primary AOI is the
        # very reason the auxiliary exists. (A live draw in the request AOI
        # could never find it.)
        alt = _aux_class_coords(drawn_id, int(drawn_value))
        if alt is not None:
            cache_id, coords = alt
            in_aoi = list(range(len(coords)))
        else:
            return None  # cache doesn't cover this region well enough; draw live

    # The other map's label at every point of the class (cached after first use;
    # keyed to the supplying cache -- primary or the auxiliary's scoped file).
    other_labels = _cached_cross_labels(cache_id, other_id, int(drawn_value), coords)

    # Deterministic spread: the cache is grouped by grid cell, so a plain head
    # take would cluster; a seeded permutation spreads the pick across cells.
    rng = np.random.default_rng(CONFIG.gmm.random_seed)
    order = [in_aoi[i] for i in rng.permutation(len(in_aoi))]

    kept: list[tuple[Coord, int | None, int | None]] = []
    for i in order:
        other = other_labels[i]
        if mode == "both":
            if other != int(compare_value):
                continue
            kept.append((coords[i], int(reference_value), other))
        elif mode == "reference":
            kept.append((coords[i], int(reference_value), other))
        else:  # "compare"
            kept.append((coords[i], other, int(compare_value)))
        if len(kept) >= n:
            break

    if mode == "both" and not kept:
        # The cell may exist only outside the class cores the cache samples
        # (e.g. along boundaries); let the live joint-mask draw look for it.
        return None
    return kept


# --------------------------------------------------------------------------- #
# Declustered stratified draw from an arbitrary mask
# --------------------------------------------------------------------------- #


def _draw_declustered(mask, overlap: Overlap, scale_m: float, n: int) -> list[Coord]:
    """Draw up to ``n`` declustered candidate coordinates from a binary mask.

    Spreads the draw across the overlap's grid cells (the same gridding Stage 2 uses
    so evidence points are not all from one field), oversamples a little so the
    min-spacing thinning still leaves enough, then thins to
    ``CONFIG.sampling.min_spacing_m`` and truncates to ``n``.
    """
    import ee

    region = overlap.ee_geometry()
    # The overlap's bbox is already known client-side; passing it skips the
    # region_bbox getInfo round trip inside _grid_cell_image.
    cells = _grid_cell_image(region, bbox=overlap.bbox).updateMask(mask)

    # Oversample: stratified draw across cells, a few per cell, then thin+truncate.
    cell_deg = CONFIG.sampling.grid_cell_deg
    n_cols = max(1, int(math.ceil((overlap.bbox[2] - overlap.bbox[0]) / cell_deg)))
    n_rows = max(1, int(math.ceil((overlap.bbox[3] - overlap.bbox[1]) / cell_deg)))
    n_cells = max(1, n_cols * n_rows)
    # Aim for ~3x the target across all cells so thinning has slack.
    per_cell = max(1, int(math.ceil(3 * n / n_cells)))

    fc = cells.stratifiedSample(
        numPoints=per_cell,
        classBand=_CELL_PROP,
        region=region,
        scale=scale_m,
        seed=CONFIG.gmm.random_seed,
        geometries=True,
        tileScale=16,
    )
    from harmonizer.registry.adapters._gee import get_info

    info = get_info(fc)
    coords: list[Coord] = []
    for feat in info.get("features", []):
        geom = feat.get("geometry") or {}
        c = geom.get("coordinates")
        if c and len(c) == 2:
            coords.append((float(c[0]), float(c[1])))

    coords = _thin_by_spacing(coords, CONFIG.sampling.min_spacing_m)
    return coords[:n]


# --------------------------------------------------------------------------- #
# Label read-back on both sides
# --------------------------------------------------------------------------- #


def _read_labels_both(
    reference_image, compare_image, coords: list[Coord], scale_m: float
) -> tuple[list[int | None], list[int | None]]:
    """Read BOTH maps' class labels at each coordinate in one server round trip.

    Stacks the two already-built annual label images as a two-band image and
    samples it once, so the expensive composites (notably Dynamic World's
    year-long per-pixel mode) are reused from the mask draw and the read-back
    costs a single ``getInfo`` instead of one per side. Points only cross the
    network. A masked / no-data pixel comes back as ``None`` -- expected in the
    row/column modes where the other side is unconstrained and may be fill there.
    """
    if not coords:
        return [], []
    from harmonizer.registry.adapters import _gee

    both = reference_image.rename("ref_label").addBands(
        compare_image.rename("cmp_label")
    )
    rows = _gee.sample_image(both, coords, scale=scale_m)

    def _get(props, band) -> int | None:
        if props is None or props.get(band) is None:
            return None
        return int(props[band])

    ref_out = [_get(props, "ref_label") for props in rows]
    cmp_out = [_get(props, "cmp_label") for props in rows]
    return ref_out, cmp_out


# --------------------------------------------------------------------------- #
# The three-mode query
# --------------------------------------------------------------------------- #


def explore_evidence(
    reference_id: str,
    compare_id: str,
    *,
    mode: Mode,
    reference_value: int | None = None,
    compare_value: int | None = None,
    overlap: Overlap | None = None,
    aoi: tuple[float, float, float, float] | None = None,
    n: int | None = None,
    oversample: float | None = None,
) -> EvidenceResult:
    """Retrieve N declustered evidence locations for a three-mode class-pair query.

    ``mode`` selects the contingency structure (docs/PIPELINE.md, section 6.2):
      * ``"both"``      -- needs both ``reference_value`` and ``compare_value``; draws
                           from the joint mask (co-located cell).
      * ``"reference"`` -- needs ``reference_value``; draws from the reference class
                           and reads the compare label back at each (a row).
      * ``"compare"``   -- needs ``compare_value``; draws from the compare class and
                           reads the reference label back at each (a column).

    Returns an :class:`EvidenceResult` of at most ``n`` (default
    ``CONFIG.review.patches_per_pair``) locations, each carrying both maps' labels and
    the fixed patch window. Only coordinates and small point tables cross the network.
    ``oversample`` scales the LIVE path's candidate draw (default
    ``CONFIG.review.live_oversample``): more candidates survive the declustering +
    exactness filter for scarce combinations, at the cost of a larger point table.
    """
    from harmonizer.registry.adapters._gee import ensure_initialized

    n = int(n or CONFIG.review.patches_per_pair)
    oversample = max(1.0, float(oversample or CONFIG.review.live_oversample))
    overlap = overlap or overlap_for_products(
        [reference_id, compare_id, "alphaearth"], aoi=aoi
    )
    scale_m = _join_scale_m(reference_id, compare_id)

    # Validate mode arguments before any GEE work.
    if mode == "both" and (reference_value is None or compare_value is None):
        raise ValueError("mode 'both' needs both reference_value and compare_value")
    if mode == "reference" and reference_value is None:
        raise ValueError("mode 'reference' needs reference_value")
    if mode == "compare" and compare_value is None:
        raise ValueError("mode 'compare' needs compare_value")
    if mode not in ("both", "reference", "compare"):  # pragma: no cover
        raise ValueError(f"unknown mode: {mode!r}")

    # Only initialise Earth Engine if a side actually needs it. A local x local
    # pair must work with no GEE credentials at all -- requiring them there
    # turned an offline-capable query into an authentication error.
    if not (_is_local(reference_id) and _is_local(compare_id)):
        ensure_initialized()

    # Cache-backed path first (the default): reuse the Stage 2 sample points --
    # the exact points that trained the drawn-from class's GMM -- and read back
    # only the other map's label (one server call). Falls back to the live
    # server-side draw when the cache is missing or doesn't cover the region.
    source = "cache"
    kept = _explore_from_cache(
        reference_id,
        compare_id,
        mode=mode,
        reference_value=reference_value,
        compare_value=compare_value,
        overlap=overlap,
        n=n,
    )

    if kept is None:
        source = "live"
        # The draw side is whichever map the mode queries; "both" needs a joint
        # mask, which only the GEE path can express server-side.
        n_draw = int(math.ceil(oversample * n))

        if _is_local(reference_id) or _is_local(compare_id):
            # Local-raster path: draw the candidates in-process from the raster,
            # then label both sides. Each side is read with whichever mechanism
            # it needs, so a local x GEE pair works as well as local x local.
            if mode == "both":
                # A joint draw needs both masks in one place; do it on the local
                # side and let the exactness filter below enforce the other.
                draw_id, draw_value = (
                    (reference_id, reference_value)
                    if _is_local(reference_id)
                    else (compare_id, compare_value)
                )
            elif mode == "reference":
                draw_id, draw_value = reference_id, reference_value
            else:
                draw_id, draw_value = compare_id, compare_value

            if _is_local(draw_id):
                coords = _local_candidates(draw_id, int(draw_value), overlap, n_draw)
            else:
                region = overlap.ee_geometry()
                region_key = tuple(round(float(x), 4) for x in overlap.bbox)
                img = _label_image_for(draw_id, region, region_key).clip(region)
                coords = _draw_declustered(
                    sampling_mask(img, int(draw_value)),
                    overlap,
                    max(scale_m, float(CONFIG.review.draw_scale_m)),
                    n_draw,
                )

            ref_labels = _labels_at(reference_id, coords, overlap)
            cmp_labels = _labels_at(compare_id, coords, overlap)
        else:
            region = overlap.ee_geometry()

            # Build each annual label image **once** and reuse it for both the
            # mask draw and the label read-back, so the expensive composite
            # (Dynamic World's year-long per-pixel mode) is not recomputed per
            # read. Both sides are built because both labels are read back at
            # every location. Bound each composite to the working region and
            # cache it by the region's bbox, so the expensive Dynamic World modal
            # composite is filtered to the AOI and reused across queries in the
            # same area.
            region_key = tuple(round(float(x), 4) for x in overlap.bbox)
            ref_img = _label_image_for(reference_id, region, region_key).clip(region)
            cmp_img = _label_image_for(compare_id, region, region_key).clip(region)

            # Build the mask to draw from, per mode.
            if mode == "both":
                mask = _joint_mask(ref_img, cmp_img, reference_value, compare_value)
            elif mode == "reference":
                # Representative samples of the reference class = a single-class
                # Stage 2 draw.
                mask = sampling_mask(ref_img, int(reference_value))
            else:  # "compare"
                mask = sampling_mask(cmp_img, int(compare_value))

            # Oversample the draw so that, after the exactness filter below, we
            # still have about ``n`` locations. Boundary points can be dropped by
            # the filter, so ask for a margin rather than exactly n.
            #
            # Find candidates at a COARSE scale so the stratifiedSample is fast --
            # the explorer only needs to locate representative places, and each
            # location's label is read back below at native resolution regardless.
            # Never finer than the join scale (don't pretend to more detail than
            # the data has).
            draw_scale_m = max(scale_m, float(CONFIG.review.draw_scale_m))
            coords = _draw_declustered(mask, overlap, draw_scale_m, n_draw)

            # Read both sides' labels back from the already-built images in a
            # SINGLE round trip (points only cross the network). Read at the finer
            # of the two products' **native** resolutions, not the (possibly
            # coarse) draw scale, so a location's reported label is what the map
            # says there.
            read_scale_m = min(
                _native_scale_m(reference_id), _native_scale_m(compare_id)
            )
            ref_labels, cmp_labels = _read_labels_both(
                ref_img, cmp_img, coords, read_scale_m
            )

        # Exactness filter. ``stratifiedSample`` (mask draw) and
        # ``sampleRegions`` (label read-back) reproject independently, so a point
        # sampled on a class boundary can read back a neighbouring label -- at
        # the join scale *or* even at native resolution. For each mode, drop any
        # location whose authoritative read-back does not satisfy the query, so
        # an evidence location always genuinely shows the queried class(es).
        # This is a data-cleaning step, never scoring (section 6.2).
        def _matches(rl: int | None, cl: int | None) -> bool:
            if mode == "both":
                return rl == int(reference_value) and cl == int(compare_value)
            if mode == "reference":
                return rl == int(reference_value)
            return cl == int(compare_value)  # "compare"

        kept = [
            (c, rl, cl)
            for c, rl, cl in zip(coords, ref_labels, cmp_labels)
            if _matches(rl, cl)
        ][:n]

    window_px = int(CONFIG.review.patch_window_px)
    window_m = float(window_px) * scale_m

    locations: list[EvidenceLocation] = []
    for (lon, lat), rl, cl in kept:
        locations.append(
            EvidenceLocation(
                lon=lon,
                lat=lat,
                reference_label=rl,
                reference_label_name=(
                    class_name(reference_id, rl) if rl is not None else None
                ),
                compare_label=cl,
                compare_label_name=(
                    class_name(compare_id, cl) if cl is not None else None
                ),
                patch_window_px=window_px,
            )
        )

    return EvidenceResult(
        mode=mode,
        reference_id=reference_id,
        compare_id=compare_id,
        reference_value=None if reference_value is None else int(reference_value),
        compare_value=None if compare_value is None else int(compare_value),
        patch_window_px=window_px,
        patch_window_m=window_m,
        reference_pixel_m=_native_scale_m(reference_id),
        compare_pixel_m=_native_scale_m(compare_id),
        source=source,
        locations=locations,
    )
