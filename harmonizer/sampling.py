"""Per-class sampling (Stage 2).

Draws clean, per-class sample points for each map independently. Because classes
are modelled separately, the two maps' point sets are unrelated -- there is no
need for co-located points across maps.

Division of labour (docs/PIPELINE.md, Stage 2, "Where the raster work happens").
For the MVP both label maps are GEE-native, so the raster work -- erosion,
homogeneous-window confirmation, and stratified sampling spread across grid cells
-- runs **server-side in GEE** and only candidate *coordinates* come back over
the network. Those coordinates are then pushed through the Stage 1 adapters to
fetch the map label (a cross-check) and the AlphaEarth embedding for each point.

Per map, per class this module:
  1. builds the annual label image (mirroring the Stage 1 adapters, without
     modifying them);
  2. discovers the classes actually present in the overlap server-side, so no
     class-code list has to be invented for a product that lacks one in config;
  3. erodes + homogeneity-masks the class (see ``buffering``), then draws
     candidate points spread across grid cells of the overlap, capped at the
     target per class;
  4. thins the candidates to the configured minimum spacing;
  5. routes candidates through the Stage 1 label adapter (cross-check) and the
     AlphaEarth adapter, dropping points with a no-data label or masked embedding;
  6. checks the surviving count against the floor and applies the
     absent-vs-buffered-away rule: short only *after* eroding -> relax the buffer
     to one pixel and resample; short *even before* eroding -> flag ``absent``.

The output per map is a point table (coordinates, class label, embedding vector)
plus per-class survival counts and buffer-relaxation / absent flags, saved to
``cache/``.

See docs/PIPELINE.md, Stage 2.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np

from harmonizer.buffering import sampling_mask
from harmonizer.config import CONFIG
from harmonizer.overlap import Overlap, default_overlap, overlap_for_products
from harmonizer.registry.products import (
    Coord,
    EmbeddingAdapter,
    LabelAdapter,
    default_registry,
)

# Property name GEE stratified sampling tags each candidate with -- unused
# downstream but present in returned features.
_CELL_PROP = "cell"

def _sample_scale_m() -> float:
    """Raster scale (metres) masks and stratified sampling run at.

    Read from config so a test run can coarsen it (30-100 m) for speed; the
    faithful value for these 10 m products is the config default of 10.0.
    """
    return CONFIG.sampling.sample_scale_m


# --------------------------------------------------------------------------- #
# Result types
# --------------------------------------------------------------------------- #


@dataclass
class ClassSample:
    """The sampled points and bookkeeping for one class of one map."""

    class_value: int
    coords: list[Coord] = field(default_factory=list)
    embeddings: list[list[float]] = field(default_factory=list)
    # Class label read back from the map at each surviving point (cross-check).
    labels: list[int] = field(default_factory=list)

    # Candidate counts used by the absent-vs-buffered-away rule.
    pre_erode_candidates: int = 0     # candidates before erosion (loose mask)
    post_erode_candidates: int = 0    # candidates after the configured erosion

    surviving: int = 0                # points that survived label+embedding checks
    buffer_relaxed: bool = False      # resampled under the 1-pixel buffer
    absent: bool = False              # genuinely rare: below floor even pre-erode
    status: str = "ok"               # "ok" | "relaxed" | "absent"


@dataclass
class MapSample:
    """All per-class samples for one map, plus the config used to draw them."""

    product_id: str
    working_year: int
    floor: int
    target: int
    classes: dict[int, ClassSample] = field(default_factory=dict)


# --------------------------------------------------------------------------- #
# Annual label images (mirror the Stage 1 adapters without importing internals)
# --------------------------------------------------------------------------- #


def _worldcover_label_image(spec, region=None):
    # WorldCover is a single global static image; a region does not change what is
    # built (it is clipped by the caller), so ``region`` is ignored here.
    import ee

    return (
        ee.ImageCollection(spec.access.asset_id)
        .first()
        .select(str(spec.band))
        .rename("label")
    )


def _dynamicworld_label_image(spec, region=None):
    # Dynamic World's annual label is a per-pixel modal composite over a year-long
    # ImageCollection -- the expensive part. When a ``region`` is given, bound the
    # collection to it with ``filterBounds`` so the mode reducer only touches scenes
    # intersecting that area, instead of reducing the global collection (a big win
    # for the small-AOI evidence explorer). With no region the global behaviour is
    # unchanged (Stage 2 clips afterwards).
    import ee

    year = CONFIG.maps.working_year
    start = f"{year}-01-01"
    end = f"{year + 1}-01-01"
    collection = (
        ee.ImageCollection(spec.access.asset_id)
        .filterDate(start, end)
        .select(str(spec.band))
    )
    if region is not None:
        collection = collection.filterBounds(region)
    return collection.reduce(ee.Reducer.mode()).rename("label")


# Cache of built label images, keyed by (product_id, region-key, working_year), so
# repeated evidence queries in the same AOI reuse the (expensive) Dynamic World
# composite instead of re-deriving it each call. Region-less calls (Stage 2) share
# a single cached global image per product.
_LABEL_IMAGE_CACHE: dict = {}


def _label_image_for(product_id: str, region=None, region_key=None):
    """Annual label image for a GEE label product.

    Asset id and band come from the product registry (per-map YAML, the single
    source of truth); the working year is a run-level config setting. An optional
    ``region`` bounds the (Dynamic World) composite to that area; ``region_key`` is
    a hashable identity for that region (e.g. its bbox tuple) used as a cache key,
    so repeated queries over the same AOI reuse the built image without any extra
    network call. ``region=None`` preserves the original global behaviour (Stage 2).
    """
    from harmonizer.registry.legends import spec as _product_spec

    spec = _product_spec(product_id)
    if spec is None or spec.access.method != "gee":
        raise ValueError(
            f"Stage 2 sampling supports the GEE label products only; got {product_id!r}"
        )

    cache_key = (product_id, region_key, CONFIG.maps.working_year)
    cached = _LABEL_IMAGE_CACHE.get(cache_key)
    if cached is not None:
        return cached

    if product_id == "worldcover":
        image = _worldcover_label_image(spec, region)
    elif product_id == "dynamicworld":
        image = _dynamicworld_label_image(spec, region)
    else:
        raise ValueError(
            f"Stage 2 sampling supports the GEE label products only; got {product_id!r}"
        )
    _LABEL_IMAGE_CACHE[cache_key] = image
    return image


# --------------------------------------------------------------------------- #
# Server-side helpers
# --------------------------------------------------------------------------- #


def present_classes(label_image, region) -> list[int]:
    """Distinct class values present in ``region``, discovered server-side.

    Uses a frequency histogram over a coarse sample so no product-specific class
    list has to be hardcoded. Discovering classes from the data also means only
    classes actually present in the overlap get sampled.

    This call is also what makes the Stage 7 absence check free: it is the run's
    one authoritative observation of which classes the AOI holds, and Stage 2
    records the outcome in the sample cache, so nothing downstream needs to ask
    Earth Engine the same question again (see ``harmonizer.absence``).
    """
    import ee

    from harmonizer.registry.adapters._gee import get_info

    hist = get_info(
        label_image.reduceRegion(
            reducer=ee.Reducer.frequencyHistogram(),
            geometry=region,
            scale=_sample_scale_m() * 30,  # coarse: only need which classes exist
            maxPixels=1e9,
            bestEffort=True,
        )
    )
    buckets = (hist or {}).get("label") or {}
    values = sorted(int(float(k)) for k in buckets.keys())
    return values


def _grid_cell_image(region, bbox: tuple[float, float, float, float] | None = None):
    """Integer grid-cell id per pixel, so sampling can be stratified by cell.

    The overlap is divided into cells of ``grid_cell_deg`` degrees. Each pixel is
    tagged with ``col + n_cols * row`` so ``stratifiedSample`` draws a quota per
    cell and one large patch cannot dominate a class's sample. When the caller
    already knows the region's bbox client-side (an Overlap does), passing it as
    ``bbox`` skips the ``region_bbox`` server round trip.
    """
    import ee

    cell_deg = CONFIG.sampling.grid_cell_deg
    lonlat = ee.Image.pixelLonLat()
    min_lon, min_lat, max_lon, max_lat = (
        bbox if bbox is not None else region_bbox(region)
    )
    n_cols = max(1, int(math.ceil((max_lon - min_lon) / cell_deg)))
    col = lonlat.select("longitude").subtract(min_lon).divide(cell_deg).floor()
    row = lonlat.select("latitude").subtract(min_lat).divide(cell_deg).floor()
    return col.add(row.multiply(n_cols)).toInt().rename(_CELL_PROP)


def region_bbox(region) -> tuple[float, float, float, float]:
    """(min_lon, min_lat, max_lon, max_lat) of an ee.Geometry, client-side."""
    from harmonizer.registry.adapters._gee import get_info

    coords = get_info(region.bounds().coordinates())[0]
    lons = [c[0] for c in coords]
    lats = [c[1] for c in coords]
    return (min(lons), min(lats), max(lons), max(lats))


def _stratified_candidates(
    label_image,
    class_value: int,
    overlap: Overlap,
    erode_pixels: int,
    target: int,
) -> list[Coord]:
    """Draw candidate (lon, lat) points for one class, spread across grid cells.

    Builds the class's sampling mask (eroded + homogeneous), tags it with the
    grid-cell id, and stratified-samples per cell so the target is spread across
    the overlap. Returns candidate coordinates only; label/embedding checks and
    min-spacing thinning happen in the caller.
    """
    from harmonizer.registry.adapters._gee import get_info

    region = overlap.ee_geometry()
    mask = sampling_mask(label_image, class_value, erode_pixels=erode_pixels)
    cells = _grid_cell_image(region).updateMask(mask)

    # Per-cell quota: spread the target across the number of cells, with a small
    # floor so sparse classes still draw a few per occupied cell.
    n_cols = max(
        1,
        int(math.ceil((overlap.bbox[2] - overlap.bbox[0]) / CONFIG.sampling.grid_cell_deg)),
    )
    n_rows = max(
        1,
        int(math.ceil((overlap.bbox[3] - overlap.bbox[1]) / CONFIG.sampling.grid_cell_deg)),
    )
    n_cells = max(1, n_cols * n_rows)
    per_cell = max(1, int(math.ceil(target / n_cells)))

    # stratifiedSample has no maxPixels arg (that belongs to sample()); work is
    # bounded by the scale and tileScale instead.
    fc = cells.stratifiedSample(
        numPoints=per_cell,
        classBand=_CELL_PROP,
        region=region,
        scale=_sample_scale_m(),
        seed=CONFIG.gmm.random_seed,
        geometries=True,
        tileScale=16,
    )
    info = get_info(fc)
    coords: list[Coord] = []
    for feat in info.get("features", []):
        geom = feat.get("geometry") or {}
        c = geom.get("coordinates")
        if c and len(c) == 2:
            coords.append((float(c[0]), float(c[1])))
    return coords


def _count_candidates_both(
    label_image, class_value: int, overlap: Overlap, erode_pixels: int
) -> tuple[int, int]:
    """Count sampling-mask pixels at zero *and* ``erode_pixels`` buffers at once.

    Returns ``(pre_erode, post_erode)`` for the absent-vs-buffered-away rule: the
    pre-erode count uses a zero buffer (raw class + homogeneity), the post-erode
    count uses the configured erosion. Both masks are reduced in a **single**
    ``reduceRegion`` (two bands) so the class needs only one server round-trip
    here instead of two -- the year-long Dynamic World mode composite is expensive,
    so halving the count calls per class is a real saving.
    """
    import ee

    from harmonizer.registry.adapters._gee import get_info

    region = overlap.ee_geometry()
    pre = sampling_mask(label_image, class_value, erode_pixels=0).rename("pre")
    post = sampling_mask(
        label_image, class_value, erode_pixels=erode_pixels
    ).rename("post")
    counts = (
        get_info(
            pre.addBands(post).reduceRegion(
                reducer=ee.Reducer.count(),
                geometry=region,
                scale=_sample_scale_m(),
                maxPixels=1e10,
                bestEffort=True,
            )
        )
    ) or {}
    return (
        int(counts.get("pre", 0) or 0),
        int(counts.get("post", 0) or 0),
    )


# --------------------------------------------------------------------------- #
# Min-spacing thinning (client-side, in metres)
# --------------------------------------------------------------------------- #


def _thin_by_spacing(coords: Sequence[Coord], min_spacing_m: float) -> list[Coord]:
    """Greedily keep points at least ``min_spacing_m`` apart (great-circle)."""
    if min_spacing_m <= 0 or len(coords) <= 1:
        return list(coords)

    kept: list[Coord] = []
    kept_rad: list[tuple[float, float]] = []
    r_earth = 6_371_000.0
    min_ang = min_spacing_m / r_earth  # small-angle threshold in radians

    for lon, lat in coords:
        lon_r = math.radians(lon)
        lat_r = math.radians(lat)
        ok = True
        for klon, klat in kept_rad:
            # Equirectangular approximation is ample at 100 m scale.
            dlon = (lon_r - klon) * math.cos(0.5 * (lat_r + klat))
            dlat = lat_r - klat
            if math.hypot(dlon, dlat) < min_ang:
                ok = False
                break
        if ok:
            kept.append((lon, lat))
            kept_rad.append((lon_r, lat_r))
    return kept


# --------------------------------------------------------------------------- #
# Per-class sampling with the absent-vs-buffered-away rule
# --------------------------------------------------------------------------- #


def _fetch_survivors(
    coords: Sequence[Coord],
    label_adapter: LabelAdapter,
    embedding_adapter: EmbeddingAdapter,
    class_value: int,
) -> tuple[list[Coord], list[list[float]], list[int]]:
    """Route candidates through Stage 1 adapters; keep points that survive.

    A point survives when its map label reads back as this class (cross-check,
    dropping no-data labels) and its AlphaEarth embedding is not masked.
    """
    if not coords:
        return [], [], []

    labels = label_adapter.sample_labels(coords)
    embeddings = embedding_adapter.sample_embeddings(coords)

    kept_coords: list[Coord] = []
    kept_emb: list[list[float]] = []
    kept_lab: list[int] = []
    for coord, lab, emb in zip(coords, labels, embeddings):
        if lab.masked or lab.label is None:
            continue
        if lab.label != int(class_value):  # cross-check against the source map
            continue
        if emb.masked or emb.vector is None:
            continue
        kept_coords.append(coord)
        kept_emb.append([float(x) for x in emb.vector])
        kept_lab.append(int(lab.label))
    return kept_coords, kept_emb, kept_lab


def sample_class(
    class_value: int,
    label_adapter: LabelAdapter,
    embedding_adapter: EmbeddingAdapter,
    *,
    count_candidates_both,
    draw_candidates,
) -> ClassSample:
    """Sample one class end-to-end, applying the absent-vs-buffered-away rule.

    The GEE and local-raster paths share this exact logic and differ only in
    *how* candidates are counted/drawn -- ``count_candidates_both(erode_pixels)
    -> (pre, post)`` and ``draw_candidates(erode_pixels) -> list[Coord]`` are
    passed in as closures over whichever image/window the caller built, so the
    absent-vs-buffered-away rule is written once and applies identically to
    both (docs/PIPELINE.md, Stage 2's rule is a single method, not one per
    access type).
    """
    cfg = CONFIG
    floor = cfg.sampling.points_floor
    target = cfg.sampling.points_target

    result = ClassSample(class_value=int(class_value))

    # Candidate availability at both buffers (mask-pixel counts), for the rule.
    result.pre_erode_candidates, result.post_erode_candidates = count_candidates_both(
        cfg.buffering.erode_pixels
    )

    def _draw(erode_pixels: int) -> tuple[list[Coord], list[list[float]], list[int]]:
        cands = draw_candidates(erode_pixels)
        cands = _thin_by_spacing(cands, cfg.sampling.min_spacing_m)
        cands = cands[:target]
        return _fetch_survivors(cands, label_adapter, embedding_adapter, class_value)

    # First pass at the configured buffer.
    coords, emb, lab = _draw(cfg.buffering.erode_pixels)

    if len(coords) < floor:
        # Decide why the class is starved.
        if result.pre_erode_candidates < floor:
            # Below floor even before eroding -> genuinely rare. Do not force-fit.
            result.absent = True
            result.status = "absent"
        else:
            # Adequate before eroding, short after -> buffered away. Relax to the
            # 1-pixel buffer and resample.
            result.buffer_relaxed = True
            result.status = "relaxed"
            coords, emb, lab = _draw(cfg.buffering.relaxed_erode_pixels)

    result.coords = coords
    result.embeddings = emb
    result.labels = lab
    result.surviving = len(coords)
    return result


# --------------------------------------------------------------------------- #
# Whole-map sampling and caching
# --------------------------------------------------------------------------- #


def _sample_map_gee(
    product_id: str,
    overlap: Overlap,
    label_adapter: LabelAdapter,
    embedding_adapter: EmbeddingAdapter,
) -> MapSample:
    """The GEE-backed path: server-side image ops, coordinates cross the wire."""
    from harmonizer.registry.adapters._gee import ensure_initialized

    ensure_initialized()

    # Build the annual label image once and clip it to the sampling region, so the
    # expensive composite (notably Dynamic World's year-long per-pixel mode) is
    # bounded to the AOI and reused across every class rather than evaluated over
    # the global asset on each call.
    region = overlap.ee_geometry()
    label_image = _label_image_for(product_id).clip(region)
    classes = present_classes(label_image, region)

    ms = MapSample(
        product_id=product_id,
        working_year=CONFIG.maps.working_year,
        floor=CONFIG.sampling.points_floor,
        target=CONFIG.sampling.points_target,
    )
    for cv in classes:
        ms.classes[cv] = sample_class(
            cv,
            label_adapter,
            embedding_adapter,
            count_candidates_both=lambda erode_pixels, cv=cv: _count_candidates_both(
                label_image, cv, overlap, erode_pixels=erode_pixels
            ),
            draw_candidates=lambda erode_pixels, cv=cv: _stratified_candidates(
                label_image,
                cv,
                overlap,
                erode_pixels=erode_pixels,
                target=CONFIG.sampling.points_target,
            ),
        )
    return ms


def _sample_map_local(
    product_id: str,
    spec,
    overlap: Overlap,
    label_adapter: LabelAdapter,
    embedding_adapter: EmbeddingAdapter,
) -> MapSample:
    """The local-raster path (docs/PIPELINE.md Stage 2's deferred local path):
    the AOI window is read into memory once and erosion/stratified sampling run
    in-process against it (harmonizer.local_sampling), mirroring the GEE path's
    logic exactly. Only coordinates and the AlphaEarth embedding still cross the
    network -- the raster itself never leaves this process.
    """
    from harmonizer import local_sampling

    band = int(spec.band) if spec.band is not None else 1
    window = local_sampling.read_label_window(spec.access.path, band, overlap.bbox)
    classes = local_sampling.present_classes(window)

    ms = MapSample(
        product_id=product_id,
        working_year=CONFIG.maps.working_year,
        floor=CONFIG.sampling.points_floor,
        target=CONFIG.sampling.points_target,
    )
    for cv in classes:
        ms.classes[cv] = sample_class(
            cv,
            label_adapter,
            embedding_adapter,
            count_candidates_both=lambda erode_pixels, cv=cv: local_sampling.count_candidates_both(
                window, cv, erode_pixels=erode_pixels
            ),
            draw_candidates=lambda erode_pixels, cv=cv: local_sampling.stratified_candidates(
                window,
                cv,
                overlap,
                erode_pixels=erode_pixels,
                target=CONFIG.sampling.points_target,
            ),
        )
    return ms


def sample_map(
    product_id: str,
    overlap: Overlap | None = None,
    aoi: tuple[float, float, float, float] | None = None,
    label_adapter: LabelAdapter | None = None,
    embedding_adapter: EmbeddingAdapter | None = None,
) -> MapSample:
    """Sample every class present in the overlap for one label product.

    When ``overlap`` is not given it is derived from the selected label product's
    footprint intersected with AlphaEarth's, so the sampling region follows the
    products in play rather than any single map's constant. For a global label
    product (WorldCover, Dynamic World) this region is global. An optional ``aoi``
    lon/lat box narrows that footprint-derived region without replacing it (e.g. a
    small test AOI); it is ignored when an explicit ``overlap`` is passed.

    Dispatches on the product's registry ``access.method``: a GEE product's
    erosion/stratified-sampling runs server-side (``_sample_map_gee``); a
    local-raster product's runs in-process against the file on disk
    (``_sample_map_local``). Either way the embedding still comes from
    AlphaEarth over GEE -- only the *label* source differs.
    """
    overlap = overlap or overlap_for_products([product_id, "alphaearth"], aoi=aoi)

    reg = default_registry()
    label_adapter = label_adapter or reg.get(product_id).adapter_factory()
    embedding_adapter = embedding_adapter or reg.get("alphaearth").adapter_factory()

    spec = reg.spec(product_id)
    if spec.access.method == "local_raster":
        return _sample_map_local(product_id, spec, overlap, label_adapter, embedding_adapter)
    return _sample_map_gee(product_id, overlap, label_adapter, embedding_adapter)


# --------------------------------------------------------------------------- #
# Persistence
# --------------------------------------------------------------------------- #


def cache_path(product_id: str) -> Path:
    return CONFIG.cache_dir / f"samples_{product_id}.npz"


def save_map_sample(ms: MapSample) -> Path:
    """Persist a MapSample to ``cache/`` as a compact ``.npz`` + JSON sidecar.

    Embeddings are stored as one stacked float array with a parallel class-value
    and label array; the per-class bookkeeping (counts, flags, status) goes to a
    JSON sidecar so it is human-readable for the verification.
    """
    CONFIG.cache_dir.mkdir(parents=True, exist_ok=True)

    all_coords: list[Coord] = []
    all_labels: list[int] = []
    all_class_values: list[int] = []
    all_emb: list[list[float]] = []
    summary: dict[str, object] = {
        "product_id": ms.product_id,
        "working_year": ms.working_year,
        "floor": ms.floor,
        "target": ms.target,
        "classes": {},
    }
    for cv, cs in ms.classes.items():
        for coord, emb, lab in zip(cs.coords, cs.embeddings, cs.labels):
            all_coords.append(coord)
            all_emb.append(emb)
            all_labels.append(lab)
            all_class_values.append(cv)
        summary["classes"][str(cv)] = {
            "class_value": cs.class_value,
            "surviving": cs.surviving,
            "pre_erode_candidates": cs.pre_erode_candidates,
            "post_erode_candidates": cs.post_erode_candidates,
            "buffer_relaxed": cs.buffer_relaxed,
            "absent": cs.absent,
            "status": cs.status,
        }

    path = cache_path(ms.product_id)
    np.savez_compressed(
        path,
        coords=np.asarray(all_coords, dtype=float).reshape(-1, 2),
        embeddings=np.asarray(all_emb, dtype=float).reshape(
            -1, CONFIG.maps.embedding_dims
        ),
        labels=np.asarray(all_labels, dtype=int),
        class_values=np.asarray(all_class_values, dtype=int),
    )
    sidecar = path.with_suffix(".json")
    sidecar.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return path
