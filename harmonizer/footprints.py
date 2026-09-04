"""Source-derived operational footprints for local-raster products (DESIGN.md 3.1).

``docs/PIPELINE.md`` section 2 prescribes that a product's authoritative
footprint is **derived from the loaded source at run time**, and that the
registry's ``footprint:`` is advisory display data. Until now only the advisory
box existed in code (``registry/products.py`` returns the YAML value verbatim),
so a run's overlap came from declared rectangles rather than from the pixels
actually on disk.

This module supplies the derived half: the rasterio bounds of a local product's
real source, reprojected to EPSG:4326. GEE products are unaffected -- they keep
returning their declared footprint (``None`` for a global product), because
there is no local file to measure.

**Why the bbox is still a bbox.** For a tile-set product the derived box is the
union of its tiles and can enclose no-data holes (the WorldCover set covers
Africa's land, not the sea between). That is deliberate and needs no polygon
machinery: the chunked sampler's per-cell class masks are simply empty over a
hole, so holes cost a cheap empty read and contribute no points.

Results are cached per product: the bounds of a file on disk do not change
within a run, and probing a 96-file mosaic is not free.
"""

from __future__ import annotations

import functools
import logging
from pathlib import Path

from harmonizer.config import CONFIG, REPO_ROOT

__all__ = ["derived_footprint", "operational_footprint", "invalidate"]

_LOG = logging.getLogger(__name__)

BBox = tuple[float, float, float, float]


def _resolve(path_str: str) -> Path:
    """A registry ``access.path`` as an absolute path."""
    path = Path(path_str)
    return path if path.is_absolute() else (REPO_ROOT / path)


@functools.lru_cache(maxsize=None)
def derived_footprint(product_id: str) -> BBox | None:
    """The rasterio bounds of a local product's source, in EPSG:4326.

    Returns ``None`` when the product is not a local raster, its source is
    missing or unreadable, or its CRS cannot be reprojected -- in every one of
    those cases the caller falls back to the declared registry footprint, which
    is the pre-existing behaviour. A derived footprint is an improvement on the
    declared box, never a precondition for running.

    Prefers the **converted COG tree** when one exists, for the same reason the
    tile path does: it is the same pixels, indexed for cheap access. For a
    mosaic the MosaicJSON already records the union bounds, so no raster is
    opened at all.
    """
    from harmonizer.registry.legends import spec as _product_spec

    spec = _product_spec(product_id)
    if spec is None or getattr(spec.access, "method", None) != "local_raster":
        return None

    box = _from_cog_tree(product_id)
    if box is not None:
        return box

    try:
        path = _resolve(spec.access.path)
        if not path.exists():
            return None
        return _bounds_4326(path)
    except Exception as exc:  # unreadable source, bad CRS, missing PROJ, ...
        _LOG.warning(
            "%s: could not derive a footprint from %s (%s: %s); "
            "falling back to the registry's declared footprint",
            product_id,
            getattr(spec.access, "path", "?"),
            type(exc).__name__,
            exc,
        )
        return None


def _from_cog_tree(product_id: str) -> BBox | None:
    """Bounds from the converted COG tree, if one has been built."""
    from harmonizer import local_tiles

    try:
        source = local_tiles.cog_source(product_id)
    except Exception:
        return None
    if source is None:
        return None
    path, is_mosaic = source
    try:
        if is_mosaic:
            import json

            doc = json.loads(Path(path).read_text())
            bounds = doc.get("bounds")
            if bounds and len(bounds) == 4:
                # MosaicJSON bounds are already lon/lat.
                return (
                    float(bounds[0]),
                    float(bounds[1]),
                    float(bounds[2]),
                    float(bounds[3]),
                )
            return None
        return _bounds_4326(Path(path))
    except Exception:
        return None


def _bounds_4326(path: Path) -> BBox | None:
    """A raster's bounds reprojected to EPSG:4326."""
    import rasterio
    from rasterio.warp import transform_bounds

    with rasterio.open(path) as ds:
        bounds = ds.bounds
        if ds.crs is None:
            return None
        if ds.crs.to_epsg() == 4326:
            left, bottom, right, top = bounds
        else:
            left, bottom, right, top = transform_bounds(
                ds.crs, "EPSG:4326", *bounds, densify_pts=21
            )
    if not (left < right and bottom < top):
        return None
    return (float(left), float(bottom), float(right), float(top))


def operational_footprint(product_id: str) -> BBox | None:
    """The footprint a run should use: derived where possible, declared otherwise.

    This is the function overlap computation should call. ``None`` means
    "unconstrained" (a global product), exactly as the declared footprint's
    ``None`` already does, so callers need no new branch.
    """
    derived = derived_footprint(product_id)
    if derived is not None:
        return derived

    from harmonizer.registry.legends import spec as _product_spec

    spec = _product_spec(product_id)
    return getattr(spec, "footprint", None) if spec is not None else None


def invalidate(product_id: str | None = None) -> None:
    """Forget cached footprints after a source is replaced or converted.

    ``lru_cache`` has no per-key eviction, so a single product is invalidated by
    clearing everything; it refills on the next lookup at one probe each.
    """
    derived_footprint.cache_clear()
