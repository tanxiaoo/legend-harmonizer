"""Overlap computation (Stages 1, 2 and 5).

Computes the valid working region for a run as the intersection of the *selected*
products' own footprints. A global product (WorldCover, Dynamic World,
AlphaEarth) does not constrain the region; a bounded product (HRLC, whose extent
is the Eastern Sahel box) does. So the region follows from the pair being run,
not from any single map's constant: WorldCover x Dynamic World are both global
and their intersection is global, whereas any pairing that includes HRLC is
clamped to HRLC's box. Shared by sampling (Stage 2), which stratifies *within*
the region, and by the API (Stage 5), which refuses early on an empty overlap.

Kept deliberately small: the overlap is expressed both as a plain
(min_lon, min_lat, max_lon, max_lat) tuple and, on demand, as an ``ee.Geometry``
rectangle for server-side GEE work. Building the ee.Geometry is lazy so importing
this module never requires Earth Engine.

See docs/PIPELINE.md, Stages 2 and 5.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

from harmonizer.config import CONFIG

# A bounding box as (min_lon, min_lat, max_lon, max_lat) in EPSG:4326.
BBox = tuple[float, float, float, float]

# The full lon/lat extent, used as the identity when intersecting footprints:
# a global product contributes this and so constrains nothing.
GLOBAL_BBOX: BBox = (-180.0, -90.0, 180.0, 90.0)


def _intersect(a: BBox, b: BBox) -> BBox | None:
    """Intersect two (min_lon, min_lat, max_lon, max_lat) boxes, or None."""
    min_lon = max(a[0], b[0])
    min_lat = max(a[1], b[1])
    max_lon = min(a[2], b[2])
    max_lat = min(a[3], b[3])
    if min_lon >= max_lon or min_lat >= max_lat:
        return None
    return (min_lon, min_lat, max_lon, max_lat)


@dataclass(frozen=True)
class Overlap:
    """The valid working region for a run, as a lon/lat bounding box.

    ``bbox`` is (min_lon, min_lat, max_lon, max_lat) in EPSG:4326. ``ee_geometry``
    returns the same rectangle as an ``ee.Geometry`` for server-side sampling and
    erosion; it is built on demand so this module imports without Earth Engine.
    ``is_global`` reports whether the region spans the whole lon/lat extent (no
    bounded product constrained it), which callers may surface to the user.
    """

    bbox: BBox

    @property
    def is_global(self) -> bool:
        return self.bbox == GLOBAL_BBOX

    def ee_geometry(self):
        import ee

        min_lon, min_lat, max_lon, max_lat = self.bbox
        return ee.Geometry.Rectangle(
            [min_lon, min_lat, max_lon, max_lat], proj=CONFIG.maps.target_crs, geodesic=False
        )


def overlap_of_footprints(
    footprints: Iterable[BBox | None], aoi: BBox | None = None
) -> Overlap:
    """Intersect a set of product footprints into the run's working region.

    Each footprint is a lon/lat box, or ``None`` for a global product. Global
    products contribute the identity (``GLOBAL_BBOX``) and so do not constrain the
    result; if every product is global the region is global. An optional ``aoi``
    box narrows the result further -- it is an extra spatial constraint the caller
    supplies (e.g. a small test AOI), *not* a product footprint, so it is applied
    on top of the footprint-derived region rather than replacing it. Raises
    ``ValueError`` if the footprints (and AOI) do not overlap.
    """
    result: BBox = GLOBAL_BBOX
    boxes = list(footprints)
    if aoi is not None:
        boxes.append(aoi)
    for box in boxes:
        box = GLOBAL_BBOX if box is None else box
        clipped = _intersect(result, box)
        if clipped is None:
            raise ValueError("selected products' footprints (and AOI) do not overlap")
        result = clipped
    return Overlap(bbox=result)


def overlap_for_products(
    product_ids: Sequence[str], aoi: BBox | None = None
) -> Overlap:
    """The working region for a named set of products, from their footprints.

    Each product contributes its **operational** footprint
    (``harmonizer.footprints``): for a local raster that is the rasterio bounds
    of the actual source reprojected to EPSG:4326, per docs/PIPELINE.md section 2
    and DESIGN.md 3.1; for a GEE product, or where the source cannot be read, the
    registry's declared box. Global products (WorldCover, Dynamic World,
    AlphaEarth) contribute ``None`` and so do not constrain the region, meaning a
    global pair yields a global region while any pairing including a bounded
    product is clamped to that product's real extent.

    Deriving from the source matters because a declared box can be stale or
    approximate -- it is hand-written in the YAML and describes whatever tile was
    present when it was written. An optional ``aoi`` box narrows the
    footprint-derived region further without changing the derivation logic.
    """
    from harmonizer.footprints import operational_footprint

    return overlap_of_footprints(
        (operational_footprint(pid) for pid in product_ids), aoi=aoi
    )


def default_overlap() -> Overlap:
    """The MVP test-swap region: WorldCover x Dynamic World (both global).

    Kept as a convenience for callers that sample the default pair. Because both
    label maps and AlphaEarth are global, this is the global region; it is *not*
    HRLC's box. A run that selects HRLC must derive its region from the selected
    products via :func:`overlap_for_products` instead.
    """
    return overlap_for_products(["worldcover", "dynamicworld", "alphaearth"])
