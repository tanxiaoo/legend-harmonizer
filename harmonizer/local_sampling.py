"""Local-raster equivalent of Stage 2 sampling (docs/PIPELINE.md, Stage 2,
"Where the raster work happens" -- the local-raster path deferred from the
original build).

``harmonizer.sampling`` and ``harmonizer.buffering`` do the erosion,
homogeneity-window check, and stratified candidate draw as **server-side GEE
image operations**, so this is the same logic against a local GeoTIFF (or VRT
mosaic) instead: read the AOI window into memory once, erode/homogeneity-mask
with ``scipy.ndimage``, and stratified-sample candidate pixels with ``numpy``.
The absent-vs-buffered-away rule, the min-spacing thinning, and everything
downstream of "candidate coordinates" are unchanged and reused as-is from
``sampling.py`` -- only the two GEE-specific steps (present_classes,
stratified candidate draw + the paired pre/post-erode count) get a local
counterpart here.

**Read-once-per-AOI, not per-class.** Erosion needs the whole class-mask array,
so the AOI window is read into memory once per call and every class's mask/
erosion/stratified draw operates on that same in-memory array -- exactly the
"annual label image built once and reused across every class" principle Stage 2
already documents for the GEE path.

**CRS handling.** The AOI/overlap bbox is always EPSG:4326 (docs/PIPELINE.md
section 2, ``target_crs``). If the raster's own CRS differs, the bbox is
reprojected to it before windowing, and sampled pixel centres are reprojected
back to EPSG:4326 for the returned coordinates -- the same responsibility the
Stage 1 local adapter already carries for point sampling.

See docs/PIPELINE.md, Stage 2.
"""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np

from harmonizer.config import CONFIG
from harmonizer.overlap import Overlap
from harmonizer.registry.products import Coord

_CELL_PROP = "cell"  # unused here; kept for symmetry with sampling.py's constant


# --------------------------------------------------------------------------- #
# Windowed read of the AOI, once per map
# --------------------------------------------------------------------------- #


class LocalLabelWindow:
    """The AOI's label pixels read into memory once, plus the geo-referencing
    needed to turn a pixel index back into an (lon, lat) coordinate.

    Built once per ``sample_map`` call (mirrors the GEE path's "annual label
    image built once, reused across every class").
    """

    def __init__(self, data: np.ndarray, transform, raster_crs, nodata) -> None:
        self.data = data  # 2-D array, shape (rows, cols)
        self.transform = transform  # affine: pixel (col, row) -> raster CRS (x, y)
        self.raster_crs = raster_crs
        self.nodata = nodata

    @property
    def shape(self) -> tuple[int, int]:
        return self.data.shape

    def pixel_to_lonlat(self, rows: np.ndarray, cols: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Pixel centres (row, col) -> (lon, lat) in EPSG:4326."""
        from pyproj import Transformer

        xs, ys = self.transform * (cols + 0.5, rows + 0.5)
        if str(self.raster_crs) in ("EPSG:4326", "epsg:4326"):
            return np.asarray(xs), np.asarray(ys)
        transformer = Transformer.from_crs(self.raster_crs, "EPSG:4326", always_xy=True)
        lons, lats = transformer.transform(xs, ys)
        return np.asarray(lons), np.asarray(lats)


def read_label_window(path: str, band: int, bbox: tuple[float, float, float, float]) -> LocalLabelWindow:
    """Read the raster's pixels over ``bbox`` (EPSG:4326) into memory.

    ``bbox`` is reprojected into the raster's own CRS before windowing if they
    differ (docs/PIPELINE.md section 2: CRS alignment is explicit, never
    assumed). Raises ``ValueError`` if the window would be empty (AOI does not
    intersect the raster).
    """
    import rasterio
    from pyproj import Transformer
    from rasterio.windows import from_bounds

    with rasterio.open(path) as ds:
        min_lon, min_lat, max_lon, max_lat = bbox
        if str(ds.crs) not in ("EPSG:4326", "epsg:4326"):
            transformer = Transformer.from_crs("EPSG:4326", ds.crs, always_xy=True)
            xs, ys = transformer.transform([min_lon, max_lon], [min_lat, max_lat])
            win_bounds = (min(xs), min(ys), max(xs), max(ys))
        else:
            win_bounds = (min_lon, min_lat, max_lon, max_lat)

        window = from_bounds(*win_bounds, transform=ds.transform)
        window = window.round_offsets().round_lengths()
        # Clip to the dataset so an AOI extending past the raster's own extent
        # doesn't ask for a negative-size or out-of-range window.
        window = window.intersection(
            rasterio.windows.Window(0, 0, ds.width, ds.height)
        )
        if window.width <= 0 or window.height <= 0:
            raise ValueError("AOI does not intersect the raster's extent")

        data = ds.read(int(band), window=window)
        win_transform = ds.window_transform(window)
        return LocalLabelWindow(
            data=data, transform=win_transform, raster_crs=ds.crs, nodata=ds.nodata
        )


# --------------------------------------------------------------------------- #
# Buffering (erosion + homogeneous window), mirroring harmonizer.buffering
# --------------------------------------------------------------------------- #


def class_mask(window: LocalLabelWindow, class_value: int) -> np.ndarray:
    """Boolean mask, True where the window's pixels equal ``class_value``."""
    mask = window.data == class_value
    if window.nodata is not None:
        mask &= window.data != window.nodata
    return mask


def _erode_square(mask: np.ndarray, radius_px: int) -> np.ndarray:
    """Binary erosion by a square structuring element of the given pixel radius.

    Mirrors ``buffering._min_over_square``: a pixel survives only if every pixel
    within ``radius_px`` (a square of side ``2*radius+1``) is also set.
    ``border_value=False`` treats pixels off the window edge as unset, exactly
    like the GEE path's ``mask.unmask(0)``.
    """
    if radius_px <= 0:
        return mask
    from scipy.ndimage import binary_erosion

    size = 2 * int(radius_px) + 1
    structure = np.ones((size, size), dtype=bool)
    return binary_erosion(mask, structure=structure, border_value=False)


def eroded_mask(window: LocalLabelWindow, class_value: int, erode_pixels: int) -> np.ndarray:
    return _erode_square(class_mask(window, class_value), erode_pixels)


def homogeneous_mask(window: LocalLabelWindow, class_value: int, homogeneous_window: int) -> np.ndarray:
    radius = (int(homogeneous_window) - 1) // 2
    return _erode_square(class_mask(window, class_value), radius)


def sampling_mask(window: LocalLabelWindow, class_value: int, erode_pixels: int) -> np.ndarray:
    """Combined mask a class is sampled from: eroded core AND homogeneous window.

    Mirrors ``buffering.sampling_mask``.
    """
    eroded = eroded_mask(window, class_value, erode_pixels)
    homogeneous = homogeneous_mask(window, class_value, CONFIG.buffering.homogeneous_window)
    return eroded & homogeneous


# --------------------------------------------------------------------------- #
# Present classes, stratified candidate draw, mirroring harmonizer.sampling
# --------------------------------------------------------------------------- #


def present_classes(window: LocalLabelWindow) -> list[int]:
    """Distinct class values present in the window (mirrors sampling.present_classes)."""
    vals = np.unique(window.data)
    if window.nodata is not None:
        vals = vals[vals != window.nodata]
    return sorted(int(v) for v in vals)


def count_candidates_both(
    window: LocalLabelWindow, class_value: int, erode_pixels: int
) -> tuple[int, int]:
    """(pre_erode, post_erode) sampling-mask pixel counts, for the
    absent-vs-buffered-away rule. Mirrors ``sampling._count_candidates_both`` --
    a local in-memory count needs no round-trip pairing, but the pair is kept for
    the same contract.
    """
    pre = sampling_mask(window, class_value, erode_pixels=0)
    post = sampling_mask(window, class_value, erode_pixels=erode_pixels)
    return int(pre.sum()), int(post.sum())


def stratified_candidates(
    window: LocalLabelWindow,
    class_value: int,
    overlap: Overlap,
    erode_pixels: int,
    target: int,
) -> list[Coord]:
    """Draw candidate (lon, lat) points for one class, spread across grid cells.

    Mirrors ``sampling._stratified_candidates``: builds the sampling mask, tags
    each set pixel with a grid-cell id (the AOI divided into
    ``grid_cell_deg``-sized cells), and draws up to a per-cell quota so the
    target is spread across the overlap rather than concentrated in one patch.
    """
    mask = sampling_mask(window, class_value, erode_pixels=erode_pixels)
    rows, cols = np.nonzero(mask)
    if rows.size == 0:
        return []

    lons, lats = window.pixel_to_lonlat(rows, cols)

    cell_deg = CONFIG.sampling.grid_cell_deg
    min_lon, min_lat, max_lon, max_lat = overlap.bbox
    n_cols = max(1, int(math.ceil((max_lon - min_lon) / cell_deg)))
    col_ids = np.floor((lons - min_lon) / cell_deg).astype(int)
    row_ids = np.floor((lats - min_lat) / cell_deg).astype(int)
    cell_ids = col_ids + row_ids * n_cols

    unique_cells = np.unique(cell_ids)
    n_cells = max(1, unique_cells.size)
    per_cell = max(1, math.ceil(target / n_cells))

    rng = np.random.default_rng(CONFIG.gmm.random_seed)
    coords: list[Coord] = []
    for cell in unique_cells:
        idx = np.nonzero(cell_ids == cell)[0]
        if idx.size > per_cell:
            idx = rng.choice(idx, size=per_cell, replace=False)
        for i in idx:
            coords.append((float(lons[i]), float(lats[i])))
    return coords
