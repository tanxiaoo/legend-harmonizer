"""Chunked, full-overlap local sampling (DESIGN.md section 3.2).

Replaces ``local_sampling``'s read-everything-once approach for local-raster
products. The old path called ``ds.read()`` over the whole AOI at native
resolution, which for a full local x local overlap (36 deg x 24 deg at 10 m is
~10^11 pixels) is physically impossible -- that single fact is why sampling could
not cover the whole overlap.

Here the overlap is walked **one 0.25 deg grid cell at a time** -- the same grid
the stratification already used, now also the unit of I/O and memory. Per cell a
single decimated windowed read at the run's ``sample_scale_m`` produces a small
array (a 0.25 deg cell at 100 m is ~280x280 px), so peak memory is bounded by one
cell no matter how large the AOI is.

**The sampling semantics are unchanged.** Erosion radius, the 3x3 homogeneity
window, per-cell stratification, min-spacing declustering, the point floor and
target, and the absent-vs-buffered-away rule all behave exactly as before and
read their constants from ``CONFIG`` (docs/PIPELINE.md section 2). What changes
is only *how far and how cheaply* the same computation runs.

Two deliberate differences, recorded rather than hidden:

* **Overviews are MODE-resampled** (``tools/to_cog.py``), so a coarse-scale pixel
  is the modal class of its footprint, whereas GEE's ``scale=`` sampling is
  nearest-at-scale. Mode is at least as defensible for label sampling -- it
  cannot fabricate a class that was not there, and it is less noisy -- but it is
  a difference in kind, not a rounding detail. See ``_read_cell``.
* **Erosion is applied in pixels at the sample scale**, matching how the GEE path
  applies ``erode_pixels`` at ``sample_scale_m``. At a coarse scale a 2-pixel
  erosion therefore removes more ground than at native scale; that is the same
  behaviour the GEE path has always had.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterator, Sequence

import numpy as np

from harmonizer.config import CONFIG
from harmonizer.overlap import Overlap
from harmonizer.registry.products import Coord

__all__ = [
    "Cell",
    "CellGrid",
    "ChunkedSampler",
    "grid_cells",
    "thin_by_spacing",
    "round_robin_truncate",
]

_LOG = logging.getLogger(__name__)

BBox = tuple[float, float, float, float]


# --------------------------------------------------------------------------- #
# The grid: the unit of I/O, memory, and stratification
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Cell:
    """One ``grid_cell_deg`` cell of the overlap, in EPSG:4326."""

    index: int
    bbox: BBox

    @property
    def area_deg2(self) -> float:
        return (self.bbox[2] - self.bbox[0]) * (self.bbox[3] - self.bbox[1])


@dataclass(frozen=True)
class CellGrid:
    n_cols: int
    n_rows: int
    cells: list[Cell]

    def __len__(self) -> int:
        return len(self.cells)


def grid_cells(bbox: BBox, cell_deg: float | None = None) -> CellGrid:
    """Divide a bbox into ``grid_cell_deg`` cells (the Stage 2 stratification grid).

    The last row/column is clipped to the bbox rather than overhanging it, so a
    cell's area is never counted outside the overlap.
    """
    cell_deg = float(cell_deg or CONFIG.sampling.grid_cell_deg)
    min_lon, min_lat, max_lon, max_lat = bbox
    n_cols = max(1, int(math.ceil((max_lon - min_lon) / cell_deg)))
    n_rows = max(1, int(math.ceil((max_lat - min_lat) / cell_deg)))

    cells: list[Cell] = []
    for row in range(n_rows):
        for col in range(n_cols):
            lo_lon = min_lon + col * cell_deg
            lo_lat = min_lat + row * cell_deg
            cells.append(
                Cell(
                    index=row * n_cols + col,
                    bbox=(
                        lo_lon,
                        lo_lat,
                        min(lo_lon + cell_deg, max_lon),
                        min(lo_lat + cell_deg, max_lat),
                    ),
                )
            )
    return CellGrid(n_cols=n_cols, n_rows=n_rows, cells=cells)


# --------------------------------------------------------------------------- #
# Declustering helpers (shared with, and faster than, sampling.py's versions)
# --------------------------------------------------------------------------- #


def thin_by_spacing(coords: Sequence[Coord], min_spacing_m: float) -> list[Coord]:
    """Greedily keep points at least ``min_spacing_m`` apart.

    Same greedy rule and same arrival order as ``sampling._thin_by_spacing`` (so
    the kept set is identical), but the neighbour test uses a ``cKDTree`` instead
    of comparing against every previously kept point. The O(n^2) loop is the
    reason large candidate pools were unusable: a full-overlap draw produces
    hundreds of thousands of candidates per class.

    Falls back to the linear scan when SciPy is unavailable, so behaviour never
    depends on the import succeeding.
    """
    if min_spacing_m <= 0 or len(coords) <= 1:
        return list(coords)

    try:
        from scipy.spatial import cKDTree
    except ImportError:  # pragma: no cover - SciPy is a hard dependency elsewhere
        from harmonizer.sampling import _thin_by_spacing

        return _thin_by_spacing(coords, min_spacing_m)

    pts = np.asarray(coords, dtype=float)
    lon = np.radians(pts[:, 0])
    lat = np.radians(pts[:, 1])
    # Equirectangular projection about the set's mean latitude, in metres -- the
    # same small-angle approximation sampling.py uses, and ample at ~100 m.
    r_earth = 6_371_000.0
    lat0 = float(np.mean(lat))
    xs = r_earth * lon * math.cos(lat0)
    ys = r_earth * lat
    xy = np.column_stack([xs, ys])

    tree = cKDTree(xy)
    # All pairs closer than the threshold, resolved greedily in arrival order.
    pairs = tree.query_pairs(r=float(min_spacing_m), output_type="ndarray")
    if pairs.size == 0:
        return list(coords)

    # Neighbour lists, so acceptance can be decided in one forward pass.
    neighbours: dict[int, list[int]] = {}
    for a, b in pairs:
        neighbours.setdefault(int(a), []).append(int(b))
        neighbours.setdefault(int(b), []).append(int(a))

    kept_flags = np.zeros(len(coords), dtype=bool)
    for i in range(len(coords)):
        if any(kept_flags[j] for j in neighbours.get(i, ())):
            continue
        kept_flags[i] = True
    return [coords[i] for i in np.nonzero(kept_flags)[0]]


def round_robin_truncate(
    per_cell: Sequence[Sequence[Coord]], target: int
) -> list[Coord]:
    """Take up to ``target`` points, cycling across cells rather than in list order.

    ``sampling.py`` truncates with ``cands[:target]`` *after* thinning, which
    takes points in arrival order -- for a large pool that means the first cells
    visited fill the quota and the rest of the overlap contributes nothing,
    partially undoing the per-cell spread the stratification just built. Cycling
    keeps the kept points spread across the whole region (DESIGN.md 3.2 step 4).
    """
    if target <= 0:
        return []
    out: list[Coord] = []
    cursors = [0] * len(per_cell)
    exhausted = False
    while len(out) < target and not exhausted:
        exhausted = True
        for c, points in enumerate(per_cell):
            if cursors[c] < len(points):
                out.append(points[cursors[c]])
                cursors[c] += 1
                exhausted = False
                if len(out) >= target:
                    break
    return out


# --------------------------------------------------------------------------- #
# Per-cell reading and masking
# --------------------------------------------------------------------------- #


def _class_codes(product_id: str) -> frozenset[int] | None:
    """The pixel values this product may sample, from its registry legend.

    ``None`` means "unknown -- accept anything but 0", which is the pre-existing
    behaviour and the right answer for a product with no legend yet.

    The sampler discovers classes from the **pixels** (``np.unique`` per cell),
    not from the legend, which is right for finding what is actually present but
    means a fill value looks exactly like a class. Copernicus LCM-10 is the case
    that exposed it: its published legend declares ``254 Unclassifiable`` ("no
    Sentinel-1/2 observations or observations of insufficient quality") and
    ``255 No Data`` ("pixels not processed"). Both occupy real pixels, so without
    this filter the pipeline would erode them, draw points from them, fetch 64-dim
    embeddings for them, fit a GMM to "places the producer could not classify",
    and emit crosswalk rows matching that non-class against real land cover in
    the other map -- confidently, and meaninglessly.

    So the rule is an **allowlist, not a blocklist**: sample a pixel value only
    if the product's registry legend names it as a class. The indexer has
    already dropped the rows the legend marks ``IsClass = FALSE``, so
    anything the raster contains but the legend does not list is either a fill
    value or an unmapped code that cannot be drawn or named -- not something to
    model either way. An allowlist also cannot be out-grown by a product whose
    codes exceed any assumed range.
    """
    try:
        from harmonizer.registry.legends import legend_classes

        classes = legend_classes(product_id)
    except Exception:
        classes = None
    if not classes:
        return None
    return frozenset(int(c.code) for c in classes)


@dataclass
class CellRead:
    """One cell's decimated label array plus what is needed to geolocate it."""

    cell: Cell
    data: np.ndarray             # 2-D, class codes at sample scale
    transform: object            # affine: (col, row) -> raster CRS
    raster_crs: object
    nodata: object

    @property
    def empty(self) -> bool:
        return self.data.size == 0


class _TransformerCache:
    """One pyproj ``Transformer`` per CRS pair, not one per call.

    ``local_sampling.pixel_to_lonlat`` built a fresh ``Transformer`` on every
    invocation; constructing one is expensive and it was happening once per class
    per cell.
    """

    def __init__(self) -> None:
        self._cache: dict[str, object] = {}

    def to_4326(self, raster_crs):
        key = str(raster_crs)
        if key in ("EPSG:4326", "epsg:4326"):
            return None  # already lon/lat; no transform needed
        cached = self._cache.get(key)
        if cached is None:
            from pyproj import Transformer

            cached = Transformer.from_crs(raster_crs, "EPSG:4326", always_xy=True)
            self._cache[key] = cached
        return cached


def _erode_square(mask: np.ndarray, radius_px: int) -> np.ndarray:
    """Binary erosion by a square structuring element of the given pixel radius.

    Mirrors ``buffering._min_over_square`` / ``local_sampling._erode_square``: a
    pixel survives only if every pixel within ``radius_px`` is also set, and
    pixels off the array edge count as unset (the GEE path's ``mask.unmask(0)``).

    Uses a **separable** pair of 1-D erosions rather than one dense (2r+1)^2
    element. Square erosion is separable, so this is mathematically identical and
    costs O(2r) per pixel instead of O(r^2) -- the dense element was something
    SciPy could not decompose on its own.
    """
    if radius_px <= 0:
        return mask
    from scipy.ndimage import binary_erosion

    size = 2 * int(radius_px) + 1
    horizontal = np.ones((1, size), dtype=bool)
    vertical = np.ones((size, 1), dtype=bool)
    out = binary_erosion(mask, structure=horizontal, border_value=False)
    return binary_erosion(out, structure=vertical, border_value=False)


# --------------------------------------------------------------------------- #
# The sampler
# --------------------------------------------------------------------------- #


@dataclass
class ClassTotals:
    """Per-class candidate counts accumulated across every cell.

    These feed the absent-vs-buffered-away rule exactly as the single-window
    counts used to, now summed over the whole region instead of one array.
    """

    pre_erode: int = 0
    post_erode: int = 0


class ChunkedSampler:
    """Streams a local product's overlap cell by cell, drawing candidates per class.

    One instance corresponds to one (product, overlap, scale) triple and may be
    reused across classes and across the relaxed-buffer second pass -- the cell
    reads are cached for the lifetime of the instance's ``scan`` call, so the
    relaxed pass costs no extra I/O for the cells it revisits.
    """

    def __init__(
        self,
        product_id: str,
        overlap: Overlap,
        *,
        sample_scale_m: float | None = None,
        progress: Callable[[float, str], None] | None = None,
    ) -> None:
        self.product_id = product_id
        self.overlap = overlap
        self.sample_scale_m = float(
            sample_scale_m if sample_scale_m is not None else CONFIG.sampling.sample_scale_m
        )
        self.grid = grid_cells(overlap.bbox)
        self._progress = progress
        self._transformers = _TransformerCache()
        self._source = None  # resolved lazily: (path, is_mosaic)
        self._native_m: float | None = None  # source pixel size, resolved lazily
        # Allowlist of samplable codes from the registry legend; None = unknown.
        self._class_codes = _class_codes(product_id)

    # -- source ------------------------------------------------------------ #

    def _resolve_source(self) -> tuple[str, bool]:
        """The file to read cells from: the converted COG tree, else the raw source.

        Prefers the same converted tree the viewer uses, because that is where the
        MODE overview pyramid lives -- the decimated reads below are only cheap,
        and only categorically correct, when they can be satisfied from it. An
        unconverted product still works (correctness first) but reads the raw
        source and is reported as degraded, matching the tile path's behaviour.
        """
        if self._source is not None:
            return self._source

        from harmonizer import local_tiles
        from harmonizer.registry.legends import spec as _product_spec

        source = local_tiles.cog_source(self.product_id)
        if source is not None:
            self._source = source
            return self._source

        spec = _product_spec(self.product_id)
        if spec is None or spec.access.method != "local_raster":
            raise KeyError(f"not a local-raster product: {self.product_id}")
        _LOG.warning(
            "%s: sampling from the raw source -- no converted COGs. Decimated "
            "reads cannot use a MODE overview pyramid, so this is slower and, if "
            "the source's own overviews are averaged, may return class codes that "
            "exist nowhere in the legend. Fix with: python tools/to_cog.py --only %s",
            self.product_id,
            self.product_id,
        )
        from harmonizer.footprints import _resolve

        self._source = (str(_resolve(spec.access.path)), False)
        return self._source

    # -- reading ----------------------------------------------------------- #

    def _read_cell(self, cell: Cell) -> CellRead | None:
        """One decimated windowed read of a cell, at the run's sample scale.

        ``out_shape`` is set to the cell's size in sample-scale pixels, which lets
        GDAL satisfy the read from the overview pyramid instead of reading native
        resolution and downsampling in the caller. That is what bounds both time
        and memory per cell.

        **Resampling is MODE via the pyramid, not nearest.** The overviews built
        by ``tools/to_cog.py`` are MODE-resampled, so a sample-scale pixel here is
        the modal class of its ground footprint. This differs from GEE's
        nearest-at-scale sampling. Mode is at least as defensible for label
        sampling -- it cannot invent a class that was not present in the footprint
        and it is less noisy than picking one arbitrary sub-pixel -- but it is a
        deliberate difference in method, not an implementation detail.
        """
        import rasterio
        from rasterio.windows import from_bounds

        path, is_mosaic = self._resolve_source()
        if is_mosaic:
            return self._read_cell_mosaic(cell)

        band = self._band()
        with rasterio.open(path) as ds:
            bounds = self._cell_bounds_in_crs(cell, ds.crs)
            if bounds is None:
                return None
            try:
                window = from_bounds(*bounds, transform=ds.transform)
            except Exception:
                return None
            window = window.round_offsets().round_lengths()
            try:
                window = window.intersection(
                    rasterio.windows.Window(0, 0, ds.width, ds.height)
                )
            except rasterio.errors.WindowError:
                # No intersection at all: the cell lies wholly outside this
                # raster. ``Window.intersection`` *raises* here rather than
                # returning a zero-size window, so this must be caught, not
                # tested for -- an uncaught WindowError aborts the whole scan
                # partway through, which is how a bbox overlap that reaches
                # beyond one product's real extent used to fail.
                #
                # This is the mechanism DESIGN.md 3.1 relies on: the run's
                # overlap is a rectangle, but the products inside it are not, so
                # cells over a no-data hole (sea, a missing tile, a corner
                # outside the footprint) must be an ordinary skip.
                return None
            if window.width <= 0 or window.height <= 0:
                return None  # degenerate sliver at the very edge

            out_h, out_w = self._out_shape(cell, window)
            if out_h <= 0 or out_w <= 0:
                return None
            data = ds.read(
                band,
                window=window,
                out_shape=(out_h, out_w),
                # Nearest at the *residual* step: the decimation itself is served
                # by the MODE pyramid, and nearest here avoids blending whatever
                # the pyramid returns. Never use an averaging resampler on codes.
                resampling=rasterio.enums.Resampling.nearest,
            )
            transform = ds.window_transform(window) * rasterio.Affine.scale(
                window.width / out_w, window.height / out_h
            )
            return CellRead(
                cell=cell,
                data=data,
                transform=transform,
                raster_crs=ds.crs,
                nodata=ds.nodata,
            )

    def _read_cell_mosaic(self, cell: Cell) -> CellRead | None:
        """A cell read from a MosaicJSON-indexed COG set.

        ``cogeo_mosaic`` merges the covering assets for a bbox; a cell with no
        covering asset is an ordinary hole and yields ``None``.
        """
        from rio_tiler.errors import NoAssetFoundError

        path, _ = self._resolve_source()
        out_h, out_w = self._out_shape_deg(cell)
        if out_h <= 0 or out_w <= 0:
            return None

        from harmonizer.local_tiles import open_mosaic

        try:
            # Not ``MosaicBackend(path)``: it dispatches on URL scheme and
            # mis-parses Windows paths. See local_tiles.open_mosaic.
            with open_mosaic(path) as mosaic:
                img, _assets = mosaic.part(
                    list(cell.bbox),
                    dst_crs="EPSG:4326",
                    height=out_h,
                    width=out_w,
                    resampling_method="nearest",
                )
        except NoAssetFoundError:
            return None
        except Exception as exc:
            _LOG.debug("%s: cell %d read failed (%s)", self.product_id, cell.index, exc)
            return None

        import rasterio

        data = img.data[0]
        min_lon, min_lat, max_lon, max_lat = cell.bbox
        transform = rasterio.transform.from_bounds(
            min_lon, min_lat, max_lon, max_lat, data.shape[1], data.shape[0]
        )
        # ImageData.mask is 255 where valid; express nodata as a masked array-free
        # sentinel by handing back the mask through ``nodata=None`` and zeroing
        # invalid pixels, which the class masks then simply never match.
        if img.mask is not None:
            data = np.where(img.mask == 0, 0, data)
        return CellRead(
            cell=cell,
            data=data,
            transform=transform,
            raster_crs="EPSG:4326",
            nodata=None,
        )

    def _band(self) -> int:
        """The band to read: 1 for a converted COG, else the registry's band."""
        from harmonizer import local_tiles

        indexes = local_tiles.band_indexes(self.product_id)
        return 1 if indexes is None else int(indexes[0])

    def _cell_bounds_in_crs(self, cell: Cell, crs) -> BBox | None:
        """A cell's bbox expressed in the raster's CRS."""
        if crs is None:
            return None
        if str(crs) in ("EPSG:4326", "epsg:4326"):
            return cell.bbox
        try:
            from rasterio.warp import transform_bounds

            return transform_bounds("EPSG:4326", crs, *cell.bbox, densify_pts=21)
        except Exception:
            return None

    def _out_shape(self, cell: Cell, window) -> tuple[int, int]:
        """Read size in pixels for a cell at the run's sample scale.

        Clamped to the window's own size, and that clamp is load-bearing rather
        than a mere memory guard: a sample scale at or below the raster's native
        resolution must read the native grid, not a resampled one of the same
        nominal ground size.

        The clamp is derived from the **decimation factor** rather than from
        ground metres alone, because the two disagree. hrlc30 is nominally "30 m"
        but its pixels are 0.00025 deg, i.e. 27.83 m at the equator: a 0.25 deg
        cell is exactly 1000x1000 native pixels, while ground-metres arithmetic
        asks for 928x918 (a degree of longitude is also short of 111320 m at
        latitude 8). Requesting the smaller grid silently decimates by ~0.85 in
        each axis, and since erosion is applied in *pixels*, it then removes
        proportionally more ground -- which showed up as post-erosion candidate
        counts at ~80% of the unchunked path's on an identical AOI.

        Computing the factor as ``sample_scale / native_scale`` and dividing the
        window by it makes a run at or below native scale read the native grid
        exactly, so it reproduces the unchunked counts, while a coarser run still
        decimates by the intended factor.

        **A product's real resolution is not its nominal one.** hrlc30 is
        catalogued as 30 m but its pixels are 27.83 m, so ``sample_scale_m=30``
        is genuinely a ~1.08x decimation and *correctly* yields ~85% of the
        native candidate count. Verified: at ``sample_scale_m`` equal to (or
        below) the true native 27.83 m, per-class pre/post-erode counts match the
        unchunked path exactly, class for class. Do not "fix" a small shortfall
        here by forcing the native grid -- check the raster's actual resolution
        first.
        """
        native_m = self._native_scale_m(window)
        if native_m is None or native_m <= 0:
            h, w = self._out_shape_deg(cell)
            return (
                max(1, min(h, int(window.height))),
                max(1, min(w, int(window.width))),
            )
        factor = max(1.0, self.sample_scale_m / native_m)
        return (
            max(1, int(round(int(window.height) / factor))),
            max(1, int(round(int(window.width) / factor))),
        )

    def _native_scale_m(self, window) -> float | None:
        """The source's own pixel size in metres, cached per sampler.

        For a geographic CRS the transform is in degrees, so it is converted at
        ~111320 m per degree of latitude -- the same conversion
        ``indexer._resolution_m`` applies when writing ``resolution_m`` into the
        registry.
        """
        if self._native_m is not None:
            return self._native_m
        import rasterio

        path, is_mosaic = self._resolve_source()
        if is_mosaic:
            return None  # mosaic reads are sized in degrees; see _out_shape_deg
        try:
            with rasterio.open(path) as ds:
                res = abs(ds.transform.a)
                if res <= 0:
                    return None
                geographic = bool(ds.crs and ds.crs.is_geographic)
                self._native_m = res * (111_320.0 if geographic else 1.0)
        except Exception:
            return None
        return self._native_m

    def _out_shape_deg(self, cell: Cell) -> tuple[int, int]:
        min_lon, min_lat, max_lon, max_lat = cell.bbox
        mid_lat = math.radians(0.5 * (min_lat + max_lat))
        m_per_deg_lat = 111_320.0
        m_per_deg_lon = 111_320.0 * max(0.05, math.cos(mid_lat))
        height_m = (max_lat - min_lat) * m_per_deg_lat
        width_m = (max_lon - min_lon) * m_per_deg_lon
        return (
            max(1, int(round(height_m / self.sample_scale_m))),
            max(1, int(round(width_m / self.sample_scale_m))),
        )

    # -- masking ----------------------------------------------------------- #

    def _class_mask(self, read: CellRead, class_value: int) -> np.ndarray:
        mask = read.data == class_value
        if read.nodata is not None:
            mask &= read.data != read.nodata
        return mask

    def _sampling_mask(
        self, read: CellRead, class_value: int, erode_pixels: int
    ) -> np.ndarray:
        """Eroded core AND homogeneous neighbourhood -- ``buffering.sampling_mask``.

        Identical rule to the unchunked path. Erosion happens per cell, so a
        class patch straddling a cell boundary is eroded from that boundary as
        well as from its true edge; at the 0.25 deg cell size this affects a
        vanishing fraction of candidates and never *adds* points, only omits a
        few near seams. Recorded here because it is a real (if tiny) difference
        from eroding one whole-region array.
        """
        base = self._class_mask(read, class_value)
        eroded = _erode_square(base, erode_pixels)
        radius = (int(CONFIG.buffering.homogeneous_window) - 1) // 2
        homogeneous = _erode_square(base, radius)
        return eroded & homogeneous

    def _to_lonlat(
        self, read: CellRead, rows: np.ndarray, cols: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Pixel centres -> (lon, lat), via one cached transformer per CRS."""
        xs, ys = read.transform * (cols + 0.5, rows + 0.5)
        transformer = self._transformers.to_4326(read.raster_crs)
        if transformer is None:
            return np.asarray(xs), np.asarray(ys)
        lons, lats = transformer.transform(xs, ys)
        return np.asarray(lons), np.asarray(lats)

    # -- the scan ---------------------------------------------------------- #

    def scan(
        self,
        class_values: Sequence[int] | None = None,
        erode_pixels: int | None = None,
        per_cell_quota: int | None = None,
    ) -> tuple[dict[int, ClassTotals], dict[int, list[list[Coord]]]]:
        """Walk every cell once, accumulating per-class counts and candidates.

        Returns ``(totals, candidates)`` where ``totals[class]`` carries the
        pre/post-erode candidate counts summed over the whole region (feeding the
        absent-vs-buffered-away rule) and ``candidates[class]`` is a list of
        per-cell coordinate lists, kept separate so the caller can truncate
        round-robin across cells rather than in arrival order.

        ``class_values`` restricts the scan to specific classes (used by the
        relaxed-buffer second pass, which only revisits starved classes).
        """
        erode_pixels = (
            CONFIG.buffering.erode_pixels if erode_pixels is None else int(erode_pixels)
        )
        target = CONFIG.sampling.points_target
        n_cells = max(1, len(self.grid))
        # Draw a little more than the even split per cell: classes are patchy, so
        # many cells contribute nothing and the surplus lets richer cells cover
        # the target. Truncation back to `target` happens round-robin afterwards.
        quota = per_cell_quota or max(1, math.ceil(2.0 * target / n_cells))

        totals: dict[int, ClassTotals] = {}
        candidates: dict[int, list[list[Coord]]] = {}
        rng = np.random.default_rng(CONFIG.gmm.random_seed)

        wanted = set(int(c) for c in class_values) if class_values is not None else None

        for i, cell in enumerate(self.grid.cells):
            if self._progress is not None:
                self._progress(
                    (i + 1) / len(self.grid),
                    f"scanning cell {i + 1}/{len(self.grid)}",
                )
            read = self._read_cell(cell)
            if read is None or read.empty:
                continue  # no data here: a hole, or outside this product

            present = np.unique(read.data)
            if read.nodata is not None:
                present = present[present != read.nodata]
            for cv in present:
                cv = int(cv)
                # 0 is fill/no-label in every product here; beyond that, only
                # codes the legend names as classes are sampled (see
                # _class_codes -- this is what keeps 'Unclassifiable' and
                # 'No Data' out of the crosswalk).
                if cv == 0:
                    continue
                if self._class_codes is not None and cv not in self._class_codes:
                    continue
                if wanted is not None and cv not in wanted:
                    continue

                totals.setdefault(cv, ClassTotals())
                candidates.setdefault(cv, [])

                pre = self._sampling_mask(read, cv, erode_pixels=0)
                post = self._sampling_mask(read, cv, erode_pixels=erode_pixels)
                totals[cv].pre_erode += int(pre.sum())
                totals[cv].post_erode += int(post.sum())

                rows, cols = np.nonzero(post)
                if rows.size == 0:
                    continue
                # Convert ONLY the drawn pixels to lon/lat. The old path converted
                # every masked pixel before subsampling, which for a large mask
                # was almost all of the cost.
                if rows.size > quota:
                    pick = rng.choice(rows.size, size=quota, replace=False)
                    rows, cols = rows[pick], cols[pick]
                lons, lats = self._to_lonlat(read, rows, cols)
                candidates[cv].append(
                    [(float(a), float(b)) for a, b in zip(lons, lats)]
                )

        return totals, candidates


def draw_for_class(
    per_cell: Sequence[Sequence[Coord]],
    *,
    min_spacing_m: float | None = None,
    target: int | None = None,
) -> list[Coord]:
    """Thin per-cell candidates to the spacing rule and truncate round-robin.

    The order matters and matches the existing path: thin first (so the spacing
    rule sees every candidate), then truncate to the target. Truncation is
    round-robin across cells rather than by arrival order, which is the fix for
    a large pool silently collapsing onto the first cells scanned.
    """
    min_spacing_m = (
        CONFIG.sampling.min_spacing_m if min_spacing_m is None else min_spacing_m
    )
    target = CONFIG.sampling.points_target if target is None else target

    thinned: list[list[Coord]] = []
    for cell_points in per_cell:
        thinned.append(thin_by_spacing(cell_points, min_spacing_m))
    return round_robin_truncate(thinned, target)
