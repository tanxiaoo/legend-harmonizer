"""Label-map tiles for local-raster products (Stage 5.2 extension).

Mirrors ``harmonizer.tiles``' public shape (``legend`` / a tile-producing call)
but for products whose ``access.method`` is ``local_raster`` instead of ``gee``:
Stage 1's :class:`~harmonizer.registry.adapters.hrlc_local.LocalRasterAdapter`
only reads single pixels, so it cannot serve tiles -- this module reads and
renders 2-D tile windows straight from the GeoTIFF with ``rio-tiler``, which
handles reprojecting the raster's native CRS into Web Mercator tile space and
picking the right overview level for a given zoom.

**Nearest-neighbour only.** The class codes are categorical (a pixel's value is
a land-cover code, not a continuous quantity), so every read here forces
``resampling_method="nearest"``. Any other resampling (bilinear, average, the
default used to build this file's own overview pyramid) blends adjacent codes
into values that were never really painted anywhere -- confirmed by comparing a
downsampled overview read (which produced spurious in-between codes such as 61-69
between the real classes 60 and 70) against a native-resolution window read
(which only ever produces the declared legend codes). Never relax this.

Colours come from the product's legend (the per-map YAML, docs/PIPELINE.md
section 2.5), exactly like ``tiles.py`` -- there is one legend per product
regardless of which access method reads it.
"""

from __future__ import annotations

import functools
import hashlib
import logging
import os
import threading
from collections import OrderedDict
from pathlib import Path
from typing import Sequence

from rio_tiler.errors import NoAssetFoundError, TileOutsideBounds

from harmonizer.config import CONFIG
from harmonizer.registry.legends import legend_classes, spec as _product_spec
from harmonizer.tiles import LegendEntry

__all__ = [
    "legend",
    "tile_png",
    "code_tile_png",
    "clear_tile_cache",
    "band_indexes",
    "max_native_zoom",
    "cog_source",
    "open_mosaic",
    "invalidate_cog_source",
    "source_state",
    "TileOutsideBounds",
]

_LOG = logging.getLogger(__name__)

# Cap GDAL's own block cache (shared across every open dataset in this
# process) rather than letting it grow unbounded -- GDAL's default is either
# 5% of physical RAM or an env-configured value, and on a shared host with a
# per-session memory cgroup that default has nothing to do with what's actually
# available, so left alone it can OOM-kill the whole server as more large
# rasters get opened in a browsing session. Set once at import, before any
# dataset is opened, so it takes effect; harmless if GDAL_CACHEMAX is already
# set in the environment (os.environ.setdefault leaves it untouched).
os.environ.setdefault("GDAL_CACHEMAX", "512")  # MB

# rio_tiler.io.Reader instances, opened lazily and kept open across requests --
# the source files are large (GB-scale) so reopening per tile would be
# wasteful. Bounded LRU: only _MAX_OPEN_READERS stay open at once (each open
# VRT/GeoTIFF holds its own GDAL dataset handle and metadata, which adds up
# across a session that browses many products), so the oldest is closed and
# evicted rather than accumulating forever.
_MAX_OPEN_READERS = 4

# Per-thread reader caches. Deliberately *not* one shared dict: see _reader().
_THREAD_LOCAL = threading.local()


def _thread_readers() -> "OrderedDict[str, tuple]":
    cache = getattr(_THREAD_LOCAL, "readers", None)
    if cache is None:
        cache = OrderedDict()
        _THREAD_LOCAL.readers = cache
    return cache

# Bounds how many tile reads/renders run at once across the whole process.
# FastAPI's sync endpoints each run in a threadpool (up to ~40 threads by
# default), and a browser panning/zooming quickly (or several map sides
# refreshing at once) can fire many concurrent tile requests.
#
# There are two separate limits because the two data sources have different
# risks:
#
# * ``_SOURCE_SEMAPHORE`` guards reads of the **original** rasters -- the
#   registry's ``access.path``, used only until a product has been converted.
#   Its size is ``CONFIG.tiles.source_concurrency``. The historical value was 1
#   (fully serialised) because those rasters lived under data/Africa/2019/,
#   mounted from another user's Lustre volume via the data/Africa symlink:
#   concurrent reads there twice crashed the whole process with a
#   SIGFPE/segfault inside libtiff -- confirmed via dmesg and correlated with a
#   Lustre "Flock LR mismatch" logged immediately before each crash. That points
#   at a filesystem-level read reliability problem under concurrent access, not
#   corrupted files (a full block-by-block scan of one flagged file found
#   nothing). Serialising does not fix the Lustre issue but removes concurrent
#   access as a trigger. The rationale is real but environment-specific, so it is
#   now a setting: **a deployment whose data/ is on a network filesystem must set
#   ``HARMONIZER_SOURCE_CONCURRENCY=1``.**
#
# * ``_COG_SEMAPHORE`` guards reads of the converted COG tree (tools/to_cog.py).
#   Those reads are ~1000x cheaper now that the files carry overview pyramids,
#   so allowing a few in flight is what keeps two map panes refreshing at once
#   from queueing behind each other.
#
# Concurrency here is only safe because each thread opens its **own** GDAL
# handle (see ``_reader``). An earlier version raised this limit while still
# sharing one Reader across threads, which produced intermittent
# ``RasterioIOError: Read failed`` and ``free(): invalid next size`` heap
# corruption that killed the process. Raise these numbers only if per-thread
# handles remain in place.
_SOURCE_SEMAPHORE = threading.Semaphore(max(1, CONFIG.tiles.source_concurrency))
_COG_SEMAPHORE = threading.Semaphore(max(1, CONFIG.tiles.cog_concurrency))

#: Products already warned about serving from the raw source, so the degraded-mode
#: warning is logged once per product instead of once per tile request.
_WARNED_DEGRADED: set[str] = set()


@functools.lru_cache(maxsize=None)
def cog_source(product_id: str) -> tuple[str, bool] | None:
    """The converted COG or MosaicJSON for a product, if one has been built.

    Returns ``(path, is_mosaic)``, or ``None`` when ``tools/to_cog.py`` has not
    been run for this product yet -- in which case the caller falls back to the
    original (much slower) source path from the registry.

    **Cached**, because this is called several times per tile request (band
    selection, semaphore choice, cache key, ETag) and each uncached call costs an
    ``exists()`` plus a directory ``glob``. The result only changes when a
    conversion job finishes, which must call :func:`invalidate_cog_source`.
    """
    return _probe_cog_source(product_id)[0]


def _probe_cog_source(product_id: str) -> tuple[tuple[str, bool] | None, str]:
    """Uncached ``(source, state)`` for a product's converted COG tree.

    ``state`` is one of:

    ``mosaic``
        A MosaicJSON over per-tile COGs -- the multi-file fast path.
    ``single``
        One converted COG -- the single-file fast path.
    ``needs-conversion``
        Nothing converted yet, **or** the pathological case this check exists
        for: >=2 COGs present but no ``mosaic.json``, which is a conversion that
        died between writing the COGs and building the index. Falling back to the
        raw source there would be silently slow forever, so it is reported.
    """
    base = CONFIG.cog_dir / product_id
    mosaic = base / "mosaic.json"
    if mosaic.exists():
        return (str(mosaic), True), "mosaic"
    if base.is_dir():
        cogs = sorted(base.glob("*.tif"))
        if len(cogs) == 1:
            return (str(cogs[0]), False), "single"
        if len(cogs) >= 2:
            # Converted tiles with no index: the mosaic build never finished.
            # Loud, because the symptom is otherwise indistinguishable from
            # "never converted" -- a map that is mysteriously slow.
            _LOG.error(
                "%s: %d COGs in %s but no mosaic.json -- the conversion did not "
                "finish. Tiles will fall back to the slow raw source. Fix with: "
                "python tools/to_cog.py --only %s",
                product_id,
                len(cogs),
                base,
                product_id,
            )
    return None, "needs-conversion"


def source_state(product_id: str) -> str:
    """``mosaic`` / ``single`` / ``needs-conversion`` for a product.

    The serving-side half of the product ready-state: a ``needs-conversion``
    product is served, but from the raw source and orders of magnitude slower.
    """
    return _probe_cog_source(product_id)[1]


def invalidate_cog_source(product_id: str | None = None) -> None:
    """Forget cached COG lookups after a conversion job writes new files.

    Without a ``product_id`` the whole cache is dropped. ``lru_cache`` has no
    per-key eviction, so a single product is invalidated by clearing everything;
    the cache refills on the next few tile requests at one directory scan each.
    """
    cog_source.cache_clear()
    if product_id is None:
        _WARNED_DEGRADED.clear()
    else:
        _WARNED_DEGRADED.discard(product_id)


def _warn_degraded_once(product_id: str) -> None:
    """Log, once per product, that tiles are coming from the unconverted source.

    Serving still works -- it is just 100-1000x slower per tile (see the
    benchmark table in ``tools/to_cog.py``). Silence here is what let this
    deployment run for weeks entirely on the slow path.
    """
    if product_id in _WARNED_DEGRADED:
        return
    _WARNED_DEGRADED.add(product_id)
    _LOG.warning(
        "%s: serving tiles from the raw source -- no converted COGs in %s. This "
        "is the slow degraded path (a zoomed-out tile reads the full-resolution "
        "raster). Fix with: python tools/to_cog.py --only %s",
        product_id,
        CONFIG.cog_dir / product_id,
        product_id,
    )


#: Backwards-compatible alias. Internal callers use the cached ``cog_source``.
_cog_source = cog_source


def band_indexes(product_id: str) -> tuple[int, ...] | None:
    """The 1-based band to render for a product, as rio-tiler ``indexes``.

    Most local rasters hold a single label band and this returns ``None`` (let
    rio-tiler use its default). Multi-band products -- an annual time series
    such as GLC_FCS30D, where each band is one year -- declare which band is the
    label band via ``band:`` in their registry YAML, and reading any other one
    silently renders the wrong year.

    Returns ``None`` for a **converted COG**, even when the registry names a
    band: ``tools/to_cog.py`` extracts that band at conversion time, so the COG
    holds one band and asking for band 20 of a 1-band file is an error. The
    registry's ``band:`` keeps pointing at the source band, and this is the one
    place that difference is resolved.

    A non-integer ``band`` (GEE products name their bands) is not meaningful for
    a local GeoTIFF, so it is ignored rather than guessed at.
    """
    s = _product_spec(product_id)
    if s is None or s.band is None:
        return None
    if cog_source(product_id) is not None:
        return None
    try:
        return (int(s.band),)
    except (TypeError, ValueError):
        return None


def open_mosaic(path: str | Path):
    """Open a local MosaicJSON, bypassing ``MosaicBackend``'s URL dispatch.

    ``cogeo_mosaic.backends.MosaicBackend`` chooses a backend by URL scheme, and
    a Windows path is not a URL: ``D:\\cache\\cog\\x\\mosaic.json`` parses as
    scheme ``"d"``, so it raises ``ValueError: 'd' is not supported``. The
    obvious workaround -- passing a ``file://`` URL -- fails differently: the
    dispatcher hands ``parsed.path`` to ``FileBackend`` **without URL-decoding
    it**, producing ``/D:/.../Land%20Cover/...`` and ``[Errno 22] Invalid
    argument`` on any path containing a space (this repo's does).

    Constructing ``FileBackend`` directly is the one form that is correct on
    every platform: it is exactly what the dispatcher does for a scheme-less
    input, minus the parsing that mangles Windows paths. Kept as a named
    function because both the tile path and the chunked sampler need it, and a
    second copy would drift.
    """
    from cogeo_mosaic.backends.file import FileBackend

    return FileBackend(str(Path(path)))


def _open_reader(product_id: str):
    """Open the best available reader for a product, preferring converted COGs.

    Order of preference:

    1. A MosaicJSON over per-tile COGs (multi-file products).
    2. A single converted COG (single-file products).
    3. The registry's original ``access.path`` -- a VRT with no overview
       pyramid, which is orders of magnitude slower and is only used until
       ``tools/to_cog.py`` has been run.
    """
    import rasterio
    from rio_tiler.io import Reader

    source = cog_source(product_id)
    if source is not None:
        path, is_mosaic = source
        if is_mosaic:
            return open_mosaic(path), True
        return Reader(path), False

    s = _product_spec(product_id)
    if s is None or s.access.method != "local_raster":
        raise KeyError(f"no local-raster tile support for product: {product_id}")
    _warn_degraded_once(product_id)
    try:
        return Reader(s.access.path), False
    except rasterio.errors.RasterioIOError as exc:
        # access.path doesn't point at an openable file yet (e.g. a placeholder
        # entry whose GeoTIFF hasn't been dropped in) -- same "not tileable"
        # contract as an unrecognised product id.
        raise KeyError(
            f"no local-raster tile support for product: {product_id} ({exc})"
        ) from exc


def _reader(product_id: str):
    """A cached ``(reader, is_mosaic)`` pair for the product.

    The cache is **thread-local**. A ``rio_tiler`` Reader wraps a single GDAL
    dataset handle, and GDAL dataset handles are not thread-safe: sharing one
    across the FastAPI threadpool produced intermittent
    ``RasterioIOError: Read failed`` and, worse, ``free(): invalid next size``
    heap corruption that killed the whole process. Each thread therefore opens
    its own handle. Open files cost little next to a crash, and the LRU bound
    still applies per thread.
    """
    cache = _thread_readers()
    entry = cache.get(product_id)
    if entry is not None:
        cache.move_to_end(product_id)
        return entry

    s = _product_spec(product_id)
    if s is None or s.access.method != "local_raster":
        raise KeyError(f"no local-raster tile support for product: {product_id}")

    entry = _open_reader(product_id)
    cache[product_id] = entry
    if len(cache) > _MAX_OPEN_READERS:
        _evicted_id, (evicted, _) = cache.popitem(last=False)
        try:
            evicted.close()
        except Exception:  # pragma: no cover - best-effort cleanup
            pass
    return entry


def max_native_zoom(product_id: str) -> int | None:
    """The finest web-mercator zoom the product actually has pixels for.

    Above this the raster holds no extra detail, so the browser should upscale
    the tiles it already has instead of asking the server for finer ones it
    would have to fabricate. Derived from the data's own ground resolution
    rather than a hardcoded constant, because these products range from 10 m to
    ~30 m.

    Returns ``None`` if it cannot be determined, in which case the caller should
    not constrain zoom.
    """
    import math

    try:
        reader, is_mosaic = _reader(product_id)

        if is_mosaic:
            # A MosaicJSON records the zoom range its tiles were indexed for.
            value = getattr(reader.mosaic_def, "maxzoom", None)
            return int(value) if value is not None else None

        # Ground resolution in metres at the equator, converted to the zoom
        # whose 256 px tiles are about that fine.
        dataset = reader.dataset
        res_x = abs(dataset.transform.a)
        if res_x <= 0:
            return None
        crs = dataset.crs
        res_m = res_x * 111_320.0 if (crs and crs.is_geographic) else res_x
        # 156543.03 m/px is zoom 0 at the equator for 256 px tiles.
        return max(0, min(22, int(round(math.log2(156_543.03 / res_m)))))
    except Exception:
        # Never let a metadata probe break tile serving -- the frontend simply
        # does not constrain zoom when this is unknown.
        return None


def legend(product_id: str) -> list[LegendEntry]:
    """The product's class legend: value, readable name, and palette colour.

    Same contract as ``tiles.legend()``, read from the same per-map YAML legend
    -- a product's legend does not depend on how it happens to be read.
    """
    s = _product_spec(product_id)
    if s is None or s.access.method != "local_raster":
        raise KeyError(f"no local-raster tile support for product: {product_id}")
    classes = legend_classes(product_id)
    if not classes:
        raise KeyError(f"no local-raster tile support for product: {product_id}")
    return [
        LegendEntry(
            value=c.code,
            name=c.name,
            color=c.color.lstrip("#"),
            description=c.description,
            observed=c.observed,
        )
        for c in classes
    ]


def _read_tile(product_id: str, z: int, x: int, y: int):
    """The raw ``ImageData`` for one tile: class codes, nearest-resampled.

    Shared by both renderers below. Nearest resampling is not negotiable here --
    see this module's docstring.
    """
    reader, is_mosaic = _reader(product_id)
    indexes = band_indexes(product_id)

    # Reads of the converted COG tree may overlap; reads of the original Lustre
    # sources must not (see the semaphore definitions above).
    using_cogs = cog_source(product_id) is not None
    guard = _COG_SEMAPHORE if using_cogs else _SOURCE_SEMAPHORE

    with guard:
        if is_mosaic:
            # MosaicBackend picks the COGs covering this tile and merges them;
            # the resampling kwarg is forwarded to each per-asset read, so
            # nearest-neighbour still applies to the class codes.
            try:
                img, _assets = reader.tile(
                    x, y, z, resampling_method="nearest", indexes=indexes
                )
            except NoAssetFoundError as exc:
                # No COG covers this tile. That is the mosaic equivalent of a
                # single reader's TileOutsideBounds -- an ordinary "nothing
                # here", not an error: Leaflet requests every tile in the
                # viewport regardless of the product's actual extent. Raised as
                # TileOutsideBounds so the API answers 404 like the single-COG
                # path; a 500 here made Leaflet treat the layer as broken and
                # leave the pane blank.
                raise TileOutsideBounds(str(exc)) from exc
        else:
            img = reader.tile(x, y, z, resampling_method="nearest", indexes=indexes)
    return img


def code_tile_png(product_id: str, z: int, x: int, y: int) -> bytes:
    """A tile encoding the raw **class code** of each pixel, for client rendering.

    This is the tile the frontend actually uses (DESIGN.md section 2.3). Unlike
    :func:`tile_png` it applies no palette and no class subset, so **one tile per
    position serves every toggle state**: switching classes on and off becomes a
    canvas pass in the browser over tiles it already holds, with no request and
    no server work. It also means the disk cache, the browser cache and the ETag
    all key on (product, z/x/y, band) alone.

    Encoding: 8-bit greyscale + alpha (PNG colour type 4).

    * The grey channel is the class code verbatim -- 0-255 covers every code in
      every registry legend (the widest is GLC_FCS30D at 220), so the 16-bit
      variant DESIGN.md allows for is not needed. **No arithmetic is applied**:
      the browser looks the value up in the legend palette, so any scaling here
      would have to be undone there, and a mismatch would silently mislabel land
      cover.
    * The alpha channel is 0 where the tile has no data and 255 elsewhere. It is
      a separate channel rather than a reserved code value because 255 is itself
      a legitimate class code in two products (``gwl_fcs30d``, ``wsf``), so no
      sentinel in the grey channel is safe.

    PNG is lossless and the browser must read exact values back, so the encoder
    is constrained accordingly -- see :func:`_encode_code_png`.
    """
    cached = _cache_get(product_id, z, x, y, None, kind="code")
    if cached is not None:
        return cached

    legend(product_id)  # validates the product (raises KeyError) before reading
    img = _read_tile(product_id, z, x, y)
    png = _encode_code_png(img)
    _cache_put(product_id, z, x, y, None, png, kind="code")
    return png


def _encode_code_png(img) -> bytes:
    """Encode ``ImageData`` as an 8-bit greyscale+alpha PNG of raw class codes.

    Written against ``rio_tiler``'s ``ImageData`` rather than its ``render()``
    because ``render`` is built around applying a colormap; here the whole point
    is to ship the codes unmodified.
    """
    import numpy as np
    from rio_tiler.utils import render as rio_render

    codes = img.data[0]
    if codes.dtype != np.uint8:
        # Every local product's codes fit in a byte (checked against all registry
        # legends). A wider dtype means values that cannot round-trip through
        # this encoding, so fail loudly instead of silently truncating a class
        # code into a different, real class.
        over = codes[codes > 255]
        if over.size:
            raise ValueError(
                f"class code {int(over.max())} exceeds 255 and cannot be encoded "
                f"in an 8-bit code tile; this product needs the 16-bit variant"
            )
        codes = codes.astype(np.uint8)

    # ImageData.mask is 255 where the pixel is valid, 0 where it is nodata --
    # exactly the alpha channel we want, so it is used as-is.
    alpha = img.mask
    if alpha is None:
        alpha = np.full(codes.shape, 255, dtype=np.uint8)
    alpha = alpha.astype(np.uint8)

    return rio_render(
        np.stack([codes]), mask=alpha, img_format="PNG", colormap=None
    )


def tile_png(
    product_id: str,
    z: int,
    x: int,
    y: int,
    visible_values: Sequence[int] | None = None,
) -> bytes:
    """A server-coloured RGBA PNG tile for the product's label map.

    ``visible_values`` optionally restricts rendering to a subset of class values
    (the per-class show/hide toggles): pixels of any other class are made
    transparent so only the requested classes are painted, letting the base map
    show through underneath -- the same behaviour ``tiles.tile_template`` gives
    GEE-backed products.

    **The frontend no longer uses this path** -- it fetches :func:`code_tile_png`
    and applies the palette itself, so a toggle costs no request (DESIGN.md 2.3).
    This is kept as the documented fallback in that section (and for any
    non-browser client that wants a ready-coloured tile): every toggle state is a
    separate render and a separate cache entry, which is exactly the cost the
    code-tile path removes.
    """
    cached = _cache_get(product_id, z, x, y, visible_values)
    if cached is not None:
        return cached

    entries = legend(product_id)
    img = _read_tile(product_id, z, x, y)

    colormap = {e.value: _hex_to_rgba(e.color) for e in entries}
    if visible_values is not None:
        keep = set(int(v) for v in visible_values)
        colormap = {v: rgba for v, rgba in colormap.items() if v in keep}
        # Values present in the raster but not selected: fully transparent
        # rather than absent, so render() doesn't fall back to a default
        # colour for them.
        for e in entries:
            if e.value not in colormap:
                colormap[e.value] = (0, 0, 0, 0)

    png = img.render(img_format="PNG", colormap=colormap)
    _cache_put(product_id, z, x, y, visible_values, png)
    return png


def _cache_key(
    product_id: str,
    z: int,
    x: int,
    y: int,
    visible: Sequence[int] | None,
    kind: str = "rgba",
) -> str:
    """A filename-safe key identifying one rendered tile.

    ``kind`` separates the two renderings of the same tile: ``code`` (raw class
    codes, what the frontend fetches) and ``rgba`` (server-coloured, the
    fallback). They are different bytes for the same address, so they must not
    collide.

    For ``rgba`` the visible-class subset is part of the key: toggling classes
    changes the colormap and therefore the rendered PNG. The subset is hashed
    rather than spelled out because it can be long enough to overflow a
    filename. **A ``code`` tile has no such component** -- that is the whole
    point of it, one cache entry shared by every toggle state.
    """
    if kind == "code":
        classes = "code"
    elif visible is None:
        classes = "all"
    else:
        joined = ",".join(str(int(v)) for v in sorted(set(visible)))
        classes = hashlib.sha1(joined.encode()).hexdigest()[:12] if joined else "none"
    # The band is part of the key: for a multi-band annual series, changing
    # ``band:`` in the registry changes what the tile shows, and without this a
    # cached PNG of the old year would be served forever for the new one.
    indexes = band_indexes(product_id)
    band = "d" if indexes is None else str(indexes[0])
    return f"{z}_{x}_{y}_b{band}_{classes}.png"


def _cache_path(
    product_id: str,
    z: int,
    x: int,
    y: int,
    visible: Sequence[int] | None,
    kind: str = "rgba",
) -> Path:
    return (
        CONFIG.cog_dir
        / "_tilecache"
        / product_id
        / _cache_key(product_id, z, x, y, visible, kind)
    )


def _cache_get(
    product_id: str,
    z: int,
    x: int,
    y: int,
    visible: Sequence[int] | None,
    kind: str = "rgba",
) -> bytes | None:
    """A previously rendered PNG for this tile, or ``None``."""
    try:
        return _cache_path(product_id, z, x, y, visible, kind).read_bytes()
    except OSError:
        return None


def _cache_put(
    product_id: str,
    z: int,
    x: int,
    y: int,
    visible: Sequence[int] | None,
    png: bytes,
    kind: str = "rgba",
) -> None:
    """Store a rendered PNG, best-effort.

    Written to a unique temporary name and renamed into place so a concurrent
    reader never observes a partially written PNG. Cache failures (a full or
    read-only volume) must never break tile serving, so every error is swallowed.
    """
    path = _cache_path(product_id, z, x, y, visible, kind)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(f".{os.getpid()}.{threading.get_ident()}.tmp")
        tmp.write_bytes(png)
        tmp.replace(path)
    except OSError:
        pass


def clear_tile_cache(product_id: str | None = None) -> int:
    """Drop cached PNGs, for when a product's legend or source data changes.

    Returns the number of files removed. Without a ``product_id`` the whole
    cache is cleared.
    """
    root = CONFIG.cog_dir / "_tilecache"
    target = root / product_id if product_id else root
    removed = 0
    if not target.is_dir():
        return 0
    for png in target.rglob("*.png"):
        try:
            png.unlink()
            removed += 1
        except OSError:
            pass
    return removed


def _hex_to_rgba(color: str) -> tuple[int, int, int, int]:
    """``#RRGGBB`` (or ``RRGGBB``) -> an opaque (r, g, b, 255) tuple."""
    h = color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return (r, g, b, 255)
