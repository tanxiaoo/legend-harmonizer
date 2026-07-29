"""Convert local-raster products to Cloud-Optimized GeoTIFFs (+ MosaicJSON).

Why this exists
---------------
The local-raster products are served as map tiles by ``harmonizer.local_tiles``,
which reads them through GDAL VRTs in ``cache/vrt/``. A VRT is only an XML index
over the source tiles -- it stores no pixel data and, crucially, **no overview
pyramid**. Without overviews a zoomed-out tile request forces GDAL to read the
*entire* raster at native resolution and downsample it on the fly. Measured on
this data, at zoom 4:

    HRLC30 (single GeoTIFF, overviews [2..256])   0.01 s
    worldcover_local.vrt (overviews [2..32])      0.40 s
    gsw_yearly.vrt       (no overviews)           7.8  s
    gwl_fcs30d.vrt       (no overviews)          24    s
    fnf4.vrt             (no overviews)          25    s
    dynamicworld_local.vrt (no overviews)        48    s
    gfc.vrt, from_glc.vrt (no overviews)         >60 s -- gfc was OOM-killed

``gfc`` is 360000x600000 px across 16 files, so one zoom-4 tile touches roughly
216 gigapixels. That is the entire cause of the slow map, and this script is the
fix: give every product a proper overview pyramid so a zoomed-out tile reads a
small pre-computed image instead of the whole raster.

Why per-tile COGs and not one big COG
-------------------------------------
Merging a product's source tiles into a single COG would mean rewriting the full
mosaic. A previous attempt to build overviews for ``worldcover_local`` this way
produced a **41 GB** uncompressed ``.ovr`` sidecar for one product. Instead each
source tile is converted independently (cheap, parallel, restartable) and the set
is indexed by a MosaicJSON that ``local_tiles`` reads through
``cogeo_mosaic``. Each tile carries its own overviews, so a zoomed-out read
touches only the coarse levels of the few tiles it intersects.

Categorical-data safety
-----------------------
These rasters are land-cover **class codes**, not continuous measurements. The
value 35 sitting between classes 30 and 40 is a different land cover, not "a bit
more than 30". So:

* Overviews are built with ``MODE`` (most common value in the block) -- never
  ``AVERAGE``/``BILINEAR``/``CUBIC``, which invent codes that were never painted
  anywhere. This mirrors the nearest-neighbour rule already enforced at read time
  in ``harmonizer/local_tiles.py``.
* ``nodata`` is written explicitly. Several source products declare none, which
  makes 0 render as a real class; where the registry knows the fill value it is
  stamped onto the output.

Safety / idempotence
--------------------
* Files under ``data/`` are **never** modified -- output goes to a separate tree
  (``HARMONIZER_COG_DIR``, see ``harmonizer.config``) that mirrors the source
  layout, so every output maps back to its source path.
* Already-valid COGs are skipped unless the source is newer or ``--force`` is
  given, so re-running is cheap and safe.
* Each tile is written to a temporary name and renamed only on success, so an
  interrupted run never leaves a half-written COG that a later run would skip.

Run::

    python tools/to_cog.py --list
    python tools/to_cog.py --only gfc
    python tools/to_cog.py --workers 8
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import sys
import time
from pathlib import Path

# Keep GDAL's per-worker block cache modest: this script may run many workers in
# parallel and the default (a share of physical RAM) is unaware of the cgroup
# memory limit on a shared host. Set before rasterio/GDAL is imported.
os.environ.setdefault("GDAL_CACHEMAX", "256")  # MB

import rasterio  # noqa: E402
from rasterio.enums import Resampling  # noqa: E402
from rio_cogeo.cogeo import cog_validate  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from harmonizer.config import CONFIG  # noqa: E402
from harmonizer.registry.legends import spec as _product_spec  # noqa: E402

__all__ = ["convert_product", "source_tiles", "cog_path_for", "build_mosaic"]

# Internal tile size of the output COGs. 512 matches the one already-fast file
# (HRLC30) and is 16x fewer block reads per map tile than the VRTs' 128x128,
# which matters over a network filesystem.
_BLOCKSIZE = 512


def _log(msg: str) -> None:
    print(msg, flush=True)


def source_tiles(product_id: str) -> list[Path]:
    """The real GeoTIFFs behind a product, resolving a VRT to its members.

    A product's ``access.path`` is either a single GeoTIFF or a VRT indexing many
    source tiles. ``rasterio``'s ``files`` reports the VRT itself alongside its
    members, so the VRT path is filtered out and only true rasters are returned.
    """
    s = _product_spec(product_id)
    if s is None or s.access.method != "local_raster":
        raise KeyError(f"not a local-raster product: {product_id}")

    path = Path(s.access.path)
    if not path.is_absolute():
        path = CONFIG.cache_dir.parent / path
    if not path.exists():
        raise FileNotFoundError(f"{product_id}: missing source at {path}")

    if path.suffix.lower() != ".vrt":
        return [path]

    with rasterio.open(path) as ds:
        members = [Path(f) for f in ds.files if Path(f).resolve() != path.resolve()]
    # A VRT can also list sidecars (.ovr/.aux.xml); keep only openable rasters.
    return [m for m in members if m.suffix.lower() in (".tif", ".tiff")]


def cog_path_for(product_id: str, src: Path) -> Path:
    """Where a source tile's COG lives, mirroring the source layout.

    The output tree is keyed by product id and then by the source file's own
    name, so every COG maps back unambiguously to the file it came from.
    """
    return CONFIG.cog_dir / product_id / (src.stem + ".tif")


def _nodata_for(product_id: str, src_nodata):
    """The nodata value to stamp on the output.

    Prefers whatever the source declares. Where the source declares none, the
    registry's fill value is used if it has one -- an explicit nodata is required
    so that 0 is not rendered as a real class.
    """
    if src_nodata is not None:
        return src_nodata
    s = _product_spec(product_id)
    for attr in ("nodata", "fill_value"):
        value = getattr(s, attr, None) if s is not None else None
        if value is not None:
            return value
    return None


def _needs_conversion(src: Path, dst: Path, force: bool) -> bool:
    """True when ``dst`` is missing, stale, or not a valid COG."""
    if force or not dst.exists():
        return True
    if dst.stat().st_mtime < src.stat().st_mtime:
        return True
    try:
        valid, _errors, _warnings = cog_validate(str(dst), quiet=True)
    except Exception:
        return True
    return not valid


def _write_cog(src: Path, dst: Path, nodata) -> None:
    """Stream ``src`` into a tiled, overviewed COG at ``dst``.

    Written against rasterio directly rather than ``rio_cogeo.cog_translate``
    because of how these sources are laid out: the JRC/GFC tiles are 120000 x
    120000 px and **striped**, one scanline per block. Decompressed that is
    ~14 GB for a single file, and rio-cogeo materialises a full-size intermediate
    (in RAM by default, and still whole-image with ``in_memory=False``), which
    the OOM killer ends. Copying window-by-window keeps peak memory to a few
    row-blocks regardless of how large the source is.

    Overviews are built with ``MODE`` -- the most common class code in each
    block. Averaging or interpolating would fabricate codes that appear nowhere
    in the legend.
    """
    from rasterio.shutil import copy as rio_copy
    from rasterio.windows import Window

    with rasterio.open(src) as ds:
        profile = ds.profile.copy()
        profile.update(
            driver="GTiff",
            tiled=True,
            blockxsize=_BLOCKSIZE,
            blockysize=_BLOCKSIZE,
            compress="deflate",
            predictor=2,
            num_threads="ALL_CPUS",
            BIGTIFF="IF_SAFER",
        )
        if nodata is not None:
            profile.update(nodata=nodata)

        # Stage a tiled copy next to the final file, then let GDAL rewrite it as
        # a COG (overviews first, then the header) via the COG driver. The
        # staging name deliberately does not end in ``.tif`` so a partially
        # written file is never picked up as a finished COG.
        staged = dst.with_name(dst.stem + ".staging.tmp")
        # Read whole block-rows at a time: with striped sources this turns
        # random access into a sequential scan, which matters over Lustre.
        rows = max(_BLOCKSIZE, ds.block_shapes[0][0])
        try:
            with rasterio.open(staged, "w", **profile) as out:
                for top in range(0, ds.height, rows):
                    height = min(rows, ds.height - top)
                    window = Window(0, top, ds.width, height)
                    out.write(ds.read(window=window), window=window)

                # MODE overviews down to roughly one screen tile, so a zoomed-out
                # request reads a small pre-computed level instead of the source.
                factors = []
                factor = 2
                while (
                    ds.width // factor > _BLOCKSIZE
                    or ds.height // factor > _BLOCKSIZE
                ):
                    factors.append(factor)
                    factor *= 2
                if factors:
                    out.build_overviews(factors, Resampling.mode)
                    out.update_tags(ns="rio_overview", resampling="mode")

            rio_copy(
                str(staged),
                str(dst),
                driver="COG",
                COMPRESS="DEFLATE",
                PREDICTOR=2,
                BLOCKSIZE=_BLOCKSIZE,
                OVERVIEW_RESAMPLING="MODE",
                BIGTIFF="IF_SAFER",
                NUM_THREADS="ALL_CPUS",
            )
        finally:
            if staged.exists():
                try:
                    staged.unlink()
                except OSError:
                    pass


def _convert_one(product_id: str, src: Path, force: bool) -> tuple[Path, str, str]:
    """Convert a single source tile. Returns ``(dst, status, detail)``."""
    dst = cog_path_for(product_id, src)
    try:
        if not _needs_conversion(src, dst, force):
            return dst, "skipped", "already a valid COG"

        dst.parent.mkdir(parents=True, exist_ok=True)

        with rasterio.open(src) as ds:
            nodata = _nodata_for(product_id, ds.nodata)

        # Write to a temp name and rename on success: an interrupted run must not
        # leave a truncated file that _needs_conversion would later accept.
        tmp = dst.with_name(dst.stem + ".partial.tmp")
        t0 = time.time()
        _write_cog(src, tmp, nodata)
        tmp.replace(dst)
        return dst, "converted", f"{time.time() - t0:.1f}s"
    except Exception as exc:  # keep one bad tile from killing the whole run
        tmp = dst.with_name(dst.stem + ".partial.tmp")
        if tmp.exists():
            try:
                tmp.unlink()
            except OSError:
                pass
        return dst, "failed", f"{type(exc).__name__}: {exc}"


def build_mosaic(product_id: str, cogs: list[Path]) -> Path | None:
    """Write a MosaicJSON indexing a product's COGs; ``None`` for single files.

    A one-file product needs no mosaic -- ``local_tiles`` reads the COG directly.
    """
    if len(cogs) < 2:
        return None

    from cogeo_mosaic.mosaic import MosaicJSON

    out = CONFIG.cog_dir / product_id / "mosaic.json"
    mosaic = MosaicJSON.from_urls([str(p) for p in sorted(cogs)], quiet=True)
    # Atomic: the tile server may be reading this file while a reindex runs, and
    # a half-written mosaic.json would break every tile for the product.
    tmp = out.with_name("mosaic.json.tmp")
    tmp.write_text(json.dumps(mosaic.model_dump(exclude_none=True)))
    tmp.replace(out)
    return out


def _clean_stale_temps(product_id: str) -> int:
    """Remove leftovers from an interrupted earlier run.

    A killed conversion can leave ``*.partial.tmp`` / ``*.staging.tmp`` files
    behind. They are never valid outputs, so drop them before converting rather
    than letting them accumulate at raster sizes.
    """
    base = CONFIG.cog_dir / product_id
    if not base.is_dir():
        return 0
    removed = 0
    for leftover in list(base.glob("*.tmp")) + list(base.glob("tmp*.tif")):
        try:
            leftover.unlink()
            removed += 1
        except OSError:
            pass
    return removed


def convert_product(product_id: str, force: bool = False, workers: int = 4) -> dict:
    """Convert every source tile of a product and index the result."""
    tiles = source_tiles(product_id)
    _log(f"\n=== {product_id}: {len(tiles)} source file(s) ===")

    stale = _clean_stale_temps(product_id)
    if stale:
        _log(f"  removed {stale} leftover temp file(s) from an interrupted run")

    done: list[Path] = []
    counts = {"converted": 0, "skipped": 0, "failed": 0}

    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(_convert_one, product_id, src, force): src for src in tiles
        }
        for i, fut in enumerate(concurrent.futures.as_completed(futures), 1):
            dst, status, detail = fut.result()
            counts[status] += 1
            if status == "failed":
                _log(f"  [{i}/{len(tiles)}] FAILED {futures[fut].name}: {detail}")
            else:
                done.append(dst)
                _log(f"  [{i}/{len(tiles)}] {status:9s} {dst.name} ({detail})")

    # Index every COG that exists for this product, not just the ones this run
    # happened to convert: a resumed run must still produce a complete mosaic.
    # ``done`` covers converted+skipped; failures are excluded because their
    # output was never renamed into place.
    mosaic = build_mosaic(product_id, sorted(set(done))) if done else None
    _log(
        f"  -> {counts['converted']} converted, {counts['skipped']} skipped, "
        f"{counts['failed']} failed"
        + (f"; mosaic {mosaic}" if mosaic else "")
    )
    return {
        "product_id": product_id,
        "counts": counts,
        "cogs": [str(p) for p in done],
        "mosaic": str(mosaic) if mosaic else None,
    }


def _local_raster_products() -> list[str]:
    from harmonizer.registry.schema import load_all_products

    return sorted(
        pid
        for pid, s in load_all_products().items()
        if getattr(s.access, "method", None) == "local_raster"
    )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--only", action="append", help="product id (repeatable)")
    ap.add_argument("--force", action="store_true", help="reconvert valid COGs")
    ap.add_argument("--workers", type=int, default=4, help="parallel conversions")
    ap.add_argument("--list", action="store_true", help="list products and exit")
    args = ap.parse_args(argv)

    products = _local_raster_products()
    if args.list:
        for pid in products:
            try:
                n = len(source_tiles(pid))
                _log(f"  {pid:24s} {n:4d} file(s)")
            except Exception as exc:
                _log(f"  {pid:24s} unavailable ({type(exc).__name__})")
        return 0

    if args.only:
        unknown = set(args.only) - set(products)
        if unknown:
            _log(f"unknown product id(s): {', '.join(sorted(unknown))}")
            return 2
        products = args.only

    _log(f"COG output tree: {CONFIG.cog_dir}")
    CONFIG.cog_dir.mkdir(parents=True, exist_ok=True)

    failed = 0
    for pid in products:
        try:
            result = convert_product(pid, force=args.force, workers=args.workers)
            failed += result["counts"]["failed"]
        except Exception as exc:
            _log(f"\n=== {pid}: ERROR {type(exc).__name__}: {exc}")
            failed += 1
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
