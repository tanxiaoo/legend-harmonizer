"""Verification for Stage V1 -- COG activation and tile-path hardening.

DESIGN.md section 6, Stage V1. Checks four things and prints a checkable
artifact for each:

1. **Tile timing.** Renders a fixed set of tiles (z4 / z8 / z12) for each local
   product and reports per-tile wall time. Run it once *before* converting a
   product and once after: the design expects the documented ~100x improvement
   on cold zoomed-out tiles.
2. **Source state.** Reports whether each product serves from a mosaic, a single
   COG, or the degraded raw-source path.
3. **The mosaic gap is loud.** Simulates a conversion that died between writing
   the COGs and building the index (by hiding ``mosaic.json``) and asserts an
   ERROR is logged rather than a silent fallback.
4. **Parallel rendering.** Renders two products' tiles concurrently and compares
   the wall time against the same work done serially -- with
   ``source_concurrency`` above 1 the concurrent run must not simply be the sum.

Run::

    python scripts/verify_v1.py                 # every available local product
    python scripts/verify_v1.py --only worldcover_2020
    python scripts/verify_v1.py --skip-timing   # just the state/warning checks
"""

from __future__ import annotations

import argparse
import concurrent.futures
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import harmonizer  # noqa: F401,E402  (repairs PROJ before rasterio loads)
from harmonizer import local_tiles  # noqa: E402
from harmonizer.config import CONFIG  # noqa: E402

# Zoom levels that between them exercise the whole pyramid: a zoomed-out tile
# (the pathological case -- without overviews it reads the entire raster), a
# regional one, and a local one. The actual x/y are derived per product from its
# own footprint (``_tiles_for``) rather than hardcoded, so the timing measures a
# real read instead of an instant "outside bounds".
_ZOOMS = (4, 8, 12)


def _lonlat_to_tile(lon: float, lat: float, z: int) -> tuple[int, int]:
    """Web-Mercator XYZ tile containing a point, at zoom ``z``."""
    import math

    n = 2**z
    lat = max(-85.05, min(85.05, lat))
    x = int((lon + 180.0) / 360.0 * n)
    rad = math.radians(lat)
    y = int((1.0 - math.asinh(math.tan(rad)) / math.pi) / 2.0 * n)
    return max(0, min(n - 1, x)), max(0, min(n - 1, y))


def _tiles_for(product_id: str) -> list[tuple[int, int, int]]:
    """One tile per zoom, centred on the product's own footprint."""
    spec = local_tiles._product_spec(product_id)
    footprint = getattr(spec, "footprint", None) if spec is not None else None
    if not footprint:
        # No declared footprint: fall back to the centre of Africa, where this
        # deployment's data sits.
        lon, lat = 26.0, 8.0
    else:
        min_lon, min_lat, max_lon, max_lat = footprint
        lon, lat = (min_lon + max_lon) / 2.0, (min_lat + max_lat) / 2.0
    out = []
    for z in _ZOOMS:
        x, y = _lonlat_to_tile(lon, lat, z)
        out.append((z, x, y))
    return out


def _local_products() -> list[str]:
    from harmonizer.registry.schema import load_all_products

    out = []
    for pid, spec in sorted(load_all_products().items()):
        if getattr(spec.access, "method", None) != "local_raster":
            continue
        try:
            local_tiles.legend(pid)
        except KeyError:
            continue
        out.append(pid)
    return out


def _render(pid: str, z: int, x: int, y: int) -> tuple[float, str]:
    """Render one tile, bypassing the PNG disk cache so the timing is real."""
    path = local_tiles._cache_path(pid, z, x, y, None)
    if path.exists():
        try:
            path.unlink()
        except OSError:
            pass
    t0 = time.time()
    try:
        png = local_tiles.tile_png(pid, z, x, y)
        return time.time() - t0, f"{len(png) / 1024:.0f} KB"
    except local_tiles.TileOutsideBounds:
        return time.time() - t0, "outside bounds"
    except Exception as exc:
        return time.time() - t0, f"{type(exc).__name__}: {exc}"


def check_timing(products: list[str]) -> None:
    print("\n=== 1. Tile render timing (disk cache bypassed) ===")
    print(f"{'product':<20} {'tile':<14} {'state':<17} {'seconds':>9}  detail")
    for pid in products:
        state = local_tiles.source_state(pid)
        for z, x, y in _tiles_for(pid):
            secs, detail = _render(pid, z, x, y)
            addr = f"z{z}/{x}/{y}"
            print(f"{pid:<20} {addr:<14} {state:<17} {secs:>9.2f}  {detail}")


def check_state(products: list[str]) -> None:
    print("\n=== 2. Source state per product ===")
    for pid in products:
        state = local_tiles.source_state(pid)
        source = local_tiles.cog_source(pid)
        where = source[0] if source else f"(raw) {local_tiles._product_spec(pid).access.path}"
        flag = "OK  " if state in ("mosaic", "single") else "SLOW"
        print(f"  [{flag}] {pid:<20} {state:<17} {where}")
    print(f"\n  COG tree: {CONFIG.cog_dir}")
    print(
        f"  concurrency: source={CONFIG.tiles.source_concurrency} "
        f"cog={CONFIG.tiles.cog_concurrency}"
    )


class _Capture(logging.Handler):
    def __init__(self) -> None:
        super().__init__()
        self.records: list[logging.LogRecord] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(record)


def check_mosaic_gap(products: list[str]) -> bool:
    """Deleting mosaic.json from a converted product must warn, not go quiet."""
    print("\n=== 3. Mosaic gap is reported, not silently slow ===")
    converted = [
        pid
        for pid in products
        if local_tiles.source_state(pid) == "mosaic"
        and len(list((CONFIG.cog_dir / pid).glob("*.tif"))) >= 2
    ]
    if not converted:
        print("  SKIP: no multi-file converted product available to test against.")
        print("        Convert one first: python tools/to_cog.py --only <product_id>")
        return True

    pid = converted[0]
    mosaic = CONFIG.cog_dir / pid / "mosaic.json"
    hidden = mosaic.with_suffix(".json.hidden")
    capture = _Capture()
    logger = logging.getLogger("harmonizer.local_tiles")
    logger.addHandler(capture)
    try:
        mosaic.rename(hidden)
        local_tiles.invalidate_cog_source()
        state = local_tiles.source_state(pid)
        errors = [r for r in capture.records if r.levelno >= logging.ERROR]
        ok = state == "needs-conversion" and bool(errors)
        print(f"  {pid}: state without mosaic.json = {state!r}")
        for r in errors:
            print(f"    logged ERROR: {r.getMessage()[:120]}")
        print(f"  {'PASS' if ok else 'FAIL'}: expected 'needs-conversion' + an ERROR log")
        return ok
    finally:
        logger.removeHandler(capture)
        if hidden.exists():
            hidden.rename(mosaic)
        local_tiles.invalidate_cog_source()


def check_parallel(products: list[str]) -> None:
    """Two panes must render at once rather than queueing behind each other."""
    print("\n=== 4. Concurrent rendering across two products ===")
    if len(products) < 2:
        print("  SKIP: need two available local products.")
        return
    a, b = products[0], products[1]
    # Both products must actually cover the tile, or one side returns instantly
    # and the comparison measures nothing. Use the centre of their overlapping
    # footprints.
    fa = getattr(local_tiles._product_spec(a), "footprint", None)
    fb = getattr(local_tiles._product_spec(b), "footprint", None)
    if fa and fb:
        lon = (max(fa[0], fb[0]) + min(fa[2], fb[2])) / 2.0
        lat = (max(fa[1], fb[1]) + min(fa[3], fb[3])) / 2.0
    else:
        lon, lat = 26.0, 8.0
    z = 8
    x, y = _lonlat_to_tile(lon, lat, z)

    serial = time.time()
    _render(a, z, x, y)
    _render(b, z, x, y)
    serial = time.time() - serial

    parallel = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        futs = [pool.submit(_render, a, z, x, y), pool.submit(_render, b, z, x, y)]
        for f in futs:
            f.result()
    parallel = time.time() - parallel

    print(f"  tile z{z}/{x}/{y} (inside both footprints)")
    print(f"  serial   {a} + {b}: {serial:.2f}s")
    print(f"  parallel {a} + {b}: {parallel:.2f}s")
    if serial > 0:
        print(f"  speedup: {serial / parallel:.2f}x (1.0 = fully serialised)")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--only", action="append", help="product id (repeatable)")
    ap.add_argument("--skip-timing", action="store_true", help="skip the slow timing pass")
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s %(name)s: %(message)s"
    )

    products = _local_products()
    if args.only:
        unknown = set(args.only) - set(products)
        if unknown:
            print(f"unknown or unavailable product id(s): {sorted(unknown)}")
            print(f"available: {products}")
            return 2
        products = args.only
    if not products:
        print("no local-raster products with readable sources on this machine")
        return 2

    print(f"Stage V1 verification -- products: {', '.join(products)}")
    check_state(products)
    ok = check_mosaic_gap(products)
    if not args.skip_timing:
        check_timing(products)
        check_parallel(products)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
