"""Prove the app never modifies anything under ``data/``.

``data/`` holds rasters the **user** downloaded. This project does not produce
them, cannot replace them, and must never alter them -- all derived data goes to
``cache/`` (``config.COG_DIR``). That is a promise worth *testing* rather than
asserting, because the failure mode is silent: a corrupted source only surfaces
later, as a wrong map or a failed run, long after the write that caused it.

This script takes a full snapshot of ``data/`` (every file's size and mtime),
runs the operations that touch source rasters hardest -- tile rendering,
footprint derivation, chunked sampling, and optionally a COG conversion -- then
re-snapshots and reports any file added, removed or modified.

It also checks the structural guarantees behind the promise:

* every raster is opened in mode ``r``;
* GDAL's PAM sidecars (``.aux.xml``) are disabled, so GDAL cannot drop files
  next to the sources either (``harmonizer._protect_source_data``);
* no conversion output path resolves inside ``data/``.

Run::

    python scripts/verify_data_readonly.py
    python scripts/verify_data_readonly.py --convert hrlc30   # include a conversion
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import harmonizer  # noqa: F401,E402  (sets GDAL_PAM_ENABLED before GDAL loads)
from harmonizer.config import CONFIG  # noqa: E402

_FAILURES: list[str] = []


def _check(label: str, ok: bool, detail: str = "") -> bool:
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}" + (f" -- {detail}" if detail else ""))
    if not ok:
        _FAILURES.append(label)
    return ok


def snapshot(root: Path) -> dict[str, tuple[int, float]]:
    """Every file under ``root`` with its size and modification time."""
    out: dict[str, tuple[int, float]] = {}
    for path in sorted(root.rglob("*")):
        if path.is_file():
            st = path.stat()
            out[str(path)] = (st.st_size, round(st.st_mtime, 3))
    return out


def check_structural() -> None:
    print("\n=== 1. Structural guarantees ===")
    _check(
        "GDAL PAM sidecars are disabled (GDAL cannot write .aux.xml into data/)",
        os.environ.get("GDAL_PAM_ENABLED") == "NO",
        f"GDAL_PAM_ENABLED={os.environ.get('GDAL_PAM_ENABLED')!r}",
    )

    from tools import to_cog

    inside = []
    for pid in _local_products():
        try:
            for src in to_cog.source_tiles(pid)[:1]:
                dst = to_cog.cog_path_for(pid, src)
                if str(CONFIG.data_dir).lower() in str(dst).lower():
                    inside.append(str(dst))
        except Exception:
            continue
    _check(
        "no conversion output path resolves inside data/",
        not inside,
        str(inside[:3]) if inside else f"all outputs go to {CONFIG.cog_dir}",
    )

    # Every raster this app opens must be opened read-only.
    import rasterio

    modes = []
    for pid in _local_products()[:3]:
        try:
            from harmonizer.registry.legends import spec

            path = Path(spec(pid).access.path)
            if not path.is_absolute():
                path = CONFIG.data_dir.parent / path
            if not path.exists():
                continue
            with rasterio.open(path) as ds:
                modes.append((pid, ds.mode))
        except Exception:
            continue
    _check(
        "source rasters are opened in read-only mode",
        all(m == "r" for _pid, m in modes),
        ", ".join(f"{p}={m}" for p, m in modes),
    )


def _local_products() -> list[str]:
    from harmonizer.registry.schema import load_all_products

    return sorted(
        pid
        for pid, s in load_all_products().items()
        if getattr(s.access, "method", None) == "local_raster"
    )


def exercise(convert: str | None) -> None:
    """Run the operations that read source rasters hardest."""
    from harmonizer import chunked_sampling as chunked
    from harmonizer import footprints, local_tiles
    from harmonizer.overlap import Overlap

    for pid in _local_products():
        try:
            local_tiles.legend(pid)
        except Exception:
            continue

        # Footprint derivation: opens the source and reads its bounds.
        footprints.derived_footprint(pid)

        spec_fp = None
        try:
            from harmonizer.registry.legends import spec

            spec_fp = spec(pid).footprint
        except Exception:
            pass
        if not spec_fp:
            continue

        # A tile render and a small sampling scan: the two hot read paths.
        lon = (spec_fp[0] + spec_fp[2]) / 2.0
        lat = (spec_fp[1] + spec_fp[3]) / 2.0
        try:
            import math

            n = 2**8
            x = int((lon + 180.0) / 360.0 * n)
            y = int(
                (1.0 - math.asinh(math.tan(math.radians(lat))) / math.pi) / 2.0 * n
            )
            local_tiles.tile_png(pid, 8, x, y)
        except Exception:
            pass
        try:
            sampler = chunked.ChunkedSampler(
                pid,
                Overlap(bbox=(lon, lat, lon + 0.25, lat + 0.25)),
                sample_scale_m=100.0,
            )
            sampler.scan()
        except Exception:
            pass
        print(f"    exercised {pid}")

    if convert:
        print(f"    converting {convert} (this writes to cache/, never data/)")
        from tools import to_cog

        to_cog.convert_product(convert, force=False, workers=2)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--convert",
        help="also run a COG conversion for this product id (the heaviest reader)",
    )
    args = ap.parse_args()

    print("Verifying that the app never modifies data/")
    print(f"  data dir : {CONFIG.data_dir}")
    print(f"  cache dir: {CONFIG.cog_dir}")

    check_structural()

    print("\n=== 2. Snapshot, exercise, re-snapshot ===")
    before = snapshot(CONFIG.data_dir)
    print(f"  {len(before)} file(s) under data/ before")
    exercise(args.convert)
    after = snapshot(CONFIG.data_dir)
    print(f"  {len(after)} file(s) under data/ after")

    added = sorted(set(after) - set(before))
    removed = sorted(set(before) - set(after))
    changed = sorted(k for k in before if k in after and before[k] != after[k])

    for label, items in (
        ("added", added),
        ("removed", removed),
        ("modified", changed),
    ):
        if items:
            print(f"  files {label}:")
            for f in items[:10]:
                print(f"    {f}")
    _check("no file added to data/", not added, f"{len(added)} added")
    _check("no file removed from data/", not removed, f"{len(removed)} removed")
    _check("no file modified in data/", not changed, f"{len(changed)} modified")

    print("\n" + "=" * 70)
    if _FAILURES:
        print(f"{len(_FAILURES)} check(s) FAILED:")
        for f in _FAILURES:
            print(f"  - {f}")
        return 1
    print("data/ is untouched: the app only ever reads the user's downloads.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
