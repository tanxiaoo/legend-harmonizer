"""Verification for Stage S1 -- chunked full-overlap sampling.

DESIGN.md section 6, Stage S1. Checks the three claims the stage makes:

1. **Semantics are unchanged.** On a small AOI the chunked sampler reproduces
   the previous implementation's per-class pre/post-erode candidate counts
   *exactly*, class for class, when run at the raster's true native scale.
   ``harmonizer.local_sampling`` is kept as that reference implementation.
2. **The full overlap is tractable in bounded memory.** A full local x local
   overlap scan completes with process memory bounded (the design says < ~2 GB),
   with per-class counts reported and progress advancing per cell.
3. **Points are spread across the whole overlap.** Round-robin truncation keeps
   the kept points distributed over the region rather than piling into the first
   cells scanned; reported as the fraction of grid cells that contributed and a
   comparison against arrival-order truncation.

Also checks the source-derived footprint (section 3.1) against the declared one,
and the cost model / scale auto-suggestion (section 3.3).

Embeddings are **not** fetched: this verifies the local raster half, which is
what S1 changes. A stub embedding adapter stands in for AlphaEarth so the script
runs without Earth Engine credentials.

Run::

    python scripts/verify_s1.py
    python scripts/verify_s1.py --reference hrlc30 --compare worldcover_2020
    python scripts/verify_s1.py --skip-full     # small-AOI checks only (fast)
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402

import harmonizer  # noqa: F401,E402  (repairs PROJ before rasterio loads)
from harmonizer import chunked_sampling as chunked  # noqa: E402
from harmonizer import local_sampling, sampling  # noqa: E402
from harmonizer.config import CONFIG  # noqa: E402
from harmonizer.overlap import Overlap, overlap_for_products  # noqa: E402
from harmonizer.registry.legends import spec as product_spec  # noqa: E402

_FAILURES: list[str] = []


def _check(label: str, ok: bool, detail: str = "") -> bool:
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}" + (f" -- {detail}" if detail else ""))
    if not ok:
        _FAILURES.append(label)
    return ok


class _StubEmbedding:
    """Stands in for AlphaEarth so no GEE credentials are needed.

    Returns an unmasked zero vector for every point, so survival depends only on
    the label cross-check -- which is exactly the half S1 changes.
    """

    class _R:
        masked = False

        def __init__(self) -> None:
            self.vector = [0.0] * CONFIG.maps.embedding_dims

    def sample_embeddings(self, coords):
        return [self._R() for _ in coords]


def _native_scale_m(product_id: str) -> float:
    """The product's true pixel size in metres, from the raster itself.

    Deliberately not the registry's ``resolution_m``: hrlc30 is catalogued as
    30 m but its pixels are 27.83 m, and the exact-match check below is only
    meaningful at the real value (see ``ChunkedSampler._out_shape``).
    """
    import rasterio

    from harmonizer import local_tiles
    from harmonizer.footprints import _resolve

    source = local_tiles.cog_source(product_id)
    path = Path(source[0]) if source and not source[1] else _resolve(
        product_spec(product_id).access.path
    )
    with rasterio.open(path) as ds:
        res = abs(ds.transform.a)
        return res * (111_320.0 if (ds.crs and ds.crs.is_geographic) else 1.0)


# --------------------------------------------------------------------------- #
# 1. Semantics unchanged vs the reference implementation
# --------------------------------------------------------------------------- #


def check_semantics(product_id: str) -> None:
    print(f"\n=== 1. Chunked == reference implementation ({product_id}) ===")
    spec = product_spec(product_id)
    footprint = spec.footprint
    # One grid cell near the footprint centre, so the old whole-window read is
    # small enough to be possible at native resolution.
    lon = round((footprint[0] + footprint[2]) / 2.0, 2)
    lat = round((footprint[1] + footprint[3]) / 2.0, 2)
    cell = CONFIG.sampling.grid_cell_deg
    bbox = (lon, lat, lon + cell, lat + cell)
    print(f"  AOI {bbox} (one {cell} deg cell)")

    native = _native_scale_m(product_id)
    print(f"  true native scale: {native:.2f} m (registry says {spec.resolution_m})")

    band = int(spec.band) if spec.band is not None else 1
    window = local_sampling.read_label_window(spec.access.path, band, bbox)
    reference = {}
    for cv in local_sampling.present_classes(window):
        if cv == 0:
            continue
        reference[cv] = local_sampling.count_candidates_both(
            window, cv, erode_pixels=CONFIG.buffering.erode_pixels
        )

    sampler = chunked.ChunkedSampler(
        product_id, Overlap(bbox=bbox), sample_scale_m=native
    )
    totals, _cands = sampler.scan()

    print(f"  {'class':>6} {'ref pre':>10} {'new pre':>10} {'ref post':>10} {'new post':>10}")
    exact = True
    for cv in sorted(set(reference) | set(totals)):
        ref = reference.get(cv, (0, 0))
        got = totals.get(cv)
        new = (got.pre_erode, got.post_erode) if got else (0, 0)
        if ref != new:
            exact = False
        print(f"  {cv:6d} {ref[0]:10d} {new[0]:10d} {ref[1]:10d} {new[1]:10d}")
    _check(
        "per-class pre/post-erode counts match the reference implementation exactly",
        exact,
    )

    # A coarser scale must decimate, not match -- proof the scale is honoured at
    # all (the old path ignored ``sample_scale_m`` entirely and always read
    # native resolution).
    #
    # The expected ratio is *at most* 1/16 for a 4x coarser scale (area scales
    # with the square), and in practice below it: these are **post-**erosion
    # masks, and erosion is applied in pixels at the sample scale, so a 2-pixel
    # buffer eats 4x more ground at 4x the scale. Measured 0.013-0.047 against a
    # 0.0625 area bound. The assertion is therefore one-sided -- anything at or
    # above the area bound would mean the scale was not applied.
    coarse = chunked.ChunkedSampler(
        product_id, Overlap(bbox=bbox), sample_scale_m=native * 4
    )
    coarse_totals, _ = coarse.scan()
    ratios = [
        coarse_totals[cv].pre_erode / totals[cv].pre_erode
        for cv in totals
        if cv in coarse_totals and totals[cv].pre_erode > 1000
    ]
    _check(
        "a 4x coarser sample_scale_m decimates the pixel counts (<= 1/16 of native)",
        bool(ratios) and all(0.0 < r <= 0.0625 for r in ratios),
        f"ratios {[round(r, 4) for r in ratios]} (area bound 0.0625)",
    )


# --------------------------------------------------------------------------- #
# 2. Full overlap in bounded memory
# --------------------------------------------------------------------------- #


def check_full_overlap(reference_id: str, compare_id: str) -> None:
    print(f"\n=== 2. Full-overlap scan ({reference_id} x {compare_id}) ===")
    ov = overlap_for_products([reference_id, compare_id, "alphaearth"])
    print(f"  derived overlap: {tuple(round(v, 3) for v in ov.bbox)}")

    from harmonizer.api import _local_pixel_estimate, _suggest_scale

    scale = _suggest_scale(ov.bbox)
    pixels = _local_pixel_estimate(ov.bbox, scale)
    print(f"  auto-suggested scale: {scale:.0f} m ({pixels / 1e9:.2f} G pixels)")

    try:
        import psutil

        proc = psutil.Process()
        baseline = proc.memory_info().rss
    except ImportError:
        proc = None
        baseline = 0

    peak = [baseline]
    ticks = [0]

    def on_progress(frac, stage):
        ticks[0] += 1
        if proc is not None:
            peak[0] = max(peak[0], proc.memory_info().rss)

    for pid in (reference_id, compare_id):
        sampler = chunked.ChunkedSampler(
            pid, ov, sample_scale_m=scale, progress=on_progress
        )
        t0 = time.time()
        totals, cands = sampler.scan()
        elapsed = time.time() - t0
        n_cells = len(sampler.grid)
        drawn = {cv: sum(len(c) for c in v) for cv, v in cands.items()}
        print(
            f"\n  {pid}: {n_cells} cells in {elapsed:.1f}s "
            f"({elapsed / max(1, n_cells) * 1000:.1f} ms/cell)"
        )
        for cv in sorted(totals):
            print(
                f"    class {cv:4d}: pre={totals[cv].pre_erode:10d} "
                f"post={totals[cv].post_erode:10d} candidates={drawn.get(cv, 0):6d}"
            )
        _check(f"{pid}: scan produced candidates", any(drawn.values()))
        _check(f"{pid}: progress reported per cell", ticks[0] >= n_cells)

    if proc is not None:
        delta_gb = (peak[0] - baseline) / 1e9
        print(f"\n  peak memory above baseline: {delta_gb:.2f} GB")
        _check("process memory stays bounded (< 2 GB above baseline)", delta_gb < 2.0)
    else:
        print("  (install psutil to measure memory)")


# --------------------------------------------------------------------------- #
# 3. Points spread across the overlap
# --------------------------------------------------------------------------- #


def check_spread(product_id: str) -> None:
    print(f"\n=== 3. Points stay spread across the overlap ({product_id}) ===")
    ov = overlap_for_products([product_id, "alphaearth"])
    from harmonizer.api import _suggest_scale

    scale = _suggest_scale(ov.bbox)
    sampler = chunked.ChunkedSampler(product_id, ov, sample_scale_m=scale)
    _totals, cands = sampler.scan()

    target = CONFIG.sampling.points_target
    print(f"  target={target}; cells a class's kept points are drawn from:")
    tested = 0
    for cv, per_cell in sorted(cands.items()):
        occupied = sum(1 for c in per_cell if c)
        if occupied < 4:
            continue
        kept = chunked.draw_for_class(per_cell)
        if not kept:
            continue
        # How many distinct cells the kept points came from, round-robin vs the
        # arrival-order truncation the old path used.
        thinned = [
            chunked.thin_by_spacing(c, CONFIG.sampling.min_spacing_m) for c in per_cell
        ]
        cells_of = {}
        for i, c in enumerate(thinned):
            for p in c:
                cells_of[p] = i
        flat = [p for c in thinned for p in c]
        arrival = flat[:target]
        rr_cells = len({cells_of[p] for p in kept})
        ar_cells = len({cells_of[p] for p in arrival})
        # Truncation only *binds* when there are more candidates than the target;
        # below that every strategy keeps everything and the comparison is vacuous.
        binds = len(flat) > target
        print(
            f"    class {cv:4d}: {occupied:4d} occupied cells, {len(flat):6d} "
            f"candidates -> kept {len(kept):5d} from {rr_cells:4d} cells "
            f"(arrival order: {ar_cells:4d}){'' if binds else '   [truncation not binding]'}"
        )
        if binds:
            tested += 1
            _check(
                f"class {cv}: round-robin spreads at least as wide as arrival order",
                rr_cells >= ar_cells,
                f"{rr_cells} vs {ar_cells} cells",
            )

    _check(
        "at least one class actually exercised the truncation",
        tested > 0,
        "otherwise the comparison above is vacuous -- every candidate was kept",
    )

    # The property in isolation, independent of whether this dataset happens to
    # produce a skewed class: one dominant cell holding more than the target,
    # plus many small cells. Arrival order collapses onto the dominant cell;
    # round-robin must not. This is the regression that DESIGN.md 3.2 step 4
    # describes ("not spatially fair and partially undoes the per-cell spread").
    synthetic = [[(0.0, float(i)) for i in range(2 * target)]] + [
        [(float(10 + j), float(k)) for k in range(10)] for j in range(50)
    ]
    idx = {}
    for i, c in enumerate(synthetic):
        for p in c:
            idx[p] = i
    rr = chunked.round_robin_truncate(synthetic, target)
    arrival = [p for c in synthetic for p in c][:target]
    rr_cells = len({idx[p] for p in rr})
    ar_cells = len({idx[p] for p in arrival})
    print(
        f"\n  synthetic skew (1 cell of {2 * target} pts + 50 cells of 10): "
        f"round-robin uses {rr_cells} cells, arrival order uses {ar_cells}"
    )
    _check(
        "round-robin beats arrival order on a skewed distribution",
        rr_cells > ar_cells,
        f"{rr_cells} vs {ar_cells} cells",
    )


# --------------------------------------------------------------------------- #
# 4. Source-derived footprint and cost model
# --------------------------------------------------------------------------- #


def check_footprints(product_ids: list[str]) -> None:
    print("\n=== 4. Source-derived footprints (3.1) and cost model (3.3) ===")
    from harmonizer import footprints

    for pid in product_ids:
        declared = product_spec(pid).footprint
        derived = footprints.derived_footprint(pid)
        print(f"  {pid}:")
        print(f"    declared {declared}")
        print(f"    derived  {tuple(round(v, 4) for v in derived) if derived else None}")
        _check(f"{pid}: footprint derived from the source", derived is not None)

    from harmonizer.api import _LOCAL_PIXEL_BUDGET, _local_pixel_estimate, _suggest_scale

    print("\n  scale suggestions:")
    for label, bbox in (
        ("full Africa-ish overlap", (26.06, 3.52, 43.29, 16.28)),
        ("1 deg AOI", (30.0, 8.0, 31.0, 9.0)),
        ("0.25 deg cell", (30.0, 8.0, 30.25, 8.25)),
    ):
        scale = _suggest_scale(bbox)
        px = _local_pixel_estimate(bbox, scale)
        ok = px <= _LOCAL_PIXEL_BUDGET
        print(f"    {label:24s} -> {scale:6.0f} m  ({px / 1e9:.3f} G px)")
        _check(f"suggested scale for {label} is within budget", ok)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--reference", default="hrlc30")
    ap.add_argument("--compare", default="worldcover_2020")
    ap.add_argument("--skip-full", action="store_true", help="skip the full-overlap scan")
    args = ap.parse_args()

    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")

    print("Stage S1 verification -- chunked full-overlap sampling")
    print(f"  reference={args.reference}  compare={args.compare}")
    print(
        f"  constants: grid_cell_deg={CONFIG.sampling.grid_cell_deg} "
        f"erode_pixels={CONFIG.buffering.erode_pixels} "
        f"floor={CONFIG.sampling.points_floor} target={CONFIG.sampling.points_target}"
    )

    check_footprints([args.reference, args.compare])
    check_semantics(args.reference)
    if not args.skip_full:
        check_full_overlap(args.reference, args.compare)
        check_spread(args.reference)

    print("\n" + "=" * 70)
    if _FAILURES:
        print(f"{len(_FAILURES)} check(s) FAILED:")
        for f in _FAILURES:
            print(f"  - {f}")
        return 1
    print("All Stage S1 checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
