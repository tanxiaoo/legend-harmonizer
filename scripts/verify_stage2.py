"""Stage 2 verification: sampling and buffering.

Samples both MVP label maps -- the test-swap reference (ESA WorldCover) and the
compare map (Dynamic World) -- over the sampling region derived from the selected
products' footprints, then prints the per-class surviving counts and the
absent-vs-buffered-away bookkeeping for each map. For this pair both label maps
and AlphaEarth are global, so the footprint-derived region is global (it is *not*
clamped to HRLC's Eastern Sahel box -- HRLC is not selected here); a deliberate
test AOI (the Nile around Khartoum) then narrows that region to keep the run
tractable on GEE. The AOI is a run-level constraint, not a pipeline constant.

For each map and class it shows: surviving points, the pre-/post-erode candidate
counts that drive the absent-vs-buffered-away rule, and the resulting status
(``ok`` / ``relaxed`` / ``absent``). The check is that every class is either at or
above the floor, or is flagged (``relaxed`` or ``absent``) -- and that counts look
reasonable relative to how common each class is in the region (docs/PIPELINE.md,
Stage 2 -> Verification).

Both point tables are also written to ``cache/`` (``samples_<product>.npz`` plus a
JSON sidecar) for Stage 3 to consume.

Run:  python scripts/verify_stage2.py

GEE must be authenticated (``earthengine authenticate``). Each map is sampled
independently, so a failure on one is reported without aborting the other. This
touches a lot of GEE compute; expect it to take a while.
"""

from __future__ import annotations

import dataclasses
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from harmonizer.config import CONFIG
from harmonizer.overlap import Overlap, overlap_for_products
from harmonizer.sampling import MapSample, sample_map, save_map_sample

# The two MVP label products to sample. AlphaEarth (the embedding) is pulled in
# automatically by sample_map.
PRODUCTS = ["worldcover", "dynamicworld"]

# Deliberate Stage-2 test AOI: the Nile around Khartoum. This is a *run-level*
# constraint passed in here to keep the global-pair verification tractable on
# GEE -- it narrows the footprint-derived region, it is not a pipeline constant
# and does not replace the footprint logic. Stored as
# (min_lon, min_lat, max_lon, max_lat) in EPSG:4326.
#
# Default to a 1x1 degree sub-box (32-33E, 14-15N) so we can see whether the
# earlier hang was area-driven; set STAGE2_FULL_AOI=1 for the full 3x2.5deg box.
_FULL_AOI = (32.0, 14.0, 35.0, 16.5)
_SUBBOX_AOI = (32.0, 14.0, 33.0, 15.0)
TEST_AOI = _FULL_AOI if os.environ.get("STAGE2_FULL_AOI") else _SUBBOX_AOI

# Coarser raster scale (metres) for the test run: sampling/eroding at ~60 m
# instead of the products' 10 m native trades spatial fidelity for speed without
# touching the real floor/target. Override with STAGE2_SCALE_M; set to 10 to run
# at native scale. The config default (10 m) is left unchanged.
TEST_SCALE_M = float(os.environ.get("STAGE2_SCALE_M", "60"))


def _describe_region(ov: Overlap) -> str:
    """A human-readable name for the derived sampling region."""
    if ov.is_global:
        return "GLOBAL (both label maps and AlphaEarth are global)"
    return f"bounded {ov.bbox}"


def _print_map(ms: MapSample) -> None:
    floor = ms.floor
    print()
    print("=" * 78)
    print(f"Map: {ms.product_id}   working year {ms.working_year}   floor {floor}")
    print("=" * 78)

    header = (
        f"{'class':>6}  {'surviving':>9}  {'pre-erode':>9}  {'post-erode':>10}  "
        f"{'status':>8}  check"
    )
    print(header)
    print("-" * len(header))

    all_ok = True
    for cv in sorted(ms.classes):
        cs = ms.classes[cv]
        # A class passes if it meets the floor, or is legitimately flagged.
        meets_floor = cs.surviving >= floor
        flagged = cs.buffer_relaxed or cs.absent
        ok = meets_floor or flagged
        all_ok = all_ok and ok
        verdict = "OK" if ok else "!! BELOW FLOOR, UNFLAGGED"
        print(
            f"{cv:>6}  {cs.surviving:>9}  {cs.pre_erode_candidates:>9}  "
            f"{cs.post_erode_candidates:>10}  {cs.status:>8}  {verdict}"
        )

    print("-" * len(header))
    print(
        f"All classes at/above floor or flagged (relaxed/absent): "
        f"{'YES' if all_ok else 'NO'}"
    )


def main() -> None:
    # Apply the coarser test scale without disturbing the real config defaults:
    # replace only sampling.sample_scale_m on the module singleton for this run.
    # floor=300/target=1500 and the 10 m default are untouched in config.py.
    global CONFIG
    new_sampling = dataclasses.replace(CONFIG.sampling, sample_scale_m=TEST_SCALE_M)
    CONFIG = dataclasses.replace(CONFIG, sampling=new_sampling)
    import harmonizer.sampling as _samp
    _samp.CONFIG = CONFIG  # sampling reads scale via CONFIG.sampling.sample_scale_m

    print("=" * 78)
    print("Stage 2 verification - sampling and buffering")
    print(
        f"floor={CONFIG.sampling.points_floor}  target={CONFIG.sampling.points_target}  "
        f"erode={CONFIG.buffering.erode_pixels}px  spacing={CONFIG.sampling.min_spacing_m}m  "
        f"grid={CONFIG.sampling.grid_cell_deg}deg"
    )
    print(
        f"TEST sample scale: {TEST_SCALE_M}m (native is 10m; config default "
        f"unchanged; override with STAGE2_SCALE_M)"
    )
    # The sampling region is derived from the selected products' footprints (this
    # global pair comes out global), then narrowed by the deliberate test AOI so
    # the verification is tractable on GEE.
    footprint_region = overlap_for_products(PRODUCTS + ["alphaearth"])
    overlap = overlap_for_products(PRODUCTS + ["alphaearth"], aoi=TEST_AOI)
    aoi_kind = "full 3x2.5deg" if TEST_AOI == _FULL_AOI else "1x1deg sub-box"
    print(f"selected products: {', '.join(PRODUCTS)} (+ alphaearth embedding)")
    print(f"footprint-derived region: {_describe_region(footprint_region)}")
    print(f"test AOI (Nile / Khartoum, {aoi_kind}): {TEST_AOI}")
    print(f"sampling region (footprint region narrowed by AOI): {overlap.bbox}")
    print("=" * 78)

    for product_id in PRODUCTS:
        try:
            ms = sample_map(product_id, overlap=overlap)
        except Exception as exc:  # noqa: BLE001 - report and continue
            print()
            print(f"{product_id}: unavailable -> {type(exc).__name__}: {exc}")
            continue
        _print_map(ms)
        path = save_map_sample(ms)
        print(f"saved -> {path}")
        print(f"       -> {path.with_suffix('.json')}")

    print()
    print(
        "Confirm: every class is at/above the floor or flagged, and counts track "
        f"how common each class is across the sampling region "
        f"(test AOI {overlap.bbox}, Nile around Khartoum)."
    )


if __name__ == "__main__":
    main()
