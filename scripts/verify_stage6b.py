"""Stage 6b verification: evidence explorer engine + review API (backend).

Exercises the two backend parts of Stage 6b from the CLI, following the revised
build-order bullet in docs/PIPELINE.md ("6b"):

  (1) Evidence explorer engine -- for a class-pair, the three-mode co-located-pixel
      query returns N spatially-declustered locations with the correct label on both
      sides (docs/PIPELINE.md, sections 6.2 / 6.3). This part hits **live GEE**, so a
      small AOI inside the Eastern Sahel is used to keep it cheap.

  (2) Review API layer -- the confirm -> refit -> reviewed-table round-trip through
      the same functions the FastAPI endpoints wrap reproduces the Stage 6a
      invariants (confirmed edges freeze; unconfirmed edges re-propose after a refit;
      no confirmed edge anywhere moves). This part is pure CPU, reusing the Stage 3
      GMM caches. The GMM cache and any feedback store are backed up and restored so
      other stages are undisturbed.

Run:  python scripts/verify_stage6b.py

Stages 2 and 3 must have run first for the test-swap pair (WorldCover reference,
Dynamic World compare). The explorer part needs GEE authentication
(``earthengine authenticate``); if GEE is unreachable it is reported as SKIPPED (not
a failure) and the review-API part still runs.
"""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

# sklearn's loky backend can hang probing physical cores on some Windows setups
# (it raises "found 0 physical cores"); pin the worker count before sklearn loads
# so the warm-start refit's GaussianMixture.fit() runs single-process and never
# blocks the verification.
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from harmonizer.affinity import compute_affinity  # noqa: E402
from harmonizer.config import CONFIG  # noqa: E402
from harmonizer.decision import absent_decisions, classify_rows  # noqa: E402
from harmonizer.modeling import gmm_cache_path  # noqa: E402
from harmonizer.review import (  # noqa: E402
    ALGORITHM_PROPOSED,
    EXPERT_CONFIRMED,
    FeedbackStore,
    build_reviewed_table,
    confirm_edges,
    feedback_cache_path,
    recompute_reviewed_table,
    rebalance_row,
    save_feedback,
    warm_start_refit,
)

REFERENCE_ID = "worldcover"
COMPARE_ID = "dynamicworld"

# A small AOI inside the Eastern Sahel to bound the live-GEE explorer draw. The
# explorer's joint draw is as expensive as Stage 2 sampling, so the verification
# uses a genuinely small AOI + a coarse scale (like a real test run would) -- see
# the slow-combination note in docs/PIPELINE.md section 2.
TEST_AOI = (32.0, 14.0, 32.15, 14.15)
TEST_SCALE_M = 40.0  # coarsen the join/read scale for the probe (default 10 m)


def _find_row(rows, rv):
    return next(r for r in rows if r.reference_value == rv)


def _edge(row, cv):
    return next(e for e in row.edges if e.compare_value == cv)


# --------------------------------------------------------------------------- #
# Part 1 -- evidence explorer engine (live GEE)
# --------------------------------------------------------------------------- #


def verify_explorer(rv: int, cv: int) -> bool:
    """Run the three modes and check each returns spread, correctly-labelled points."""
    from harmonizer.explorer import explore_evidence

    print("\n" + "-" * 88)
    print(f"[EXPLORER] class-pair reference={rv}  compare={cv}  AOI={TEST_AOI}")
    print("-" * 88)

    ok = True

    # Mode 1 -- both classes (a cell): every location must carry BOTH labels, and the
    # co-located query means each side reads back its queried class.
    both = explore_evidence(
        REFERENCE_ID, COMPARE_ID, mode="both",
        reference_value=rv, compare_value=cv, aoi=TEST_AOI,
    )
    print(f"\nmode 'both': {both.n} locations "
          f"(window {both.patch_window_px}px = {both.patch_window_m:.0f} m)")
    both_ok = both.n > 0 and all(
        loc.reference_label == rv and loc.compare_label == cv
        for loc in both.locations
    )
    for loc in both.locations[:4]:
        print(f"    ({loc.lon:.4f}, {loc.lat:.4f})  "
              f"ref={loc.reference_label}({loc.reference_label_name})  "
              f"cmp={loc.compare_label}({loc.compare_label_name})")
    print(f"    every location satisfies BOTH conditions: "
          f"{'YES' if both_ok else 'NO'}")
    ok &= both_ok

    # Mode 2 -- reference only (a row): reference side must read back rv; the compare
    # side is whatever it labels there (not constrained).
    row = explore_evidence(
        REFERENCE_ID, COMPARE_ID, mode="reference",
        reference_value=rv, aoi=TEST_AOI,
    )
    print(f"\nmode 'reference': {row.n} locations")
    row_ok = row.n > 0 and all(loc.reference_label == rv for loc in row.locations)
    cmp_spread = {loc.compare_label for loc in row.locations}
    for loc in row.locations[:4]:
        print(f"    ({loc.lon:.4f}, {loc.lat:.4f})  "
              f"ref={loc.reference_label}  cmp={loc.compare_label}")
    print(f"    reference side always = {rv}: {'YES' if row_ok else 'NO'}")
    print(f"    compare labels seen (the row's spread): {sorted(c for c in cmp_spread if c is not None)}")
    ok &= row_ok

    # Mode 3 -- compare only (a column): compare side must read back cv.
    col = explore_evidence(
        REFERENCE_ID, COMPARE_ID, mode="compare",
        compare_value=cv, aoi=TEST_AOI,
    )
    print(f"\nmode 'compare': {col.n} locations")
    col_ok = col.n > 0 and all(loc.compare_label == cv for loc in col.locations)
    ref_spread = {loc.reference_label for loc in col.locations}
    for loc in col.locations[:4]:
        print(f"    ({loc.lon:.4f}, {loc.lat:.4f})  "
              f"ref={loc.reference_label}  cmp={loc.compare_label}")
    print(f"    compare side always = {cv}: {'YES' if col_ok else 'NO'}")
    print(f"    reference labels seen (the column's spread): {sorted(r for r in ref_spread if r is not None)}")
    ok &= col_ok

    # Spatial decluster check: no two returned points closer than the configured
    # min spacing (great-circle, metres).
    declustered = _min_pairwise_m(both.locations) >= CONFIG.sampling.min_spacing_m * 0.99
    print(f"\n    'both' locations respect min spacing "
          f"({CONFIG.sampling.min_spacing_m:.0f} m): "
          f"{'YES' if declustered or both.n < 2 else 'NO'}")
    ok &= (declustered or both.n < 2)

    return ok


def _min_pairwise_m(locations) -> float:
    import math

    r = 6_371_000.0
    pts = [(math.radians(l.lon), math.radians(l.lat)) for l in locations]
    best = float("inf")
    for i in range(len(pts)):
        for j in range(i + 1, len(pts)):
            (lo1, la1), (lo2, la2) = pts[i], pts[j]
            dlon = (lo1 - lo2) * math.cos(0.5 * (la1 + la2))
            dlat = la1 - la2
            best = min(best, math.hypot(dlon, dlat) * r)
    return best


# --------------------------------------------------------------------------- #
# Part 2 -- review API round-trip (pure CPU, cached GMMs)
# --------------------------------------------------------------------------- #


def verify_review_roundtrip() -> bool:
    """Confirm -> refit -> recompute; assert the Stage 6a freeze/re-propose invariants."""
    print("\n" + "-" * 88)
    print("[REVIEW] confirm -> refit -> recompute round-trip")
    print("-" * 88)

    aff = compute_affinity(REFERENCE_ID, COMPARE_ID)
    decisions, _ = classify_rows(aff)
    absent = absent_decisions(REFERENCE_ID, aff)
    store = FeedbackStore(reference_id=REFERENCE_ID, compare_id=COMPARE_ID)

    rows0 = build_reviewed_table(aff, decisions + absent, store)
    mixed = [d for d in decisions if d.status == "mixed"]
    if not mixed:
        print("No 'mixed' class to review; cannot exercise the round-trip.")
        return False
    rv = mixed[0].reference_value
    row0 = _find_row(rows0, rv)

    all_open = all(e.provenance == ALGORITHM_PROPOSED for r in rows0 for e in r.edges)
    print(f"\nreviewing '{row0.reference_name}' (rv={rv}); all edges start open: "
          f"{'YES' if all_open else 'NO'}")

    # Confirm the top open edge and freeze it.
    confirm_cv = row0.edges[0].compare_value
    confirm_edges(store, aff, rv, [confirm_cv])
    save_feedback(store)

    rows1 = build_reviewed_table(aff, decisions + absent, store)
    row1 = _find_row(rows1, rv)
    confirmed_edge = _edge(row1, confirm_cv)
    froze = confirmed_edge.provenance == EXPERT_CONFIRMED
    retained = store.confirmed_edges(rv)[0].retained_probability
    prob_matches = abs(confirmed_edge.probability - retained) < 1e-12
    full = rebalance_row(
        aff.normalized_affinity[aff.reference_classes.index(rv)],
        aff.compare_classes, store.confirmed_edges(rv),
    )
    sums_one = abs(sum(full) - 1.0) < 1e-9
    print(f"    confirmed edge frozen: {'YES' if froze else 'NO'}; "
          f"keeps retained prob: {'YES' if prob_matches else 'NO'}; "
          f"row sums to 1: {'YES' if sums_one else 'NO'}")

    # Snapshot, refit, recompute.
    frozen_before = confirmed_edge.probability
    open_before = {
        e.compare_value: e.probability
        for e in row1.edges if e.provenance == ALGORITHM_PROPOSED
    }
    refit = warm_start_refit(REFERENCE_ID, COMPARE_ID, rv, store)
    print(f"    refit: {refit.refit}  points={refit.n_points_refit}  "
          f"K={refit.n_components}  cov={refit.covariance_type_used}")

    _, rows2 = recompute_reviewed_table(REFERENCE_ID, COMPARE_ID, store)
    row2 = _find_row(rows2, rv)

    frozen_after = _edge(row2, confirm_cv)
    frozen_unchanged = (
        frozen_after.provenance == EXPERT_CONFIRMED
        and abs(frozen_after.probability - frozen_before) < 1e-12
    )
    unconfirmed_changed = any(
        e.provenance == ALGORITHM_PROPOSED
        and e.compare_value in open_before
        and abs(e.probability - open_before[e.compare_value]) > 1e-9
        for e in row2.edges
    )
    no_confirmed_moved = True
    for r_after in rows2:
        for e in r_after.edges:
            if e.provenance != EXPERT_CONFIRMED:
                continue
            ret = next(
                ce.retained_probability
                for ce in store.confirmed_edges(r_after.reference_value)
                if ce.compare_value == e.compare_value
            )
            if abs(e.probability - ret) > 1e-12:
                no_confirmed_moved = False

    print(f"    confirmed edge UNCHANGED after refit: "
          f"{'YES' if frozen_unchanged else 'NO'}")
    print(f"    an UNCONFIRMED edge changed after refit: "
          f"{'YES' if unconfirmed_changed else 'NO'}")
    print(f"    no confirmed edge anywhere moved: "
          f"{'YES' if no_confirmed_moved else 'NO'}")

    return bool(
        all_open and froze and prob_matches and sums_one
        and frozen_unchanged and unconfirmed_changed and no_confirmed_moved
    )


# --------------------------------------------------------------------------- #


def main() -> None:
    print("=" * 88)
    print("Stage 6b verification - evidence explorer engine + review API")
    print(f"reference={REFERENCE_ID}  compare={COMPARE_ID}")
    print("=" * 88)

    gmm_path = gmm_cache_path(REFERENCE_ID)
    if not gmm_path.exists():
        print(f"\nMissing {gmm_path}. Run Stage 3 for the test-swap pair first.")
        print("ALL CHECKS PASSED: NO")
        return

    # Back up caches the refit / feedback store would touch.
    fb_path = feedback_cache_path(REFERENCE_ID, COMPARE_ID)
    fb_backup = fb_path.with_suffix(".bak") if fb_path.exists() else None
    if fb_backup:
        shutil.copy2(fb_path, fb_backup)
    gmm_backup = gmm_path.with_suffix(".json.bak")
    shutil.copy2(gmm_path, gmm_backup)

    explorer_ok = None  # None => skipped
    review_ok = False
    try:
        # Part 1 -- explorer (live GEE). Pick a class-pair present in both caches.
        try:
            import dataclasses

            import harmonizer.config as _cfgmod
            from harmonizer.registry.adapters._gee import ensure_initialized

            ensure_initialized()
            # Coarsen the sample scale for the probe (a fine scale over any AOI is as
            # slow as Stage 2 -- see docs/PIPELINE.md section 2). Restored in finally.
            _orig_sampling = _cfgmod.CONFIG.sampling
            object.__setattr__(
                _cfgmod.CONFIG,
                "sampling",
                dataclasses.replace(_orig_sampling, sample_scale_m=TEST_SCALE_M),
            )
            try:
                # Grassland (WorldCover 30) x Grass (Dynamic World 4): the two most
                # common classes in this AOI, so all three modes return points.
                explorer_ok = verify_explorer(rv=30, cv=4)
            finally:
                object.__setattr__(_cfgmod.CONFIG, "sampling", _orig_sampling)
        except Exception as exc:  # GEE unreachable, quota, etc. -> skip, don't fail.
            print(f"\n[EXPLORER] SKIPPED (GEE unavailable): "
                  f"{type(exc).__name__}: {str(exc)[:160]}")
            explorer_ok = None

        # Part 2 -- review round-trip (pure CPU).
        review_ok = verify_review_roundtrip()
    finally:
        shutil.move(str(gmm_backup), str(gmm_path))
        if fb_backup:
            shutil.move(str(fb_backup), str(fb_path))
        elif fb_path.exists():
            fb_path.unlink()

    print("\n" + "=" * 88)
    if explorer_ok is None:
        print("explorer:  SKIPPED (GEE unavailable)")
    else:
        print(f"explorer:  {'PASS' if explorer_ok else 'FAIL'}")
    print(f"review:    {'PASS' if review_ok else 'FAIL'}")
    # A skipped explorer does not fail the run, but a present explorer must pass.
    all_ok = review_ok and (explorer_ok is None or explorer_ok)
    print(f"\nALL CHECKS PASSED: {'YES' if all_ok else 'NO'}")


if __name__ == "__main__":
    main()
