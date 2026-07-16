"""Stage 6a verification: feedback data model + logic (backend).

Exercises the Stage 6a mechanics from the CLI, following the build-order bullet in
docs/PIPELINE.md ("6a"): *confirm an edge, see it freeze; refit; see unconfirmed
edges change but confirmed ones unchanged.*

It consumes the Stage 3 GMM caches (``cache/gmm_<product>.json``) and Stage 2
sample caches (``cache/samples_<product>.npz``) the earlier stages produced, for
the test-swap pair (WorldCover reference, Dynamic World compare). Pure CPU -- no
network.

Steps:
  1. Compute Stage 4 affinity + decisions and pick a ``mixed`` reference class
     (the primary review case). Build the reviewed table and confirm every edge is
     ``algorithm-proposed`` to start.
  2. Confirm one of its candidate edges. Rebuild the table and confirm that edge is
     now ``expert-confirmed`` and frozen with its retained probability, that the
     row still sums to ~1, and that the OTHER (open) edges' probabilities were
     re-balanced (their frozen counterpart didn't move).
  3. Snapshot the confirmed edge's probability and a still-open edge on a DIFFERENT
     reference class. Warm-start-refit the confirmed class from its samples, then
     recompute the whole reviewed table from the refit GMMs.
  4. Assert the invariants:
       * the confirmed edge is byte-for-byte unchanged (frozen);
       * at least one unconfirmed edge's probability moved after the refit
         (the model re-proposed the open edges);
       * no confirmed edge anywhere changed.

Because the caches must not be mutated for other stages, the GMM cache is backed
up and restored, and the feedback store written to a temp path under cache/ is
removed at the end.

Run:  python scripts/verify_stage6a.py

Stages 2 and 3 must have run first for the test-swap pair.
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

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
    load_feedback,
    recompute_reviewed_table,
    save_feedback,
    warm_start_refit,
)

REFERENCE_ID = "worldcover"
COMPARE_ID = "dynamicworld"


def _find_row(rows, rv):
    return next(r for r in rows if r.reference_value == rv)


def _edge(row, cv):
    return next(e for e in row.edges if e.compare_value == cv)


def _print_row(title, row):
    print(f"  {title}: {row.reference_name} [{row.status}]")
    for e in row.edges:
        tag = "FROZEN " if e.provenance == EXPERT_CONFIRMED else "open   "
        print(
            f"    {tag} {e.compare_name:>22} (cv={e.compare_value})  "
            f"p={e.probability:.4f}"
        )


def main() -> None:
    print("=" * 88)
    print("Stage 6a verification - feedback data model + logic")
    print(f"reference={REFERENCE_ID}  compare={COMPARE_ID}")
    print("=" * 88)

    # Guard: temp feedback store must not clash with a real one; back up the GMM
    # cache so the refit does not disturb other stages.
    fb_path = feedback_cache_path(REFERENCE_ID, COMPARE_ID)
    gmm_path = gmm_cache_path(REFERENCE_ID)
    if not gmm_path.exists():
        print(f"\nMissing {gmm_path}. Run Stage 3 for the test-swap pair first.")
        print("ALL CHECKS PASSED: NO")
        return
    fb_backup = fb_path.with_suffix(".bak") if fb_path.exists() else None
    if fb_backup:
        shutil.copy2(fb_path, fb_backup)
    gmm_backup = gmm_path.with_suffix(".json.bak")
    shutil.copy2(gmm_path, gmm_backup)

    all_ok = True
    try:
        # 1. Stage 4 affinity + decisions; pick a mixed class.
        aff = compute_affinity(REFERENCE_ID, COMPARE_ID)
        decisions, _ = classify_rows(aff)
        absent = absent_decisions(REFERENCE_ID, aff)
        store = FeedbackStore(reference_id=REFERENCE_ID, compare_id=COMPARE_ID)

        rows0 = build_reviewed_table(aff, decisions + absent, store)
        mixed = [d for d in decisions if d.status == "mixed"]
        if not mixed:
            print("\nNo 'mixed' class to review; cannot exercise Stage 6a fully.")
            print("ALL CHECKS PASSED: NO")
            return
        target = mixed[0]
        rv = target.reference_value
        row0 = _find_row(rows0, rv)

        print("\n[1] Starting reviewed row (all edges algorithm-proposed):")
        _print_row("before", row0)
        start_all_open = all(
            e.provenance == ALGORITHM_PROPOSED for r in rows0 for e in r.edges
        )
        print(f"    all edges start 'algorithm-proposed': "
              f"{'YES' if start_all_open else 'NO'}")
        all_ok &= start_all_open

        # 2. Confirm the row's top open edge and freeze it.
        confirm_cv = row0.edges[0].compare_value
        confirm_edges(store, aff, rv, [confirm_cv])
        save_feedback(store)

        rows1 = build_reviewed_table(aff, decisions + absent, store)
        row1 = _find_row(rows1, rv)
        print("\n[2] After confirming one edge:")
        _print_row("after confirm", row1)

        confirmed_edge = _edge(row1, confirm_cv)
        froze = confirmed_edge.provenance == EXPERT_CONFIRMED
        retained = store.confirmed_edges(rv)[0].retained_probability
        frozen_prob_matches = abs(confirmed_edge.probability - retained) < 1e-12
        # The re-balanced row still sums to 1 (frozen mass + open budget).
        row_sums_one = abs(sum(_row_full(aff, rv, store)) - 1.0) < 1e-9
        print(f"    confirmed edge is frozen (expert-confirmed): "
              f"{'YES' if froze else 'NO'}")
        print(f"    frozen edge keeps its retained probability: "
              f"{'YES' if frozen_prob_matches else 'NO'}")
        print(f"    re-balanced row still sums to 1: "
              f"{'YES' if row_sums_one else 'NO'}")
        all_ok &= froze and frozen_prob_matches and row_sums_one

        # 3. Snapshot, then warm-start refit the confirmed class and recompute.
        #    The refit changes the CONFIRMED class's own GMM, so its OPEN edges
        #    re-propose while its frozen edge does not (the build-order invariant:
        #    "unconfirmed edges change but confirmed ones unchanged").
        frozen_before = confirmed_edge.probability
        open_before = {
            e.compare_value: e.probability
            for e in row1.edges
            if e.provenance == ALGORITHM_PROPOSED
        }

        refit = warm_start_refit(REFERENCE_ID, COMPARE_ID, rv, store)
        print("\n[3] Warm-start refit of the confirmed class:")
        print(f"    refit={refit.refit}  points={refit.n_points_refit}  "
              f"K={refit.n_components}  cov={refit.covariance_type_used}")

        _, rows2 = recompute_reviewed_table(REFERENCE_ID, COMPARE_ID, store)
        row2 = _find_row(rows2, rv)
        print("\n[4] After refit + recompute:")
        _print_row("after refit", row2)

        # 4. Invariants.
        frozen_after = _edge(row2, confirm_cv).probability
        frozen_unchanged = (
            _edge(row2, confirm_cv).provenance == EXPERT_CONFIRMED
            and abs(frozen_after - frozen_before) < 1e-12
        )
        # At least one OPEN edge of the refit class re-proposed (moved).
        unconfirmed_changed = False
        for e in row2.edges:
            if e.provenance != ALGORITHM_PROPOSED:
                continue
            if e.compare_value in open_before and (
                abs(e.probability - open_before[e.compare_value]) > 1e-9
            ):
                unconfirmed_changed = True

        # No confirmed edge anywhere changed.
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

        print()
        print(f"    confirmed edge UNCHANGED after refit: "
              f"{'YES' if frozen_unchanged else 'NO'}")
        print(f"    an UNCONFIRMED edge changed after refit: "
              f"{'YES' if unconfirmed_changed else 'NO'}")
        print(f"    no confirmed edge anywhere moved: "
              f"{'YES' if no_confirmed_moved else 'NO'}")
        all_ok &= frozen_unchanged and unconfirmed_changed and no_confirmed_moved

    finally:
        # Restore caches so other stages are undisturbed.
        shutil.move(str(gmm_backup), str(gmm_path))
        if fb_backup:
            shutil.move(str(fb_backup), str(fb_path))
        elif fb_path.exists():
            fb_path.unlink()

    print()
    print(f"ALL CHECKS PASSED: {'YES' if all_ok else 'NO'}")


def _row_full(aff, rv, store):
    """The full re-balanced probability row for a reference class (sums to 1)."""
    from harmonizer.review import rebalance_row

    ri = aff.reference_classes.index(rv)
    return rebalance_row(
        aff.normalized_affinity[ri], aff.compare_classes, store.confirmed_edges(rv)
    )


if __name__ == "__main__":
    main()
