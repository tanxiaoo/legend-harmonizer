"""Stage 4 verification: affinity matrix, decision, and matching table.

Consumes the fitted GMMs Stage 3 cached (``cache/gmm_<product>.json``) for the
reference (test-swap ESA WorldCover) and compare (Dynamic World) maps, and:

  1. Runs a **self-check** that the MW2 mixture distance reduces to the plain
     Bures-Wasserstein distance when both GMMs have a single component -- the
     sanity check called for in docs/PIPELINE.md, Stage 4.
  2. Computes the affinity matrices and **emits the raw-similarity distribution**
     (per-row best plus min/median/max and a histogram) needed to calibrate the
     absolute-affinity floor -- a required output of the stage.
  3. Runs the decision **once with the floor uncalibrated** (``None``) to show the
     guard: the orphan test is skipped and statuses come back ``unassigned``.
  4. Re-runs the decision with the **calibrated values now in CONFIG** (floor
     0.60, entropy threshold 0.65, softmax temperature 0.25 -- read off this same
     raw-similarity distribution) and prints the resulting statuses.
  5. Saves the three CSVs (``raw_similarity.csv``, ``normalized_affinity.csv``,
     ``matching_table.csv``) to ``cache/`` and prints the matching table.
  6. Demonstrates the **direction toggle** by printing the compare-to-reference
     best matches too.

Checks to eyeball (docs/PIPELINE.md, Stage 4 -> Verification): obvious pairs read
correctly (water->water, forest/tree-cover->trees as ``strong``), at least some
classes come back ``mixed`` with a sensible split, and entropy/status are
consistent with the probabilities.

Run:  python scripts/verify_stage4.py

Stage 3 must have run first (``cache/gmm_worldcover.json`` and
``cache/gmm_dynamicworld.json`` must exist). Pure CPU -- no network.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from harmonizer.affinity import (  # noqa: E402
    ClassComponents,
    bures_wasserstein_sq,
    class_name,
    compute_affinity,
    mixture_wasserstein,
)
from harmonizer.config import CONFIG  # noqa: E402
from harmonizer.decision import (  # noqa: E402
    absent_decisions,
    build_matching_table,
    classify_rows,
    compute_affinity_directed,
    raw_similarity_summary,
    save_matching_table_csv,
    save_normalized_affinity_csv,
    save_raw_similarity_csv,
)

REFERENCE_ID = "worldcover"
COMPARE_ID = "dynamicworld"


# --------------------------------------------------------------------------- #
# 1. Single-component sanity check: MW2 reduces to Bures-Wasserstein
# --------------------------------------------------------------------------- #


def _single_component_check() -> bool:
    rng = np.random.default_rng(0)
    d = 5
    m1 = rng.normal(size=d)
    m2 = rng.normal(size=d)
    a1 = rng.normal(size=(d, d))
    a2 = rng.normal(size=(d, d))
    cov1 = a1 @ a1.T + np.eye(d)  # SPD
    cov2 = a2 @ a2.T + np.eye(d)

    ca = ClassComponents(
        product_id="a", class_value=0,
        weights=np.array([1.0]), means=m1[None, :], covariances=cov1[None, :, :],
        low_confidence=False, covariance_type_used="full", n_points=1000, status="ok",
    )
    cb = ClassComponents(
        product_id="b", class_value=0,
        weights=np.array([1.0]), means=m2[None, :], covariances=cov2[None, :, :],
        low_confidence=False, covariance_type_used="full", n_points=1000, status="ok",
    )

    mw = mixture_wasserstein(ca, cb)
    bw = float(np.sqrt(bures_wasserstein_sq(m1, cov1, m2, cov2)))
    ok = abs(mw - bw) < 1e-8

    print("Self-check: single-component MW2 == Bures-Wasserstein")
    print(f"  MW2 distance          = {mw:.10f}")
    print(f"  Bures-Wasserstein     = {bw:.10f}")
    print(f"  match (|diff| < 1e-8) : {'YES' if ok else 'NO'}")
    return ok


# --------------------------------------------------------------------------- #
# Printing helpers
# --------------------------------------------------------------------------- #


def _print_raw_distribution(summary) -> None:
    print()
    print("Raw-similarity distribution (for calibrating the absolute-affinity floor)")
    print(
        f"  whole matrix: min={summary.all_min:.4f}  "
        f"median={summary.all_median:.4f}  max={summary.all_max:.4f}"
    )
    print("  per-row best raw similarity:")
    print(
        "    "
        + "  ".join(f"{x:.3f}" for x in sorted(summary.best_per_row, reverse=True))
    )
    print("  histogram of all raw similarities (bins 0..1):")
    for k, c in enumerate(summary.histogram_counts):
        lo = summary.histogram_edges[k]
        hi = summary.histogram_edges[k + 1]
        bar = "#" * c
        print(f"    [{lo:.1f},{hi:.1f})  {c:>4}  {bar}")


def _print_decisions(decisions, title: str) -> None:
    print()
    print(title)
    header = (
        f"{'ref':>4}  {'reference':>22}  {'status':>10}  {'best compare':>18}  "
        f"{'raw':>6}  {'prob':>6}  {'margin':>6}  {'H':>5}  low-conf"
    )
    print(header)
    print("-" * len(header))
    for d in decisions:
        print(
            f"{d.reference_value:>4}  {d.reference_name:>22}  {d.status:>10}  "
            f"{d.best_compare_name:>18}  {d.best_raw_similarity:>6.3f}  "
            f"{d.best_probability:>6.3f}  {d.margin:>6.3f}  {d.entropy:>5.2f}  "
            f"{'yes' if d.low_confidence else 'no'}"
        )


def _print_matching_table(rows) -> None:
    print()
    print("Matching table (primary deliverable)")
    for r in rows:
        parts = ", ".join(
            f"{nm} ({p:.2f})" for nm, p in zip(r.compare_names, r.probabilities)
        )
        lc = " [low-confidence]" if r.reference_low_confidence else ""
        print(
            f"  {r.reference_name:>22} -> {r.status:>10}: "
            f"{parts if parts else '(no candidate)'}{lc}"
        )


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #


def main() -> None:
    print("=" * 92)
    print("Stage 4 verification - affinity matrix, decision, and matching table")
    print(
        f"reference={REFERENCE_ID}  compare={COMPARE_ID}  "
        f"entropy threshold={CONFIG.affinity.high_entropy_threshold}  "
        f"floor(config)={CONFIG.affinity.absolute_affinity_floor}"
    )
    print("=" * 92)

    check_ok = _single_component_check()

    try:
        aff = compute_affinity(REFERENCE_ID, COMPARE_ID)
    except FileNotFoundError as exc:
        print()
        print(str(exc))
        print("ALL CHECKS PASSED: NO")
        return

    # 2. Emit the raw-similarity distribution.
    summary = raw_similarity_summary(aff)
    _print_raw_distribution(summary)

    # 3. Decision with the floor UNCALIBRATED (guard demonstration).
    print()
    print("-" * 92)
    print("Pass 1 - floor uncalibrated (None): orphan test skipped, statuses unassigned")
    print("-" * 92)
    import warnings

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        decisions_uncal, calibrated = classify_rows(aff, floor=None)
    guard_ok = (not calibrated) and all(
        d.status == "unassigned" for d in decisions_uncal
    )
    if caught:
        print(f"  warning raised: {caught[0].message}")
    print(f"  guard held (all 'unassigned', orphan test skipped): "
          f"{'YES' if guard_ok else 'NO'}")

    # 4. Re-run with the calibrated floor now in CONFIG (set from this very
    #    distribution: raw similarities bunch 0.47-0.91, so 0.60 sits below every
    #    genuine best match). classify_rows reads CONFIG.affinity by default.
    calibrated_floor = CONFIG.affinity.absolute_affinity_floor
    print()
    print("-" * 92)
    print(
        f"Pass 2 - calibrated floor (from CONFIG) = {calibrated_floor}  "
        f"margin threshold = {CONFIG.affinity.margin_threshold}  "
        f"softmax T = {CONFIG.affinity.softmax_temperature}  "
        f"(entropy reported, not the classifier)"
    )
    print("-" * 92)
    decisions, calibrated = classify_rows(aff)
    _print_decisions(decisions, "Reference-to-compare decisions")

    # 5. Build and save the matching table + both matrices.
    absent = absent_decisions(REFERENCE_ID, aff)
    rows = build_matching_table(aff, decisions, include_absent=absent)
    _print_matching_table(rows)

    raw_path = save_raw_similarity_csv(aff)
    norm_path = save_normalized_affinity_csv(aff)
    table_path = save_matching_table_csv(rows)
    print()
    print("Saved artifacts:")
    print(f"  raw similarity     -> {raw_path}")
    print(f"  normalized affinity-> {norm_path}")
    print(f"  matching table     -> {table_path}")

    # 6. Direction toggle: compare-to-reference best matches.
    aff_rev = compute_affinity_directed(
        REFERENCE_ID, COMPARE_ID, direction="compare_to_reference"
    )
    dec_rev, _ = classify_rows(aff_rev)
    _print_decisions(
        dec_rev, "Direction toggle - compare-to-reference decisions"
    )

    # Consistency: strong/mixed must track the margin threshold (the classifier).
    thr = CONFIG.affinity.margin_threshold
    consistent = all(
        (d.status != "strong" or d.margin >= thr)
        and (d.status != "mixed" or d.margin < thr)
        for d in decisions
        if d.status in ("strong", "mixed")
    )
    has_mixed = any(d.status == "mixed" for d in decisions)
    has_strong = any(d.status == "strong" for d in decisions)
    print()
    print(f"margin/status consistent with margin threshold {thr}: "
          f"{'YES' if consistent else 'NO'}")
    print(f"at least one 'mixed' and one 'strong' class present: "
          f"{'YES' if (has_mixed and has_strong) else 'NO'}")

    print()
    print(
        "Eyeball: confirm obvious pairs read correctly (water->Water as 'strong'), "
        "that clean single winners (Tree cover->Trees, Cropland->Crops) now read "
        "'strong' by margin, and that genuine two-way ties come back 'mixed'."
    )
    print()
    all_ok = check_ok and guard_ok and consistent
    print(f"ALL AUTOMATED CHECKS PASSED: {'YES' if all_ok else 'NO'}")


if __name__ == "__main__":
    main()
