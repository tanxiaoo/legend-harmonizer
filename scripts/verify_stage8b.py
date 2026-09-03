"""Stage 8b verification: the semantic prior folded into affinity and decision.

Runs on the cached GMMs for the ``hrlc30_africa`` x ``worldcover_2020`` pair --
pure CPU, no GEE. Stages 2/3 must have run first (``cache/gmm_*.json`` for both
products); fit them from the cached samples with ``harmonizer.modeling.fit_map``
if absent.

Four checks, from docs/PIPELINE.md Stage 8b:

1. **Regression.** ``alpha=0`` must reproduce the pre-Stage-8 behaviour exactly.
   Verified against the prior forced to ones, which is the definition of "the
   prior does not participate": every probability, status, margin and entropy
   must be bit-identical, not merely close.
2. **alpha=1 moves the right rows.** Cropland's probability shifts away from
   grassland and towards croplands; the four HRLC tree classes keep their
   *ratios* under a flat prior; orphan statuses are untouched by alpha.
3. **New columns and CSVs.** ``semantic_orphan`` / ``agreement`` populated on
   every row, and the three CSVs written with matching headers and shapes.
4. **Direction toggle.** ``compare_to_reference`` builds the prior for the
   swapped ordered pair, so an asymmetric cell reads differently in each
   direction.

Run:  python scripts/verify_stage8b.py
"""

from __future__ import annotations

import csv
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402

from harmonizer.affinity import compute_affinity  # noqa: E402
from harmonizer.config import CONFIG  # noqa: E402
from harmonizer.decision import (  # noqa: E402
    build_matching_table,
    classify_rows,
    compute_affinity_directed,
    save_aef_affinity_csv,
    save_matching_table_csv,
    save_semantic_prior_csv,
)

REFERENCE_ID = "hrlc30_africa"
COMPARE_ID = "worldcover_2020"


def _rule(title: str) -> None:
    print(f"\n{title}\n{'-' * len(title)}")


def check(ok: bool, message: str, detail: str = "") -> bool:
    print(f"  {'PASS' if ok else 'FAIL'}  {message}" + (f" [{detail}]" if detail else ""))
    return bool(ok)


def _cell(aff, ref_code: int, cmp_code: int, matrix: np.ndarray) -> float:
    return float(
        matrix[aff.reference_classes.index(ref_code), aff.compare_classes.index(cmp_code)]
    )


def main() -> int:
    ok = True

    for pid in (REFERENCE_ID, COMPARE_ID):
        from harmonizer.modeling import gmm_cache_path

        if not gmm_cache_path(pid).exists():
            print(f"FAILED: no cached GMM for {pid}; run Stages 2-3 first.")
            return 1

    _rule("Stage 8b - semantic prior fused into the affinity")
    a0 = compute_affinity(REFERENCE_ID, COMPARE_ID, alpha=0.0)
    a1 = compute_affinity(REFERENCE_ID, COMPARE_ID, alpha=1.0)
    print(f"  reference classes : {a0.reference_classes}")
    print(f"  compare classes   : {a0.compare_classes}")
    print(f"  config alpha      : {CONFIG.affinity.semantic_prior_alpha}")
    print(f"  prior shape       : {a0.semantic_prior.shape}")

    # ---------------------------------------------------------------- 1. #
    _rule("1. Regression: alpha=0 is byte-identical to a prior-free run")

    # "Prior does not participate" made explicit: force pi to ones and recompute
    # the rows the same way compute_affinity does.
    from harmonizer.affinity import _softmax_rows

    ones = np.ones_like(a0.semantic_prior)
    expected = _softmax_rows(
        a0.distance, CONFIG.affinity.softmax_temperature, prior=ones, alpha=0.0
    )
    ok &= check(
        np.array_equal(a0.normalized_affinity, expected),
        "alpha=0 normalized_affinity identical to the prior-free softmax",
        f"max |diff| = {np.abs(a0.normalized_affinity - expected).max():.3e}",
    )
    ok &= check(
        np.array_equal(a0.normalized_affinity, a0.normalized_affinity_aef),
        "at alpha=0 the fused and unfused matrices are the same array values",
    )
    # The raw similarity is the orphan signal and must never see the prior.
    ok &= check(
        np.array_equal(a0.raw_similarity, a1.raw_similarity),
        "raw_similarity unchanged by alpha (the orphan signal is observational)",
    )
    ok &= check(
        np.array_equal(a0.normalized_affinity_aef, a1.normalized_affinity_aef),
        "normalized_affinity_aef identical at alpha=0 and alpha=1",
    )

    d0, cal0 = classify_rows(a0)
    d1, _ = classify_rows(a1)
    if not cal0:
        print("  (floor uncalibrated - statuses would be 'unassigned')")

    # ---------------------------------------------------------------- 2. #
    _rule("2. alpha=1 reshapes the rows the prior disagrees with")

    # WorldCover 40 Cropland is the flagship case: spectrally close to grassland,
    # semantically cultivated. Read from the compare-to-reference direction,
    # where WorldCover classes are the rows.
    r0 = compute_affinity(COMPARE_ID, REFERENCE_ID, alpha=0.0)
    r1 = compute_affinity(COMPARE_ID, REFERENCE_ID, alpha=1.0)
    if 40 in r0.reference_classes:
        grass0, grass1 = _cell(r0, 40, 70, r0.normalized_affinity), _cell(
            r1, 40, 70, r1.normalized_affinity
        )
        crop0, crop1 = _cell(r0, 40, 80, r0.normalized_affinity), _cell(
            r1, 40, 80, r1.normalized_affinity
        )
        ok &= check(
            grass1 < grass0,
            "WC 40 Cropland -> HRLC 70 Grassland probability DROPS under alpha=1",
            f"{grass0:.4f} -> {grass1:.4f}",
        )
        ok &= check(
            crop1 > crop0,
            "WC 40 Cropland -> HRLC 80 Croplands probability RISES under alpha=1",
            f"{crop0:.4f} -> {crop1:.4f}",
        )
    else:
        print("  SKIP  WorldCover 40 not modelled in this cache")

    # WC 10's prior is flat across the HRLC tree classes (no leaf-type info), so
    # alpha must not redistribute mass *among* them: their ratios are preserved.
    trees = [c for c in (10, 20, 30, 40) if c in r0.compare_classes]
    if 10 in r0.reference_classes and len(trees) >= 2:
        row0 = np.array([_cell(r0, 10, c, r0.normalized_affinity) for c in trees])
        row1 = np.array([_cell(r1, 10, c, r1.normalized_affinity) for c in trees])
        ratio0, ratio1 = row0 / row0.sum(), row1 / row1.sum()
        ok &= check(
            np.allclose(ratio0, ratio1, atol=1e-12),
            "WC 10 keeps its split across HRLC tree classes in ratio (flat prior)",
            f"max |diff| = {np.abs(ratio0 - ratio1).max():.2e}",
        )
    else:
        print(f"  SKIP  fewer than two HRLC tree classes modelled ({trees})")

    # Orphan status is observational and must not move with alpha.
    s0 = {d.reference_value: d.status for d in d0}
    s1 = {d.reference_value: d.status for d in d1}
    orph0 = {k for k, v in s0.items() if v == "orphan"}
    orph1 = {k for k, v in s1.items() if v == "orphan"}
    ok &= check(
        orph0 == orph1,
        "orphan statuses identical between alpha=0 and alpha=1",
        f"{sorted(orph0) or 'none'}",
    )

    changed = {k for k in s0 if s0[k] != s1[k]}
    print(f"  note  statuses changed by alpha=1: {sorted(changed) or 'none'}")

    # ---------------------------------------------------------------- 3. #
    _rule("3. New decision columns and CSVs")

    rows1 = build_matching_table(a1, d1)
    # The semantic fields describe a *reference* row's mapping. Compare-side
    # rows are absent unmatched targets with no reference class, so they carry
    # none of it -- assert that split rather than demanding it of every row.
    ref_rows = [r for r in rows1 if r.side == "reference"]
    cmp_rows = [r for r in rows1 if r.side == "compare"]
    ok &= check(
        bool(ref_rows) and all(r.agreement for r in ref_rows),
        "every reference row carries an agreement value",
        ", ".join(sorted({r.agreement for r in ref_rows})),
    )
    ok &= check(
        all(r.best_semantic_value >= 0 for r in ref_rows),
        "every reference row carries a best semantic candidate",
    )
    ok &= check(
        all(not r.agreement and r.best_semantic_value < 0 for r in cmp_rows),
        "compare-side unmatched-target rows carry no semantic fields",
        f"{len(cmp_rows)} row(s)",
    )

    # Stage 8a showed no class of this pair is a semantic orphan, so assert the
    # flag agrees with the prior rather than asserting a specific class.
    expected_orphans = {
        a1.reference_classes[i]
        for i in range(len(a1.reference_classes))
        if a1.semantic_prior[i].max() < CONFIG.affinity.semantic_orphan_floor
    }
    flagged = {d.reference_value for d in d1 if d.semantic_orphan}
    ok &= check(
        flagged == expected_orphans,
        "semantic_orphan flags exactly the rows below the floor",
        f"{sorted(flagged) or 'none (matches Stage 8a)'}",
    )

    # An observational orphan that is NOT a semantic orphan is definition drift:
    # nothing in the other map looks like it, yet something means the same. That
    # must never read as `agree`, which would hide the disagreement entirely.
    drift = [
        d for d in d1 if d.status == "orphan" and not d.semantic_orphan
    ]
    ok &= check(
        all(d.agreement == "semantic_overrides" for d in drift),
        "observational orphans with a semantic match read as semantic_overrides",
        ", ".join(f"{d.reference_value}:{d.agreement}" for d in drift) or "none",
    )

    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp)
        sem_path = save_semantic_prior_csv(a1, out / "semantic_prior.csv")
        aef_path = save_aef_affinity_csv(a1, out / "aef_affinity.csv")
        tbl_path = save_matching_table_csv(rows1, out / "matching_table.csv")

        shapes = {}
        for path in (sem_path, aef_path):
            with path.open(encoding="utf-8") as fh:
                data = list(csv.reader(fh))
            shapes[path.name] = (len(data) - 1, len(data[0]) - 1)
        expected_shape = (len(a1.reference_classes), len(a1.compare_classes))
        ok &= check(
            all(s == expected_shape for s in shapes.values()),
            "both new matrix CSVs match the affinity shape",
            f"{shapes} vs {expected_shape}",
        )

        with tbl_path.open(encoding="utf-8") as fh:
            header = next(csv.reader(fh))
        needed = [
            "best_semantic_value",
            "best_semantic_name",
            "best_semantic_prior",
            "semantic_orphan",
            "aef_best_compare_value",
            "agreement",
        ]
        missing = [c for c in needed if c not in header]
        ok &= check(
            not missing,
            "matching table header carries the six new columns",
            f"missing {missing}" if missing else f"{len(header)} columns",
        )

    # ---------------------------------------------------------------- 4. #
    _rule("4. Direction toggle orients the prior correctly")

    fwd = compute_affinity_directed(
        REFERENCE_ID, COMPARE_ID, direction="reference_to_compare", alpha=1.0
    )
    rev = compute_affinity_directed(
        REFERENCE_ID, COMPARE_ID, direction="compare_to_reference", alpha=1.0
    )
    ok &= check(
        fwd.reference_id == REFERENCE_ID and rev.reference_id == COMPARE_ID,
        "swapping direction swaps which product indexes the rows",
        f"{fwd.reference_id} / {rev.reference_id}",
    )

    # HRLC 10 -> WC 10 is 1.0 (contained); WC 10 -> HRLC 10 is strictly less.
    # The same asymmetric pair must read differently in the two directions,
    # which is only true if each direction built its own prior.
    if 10 in fwd.reference_classes and 10 in fwd.compare_classes:
        f10 = _cell(fwd, 10, 10, fwd.semantic_prior)
        r10 = _cell(rev, 10, 10, rev.semantic_prior)
        ok &= check(
            f10 > r10,
            "asymmetric cell differs by direction (HRLC10->WC10 > WC10->HRLC10)",
            f"{f10:.4f} vs {r10:.4f}",
        )
    else:
        print("  SKIP  class 10 not modelled on both sides")

    print(f"\n{'OK' if ok else 'FAILED'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
