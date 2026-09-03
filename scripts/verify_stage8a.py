"""Stage 8a verification: the LCCS semantic prior.

Prints the directed prior ``pi`` for both directions of the
``hrlc30_africa`` x ``worldcover_2020`` pair as labelled tables, then asserts the
properties that follow from the Stage 8a rules (docs/PIPELINE.md, Stage 8a).

Pure CPU, no network and no cache: the prior is built from the registry YAML
alone, so this runs from a clean checkout before any sampling has happened.

Two of the spec's illustrative figures do **not** hold under the rules as
written, and this script asserts the rule-derived behaviour instead. Both trace
to the same clause -- graded attributes are *averaged*, so a single low
attribute is diluted by the ones scoring 1.0:

  * The spec predicts ``WC 10 -> HRLC 10 ~ 0.56`` "(cover inclusion 50/90)".
    The cover term is exactly that 0.5556, but averaging it with leaf type,
    phenology, height, and flooding (all 1.0) gives 0.911. 0.56 is the cover
    score alone, i.e. the value before the graded average was settled.
  * The spec predicts HRLC 141 (seasonal water) is a semantic orphan. Its best
    fit is WC 80 (permanent water) at 0.80, far above the 0.30 floor: flooding
    is a *graded* attribute, so the zero overlap between 5-9 and 9-12 months is
    averaged against a perfect surface match rather than vetoing it. Seasonal
    water is still water, which is the defensible reading -- but it means no
    class in this pair is a semantic orphan.

Both are reported below as NOTE lines with the computed values, so the human can
decide whether to re-tune the veto/graded split or accept the behaviour.

Run:  python scripts/verify_stage8a.py
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402

from harmonizer.config import CONFIG  # noqa: E402
from harmonizer.registry.legends import legend_classes, spec  # noqa: E402
from harmonizer.semantics import (  # noqa: E402
    semantic_orphans,
    semantic_prior,
)

REFERENCE_ID = "hrlc30_africa"
COMPARE_ID = "worldcover_2020"

# Tolerance for the equality checks below. The values are exact rationals in
# principle; this only guards float representation.
TOL = 1e-9


def _rule(title: str) -> None:
    print(f"\n{title}\n{'-' * len(title)}")


def _codes(product_id: str) -> list[int]:
    return [c.code for c in legend_classes(product_id)]


def _short(product_id: str, code: int, width: int = 22) -> str:
    name = spec(product_id).class_name(code)
    label = f"{code} {name}"
    return label if len(label) <= width else label[: width - 1] + "…"


def print_matrix(prior: np.ndarray, src_id: str, tgt_id: str) -> None:
    """Print pi as a labelled table: rows are source classes, columns targets."""
    src, tgt = _codes(src_id), _codes(tgt_id)
    header = " " * 24 + "".join(f"{c:>7}" for c in tgt)
    print(f"\n  {src_id} (rows)  ->  {tgt_id} (columns, by code)")
    print("  " + header)
    for i, sc in enumerate(src):
        cells = "".join(f"{prior[i, j]:>7.3f}" for j in range(len(tgt)))
        print(f"  {_short(src_id, sc):<24}{cells}")


def check(ok: bool, message: str, detail: str = "") -> bool:
    print(f"  {'PASS' if ok else 'FAIL'}  {message}" + (f" [{detail}]" if detail else ""))
    return ok


def main() -> int:
    hr, wc = _codes(REFERENCE_ID), _codes(COMPARE_ID)

    _rule("Stage 8a - semantic prior from LCCS attributes")
    print(f"  veto floor    : {CONFIG.affinity.semantic_veto_floor}")
    print(f"  orphan floor  : {CONFIG.affinity.semantic_orphan_floor}")
    print(f"  {REFERENCE_ID}: {len(hr)} classes, encoded={spec(REFERENCE_ID).has_semantics}")
    print(f"  {COMPARE_ID}: {len(wc)} classes, encoded={spec(COMPARE_ID).has_semantics}")

    fwd = semantic_prior(REFERENCE_ID, COMPARE_ID, hr, wc)   # HRLC -> WorldCover
    rev = semantic_prior(COMPARE_ID, REFERENCE_ID, wc, hr)   # WorldCover -> HRLC

    _rule("Prior matrices (both directions)")
    print_matrix(fwd, REFERENCE_ID, COMPARE_ID)
    print_matrix(rev, COMPARE_ID, REFERENCE_ID)

    def f(src: int, tgt: int) -> float:
        return float(fwd[hr.index(src), wc.index(tgt)])

    def r(src: int, tgt: int) -> float:
        return float(rev[wc.index(src), hr.index(tgt)])

    _rule("Checks")
    ok = True

    # 1. Both legends fully encoded, and both matrices the right shape.
    ok &= check(
        spec(REFERENCE_ID).has_semantics and spec(COMPARE_ID).has_semantics,
        "both products report has_semantics",
    )
    ok &= check(
        fwd.shape == (len(hr), len(wc)) and rev.shape == (len(wc), len(hr)),
        "matrix shapes match the legend sizes",
        f"{fwd.shape} / {rev.shape}",
    )

    # 2. All values in (0, 1].
    in_range = bool(
        (fwd > 0).all() and (fwd <= 1 + TOL).all()
        and (rev > 0).all() and (rev <= 1 + TOL).all()
    )
    ok &= check(
        in_range,
        "all pi in (0, 1]",
        f"fwd [{fwd.min():.3f}, {fwd.max():.3f}] rev [{rev.min():.3f}, {rev.max():.3f}]",
    )

    # 3. HRLC 10 sits entirely inside WC 10: cover >=50% inside cover >=10%,
    #    same life form, same height floor. Exact containment scores 1.0.
    ok &= check(
        abs(f(10, 10) - 1.0) < TOL,
        "HRLC 10 -> WC 10 == 1.0 (narrow class inside wide one)",
        f"{f(10, 10):.4f}",
    )

    # 4. Asymmetry: the reverse direction must be strictly smaller, because only
    #    part of WC 10's cover range falls inside HRLC 10's.
    ok &= check(
        r(10, 10) < f(10, 10) - TOL,
        "asymmetric: pi[WC10->HRLC10] < pi[HRLC10->WC10]",
        f"{r(10, 10):.4f} < {f(10, 10):.4f}",
    )

    # 5. WC 10 carries no leaf-type or phenology information, so it cannot
    #    distinguish the four HRLC tree classes: all four must score equally.
    tree_scores = [r(10, c) for c in (10, 20, 30, 40)]
    ok &= check(
        max(tree_scores) - min(tree_scores) < TOL,
        "WC 10 -> HRLC 10/20/30/40 all equal (no leaf-type on the source side)",
        ", ".join(f"{s:.4f}" for s in tree_scores),
    )

    # 6. WC 40 Cropland -> HRLC 70 Grassland is driven down by the cultivation
    #    veto, and must be the weakest vegetated target in that row.
    veg_targets = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110]
    grass = r(40, 70)
    others = {c: r(40, c) for c in veg_targets if c != 70}
    ok &= check(
        all(grass <= v + TOL for v in others.values()),
        "WC 40 -> HRLC 70 is the weakest vegetated target in its row",
        f"grassland {grass:.3f} vs min other {min(others.values()):.3f}",
    )
    # It is the cultivation veto doing this, not an interval: natural/cultivated
    # scores 0.3, so the row's cropland target should score far higher.
    ok &= check(
        r(40, 80) > grass + 0.3,
        "WC 40 -> HRLC 80 Croplands far exceeds the grassland cell",
        f"{r(40, 80):.3f} vs {grass:.3f}",
    )

    # 7. WC 95 Mangroves -> HRLC 90 (flooded woody vegetation) is the row max.
    mang = {c: r(95, c) for c in hr}
    best = max(mang, key=lambda c: mang[c])
    ok &= check(
        best == 90,
        "WC 95 Mangroves -> HRLC 90 is the row maximum",
        f"argmax = {best} ({mang[best]:.3f})",
    )

    # 8. HRLC 142 permanent water sits exactly inside WC 80 permanent water.
    ok &= check(
        abs(f(142, 80) - 1.0) < TOL,
        "HRLC 142 -> WC 80 == 1.0 (permanent water, same 9-12 month regime)",
        f"{f(142, 80):.4f}",
    )

    # 9. Seasonal water is the weakest-fitting HRLC row, even though it clears
    #    the orphan floor (see the NOTE below).
    maxima = {c: float(fwd[i].max()) for i, c in enumerate(hr)}
    ok &= check(
        maxima[141] == min(maxima.values()),
        "HRLC 141 seasonal water has the lowest best-fit of any HRLC class",
        f"{maxima[141]:.3f}",
    )

    # 10. A product with no encoding yields ones and warns.
    dw = _codes("dynamicworld")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        uniform = semantic_prior(REFERENCE_ID, "dynamicworld", hr, dw)
    ok &= check(
        bool((uniform == 1.0).all()) and len(caught) == 1,
        "an unencoded product (dynamicworld) yields all-ones + one warning",
        f"{len(caught)} warning(s), all_ones={bool((uniform == 1.0).all())}",
    )

    _rule("Notes (spec figures that the stated rules do not reproduce)")
    print(
        f"  NOTE  spec predicts WC 10 -> HRLC 10 ~ 0.56; computed {r(10, 10):.4f}.\n"
        "        0.56 is the cover term (50/90) alone. Averaging it with leaf type,\n"
        "        phenology, height and flooding -- all 1.0 -- gives the value above.\n"
        "        The graded-average rule is stated twice in the spec, so it wins."
    )
    orphans = semantic_orphans(fwd)
    named = [hr[i] for i in range(len(hr)) if orphans[i]]
    print(
        f"  NOTE  spec predicts HRLC 141 is a semantic orphan; its best fit is\n"
        f"        WC 80 at {f(141, 80):.3f}, above the "
        f"{CONFIG.affinity.semantic_orphan_floor} floor. Flooding is graded, not a\n"
        "        veto, so 5-9 vs 9-12 months does not disqualify a water/water pair.\n"
        f"        Semantic orphans in this pair: {named or 'none'}."
    )

    print(f"\n{'OK' if ok else 'FAILED'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
