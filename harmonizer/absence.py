"""Symmetric absence accounting (Stage 7a).

Every class **declared** in a map's registry legend must account for itself in the
output -- on both sides. Stage 2 builds its class list from the codes *observed* in
the AOI (``sampling.present_classes``), so a legend class with no pixels there never
enters sampling, never reaches Stages 3-4, and never appears in the matching table:
not even as ``absent``. A reader cannot then tell a class that was *never considered*
from one that was considered and matched.

This module closes that gap by reconciling **declared** (registry YAML, section 2.5)
against **modelled** (the Stage 3 GMM cache), and assigning every unmodelled class a
status of ``absent`` plus an explicit **reason** (docs/PIPELINE.md, section 2 ->
Multi-AOI absence handling):

  * ``not_in_aoi`` -- declared but never observed: no sample entry at all. An
    auxiliary AOI covering the class would help (Stage 7c).
  * ``too_rare``   -- observed but below the point floor even before eroding, so
    Stage 2 flagged it ``absent`` and Stage 3 skipped it. A bigger AOI might help;
    relaxing the buffer would not.

**Symmetry.** The pre-Stage-7 pipeline only carried ``absent`` for *reference*
classes (matrix rows). An absent **compare** class simply lost its column, which
also silently changed the softmax normalisation of every row. Here both sides are
resolved the same way: :func:`absent_classes` works off any product id, and
:func:`compare_absent_classes` reports absent compare classes as unmatched targets
rather than dropping them.

Distinguishing the two reasons needs the Stage 2 sample sidecar, not just the Stage
3 GMM cache: a class the GMM cache does not mention at all was never sampled
(``not_in_aoi``), whereas one it marks ``absent`` was sampled and starved
(``too_rare``). Both caches are read; neither is written.

**No Earth Engine access, by design.** Sampling *is* the observation: Stage 2 must
call ``present_classes`` server-side to know which classes to sample, so once a run
has sampled an AOI, what that AOI lacks is already recorded in the caches. Querying
GEE again purely to ask "which classes are here" would re-pay for an answer already
on disk -- and give a *worse* one, since a pre-run probe cannot separate
``not_in_aoi`` from ``too_rare`` (rarity depends on the point floor after erosion,
which only sampling establishes). The Stage 7b dialog therefore runs **after**
Stage 2, off these caches; an auxiliary AOI (Stage 7c) costs GEE only for the
classes it is actually there to sample.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

from harmonizer.config import CONFIG, REASON_NOT_IN_AOI, REASON_TOO_RARE
from harmonizer.modeling import gmm_cache_path
from harmonizer.registry.legends import class_name, legend_classes
from harmonizer.sampling import cache_path as sample_cache_path


@dataclass(frozen=True)
class AbsentClass:
    """A declared legend class that carries status ``absent``, and why.

    ``observed`` records whether the class was seen in the AOI at all, which is
    what separates the two reasons: an unobserved class may be recoverable with an
    auxiliary AOI (Stage 7c), a starved one needs a larger AOI or stays absent.
    """

    product_id: str
    class_value: int
    class_name: str
    reason: str            # REASON_NOT_IN_AOI | REASON_TOO_RARE
    observed: bool         # was the class present in the AOI at all
    n_points: int = 0      # points that survived sampling (0 when not observed)


def _sample_meta(product_id: str) -> dict[int, dict]:
    """Per-class Stage 2 sidecar entries, or {} when the map was never sampled."""
    sidecar = sample_cache_path(product_id).with_suffix(".json")
    if not sidecar.exists():
        return {}
    payload = json.loads(sidecar.read_text(encoding="utf-8"))
    return {int(cv): meta for cv, meta in payload.get("classes", {}).items()}


def _modelled_classes(product_id: str) -> set[int]:
    """Class values with a fitted GMM in the Stage 3 cache (absent ones excluded)."""
    path = gmm_cache_path(product_id)
    if not path.exists():
        return set()
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {
        int(cv)
        for cv, cs in payload.get("classes", {}).items()
        if cs.get("fitted", False) and not cs.get("absent", False)
    }


def _sampled_ok_classes(product_id: str) -> set[int]:
    """Class values healthily sampled in Stage 2 (enough points to fit later)."""
    return {
        cv for cv, m in _sample_meta(product_id).items()
        if not m.get("absent", False)
    }


def covered_classes(product_id: str) -> set[int]:
    """Classes this product's caches account for.

    When Stage 3 has run (a GMM cache exists), the fitted GMMs are the truth.
    Before that — a sample-only run, where the points are collected but the
    models are fitted later by "Run all" — a class healthily sampled in Stage 2
    counts as covered, so absence (and auxiliary targeting, Stage 7c) can be
    judged from the points alone without pretending everything is absent.
    """
    if gmm_cache_path(product_id).exists():
        return _modelled_classes(product_id)
    return _sampled_ok_classes(product_id)


def declared_classes(product_id: str) -> list[int]:
    """Class values declared in the product's registry legend, in declared order."""
    return [lc.code for lc in legend_classes(product_id)]


def absent_classes(product_id: str) -> list[AbsentClass]:
    """Every declared class of ``product_id`` that has no fitted GMM, with its reason.

    Works for either side of the comparison -- pass the reference or the compare
    product id. Returns declared-order entries so output ordering follows the
    legend rather than dict iteration order.
    """
    covered = covered_classes(product_id)
    meta = _sample_meta(product_id)

    out: list[AbsentClass] = []
    for cv in declared_classes(product_id):
        if cv in covered:
            continue
        entry = meta.get(cv)
        # A class the sample sidecar never mentions was not observed in the AOI at
        # all; one it mentions was observed but starved below the floor pre-erode.
        if entry is None:
            reason, observed, n_points = REASON_NOT_IN_AOI, False, 0
        else:
            reason, observed = REASON_TOO_RARE, True
            n_points = int(entry.get("surviving", 0))
        out.append(
            AbsentClass(
                product_id=product_id,
                class_value=cv,
                class_name=class_name(product_id, cv),
                reason=reason,
                observed=observed,
                n_points=n_points,
            )
        )
    return out


def compare_absent_classes(compare_id: str) -> list[AbsentClass]:
    """Absent classes on the **compare** side -- unmatched targets.

    These have no column in the affinity matrix, so nothing can map to them. Before
    Stage 7 they vanished silently; they are reported so a reader sees which compare
    classes this run could not offer as candidates.
    """
    return absent_classes(compare_id)


def undeclared_codes(product_id: str) -> list[int]:
    """Class codes sampled from the data but missing from the registry legend.

    The registry reconciliation check's other signal (section 2.5): a code in the
    data with no legend entry means the YAML is incomplete -- the class has no name
    or colour. Surfaced loudly as a registry bug, distinct from absence.
    """
    declared = set(declared_classes(product_id))
    return sorted(cv for cv in _sample_meta(product_id) if cv not in declared)


def absence_report(reference_id: str, compare_id: str) -> dict[str, object]:
    """Both sides' absence accounting, for the CLI verification and the API.

    Keyed by side so a caller can render "WorldCover: Snow and ice, Mangroves, Moss
    and lichen -- Dynamic World: Snow and ice" without re-deriving which product is
    which, and shaped for the Stage 7b dialog (``any_absent`` drives whether it is
    raised at all).

    Reads the caches Stage 2/3 already wrote -- **no GEE**. Sampling is itself the
    observation: Stage 2 must call ``present_classes`` to know what to sample, so by
    the time a run finishes, which classes the AOI lacks is already established. A
    separate live query would only re-ask a question the sample data answers, and
    answer it *worse* -- a pre-run probe cannot tell ``not_in_aoi`` from
    ``too_rare``, because rarity depends on the point floor after erosion, which
    only sampling establishes.
    """
    ref_absent = absent_classes(reference_id)
    cmp_absent = compare_absent_classes(compare_id)
    return {
        "reference": {
            "product_id": reference_id,
            "declared": len(declared_classes(reference_id)),
            "absent": [_as_dict(a) for a in ref_absent],
            "undeclared_codes": undeclared_codes(reference_id),
        },
        "compare": {
            "product_id": compare_id,
            "declared": len(declared_classes(compare_id)),
            "absent": [_as_dict(a) for a in cmp_absent],
            "undeclared_codes": undeclared_codes(compare_id),
        },
        "any_absent": bool(ref_absent or cmp_absent),
    }


def _as_dict(a: AbsentClass) -> dict[str, object]:
    return {
        "class_value": a.class_value,
        "class_name": a.class_name,
        "reason": a.reason,
        "observed": a.observed,
        "n_points": a.n_points,
    }
