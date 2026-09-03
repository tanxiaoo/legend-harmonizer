"""Semantic prior from LCCS attributes (Stage 8a).

The AEF + GMM pipeline measures whether pixels labelled X in map A *look like*
pixels labelled Y in map B. It cannot separate classes that are semantically
distinct but spectrally similar -- cropland vs grassland, irrigated crops vs
herbaceous wetland. This module supplies the missing signal: a directed prior
``pi[i, j]`` saying how well the *definition* of reference class ``i`` fits
inside the definition of compare class ``j``, built from the structured LCCS
attribute encodings carried in the registry YAML.

The prior is deliberately **asymmetric**. It measures inclusion (source into
target), not likeness, because a legend relationship usually is asymmetric: HRLC
"Tree cover evergreen broadleaf" (canopy cover >= 50%) sits entirely inside
WorldCover "Tree cover" (>= 10%), so that direction scores 1.0, while the reverse
scores the fraction of WorldCover's wider interval that falls inside HRLC's
narrower one.

Three rules do the work (docs/PIPELINE.md, Stage 8a.3):

* **Categorical attributes** score through the FAO correspondence table
  (``config.SEMANTIC_CORRESPONDENCE``): identical values score 1.0, related ones
  a tabulated fraction, and *unspecified on either side* scores 1.0 -- the legend
  places no constraint there, which is not the same as a mismatch.
* **Interval attributes** score by directed inclusion, the fraction of the source
  interval lying inside the target, under a stated uniform-density assumption.
* **Veto vs graded.** Surface, cultivation, and life form are *veto* attributes:
  they multiply, so getting one wrong is close to disqualifying. They are clipped
  from below at ``semantic_veto_floor`` rather than allowed to reach zero,
  because real legends leak -- WorldCover grassland may contain uncultivated
  cropland, and HRLC croplands include annual pastures. Leaf type, phenology,
  cover, height, and flooding are *graded*: they average, so a single mismatch
  degrades the score without destroying it.

A class may carry several OR-branch ``alternatives`` ("either dry cropland or
aquatic cropland"). Source-side alternatives take the **max** -- the class
matches if any of its meanings does. Target-side alternatives are **merged
attribute-wise** first, because a target that admits either meaning admits their
union.

Products whose legend is not fully encoded get a uniform prior of ones and a
warning, so every pair outside the Stage 8 scope keeps today's behaviour exactly.

See docs/PIPELINE.md, Stage 8.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np

from harmonizer.config import (
    CONFIG,
    SEMANTIC_CORRESPONDENCE,
    SEMANTIC_INTERVAL_MAX,
)
from harmonizer.registry.legends import spec as _product_spec
from harmonizer.registry.schema import (
    SEMANTIC_ENUMS,
    SEMANTIC_INTERVALS,
    ClassSemantics,
    SemanticAlternative,
)

# Attributes whose mismatch is close to disqualifying: they multiply, clipped
# from below at ``semantic_veto_floor``.
VETO_ATTRIBUTES: tuple[str, ...] = ("surface", "cultivation", "life_form")

# Attributes that shade a match rather than decide it: they are averaged.
GRADED_ATTRIBUTES: tuple[str, ...] = (
    "leaf_type",
    "phenology",
    "cover",
    "height",
    "flooding",
)


# --------------------------------------------------------------------------- #
# Merged target-side view
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class MergedAlternative:
    """A target class's alternatives collapsed into one attribute-wise union.

    A categorical attribute becomes the **set** of values any branch allows (a
    source value scores the max over that set); an interval becomes the **hull**
    of the branches' intervals. Either may be ``None``, meaning at least one
    branch left the attribute unspecified -- and an unconstrained branch makes the
    whole target unconstrained on that attribute.
    """

    categorical: dict[str, frozenset[str] | None]
    intervals: dict[str, tuple[float, float] | None]


def _interval_bounds(attr: str, interval) -> tuple[float, float]:
    """Resolve an interval's open-ended bounds against the attribute's caps."""
    low, high = interval
    return (
        0.0 if low is None else float(low),
        SEMANTIC_INTERVAL_MAX[attr] if high is None else float(high),
    )


def merge_alternatives(alternatives: Sequence[SemanticAlternative]) -> MergedAlternative:
    """Collapse a target class's OR-branches into one merged constraint.

    Categorical attributes union into a set; intervals take the hull. If **any**
    branch leaves an attribute unspecified the merged attribute is unspecified,
    since that branch already admits anything.
    """
    if not alternatives:
        raise ValueError("cannot merge an empty list of alternatives")

    categorical: dict[str, frozenset[str] | None] = {}
    for attr in SEMANTIC_ENUMS:
        values = [getattr(alt, attr) for alt in alternatives]
        categorical[attr] = None if any(v is None for v in values) else frozenset(values)

    intervals: dict[str, tuple[float, float] | None] = {}
    for attr in SEMANTIC_INTERVALS:
        raw = [getattr(alt, attr) for alt in alternatives]
        if any(v is None for v in raw):
            intervals[attr] = None
            continue
        bounds = [_interval_bounds(attr, v) for v in raw]
        intervals[attr] = (min(b[0] for b in bounds), max(b[1] for b in bounds))

    return MergedAlternative(categorical=categorical, intervals=intervals)


# --------------------------------------------------------------------------- #
# Attribute scoring
# --------------------------------------------------------------------------- #


def categorical_score(attr: str, src: str | None, tgt: frozenset[str] | str | None) -> float:
    """Likeness of a source value against a target value or set of values.

    Unspecified on either side scores 1.0: the legend imposes no constraint on
    that attribute, which must not be read as a mismatch. Otherwise the score is
    the FAO correspondence entry, or 1.0 for an identical value, taking the best
    over a target set (the target admits any of them).
    """
    if src is None or tgt is None:
        return 1.0
    targets = frozenset({tgt}) if isinstance(tgt, str) else tgt
    if not targets:
        return 1.0

    table = SEMANTIC_CORRESPONDENCE.get(attr, {})
    best = 0.0
    for value in targets:
        if value == src:
            return 1.0
        best = max(best, table.get(frozenset({src, value}), 0.0))
    return best


def interval_score(attr: str, src, tgt) -> float:
    """Directed inclusion ``|src n tgt| / |src|`` of two interval attributes.

    Unspecified on either side scores 1.0. A degenerate source interval (a single
    point, e.g. ``flooding: [0, 0]`` for "dry") is scored by membership, since its
    width is zero and the ratio would be undefined. Open-ended bounds are capped
    at the attribute's natural maximum before measuring, so "5 m or taller" is a
    finite interval rather than an infinite one that swamps every ratio.

    The ratio assumes the source class is distributed uniformly across its own
    interval. That is an approximation, stated rather than hidden: real cover
    distributions are not uniform, but nothing in the legend tells us their shape.
    """
    if src is None or tgt is None:
        return 1.0

    src_low, src_high = _interval_bounds(attr, src)
    tgt_low, tgt_high = _interval_bounds(attr, tgt)

    overlap = min(src_high, tgt_high) - max(src_low, tgt_low)
    width = src_high - src_low
    if width <= 0.0:
        # Degenerate source: inclusion is membership of the single point.
        return 1.0 if tgt_low <= src_low <= tgt_high else 0.0
    return float(np.clip(overlap / width, 0.0, 1.0))


def attribute_score(attr: str, src, tgt) -> float:
    """Score one attribute, dispatching on whether it is categorical or interval."""
    if attr in SEMANTIC_ENUMS:
        return categorical_score(attr, src, tgt)
    if attr in SEMANTIC_INTERVALS:
        return interval_score(attr, src, tgt)
    raise KeyError(f"unknown semantic attribute {attr!r}")


# --------------------------------------------------------------------------- #
# Inclusion of one alternative in a merged target
# --------------------------------------------------------------------------- #


def _target_value(merged: MergedAlternative, attr: str):
    if attr in SEMANTIC_ENUMS:
        return merged.categorical[attr]
    return merged.intervals[attr]


def inclusion(
    src_alt: SemanticAlternative,
    tgt_merged: MergedAlternative,
    *,
    veto_floor: float | None = None,
) -> float:
    """How far one source alternative falls inside a merged target class.

    ``prod_veto max(score, veto_floor) * mean_graded score``: the veto attributes
    multiply so a wrong surface or life form dominates, but each is clipped at
    ``veto_floor`` so a mismatch is a strong penalty rather than an impossibility.
    The graded attributes average, so they shade the result without vetoing it.
    """
    if veto_floor is None:
        veto_floor = CONFIG.affinity.semantic_veto_floor

    score = 1.0
    for attr in VETO_ATTRIBUTES:
        raw = attribute_score(attr, getattr(src_alt, attr), _target_value(tgt_merged, attr))
        score *= max(raw, veto_floor)

    graded = [
        attribute_score(attr, getattr(src_alt, attr), _target_value(tgt_merged, attr))
        for attr in GRADED_ATTRIBUTES
    ]
    if graded:
        score *= float(np.mean(graded))
    return float(score)


# --------------------------------------------------------------------------- #
# The prior matrix
# --------------------------------------------------------------------------- #


#: Separator that scopes a product id to one auxiliary AOI
#: (``harmonizer.auxiliary.aux_scoped_id``, e.g. ``worldcover_2020__aux_coast``).
_AUX_SCOPE = "__aux_"


def base_product_id(product_id: str) -> str:
    """Strip an auxiliary-AOI scope suffix to get the registry product id.

    An auxiliary AOI reuses the Stage 2/3 machinery under a scoped id so its
    per-AOI caches sit in their own files. That id has no registry entry, so a
    naive lookup would find no legend, fall back to a uniform prior, and leave
    auxiliary rows observational while primary rows are fused -- in the same
    merged table. The legend is a property of the *product*, not of which AOI
    it was sampled in, so the scope is dropped here.
    """
    head, sep, _ = product_id.partition(_AUX_SCOPE)
    return head if sep else product_id


def _semantics_by_code(product_id: str) -> dict[int, ClassSemantics] | None:
    """Every legend class's encoding for a product, or None if not fully encoded."""
    spec = _product_spec(base_product_id(product_id))
    if spec is None or not spec.has_semantics:
        return None
    return {c.code: c.semantics for c in spec.legend if c.semantics is not None}


def semantic_prior(
    reference_id: str,
    compare_id: str,
    ref_classes: Iterable[int],
    cmp_classes: Iterable[int],
    *,
    veto_floor: float | None = None,
) -> np.ndarray:
    """The directed M x N prior for an ordered product pair.

    Rows are ``ref_classes`` (the source), columns ``cmp_classes`` (the target),
    in the order given -- the same order the caller uses for the distance matrix,
    so the two align cell for cell. Each entry is
    ``max_k inclusion(alternative_k of i, merged(j))``: the class matches if any
    of its meanings fits inside the target.

    Values lie in (0, 1]. If either product's legend is not fully encoded the
    result is all ones -- a uniform prior that leaves the fused logits untouched
    -- and a warning is issued, so an unencoded pair degrades to today's
    AEF-only behaviour rather than to a silently wrong prior.
    """
    ref_codes = [int(c) for c in ref_classes]
    cmp_codes = [int(c) for c in cmp_classes]
    shape = (len(ref_codes), len(cmp_codes))

    ref_sem = _semantics_by_code(reference_id)
    cmp_sem = _semantics_by_code(compare_id)
    missing = [
        base_product_id(pid)
        for pid, sem in ((reference_id, ref_sem), (compare_id, cmp_sem))
        if sem is None
    ]
    if missing:
        warnings.warn(
            f"no LCCS attribute encoding for {', '.join(missing)}; "
            "using a uniform semantic prior (pi = 1) for this pair",
            stacklevel=2,
        )
        return np.ones(shape, dtype=float)

    # Merge each target class once, not once per row.
    merged = {
        code: merge_alternatives(sem.alternatives) for code, sem in cmp_sem.items()
    }

    prior = np.ones(shape, dtype=float)
    for i, rc in enumerate(ref_codes):
        source = ref_sem.get(rc)
        for j, cc in enumerate(cmp_codes):
            target = merged.get(cc)
            if source is None or target is None:
                # A class absent from the legend cannot be scored; leave it
                # uniform rather than inventing a constraint for it.
                continue
            prior[i, j] = max(
                inclusion(alt, target, veto_floor=veto_floor)
                for alt in source.alternatives
            )
    return prior


def semantic_orphans(prior: np.ndarray, *, floor: float | None = None) -> np.ndarray:
    """Rows whose best semantic fit falls below ``semantic_orphan_floor``.

    A semantic orphan is a class whose *definition* finds no good home in the
    other legend, independent of how its pixels look. Reported alongside the
    observational orphan status, never replacing it.
    """
    if floor is None:
        floor = CONFIG.affinity.semantic_orphan_floor
    if prior.size == 0:
        return np.zeros(prior.shape[0], dtype=bool)
    return prior.max(axis=1) < float(floor)
