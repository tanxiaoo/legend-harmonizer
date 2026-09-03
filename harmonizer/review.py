"""Human-in-the-loop review -- feedback data model + logic (Stage 6a).

This is the **backend** half of Stage 6: per-edge provenance in the matching
table, the confirm/freeze operation, the warm-start refit on confirmed samples,
and the rule that any recompute/re-balance touches **unconfirmed edges only**
(docs/PIPELINE.md, Stage 6.5 / 6.6 / build-order bullet "6a"). The evidence
explorer (co-located-pixel queries, patch sampling) is Stage 6b and the Review UI
is Stage 6c; neither is built here.

An expert confirmation does two distinct things, kept from colliding by different
rules (Stage 6.6):

1. **Authoritative crosswalk edge.** A confirmed edge is written to the matching
   table as ``expert-confirmed`` and **frozen** -- never overwritten by any later
   refit or recompute, and it keeps its retained probability.
2. **Training signal.** The samples validated by that confirmation warm-start a
   refit of the reference class's GMM, improving the model's *proposals* for the
   edges the expert has **not** confirmed.

The two never collide because:
  * the refit and any subsequent affinity recompute apply to **unconfirmed edges
    only** -- confirmed edges are excluded from recompute and re-normalisation;
  * the model never overwrites a confirmed edge; it may only re-propose or
    re-weight the open edges around it.

**Hard boundary.** This module implements the *feedback* mechanics only. It never
derives or weights correspondences from co-occurrence -- that remains the
embedding + GMM + Bures-Wasserstein method of Stages 3-4. Confirmation is an
expert act recorded here; it does not feed a co-occurrence vote (docs/PIPELINE.md,
Stage 6.2 "co-occurrence is evidence retrieval only, never scoring").

Provenance vocabulary (Stage 6.5): every edge in the matching table carries
``expert-confirmed`` (frozen) or ``algorithm-proposed`` (open).

Constants come from ``CONFIG``; nothing tunable is hardcoded here.
"""

from __future__ import annotations

import json
import warnings
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
from sklearn.mixture import GaussianMixture

from harmonizer.affinity import (
    AffinityResult,
    class_name,
    compute_affinity,
    load_map_gmm,
)
from harmonizer.config import CONFIG
from harmonizer.decision import (
    ClassDecision,
    absent_decisions,
    build_matching_table,
    classify_rows,
)
from harmonizer.modeling import gmm_cache_path, load_samples

# Provenance values carried by every edge (docs/PIPELINE.md, Stage 6.5).
EXPERT_CONFIRMED = "expert-confirmed"   # frozen; never overwritten by a recompute
ALGORITHM_PROPOSED = "algorithm-proposed"  # open; free to re-propose / re-weight


# --------------------------------------------------------------------------- #
# Feedback data model
# --------------------------------------------------------------------------- #


@dataclass
class ConfirmedEdge:
    """One expert-confirmed edge (reference class -> compare class), frozen.

    ``retained_probability`` is the mapping probability the algorithm had assigned
    to this edge at confirmation time, kept so the crosswalk stays quantitative
    rather than collapsing to a flat set (docs/PIPELINE.md, Stage 6.5
    "Multi-select candidates"). It is *frozen*: later recomputes do not touch it.
    """

    reference_value: int
    compare_value: int
    retained_probability: float
    provenance: str = EXPERT_CONFIRMED


@dataclass
class FeedbackStore:
    """All expert feedback for one reference x compare product pair.

    Persisted as JSON beside the GMM cache (``cache/feedback_<ref>__<cmp>.json``).
    Holds the confirmed edges keyed by reference class, so recompute can exclude a
    reference class's confirmed edges and re-balance only its open ones.
    """

    reference_id: str
    compare_id: str
    # reference_value -> list of confirmed edges for that reference class.
    confirmed: dict[int, list[ConfirmedEdge]] = field(default_factory=dict)
    # Reference classes the expert marked COMPLETE: the confirmed edges are the
    # row's ONLY mapping. A complete row's confirmed edges are renormalised to
    # sum to 1 and its open edges drop to 0 (option A -- exclusive confirmation),
    # instead of the default "frozen sliver + open remainder" re-balance.
    complete: set[int] = field(default_factory=set)

    # -- confirmed-edge queries -------------------------------------------- #

    def confirmed_edges(self, reference_value: int) -> list[ConfirmedEdge]:
        return list(self.confirmed.get(int(reference_value), []))

    def is_complete(self, reference_value: int) -> bool:
        return int(reference_value) in self.complete

    def confirmed_compare_values(self, reference_value: int) -> set[int]:
        return {e.compare_value for e in self.confirmed_edges(reference_value)}

    def is_confirmed(self, reference_value: int, compare_value: int) -> bool:
        return int(compare_value) in self.confirmed_compare_values(reference_value)

    def has_any(self, reference_value: int) -> bool:
        return bool(self.confirmed.get(int(reference_value)))


# --------------------------------------------------------------------------- #
# Persistence
# --------------------------------------------------------------------------- #


def feedback_cache_path(reference_id: str, compare_id: str) -> Path:
    """Path to the feedback store for a reference x compare pair.

    Direction-specific (reference vs compare matters), so a swapped pair gets its
    own store -- matching how the affinity itself is direction-specific.
    """
    return CONFIG.cache_dir / f"feedback_{reference_id}__{compare_id}.json"


def load_feedback(reference_id: str, compare_id: str) -> FeedbackStore:
    """Load the feedback store for a pair, or an empty one if none exists yet."""
    path = feedback_cache_path(reference_id, compare_id)
    if not path.exists():
        return FeedbackStore(reference_id=reference_id, compare_id=compare_id)
    payload = json.loads(path.read_text(encoding="utf-8"))
    confirmed: dict[int, list[ConfirmedEdge]] = {}
    for rv_str, edges in payload.get("confirmed", {}).items():
        confirmed[int(rv_str)] = [
            ConfirmedEdge(
                reference_value=int(e["reference_value"]),
                compare_value=int(e["compare_value"]),
                retained_probability=float(e["retained_probability"]),
                provenance=str(e.get("provenance", EXPERT_CONFIRMED)),
            )
            for e in edges
        ]
    return FeedbackStore(
        reference_id=str(payload["reference_id"]),
        compare_id=str(payload["compare_id"]),
        confirmed=confirmed,
        complete={int(rv) for rv in payload.get("complete", [])},
    )


def save_feedback(store: FeedbackStore) -> Path:
    """Persist a feedback store to ``cache/`` as JSON."""
    CONFIG.cache_dir.mkdir(parents=True, exist_ok=True)
    path = feedback_cache_path(store.reference_id, store.compare_id)
    payload = {
        "reference_id": store.reference_id,
        "compare_id": store.compare_id,
        "confirmed": {
            str(rv): [asdict(e) for e in edges]
            for rv, edges in store.confirmed.items()
        },
        "complete": sorted(store.complete),
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


# --------------------------------------------------------------------------- #
# Affinity memo -- recomputing Bures-Wasserstein over every class pair takes
# seconds, and the interactive review path (table fetch, confirm, unconfirm)
# needs it on every request. The GMM caches on disk are the single source of
# truth, so memoise by their mtimes: any refit rewrites the file and naturally
# invalidates the memo. Local single-user server, so a plain dict suffices.
# --------------------------------------------------------------------------- #

_AFFINITY_MEMO: dict[tuple, AffinityResult] = {}


def cached_affinity(
    reference_id: str, compare_id: str, alpha: float | None = None
) -> AffinityResult:
    """Stage 4 affinity for a pair, memoised on the two GMM caches' mtimes.

    ``alpha`` (Stage 8d) is part of the memo key: the same cached models yield a
    different fused table at a different semantic-prior weight, so keying only
    on the mtimes would serve one alpha's result for another's request. ``None``
    means the calibrated ``CONFIG.affinity.semantic_prior_alpha``, resolved here
    so the key is the *effective* value and an omitted alpha hits the same entry
    as its explicit equivalent.
    """
    from harmonizer.config import CONFIG

    effective = CONFIG.affinity.semantic_prior_alpha if alpha is None else float(alpha)
    key = (
        reference_id,
        compare_id,
        gmm_cache_path(reference_id).stat().st_mtime_ns,
        gmm_cache_path(compare_id).stat().st_mtime_ns,
        effective,
    )
    aff = _AFFINITY_MEMO.get(key)
    if aff is None:
        aff = compute_affinity(reference_id, compare_id, alpha=effective)
        # Keep only entries for the CURRENT caches; a stale-mtime entry can
        # never be requested again, but several alphas of the current one can.
        for stale in [k for k in _AFFINITY_MEMO if k[2:4] != key[2:4]]:
            del _AFFINITY_MEMO[stale]
        _AFFINITY_MEMO[key] = aff
    return aff


# --------------------------------------------------------------------------- #
# Confirm / freeze (the expert decision, edge-level)
# --------------------------------------------------------------------------- #


def confirm_edges(
    store: FeedbackStore,
    aff: AffinityResult,
    reference_value: int,
    compare_values: list[int],
) -> FeedbackStore:
    """Confirm one or more edges for a reference class and freeze them.

    Confirmation is at the **edge level, not the class row** (docs/PIPELINE.md,
    Stage 6.5): the expert may confirm several compare classes for one reference
    class (multi-select), and confirming does not touch that reference class's
    other, still-open edges. Each confirmed edge freezes with the mapping
    probability the algorithm currently assigns it (``retained_probability``), so
    the crosswalk stays quantitative among the confirmed candidates.

    The ranked affinity is a *hint, not a constraint*: an expert may confirm a
    compare class the algorithm ranked low -- or one absent from the affinity
    matrix entirely (unfitted / absent in the AOI), which freezes with a retained
    probability of 0.0 since the algorithm assigned it no mass. Mutates and
    returns ``store`` (also persist it with :func:`save_feedback`).

    **Absent reference classes are confirmable too** (Stage 7.5). A class with no
    pixels in the AOI has no matrix row, but it is exactly the case only an expert
    can resolve -- the AOI cannot evidence it, so the algorithm never will. Its
    edges freeze at a retained probability of 0.0, the same rule already applied to
    an absent *compare* class. Rejecting it here (as this did before Stage 7) made
    the row unreviewable on both the API and the UI, which is the asymmetry Stage 7
    exists to remove. A class the registry does not declare at all is still an
    error: that is a bad request, not an expert judgement.
    """
    rv = int(reference_value)
    in_matrix = rv in aff.reference_classes
    if not in_matrix:
        from harmonizer.absence import declared_classes

        if rv not in declared_classes(aff.reference_id):
            raise ValueError(
                f"reference class {rv} is not declared in the registry legend for "
                f"{aff.reference_id!r}."
            )
    prob_row = aff.normalized_affinity[aff.reference_classes.index(rv)] if in_matrix else None

    existing = {e.compare_value for e in store.confirmed.get(rv, [])}
    edges = store.confirmed.setdefault(rv, [])
    for cv in compare_values:
        cv = int(cv)
        if cv in existing:
            continue  # already confirmed; confirming again is a no-op
        if prob_row is not None and cv in aff.compare_classes:
            retained = float(prob_row[aff.compare_classes.index(cv)])
        else:
            # No mass to retain: either the compare class is outside the matrix, or
            # the reference class is absent so it has no row at all. The expert may
            # still assert the correspondence.
            retained = 0.0
        edges.append(
            ConfirmedEdge(
                reference_value=rv,
                compare_value=cv,
                retained_probability=retained,
            )
        )
        existing.add(cv)
    return store


def unconfirm_edge(
    store: FeedbackStore, reference_value: int, compare_value: int
) -> FeedbackStore:
    """Remove a previously confirmed edge (an expert changing their mind).

    Reopens the edge so it becomes ``algorithm-proposed`` again on the next table
    build. Mutates and returns ``store``.
    """
    rv = int(reference_value)
    cv = int(compare_value)
    if rv in store.confirmed:
        store.confirmed[rv] = [
            e for e in store.confirmed[rv] if e.compare_value != cv
        ]
        if not store.confirmed[rv]:
            del store.confirmed[rv]
            # A row with no confirmed edges cannot be "complete".
            store.complete.discard(rv)
    return store


# --------------------------------------------------------------------------- #
# Recompute / re-balance -- unconfirmed edges only
# --------------------------------------------------------------------------- #
# Confirmed edges are frozen: excluded from re-normalisation and never
# overwritten. Only a reference row's *open* mass is re-balanced when the affinity
# is recomputed after a refit (docs/PIPELINE.md, Stage 6.6).


def rebalance_row(
    prob_row: np.ndarray,
    compare_classes: list[int],
    confirmed: list[ConfirmedEdge],
) -> np.ndarray:
    """Re-balance one reference row leaving its confirmed edges frozen.

    The confirmed edges keep exactly their ``retained_probability``; the remaining
    ("open") probability mass ``1 - sum(retained)`` is redistributed across the
    **unconfirmed** compare classes in proportion to their freshly recomputed
    probabilities. This is the "recompute touches unconfirmed edges only" rule made
    concrete: a recompute can move probability among open edges but cannot change a
    frozen edge's share (docs/PIPELINE.md, Stage 6.6).

    Returns a new probability row (same length/order as ``compare_classes``) that
    still sums to 1. If the confirmed probabilities already sum to >= 1 (e.g. an
    expert confirmed the whole row), the open edges get 0.
    """
    row = np.array(prob_row, dtype=float)
    if not confirmed:
        return row

    frozen = {e.compare_value: e.retained_probability for e in confirmed}
    frozen_mass = float(sum(frozen.values()))
    open_budget = max(0.0, 1.0 - frozen_mass)

    open_idx = [j for j, cc in enumerate(compare_classes) if cc not in frozen]
    open_mass = float(sum(row[j] for j in open_idx))

    out = np.zeros_like(row)
    for j, cc in enumerate(compare_classes):
        if cc in frozen:
            out[j] = frozen[cc]
        elif open_mass > 0.0:
            out[j] = open_budget * (row[j] / open_mass)
        else:
            # No open probability to distribute: spread the budget evenly.
            out[j] = open_budget / len(open_idx) if open_idx else 0.0
    return out


# --------------------------------------------------------------------------- #
# Warm-start refit on confirmed samples (the training signal)
# --------------------------------------------------------------------------- #
# The samples validated by a confirmation are used to warm-start-refit the
# reference class's GMM. The refit improves the model's *proposals* for the open
# edges; it does not touch the confirmed edges (those are frozen in the table).
# Warm start: seed the new fit from the existing fitted GMM's parameters so the
# refit is an incremental adjustment, not a fresh random fit.


def _warm_start_gmm(prev: dict, n_points: int, cov_type: str) -> GaussianMixture:
    """A GaussianMixture pre-seeded from a previously fitted class's parameters.

    Passing ``means_init``/``weights_init``/``precisions_init`` makes the EM
    iterations start from the prior fit rather than a random k-means seed, so the
    refit warm-starts (incremental adjustment) instead of re-fitting from scratch.
    K is taken from the prior fit's component count (already capped in Stage 3);
    the covariance type matches the prior fit so the seeds are shape-compatible.
    """
    weights = np.asarray(prev["weights"], dtype=float)
    means = np.asarray(prev["means"], dtype=float)
    k = means.shape[0]

    gmm = GaussianMixture(
        n_components=k,
        covariance_type=cov_type,
        reg_covar=CONFIG.gmm.reg_covar,
        random_state=CONFIG.gmm.random_seed,
        weights_init=weights / weights.sum(),
        means_init=means,
    )
    return gmm


@dataclass
class RefitResult:
    """Outcome of a warm-start refit for one reference class."""

    reference_value: int
    n_confirmed_edges: int
    n_points_refit: int
    n_components: int
    covariance_type_used: str
    refit: bool  # False when there was nothing to refit (skipped)


def warm_start_refit(
    reference_id: str,
    compare_id: str,
    reference_value: int,
    store: FeedbackStore | None = None,
) -> RefitResult:
    """Warm-start-refit a reference class's GMM from its confirmed samples.

    The confirmed edges validate that this reference class's sampled points are
    genuine members of the class, so the class's embedding vectors (from the Stage
    2 sample cache) are used to refit its GMM, warm-started from the current fit.
    The refit is written back to ``cache/gmm_<reference>.json`` so the next affinity
    recompute proposes from the improved model -- but only for the **open** edges;
    the confirmed edges stay frozen in the matching table regardless
    (docs/PIPELINE.md, Stage 6.6).

    Requires at least one confirmed edge for the reference class (there is no
    training signal otherwise). Returns a :class:`RefitResult`; call it after
    :func:`save_feedback`.
    """
    store = store or load_feedback(reference_id, compare_id)
    rv = int(reference_value)
    edges = store.confirmed_edges(rv)
    if not edges:
        return RefitResult(rv, 0, 0, 0, "", refit=False)

    # The class's embedding vectors validated by the confirmation.
    samples = load_samples(reference_id)
    embeddings = samples.embeddings.get(rv)
    if embeddings is None or embeddings.shape[0] == 0:
        warnings.warn(
            f"no cached samples for reference class {rv} in {reference_id!r}; "
            "cannot warm-start refit.",
            stacklevel=2,
        )
        return RefitResult(rv, len(edges), 0, 0, "", refit=False)

    # The current fit, whose parameters seed the warm start.
    gmm_path = gmm_cache_path(reference_id)
    payload = json.loads(gmm_path.read_text(encoding="utf-8"))
    prev = payload["classes"].get(str(rv))
    if prev is None or not prev.get("fitted", False):
        warnings.warn(
            f"reference class {rv} in {reference_id!r} has no fitted GMM to "
            "warm-start from; skipping refit.",
            stacklevel=2,
        )
        return RefitResult(rv, len(edges), int(embeddings.shape[0]), 0, "", refit=False)

    cov_type = str(prev["covariance_type_used"])
    gmm = _warm_start_gmm(prev, embeddings.shape[0], cov_type)
    gmm.fit(embeddings)

    # Write the refit back, preserving the class's Stage 2/3 bookkeeping.
    prev["weights"] = gmm.weights_.tolist()
    prev["means"] = gmm.means_.tolist()
    prev["covariances"] = gmm.covariances_.tolist()
    prev["n_components_used"] = int(gmm.n_components)
    prev["refit_from_feedback"] = True
    payload["classes"][str(rv)] = prev
    gmm_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    return RefitResult(
        reference_value=rv,
        n_confirmed_edges=len(edges),
        n_points_refit=int(embeddings.shape[0]),
        n_components=int(gmm.n_components),
        covariance_type_used=cov_type,
        refit=True,
    )


# --------------------------------------------------------------------------- #
# Matching table with provenance (the Stage 6a output artifact)
# --------------------------------------------------------------------------- #


@dataclass
class ReviewedEdge:
    """One edge in the reviewed matching table, carrying provenance.

    ``provenance`` is ``expert-confirmed`` (frozen) or ``algorithm-proposed``
    (open). ``probability`` is the re-balanced mapping probability -- a confirmed
    edge keeps its frozen ``retained_probability``; an open edge carries its
    recomputed, re-balanced share (docs/PIPELINE.md, Stage 6.5 / 6.6).
    """

    reference_value: int
    reference_name: str
    compare_value: int
    compare_name: str
    probability: float
    provenance: str


@dataclass
class ReviewedRow:
    """A reference class's reviewed mapping: its edges, each with provenance.

    ``all_probabilities`` carries the FULL re-balanced row (every compare class,
    not just the kept edges), so a UI can show the algorithm's mass for any
    compare class the expert reaches for -- including ones below the display
    cutoff (docs/PIPELINE.md, Stage 6.5 "any valid compare class is accepted").
    """

    reference_value: int
    reference_name: str
    status: str
    edges: list[ReviewedEdge] = field(default_factory=list)
    all_probabilities: dict[int, float] = field(default_factory=dict)
    # Expert marked the row complete: the confirmed edges are its ONLY mapping
    # (renormalised to sum 1, open edges zeroed) -- option A.
    complete: bool = False
    # Stage 7: why an ``absent`` class could not be modelled (not_in_aoi |
    # too_rare); empty for every other status. Carried so the crosswalk export can
    # distinguish "this class does not occur here" from "not reviewed yet" -- both
    # of which otherwise have no edges.
    absence_reason: str = ""

    @property
    def has_confirmed(self) -> bool:
        return any(e.provenance == EXPERT_CONFIRMED for e in self.edges)


def build_reviewed_table(
    aff: AffinityResult,
    decisions: list[ClassDecision],
    store: FeedbackStore,
    *,
    prob_cutoff: float = 0.25,
) -> list[ReviewedRow]:
    """Build the matching table with per-edge provenance and re-balanced rows.

    For each reference row: freeze its confirmed edges (provenance
    ``expert-confirmed``, keeping ``retained_probability``), re-balance the open
    mass across the unconfirmed compare classes, then keep the meaningful edges the
    same way Stage 4's :func:`build_matching_table` does (top candidate plus any
    within ``prob_cutoff`` of it), so a ``mixed`` row still shows its split. Every
    confirmed edge is always kept regardless of the cutoff (the expert asked for
    it).

    This is the authoritative crosswalk: confirmed edges are frozen and excluded
    from re-normalisation; only the open edges were re-balanced (docs/PIPELINE.md,
    Stage 6.6).
    """
    rows: list[ReviewedRow] = []
    for d in decisions:
        rv = d.reference_value

        # Absent classes have no matrix row, so the algorithm proposes nothing for
        # them -- but an expert may still have confirmed edges by hand, which is the
        # only way an absent class ever gets a mapping (Stage 7.5). Surface those;
        # dropping them would silently discard the expert's decision. The reason
        # rides along so the export can say why, rather than emitting a blank that
        # reads as "not reviewed yet".
        if rv not in aff.reference_classes:
            rows.append(
                ReviewedRow(
                    reference_value=rv,
                    reference_name=d.reference_name,
                    status=d.status,
                    edges=[
                        ReviewedEdge(
                            reference_value=rv,
                            reference_name=d.reference_name,
                            compare_value=e.compare_value,
                            compare_name=class_name(aff.compare_id, e.compare_value),
                            probability=e.retained_probability,
                            provenance=EXPERT_CONFIRMED,
                        )
                        for e in store.confirmed_edges(rv)
                    ],
                    absence_reason=d.absence_reason,
                    complete=store.is_complete(rv),
                )
            )
            continue

        ri = aff.reference_classes.index(rv)
        confirmed = store.confirmed_edges(rv)
        confirmed_cvs = {e.compare_value for e in confirmed}

        # COMPLETE row (option A, legacy): the confirmed edges are the row's
        # only mapping -- renormalise them to sum to 1 and zero every open
        # edge. The current UI never saves this shape; it survives for stores
        # written before the no-match redesign. A complete row with NO
        # confirmed edges ("no matching class") instead flows through the
        # normal path below, keeping its proposals and probabilities, with
        # only the row's ``complete`` flag marking the decision -- so it stays
        # a normal, fully reversible choice.
        if confirmed and store.is_complete(rv):
            total = sum(e.retained_probability for e in confirmed)
            norm = {
                e.compare_value: (
                    e.retained_probability / total
                    if total > 0
                    else 1.0 / len(confirmed)
                )
                for e in confirmed
            }
            edges = [
                ReviewedEdge(
                    reference_value=rv,
                    reference_name=d.reference_name,
                    compare_value=cv,
                    compare_name=class_name(aff.compare_id, cv),
                    probability=p,
                    provenance=EXPERT_CONFIRMED,
                )
                for cv, p in sorted(norm.items(), key=lambda kv: -kv[1])
            ]
            rows.append(
                ReviewedRow(
                    reference_value=rv,
                    reference_name=d.reference_name,
                    status=d.status,
                    edges=edges,
                    all_probabilities={
                        int(cc): float(norm.get(cc, 0.0))
                        for cc in aff.compare_classes
                    },
                    complete=True,
                )
            )
            continue

        balanced = rebalance_row(
            aff.normalized_affinity[ri], aff.compare_classes, confirmed
        )

        # Rank candidates by re-balanced probability; keep the top and anything
        # within prob_cutoff of it, plus every confirmed edge unconditionally.
        ranked = sorted(
            (
                (cc, float(balanced[j]))
                for j, cc in enumerate(aff.compare_classes)
            ),
            key=lambda kv: kv[1],
            reverse=True,
        )
        top_prob = ranked[0][1] if ranked else 0.0
        threshold = max(prob_cutoff * top_prob, 0.0)
        kept_cvs = {
            cc for cc, p in ranked if p >= threshold
        } | confirmed_cvs
        if ranked and not kept_cvs:
            kept_cvs = {ranked[0][0]}

        edges: list[ReviewedEdge] = []
        for cc, p in ranked:
            if cc not in kept_cvs:
                continue
            edges.append(
                ReviewedEdge(
                    reference_value=rv,
                    reference_name=d.reference_name,
                    compare_value=cc,
                    compare_name=class_name(aff.compare_id, cc),
                    probability=p,
                    provenance=(
                        EXPERT_CONFIRMED if cc in confirmed_cvs
                        else ALGORITHM_PROPOSED
                    ),
                )
            )
        # A confirmed edge whose compare class is outside the affinity matrix
        # (absent / unfitted) has no ranked entry; keep it anyway -- the expert
        # asked for it -- at its frozen retained probability.
        in_matrix = set(aff.compare_classes)
        for e in confirmed:
            if e.compare_value in in_matrix:
                continue
            edges.append(
                ReviewedEdge(
                    reference_value=rv,
                    reference_name=d.reference_name,
                    compare_value=e.compare_value,
                    compare_name=class_name(aff.compare_id, e.compare_value),
                    probability=e.retained_probability,
                    provenance=EXPERT_CONFIRMED,
                )
            )
        rows.append(
            ReviewedRow(
                reference_value=rv,
                reference_name=d.reference_name,
                status=d.status,
                edges=edges,
                all_probabilities={
                    int(cc): float(balanced[j])
                    for j, cc in enumerate(aff.compare_classes)
                },
                # With no confirmed edges this is the saved "no matching
                # class" decision (the UI's no-match choice).
                complete=store.is_complete(rv),
            )
        )
    return rows


# --------------------------------------------------------------------------- #
# Convenience: recompute affinity + decisions + reviewed table for a pair
# --------------------------------------------------------------------------- #


def recompute_reviewed_table(
    reference_id: str,
    compare_id: str,
    store: FeedbackStore | None = None,
    aff: AffinityResult | None = None,
) -> tuple[AffinityResult, list[ReviewedRow]]:
    """Recompute affinity/decisions from the (possibly refit) GMMs and re-review.

    Reads the current GMM caches -- which may have been warm-start-refit by
    :func:`warm_start_refit` -- recomputes Stage 4 affinity (memoised on the GMM
    caches' mtimes; a caller that already holds the current result may pass it as
    ``aff``) and decisions, then builds the reviewed table honouring the feedback
    store's confirmed edges. This is the "global re-balancing" of Stage 6.6, and
    it touches unconfirmed edges only: confirmed edges stay frozen in the
    returned table.
    """
    store = store or load_feedback(reference_id, compare_id)
    aff = aff or cached_affinity(reference_id, compare_id)
    decisions, _ = classify_rows(aff)
    absent = absent_decisions(reference_id, aff)
    rows = build_reviewed_table(aff, decisions + absent, store)
    return aff, rows


def refit_all_confirmed(
    reference_id: str,
    compare_id: str,
    store: FeedbackStore | None = None,
) -> list[RefitResult]:
    """Warm-start-refit every reference class that has confirmed edges.

    The explicit "retrain" action: confirming edges is a cheap save-only step;
    when the expert is ready, this refits each confirmed class's GMM from its
    validated samples in one pass (Stage 6.6). The rewritten GMM cache invalidates
    the affinity memo, so the next table recompute re-proposes the open edges.
    """
    store = store or load_feedback(reference_id, compare_id)
    return [
        warm_start_refit(reference_id, compare_id, rv, store)
        for rv in sorted(store.confirmed)
    ]
