"""Decision and matching table (Stage 4, part 2).

Turns the affinity matrices from ``harmonizer.affinity`` into the primary
deliverable: a per-class classification (``strong`` / ``mixed`` / ``orphan`` /
``absent``) and the matching table. Also emits the raw-similarity distribution
needed to calibrate the absolute-affinity floor, and writes the three Stage 4
CSVs (docs/PIPELINE.md, Stage 4).

Two signals drive the classification, used together (section 2 -> Output
statuses):
  * the row's **best raw similarity** vs the absolute-affinity floor -- is there a
    real match at all;
  * the row's **normalised entropy** vs the high-entropy threshold -- is the match
    focused or spread.

Mapping:
  * best raw >= floor  and entropy <  threshold -> ``strong``
  * best raw >= floor  and entropy >= threshold -> ``mixed``
  * best raw <  floor                            -> ``orphan``
  * declared in the legend but never modelled    -> ``absent`` (+ a reason)

**Symmetric absence (Stage 7a).** Statuses are assigned to every class **declared**
in either map's registry legend, not only to those observed in the AOI, and on both
sides. An ``absent`` class carries a reason (``not_in_aoi`` / ``too_rare``, see
``harmonizer.absence``); absent compare classes appear as unmatched-target rows
(``side="compare"``) rather than silently losing their matrix column.

**Uncalibrated-floor guard.** The absolute-affinity floor defaults to ``None``
(section 2) and can only be set from real data. While it is ``None`` the orphan
test must not run: this module warns and leaves orphan status **unassigned**
(``status = "unassigned"``) rather than defaulting the floor to 0 (which would
pass everything) or to some value (which could mark every class an orphan). The
intended flow is: run once to emit the raw-similarity distribution, set
``CONFIG.affinity.absolute_affinity_floor`` from it, then re-run to assign
orphan status.

A **direction toggle** lets the same underlying matrix be read
compare-to-reference as well as reference-to-compare: the affinity is recomputed
with the roles swapped (rows become the other map's classes), because the
row-normalisation and entropy are direction-specific and cannot be recovered by
transposing the normalised matrix.

Constants (entropy threshold, absolute-affinity floor) come from
``CONFIG.affinity``; nothing is hardcoded here.
"""

from __future__ import annotations

import csv
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from harmonizer.affinity import AffinityResult, class_name, compute_affinity
from harmonizer.config import CONFIG

# Status returned when the floor is uncalibrated so the orphan test cannot run.
UNASSIGNED = "unassigned"

# Sentinel for "floor argument not supplied" so callers can pass floor=None
# explicitly to *force* the uncalibrated path, distinct from omitting the
# argument (which reads CONFIG.affinity.absolute_affinity_floor).
_USE_CONFIG = object()


# --------------------------------------------------------------------------- #
# Raw-similarity distribution (for calibrating the absolute-affinity floor)
# --------------------------------------------------------------------------- #


@dataclass
class RawSimilaritySummary:
    """The raw (pre-normalisation) similarity distribution emitted for calibration.

    ``best_per_row`` is each reference row's best raw similarity across all compare
    classes -- the exact quantity the orphan test checks against the floor -- so
    the floor can be read off real data. ``min``/``median``/``max`` and a small
    histogram summarise the whole matrix.
    """

    best_per_row: list[float]
    all_min: float
    all_median: float
    all_max: float
    histogram_counts: list[int]
    histogram_edges: list[float]


def raw_similarity_summary(aff: AffinityResult, n_bins: int = 10) -> RawSimilaritySummary:
    """Summarise the raw-similarity distribution of an affinity result."""
    raw = aff.raw_similarity
    best_per_row = raw.max(axis=1) if raw.size else np.array([])
    flat = raw.reshape(-1)
    counts, edges = np.histogram(flat, bins=n_bins, range=(0.0, 1.0))
    return RawSimilaritySummary(
        best_per_row=[float(x) for x in best_per_row],
        all_min=float(flat.min()) if flat.size else float("nan"),
        all_median=float(np.median(flat)) if flat.size else float("nan"),
        all_max=float(flat.max()) if flat.size else float("nan"),
        histogram_counts=[int(c) for c in counts],
        histogram_edges=[float(e) for e in edges],
    )


# --------------------------------------------------------------------------- #
# Classification
# --------------------------------------------------------------------------- #


@dataclass
class ClassDecision:
    """The Stage 4 decision for a single reference row."""

    reference_value: int
    reference_name: str
    status: str                      # strong | mixed | orphan | absent | unassigned
    best_compare_value: int
    best_compare_name: str
    best_raw_similarity: float
    best_probability: float
    margin: float                    # top1 - top2 of the normalised row; the classifier
    entropy: float                   # reported only, not the classifier
    low_confidence: bool             # propagated from the reference class GMM
    # Why an ``absent`` class could not be modelled (Stage 7): not_in_aoi | too_rare.
    # Empty for every other status.
    absence_reason: str = ""
    # Full normalised probability row over compare classes, for the table split.
    probabilities: dict[int, float] = field(default_factory=dict)


def classify_rows(
    aff: AffinityResult,
    *,
    floor: float | None = _USE_CONFIG,
    margin_threshold: float | None = None,
) -> tuple[list[ClassDecision], bool]:
    """Classify every reference row into a status.

    ``floor`` omitted reads ``CONFIG.affinity.absolute_affinity_floor``; passing
    ``floor=None`` explicitly forces the uncalibrated path (to exercise the
    guard), and passing a number overrides the config. ``margin_threshold``
    defaults to ``CONFIG.affinity.margin_threshold``. Returns
    the decisions and a flag ``floor_calibrated`` -- ``False`` when the floor is
    ``None`` and the orphan test was therefore skipped (statuses are
    ``unassigned`` for rows that would otherwise be orphans).

    strong/mixed comes from the **top1-top2 margin** of the normalised (softmax)
    row, not entropy: ``margin >= margin_threshold`` is ``strong``, else
    ``mixed``. Entropy inverts on two-way ties (a clean single winner with a thin
    tail can out-score a genuine two-way split), so it is reported but not used to
    classify. orphan still comes from the absolute-affinity floor on the raw
    similarity.

    Uncalibrated-floor guard: when ``floor`` is ``None`` the orphan test cannot
    run, so *no* status is trustworthy (a would-be orphan cannot be told apart
    from a strong match without the floor). All rows are marked ``unassigned`` and
    a warning is emitted, matching the intended two-pass flow.
    """
    if floor is _USE_CONFIG:
        floor = CONFIG.affinity.absolute_affinity_floor
    if margin_threshold is None:
        margin_threshold = CONFIG.affinity.margin_threshold

    floor_calibrated = floor is not None
    if not floor_calibrated:
        warnings.warn(
            "absolute_affinity_floor is None (uncalibrated): the orphan test "
            "cannot run, so statuses are left 'unassigned'. Read the floor off the "
            "emitted raw-similarity distribution, set "
            "CONFIG.affinity.absolute_affinity_floor, then re-run.",
            stacklevel=2,
        )

    decisions: list[ClassDecision] = []
    for i, rc in enumerate(aff.reference_classes):
        raw_row = aff.raw_similarity[i]
        prob_row = aff.normalized_affinity[i]
        entropy = float(aff.row_entropy[i])

        best_j = int(np.argmax(raw_row))
        best_cc = aff.compare_classes[best_j]
        best_raw = float(raw_row[best_j])
        best_prob = float(prob_row[best_j])

        # top1 - top2 margin of the normalised row (0.0 if only one compare class).
        if prob_row.shape[0] >= 2:
            top2 = np.partition(prob_row, -2)[-2:]
            margin = float(top2[1] - top2[0])
        else:
            margin = float(prob_row[best_j]) if prob_row.size else 0.0

        if not floor_calibrated:
            status = UNASSIGNED
        elif best_raw < floor:
            status = "orphan"
        elif margin >= margin_threshold:
            status = "strong"
        else:
            status = "mixed"

        decisions.append(
            ClassDecision(
                reference_value=rc,
                reference_name=class_name(aff.reference_id, rc),
                status=status,
                best_compare_value=best_cc,
                best_compare_name=class_name(aff.compare_id, best_cc),
                best_raw_similarity=best_raw,
                best_probability=best_prob,
                margin=margin,
                entropy=entropy,
                low_confidence=bool(aff.reference_low_confidence.get(rc, False)),
                probabilities={
                    cc: float(prob_row[j])
                    for j, cc in enumerate(aff.compare_classes)
                },
            )
        )

    return decisions, floor_calibrated


# --------------------------------------------------------------------------- #
# Absent classes (carried through from Stage 3, never fitted)
# --------------------------------------------------------------------------- #


def absent_decisions(reference_id: str, aff: AffinityResult) -> list[ClassDecision]:
    """Decisions for reference classes that carry status ``absent`` (no GMM, no row).

    Driven by the **declared** registry legend, not by what was observed (Stage 7a):
    a class with no pixels in the AOI is never sampled and so never reaches the
    Stage 3 cache at all. Reading only that cache -- as this did before Stage 7 --
    therefore missed exactly the classes that most needed reporting, and they
    vanished from the matching table with no trace.

    ``harmonizer.absence`` reconciles declared against modelled and supplies the
    reason (``not_in_aoi`` / ``too_rare``) each absent class carries through.
    """
    from harmonizer.absence import absent_classes

    out: list[ClassDecision] = []
    for a in absent_classes(reference_id):
        out.append(
            ClassDecision(
                reference_value=a.class_value,
                reference_name=a.class_name,
                status="absent",
                best_compare_value=-1,
                best_compare_name="",
                best_raw_similarity=float("nan"),
                best_probability=float("nan"),
                margin=float("nan"),
                entropy=float("nan"),
                low_confidence=True,  # absent classes are inherently low-confidence
                absence_reason=a.reason,
            )
        )
    return out


# --------------------------------------------------------------------------- #
# Matching table
# --------------------------------------------------------------------------- #


@dataclass
class MatchingRow:
    """One row of the matching table (one reference class's mapping).

    ``compare_values``/``compare_names``/``probabilities`` list the compare
    class(es) this reference class maps to, ranked by probability. For ``strong``
    the top candidate carries the mass; for ``mixed`` several do; for ``orphan``
    the best is below the floor; for ``absent`` there is no candidate.
    """

    reference_value: int
    reference_name: str
    status: str
    compare_values: list[int]
    compare_names: list[str]
    probabilities: list[float]
    best_raw_similarity: float
    margin: float                    # top1-top2 of the normalised row (the classifier)
    entropy: float                   # reported only
    reference_low_confidence: bool
    compare_low_confidence: list[bool]
    # Stage 7: why an ``absent`` row could not be modelled (empty otherwise), and
    # which side the row describes -- "reference" for a normal row, "compare" for an
    # absent compare class reported as an unmatched target.
    absence_reason: str = ""
    side: str = "reference"
    # Stage 7c: which AOI's evidence produced this row -- "primary" (default),
    # an auxiliary AOI's name, "none" for an expert-declared mapping on a class
    # no AOI could evidence, or "" for an absent row with no evidence at all.
    # Probabilities normalise within the named AOI's sub-matrix, so rows are
    # only comparable to other rows carrying the same tag (docs 7.4).
    evidence_aoi: str = "primary"


def _rank_candidates(
    decision: ClassDecision,
    aff: AffinityResult,
    *,
    prob_cutoff: float,
) -> tuple[list[int], list[str], list[float], list[bool]]:
    """Rank a row's compare candidates by probability, keeping meaningful ones.

    Always keeps the top candidate; also keeps any other candidate whose
    probability is at least ``prob_cutoff`` of the top, so a ``mixed`` row shows
    its genuine split rather than a long tail of near-zeros.
    """
    items = sorted(
        decision.probabilities.items(), key=lambda kv: kv[1], reverse=True
    )
    if not items:
        return [], [], [], []
    top_prob = items[0][1]
    threshold = max(prob_cutoff * top_prob, 0.0)
    kept = [(cc, p) for cc, p in items if p >= threshold] or [items[0]]

    values = [cc for cc, _ in kept]
    names = [class_name(aff.compare_id, cc) for cc in values]
    probs = [round(p, 6) for _, p in kept]
    low_conf = [bool(aff.compare_low_confidence.get(cc, False)) for cc in values]
    return values, names, probs, low_conf


def build_matching_table(
    aff: AffinityResult,
    decisions: list[ClassDecision],
    *,
    prob_cutoff: float = 0.25,
    include_absent: list[ClassDecision] | None = None,
    include_compare_absent: bool = True,
) -> list[MatchingRow]:
    """Assemble the matching table from classified rows.

    ``prob_cutoff`` controls how many compare candidates a row lists (a candidate
    is kept if its probability is at least ``prob_cutoff`` of the top candidate's).
    ``include_absent`` appends ``absent`` reference classes (which have no matrix
    row) as candidate-less rows.

    ``include_compare_absent`` (Stage 7a) appends the **compare** side's absent
    classes as unmatched-target rows (``side="compare"``). Before Stage 7 an absent
    compare class simply lost its matrix column and disappeared, so a reader could
    not see which compare classes the run was unable to offer as candidates. These
    rows describe a compare class, so their ``reference_*`` fields are empty.
    """
    rows: list[MatchingRow] = []
    for d in decisions:
        values, names, probs, low_conf = _rank_candidates(
            d, aff, prob_cutoff=prob_cutoff
        )
        rows.append(
            MatchingRow(
                reference_value=d.reference_value,
                reference_name=d.reference_name,
                status=d.status,
                compare_values=values,
                compare_names=names,
                probabilities=probs,
                best_raw_similarity=d.best_raw_similarity,
                margin=d.margin,
                entropy=d.entropy,
                reference_low_confidence=d.low_confidence,
                compare_low_confidence=low_conf,
                absence_reason=d.absence_reason,
            )
        )

    for d in include_absent or []:
        rows.append(
            MatchingRow(
                reference_value=d.reference_value,
                reference_name=d.reference_name,
                status="absent",
                compare_values=[],
                compare_names=[],
                probabilities=[],
                best_raw_similarity=d.best_raw_similarity,
                margin=d.margin,
                entropy=d.entropy,
                reference_low_confidence=d.low_confidence,
                compare_low_confidence=[],
                absence_reason=d.absence_reason,
                # No AOI evidenced this class (Stage 7c: absent rows carry no
                # evidence_aoi; an expert edge later tags it "none" instead).
                evidence_aoi="",
            )
        )

    if include_compare_absent:
        from harmonizer.absence import compare_absent_classes

        for a in compare_absent_classes(aff.compare_id):
            rows.append(
                MatchingRow(
                    reference_value=-1,
                    reference_name="",
                    status="absent",
                    compare_values=[a.class_value],
                    compare_names=[a.class_name],
                    probabilities=[],
                    best_raw_similarity=float("nan"),
                    margin=float("nan"),
                    entropy=float("nan"),
                    reference_low_confidence=False,
                    compare_low_confidence=[True],
                    absence_reason=a.reason,
                    side="compare",
                    evidence_aoi="",
                )
            )
    return rows


# --------------------------------------------------------------------------- #
# CSV output (the three Stage 4 artifacts)
# --------------------------------------------------------------------------- #


def _matrix_csv(
    path: Path,
    matrix: np.ndarray,
    aff: AffinityResult,
) -> None:
    """Write an M x N matrix with reference-class row headers and compare headers."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        header = ["reference"] + [
            f"{cc}:{class_name(aff.compare_id, cc)}" for cc in aff.compare_classes
        ]
        w.writerow(header)
        for i, rc in enumerate(aff.reference_classes):
            label = f"{rc}:{class_name(aff.reference_id, rc)}"
            w.writerow([label] + [f"{v:.6f}" for v in matrix[i]])


def save_raw_similarity_csv(aff: AffinityResult, path: Path | None = None) -> Path:
    """Write ``raw_similarity.csv`` (the calibration/orphan-detection matrix)."""
    path = path or (CONFIG.cache_dir / "raw_similarity.csv")
    _matrix_csv(path, aff.raw_similarity, aff)
    return path


def save_normalized_affinity_csv(aff: AffinityResult, path: Path | None = None) -> Path:
    """Write ``normalized_affinity.csv`` (the probability/entropy matrix)."""
    path = path or (CONFIG.cache_dir / "normalized_affinity.csv")
    _matrix_csv(path, aff.normalized_affinity, aff)
    return path


def save_matching_table_csv(
    rows: list[MatchingRow], path: Path | None = None
) -> Path:
    """Write the matching table CSV -- the primary Stage 4 deliverable.

    Compare candidates and their probabilities are ``|``-joined so one row per
    reference class stays one CSV row. The confidence columns carry the propagated
    low-confidence markers (section 2 -> Confidence column).
    """
    path = path or (CONFIG.cache_dir / "matching_table.csv")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(
            [
                "side",
                "evidence_aoi",
                "reference_value",
                "reference_name",
                "status",
                "absence_reason",
                "compare_values",
                "compare_names",
                "mapping_probabilities",
                "best_raw_similarity",
                "margin",
                "entropy",
                "reference_low_confidence",
                "compare_low_confidence",
            ]
        )
        for r in rows:
            w.writerow(
                [
                    r.side,
                    r.evidence_aoi,
                    "" if r.reference_value < 0 else r.reference_value,
                    r.reference_name,
                    r.status,
                    r.absence_reason,
                    "|".join(str(v) for v in r.compare_values),
                    "|".join(r.compare_names),
                    "|".join(f"{p:.4f}" for p in r.probabilities),
                    "" if np.isnan(r.best_raw_similarity) else f"{r.best_raw_similarity:.6f}",
                    "" if np.isnan(r.margin) else f"{r.margin:.4f}",
                    "" if np.isnan(r.entropy) else f"{r.entropy:.4f}",
                    str(r.reference_low_confidence),
                    "|".join(str(b) for b in r.compare_low_confidence),
                ]
            )
    return path


# --------------------------------------------------------------------------- #
# Direction toggle
# --------------------------------------------------------------------------- #


def compute_affinity_directed(
    reference_id: str,
    compare_id: str,
    *,
    direction: str = "reference_to_compare",
) -> AffinityResult:
    """Affinity for a chosen reading direction.

    ``reference_to_compare`` (default) puts reference classes on the rows;
    ``compare_to_reference`` swaps the roles so compare classes are the rows. The
    affinity is *recomputed* with roles swapped rather than transposed, because
    row-normalisation and entropy are direction-specific (docs/PIPELINE.md,
    Stage 4 -> direction toggle).
    """
    if direction == "reference_to_compare":
        return compute_affinity(reference_id, compare_id)
    if direction == "compare_to_reference":
        return compute_affinity(compare_id, reference_id)
    raise ValueError(
        f"Unknown direction {direction!r}; expected 'reference_to_compare' or "
        "'compare_to_reference'."
    )
