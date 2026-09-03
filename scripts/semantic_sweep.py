"""Alpha sweep for the semantic prior (Stage 8c).

Writes one matching table per (alpha, direction) so the effect of the prior can
be read off a one-dimensional curve, with the softmax temperature held fixed at
its section-2 value. Alpha is never co-tuned with T: T shapes how sharply the
distances alone separate classes, and letting both move at once makes neither
attributable.

With ``--reference path.csv`` the sweep also scores each alpha against a
hand-made crosswalk. That reference is an **illustration of the cardinalities the
tool must handle** (one-to-many, many-to-one, zero), not a tuning target: the
pipeline produces its own table, and the reference is used only to read an alpha
off the curve. Treat a high score as "this alpha reproduces known relationships",
never as ground truth.

Reference CSV format, one row per reference class::

    reference_value,compare_values
    10,10|20
    141,

The column pair may also be named ``<Anything>_Code,<Anything>_Code`` (the first
column is the reference side), which is how a hand-made crosswalk usually comes
out of a spreadsheet. Several rows may repeat the same reference class to express
a one-to-many mapping; their targets are unioned.

An empty ``compare_values`` -- blank, ``-``, ``none`` or ``na`` -- means "this
class should have no match". Such a row counts as correct when the pipeline
reports ``orphan`` or ``semantic_orphan``, which is how a legitimate
non-correspondence is scored rather than punished.

Two metrics, per direction:

* **Jaccard** -- per-row overlap between the candidates the table lists and the
  reference set, averaged over scored rows. Rewards getting the whole set right,
  so a one-to-many relationship is not scored as a miss for naming two classes.
* **Top-1 hit rate** -- how often the highest-probability candidate is in the
  reference set. Insensitive to the tail, so it says whether the *primary*
  correspondence is right even when the split is imperfect.

Run:  python scripts/semantic_sweep.py [--reference crosswalk.csv]
      python scripts/semantic_sweep.py --alphas 0,0.5,1.0 --out-dir cache/sweep

Pure CPU: reads the cached GMMs, no GEE. Stages 2-3 must have run for the pair.
"""

from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from harmonizer.affinity import compute_affinity  # noqa: E402
from harmonizer.config import CONFIG  # noqa: E402
from harmonizer.decision import (  # noqa: E402
    absent_decisions,
    build_matching_table,
    classify_rows,
    save_matching_table_csv,
)

DEFAULT_REFERENCE_ID = "hrlc30_africa"
DEFAULT_COMPARE_ID = "worldcover_2020"
DEFAULT_ALPHAS = (0.0, 0.25, 0.5, 0.75, 1.0)
DIRECTIONS = ("reference_to_compare", "compare_to_reference")


@dataclass
class DirectionScore:
    """How one (alpha, direction) table compares to the reference crosswalk."""

    alpha: float
    direction: str
    n_scored: int
    mean_jaccard: float
    top1_hit_rate: float


def _alpha_tag(alpha: float) -> str:
    """A filename-safe alpha tag: 0.25 -> '0p25'."""
    return f"{alpha:g}".replace(".", "p")


# Cell values that all mean "this class has no counterpart". A hand-made
# crosswalk writes this several ways, and reading one of them as a class code
# would silently score a real non-correspondence as a miss.
_NO_MATCH = {"", "-", "--", "none", "na", "n/a", "null"}


def load_reference(path: Path) -> dict[int, set[int]]:
    """Read a reference crosswalk CSV into {reference_value: {compare_values}}.

    Accepts either the canonical ``reference_value,compare_values`` header or a
    ``<X>_Code,<Y>_Code`` pair, taking the first column as the reference side.
    Repeated reference classes are unioned, so a one-to-many mapping may be
    written either as ``10,20|30`` on one row or as two rows.
    """
    out: dict[int, set[int]] = {}
    with path.open(encoding="utf-8-sig", newline="") as fh:
        reader = csv.reader(fh)
        header = next(reader, None)
        if header is None:
            raise ValueError(f"crosswalk is empty: {path}")

        cols = [h.strip().lower() for h in header]
        if "reference_value" in cols:
            ref_i = cols.index("reference_value")
            cmp_i = cols.index("compare_values") if "compare_values" in cols else 1
        elif len(cols) >= 2:
            # A spreadsheet-style header such as HRLC_Code,WorldCover_Code.
            ref_i, cmp_i = 0, 1
        else:
            raise ValueError(
                f"crosswalk needs two columns (reference, compare); got {header!r}"
            )

        for row in reader:
            if len(row) <= max(ref_i, cmp_i):
                continue
            key = row[ref_i].strip()
            if not key or key.lower() in _NO_MATCH:
                continue
            raw = row[cmp_i].strip()
            values = {
                int(v)
                for v in raw.replace(",", "|").split("|")
                if v.strip() and v.strip().lower() not in _NO_MATCH
            }
            # Union rather than overwrite: a one-to-many mapping is often
            # written as one row per target.
            out.setdefault(int(key), set()).update(values)
    return out


def score_rows(rows, reference: dict[int, set[int]]) -> DirectionScore | None:
    """Score one direction's matching table against the reference crosswalk.

    Only reference-side rows whose class appears in the crosswalk are scored;
    everything else is left out rather than counted as a miss, since the
    crosswalk is a partial illustration, not a complete key.
    """
    jaccards: list[float] = []
    hits: list[bool] = []

    for r in rows:
        if r.side != "reference" or r.reference_value not in reference:
            continue
        expected = reference[r.reference_value]
        predicted = set(r.compare_values)

        if not expected:
            # "Should have no match": correct when the pipeline says so too.
            unmatched = r.status in ("orphan", "absent") or r.semantic_orphan
            jaccards.append(1.0 if unmatched else 0.0)
            hits.append(bool(unmatched))
            continue

        union = expected | predicted
        jaccards.append(len(expected & predicted) / len(union) if union else 1.0)
        # Candidates are ranked by probability, so the first is the top-1.
        hits.append(bool(r.compare_values) and r.compare_values[0] in expected)

    if not jaccards:
        return None
    return DirectionScore(
        alpha=float("nan"),
        direction="",
        n_scored=len(jaccards),
        mean_jaccard=sum(jaccards) / len(jaccards),
        top1_hit_rate=sum(hits) / len(hits),
    )


def run_one(
    reference_id: str,
    compare_id: str,
    alpha: float,
    direction: str,
    out_dir: Path,
) -> tuple[Path, list]:
    """Compute one (alpha, direction) table and write it. Returns (path, rows)."""
    # The direction toggle swaps which product indexes the rows; the prior is
    # directed, so building it for the swapped ordered pair orients it correctly.
    if direction == "reference_to_compare":
        row_id, col_id = reference_id, compare_id
    else:
        row_id, col_id = compare_id, reference_id

    aff = compute_affinity(row_id, col_id, alpha=alpha)
    decisions, floor_calibrated = classify_rows(aff)
    rows = build_matching_table(
        aff, decisions, include_absent=absent_decisions(row_id, aff)
    )

    tag = "ref2cmp" if direction == "reference_to_compare" else "cmp2ref"
    path = out_dir / f"matching_table_alpha{_alpha_tag(alpha)}_{tag}.csv"
    save_matching_table_csv(rows, path)
    if not floor_calibrated:
        print("    (floor uncalibrated - statuses are 'unassigned')")
    return path, rows


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--reference-id", default=DEFAULT_REFERENCE_ID)
    ap.add_argument("--compare-id", default=DEFAULT_COMPARE_ID)
    ap.add_argument(
        "--alphas",
        default=",".join(f"{a:g}" for a in DEFAULT_ALPHAS),
        help="comma-separated alpha values (default: 0,0.25,0.5,0.75,1)",
    )
    ap.add_argument(
        "--reference",
        type=Path,
        default=None,
        help="optional crosswalk CSV (reference_value,compare_values) to score against",
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="where to write the tables (default: cache/sweep)",
    )
    args = ap.parse_args(argv)

    alphas = [float(a) for a in args.alphas.split(",") if a.strip()]
    out_dir = args.out_dir or (CONFIG.cache_dir / "sweep")
    out_dir.mkdir(parents=True, exist_ok=True)

    reference = load_reference(args.reference) if args.reference else None

    print(f"pair        : {args.reference_id} x {args.compare_id}")
    print(f"alphas      : {', '.join(f'{a:g}' for a in alphas)}")
    print(f"temperature : {CONFIG.affinity.softmax_temperature} (held fixed)")
    print(f"out dir     : {out_dir}")
    if reference is not None:
        print(f"reference   : {args.reference} ({len(reference)} classes)")

    scores: list[DirectionScore] = []
    for alpha in alphas:
        print(f"\nalpha = {alpha:g}")
        for direction in DIRECTIONS:
            path, rows = run_one(
                args.reference_id, args.compare_id, alpha, direction, out_dir
            )
            print(f"  {direction:<22} -> {path.name}")
            if reference is None or direction != "reference_to_compare":
                # The crosswalk is keyed by reference class, so it only scores
                # the direction whose rows *are* reference classes.
                continue
            s = score_rows(rows, reference)
            if s is None:
                print("    (no scored rows: crosswalk names no class in this table)")
                continue
            scores.append(
                DirectionScore(
                    alpha=alpha,
                    direction=direction,
                    n_scored=s.n_scored,
                    mean_jaccard=s.mean_jaccard,
                    top1_hit_rate=s.top1_hit_rate,
                )
            )
            print(
                f"    scored {s.n_scored} rows: "
                f"Jaccard {s.mean_jaccard:.3f}, top-1 {s.top1_hit_rate:.3f}"
            )

    if scores:
        print("\nCalibration curve (reference_to_compare)")
        print(f"  {'alpha':>7} {'rows':>6} {'Jaccard':>9} {'top-1':>8}")
        for s in scores:
            print(
                f"  {s.alpha:>7.2f} {s.n_scored:>6} "
                f"{s.mean_jaccard:>9.3f} {s.top1_hit_rate:>8.3f}"
            )
        best = max(scores, key=lambda s: (s.mean_jaccard, s.top1_hit_rate))
        print(
            f"\n  best Jaccard at alpha = {best.alpha:g} "
            f"({best.mean_jaccard:.3f}). Set semantic_prior_alpha in config.py "
            "after inspecting the tables, not from this number alone."
        )
    elif reference is not None:
        print("\n  no rows scored - check the crosswalk's reference_value column")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
