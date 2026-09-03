"""Disagreement report: where the pixels and the definitions differ (Stage 8c).

Every class pair carries two independent signals:

* ``s_aef`` -- the raw similarity ``1 / (1 + d)`` from the AlphaEarth embedding
  and the fitted GMMs. Whether the two classes *look* alike.
* ``s_sem`` -- the semantic prior pi from the LCCS attribute encodings. Whether
  one class's *definition* fits inside the other's.

Splitting the pairs on the existing thresholds -- ``absolute_affinity_floor`` for
the observational signal, ``semantic_orphan_floor`` for the semantic one -- gives
four quadrants, and the two off-diagonal ones are the interesting output:

* **agree-match** (both high) -- the classes look alike and mean the same thing.
* **agree-nonmatch** (both low) -- unrelated on both counts. The bulk of any
  matrix, since most class pairs are simply unrelated.
* **spectral-confusion** (aef high, sem low) -- they look alike but mean
  different things. The failure mode Stage 8 exists to catch: cropland and
  grassland in the same season, or a dry herbaceous wetland.
* **definition-drift** (sem high, aef low) -- they mean the same thing but do not
  look alike. Usually a sampling or seasonality artefact rather than a legend
  problem: too few points, a mixed sample, or a class observed at the wrong time
  of year.

Writes ``disagreement.csv`` (one row per class pair) and a scatter PNG with the
two thresholds drawn as quadrant lines.

Run:  python scripts/disagreement_report.py
      python scripts/disagreement_report.py --alpha 1.0 --out-dir cache

Pure CPU: reads the cached GMMs, no GEE. Stages 2-3 must have run for the pair.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402

from harmonizer.affinity import class_name, compute_affinity  # noqa: E402
from harmonizer.config import CONFIG  # noqa: E402

DEFAULT_REFERENCE_ID = "hrlc30_africa"
DEFAULT_COMPARE_ID = "worldcover_2020"

QUADRANTS = (
    "agree-match",
    "agree-nonmatch",
    "spectral-confusion",
    "definition-drift",
)


def quadrant(s_aef: float, s_sem: float, aef_floor: float, sem_floor: float) -> str:
    """Which of the four quadrants a class pair falls in."""
    aef_high = s_aef >= aef_floor
    sem_high = s_sem >= sem_floor
    if aef_high and sem_high:
        return "agree-match"
    if not aef_high and not sem_high:
        return "agree-nonmatch"
    if aef_high:
        return "spectral-confusion"
    return "definition-drift"


def build_rows(aff, aef_floor: float, sem_floor: float) -> list[dict]:
    """One record per class pair, with both signals and the quadrant."""
    rows: list[dict] = []
    for i, rc in enumerate(aff.reference_classes):
        for j, cc in enumerate(aff.compare_classes):
            s_aef = float(aff.raw_similarity[i, j])
            s_sem = float(aff.semantic_prior[i, j])
            rows.append(
                {
                    "reference_value": rc,
                    "reference_name": class_name(aff.reference_id, rc),
                    "compare_value": cc,
                    "compare_name": class_name(aff.compare_id, cc),
                    "s_aef": round(s_aef, 6),
                    "s_sem": round(s_sem, 6),
                    "distance": round(float(aff.distance[i, j]), 6),
                    "quadrant": quadrant(s_aef, s_sem, aef_floor, sem_floor),
                }
            )
    return rows


def save_csv(rows: list[dict], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    return path


def save_scatter(
    rows: list[dict],
    path: Path,
    *,
    aef_floor: float,
    sem_floor: float,
    title: str,
) -> Path | None:
    """Scatter the pairs in (s_aef, s_sem) space with the quadrant lines drawn."""
    try:
        import matplotlib

        matplotlib.use("Agg")  # headless: never try to open a window
        import matplotlib.pyplot as plt
    except ImportError:
        print("  (matplotlib not installed - skipping the scatter PNG)")
        return None

    colors = {
        "agree-match": "#2a9d3f",
        "agree-nonmatch": "#9aa0a6",
        "spectral-confusion": "#d1495b",
        "definition-drift": "#3d7dd8",
    }

    fig, ax = plt.subplots(figsize=(8, 6.5), dpi=140)
    for q in QUADRANTS:
        pts = [r for r in rows if r["quadrant"] == q]
        if not pts:
            continue
        ax.scatter(
            [r["s_aef"] for r in pts],
            [r["s_sem"] for r in pts],
            s=42,
            c=colors[q],
            edgecolor="white",
            linewidth=0.6,
            label=f"{q} ({len(pts)})",
            zorder=3,
        )

    ax.axvline(aef_floor, color="#444", lw=1, ls="--", zorder=2)
    ax.axhline(sem_floor, color="#444", lw=1, ls="--", zorder=2)
    # Both threshold labels sit *inside* the axes: outside the top edge they
    # collide with the title, and outside the right edge they get clipped.
    # Low and to the right of the line: the bottom of the spectral-confusion
    # quadrant is the emptiest part of the plot, so the label lands on nothing.
    ax.text(
        aef_floor + 0.004, 0.02, f"observational floor {aef_floor:g}",
        fontsize=7.5, color="#444", ha="left", va="bottom",
        transform=ax.get_xaxis_transform(),
    )
    ax.text(
        0.995, sem_floor + 0.012, f"semantic floor {sem_floor:g}",
        fontsize=7.5, color="#444", ha="right", va="bottom",
        transform=ax.get_yaxis_transform(),
    )

    # Label only the off-diagonal pairs: those are the ones worth reading, and
    # labelling all 90 would be unreadable. Even so, definition-drift clusters
    # hard along the semantic floor, so nudge each label off its neighbours --
    # a plain annotate() there produces overlapping clumps that read as noise.
    labelled = [
        r for r in rows
        if r["quadrant"] in ("spectral-confusion", "definition-drift")
    ]
    placed: list[tuple[float, float]] = []
    for r in sorted(labelled, key=lambda r: (-r["s_sem"], -r["s_aef"])):
        x, y = r["s_aef"], r["s_sem"]
        dx, dy = 6.0, 4.0
        # Push the label up in small steps until it clears the ones already
        # placed near this point (data coords: ~0.012 in y is one text line).
        for _ in range(12):
            if all(
                abs(x - px) > 0.022 or abs((y + dy / 380) - py) > 0.028
                for px, py in placed
            ):
                break
            dy += 9.0
        placed.append((x, y + dy / 380))
        ax.annotate(
            f"{r['reference_value']}->{r['compare_value']}",
            (x, y),
            textcoords="offset points",
            xytext=(dx, dy),
            fontsize=6.5,
            color="#222",
            zorder=4,
        )

    ax.set_xlabel("observational similarity  s_aef = 1 / (1 + d)")
    ax.set_ylabel("semantic prior  s_sem = pi")
    ax.set_title(title, fontsize=11)
    ax.set_ylim(-0.02, 1.05)
    ax.grid(alpha=0.25, zorder=1)
    ax.legend(loc="lower left", fontsize=8, framealpha=0.95)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    return path


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--reference-id", default=DEFAULT_REFERENCE_ID)
    ap.add_argument("--compare-id", default=DEFAULT_COMPARE_ID)
    ap.add_argument(
        "--alpha",
        type=float,
        default=None,
        help="alpha for the run (affects only the fused rows, not the two signals)",
    )
    ap.add_argument("--out-dir", type=Path, default=None)
    args = ap.parse_args(argv)

    out_dir = args.out_dir or CONFIG.cache_dir
    aef_floor = CONFIG.affinity.absolute_affinity_floor
    sem_floor = CONFIG.affinity.semantic_orphan_floor

    if aef_floor is None:
        print(
            "FAILED: absolute_affinity_floor is None (uncalibrated), so the\n"
            "        observational axis has no threshold to split on. Set it in\n"
            "        config.py from the raw-similarity distribution first."
        )
        return 1

    aff = compute_affinity(args.reference_id, args.compare_id, alpha=args.alpha)
    rows = build_rows(aff, aef_floor, sem_floor)

    print(f"pair            : {args.reference_id} x {args.compare_id}")
    print(f"class pairs     : {len(rows)} ({len(aff.reference_classes)} x {len(aff.compare_classes)})")
    print(f"observational floor : {aef_floor}")
    print(f"semantic floor      : {sem_floor}")

    counts = {q: sum(1 for r in rows if r["quadrant"] == q) for q in QUADRANTS}
    print("\nQuadrants")
    for q in QUADRANTS:
        print(f"  {q:<20} {counts[q]:>4}")
    print(f"  {'total':<20} {sum(counts.values()):>4}")

    for q in ("spectral-confusion", "definition-drift"):
        pairs = [r for r in rows if r["quadrant"] == q]
        if not pairs:
            continue
        print(f"\n{q} ({len(pairs)})")
        for r in sorted(pairs, key=lambda r: -r["s_aef"]):
            print(
                f"  {r['reference_value']:>4} {r['reference_name'][:26]:<26} -> "
                f"{r['compare_value']:>4} {r['compare_name'][:22]:<22} "
                f"aef {r['s_aef']:.3f}  sem {r['s_sem']:.3f}"
            )

    csv_path = save_csv(rows, out_dir / "disagreement.csv")
    print(f"\nwrote {csv_path}")
    png_path = save_scatter(
        rows,
        out_dir / "disagreement.png",
        aef_floor=aef_floor,
        sem_floor=sem_floor,
        title=f"{args.reference_id} -> {args.compare_id}: observational vs semantic",
    )
    if png_path:
        print(f"wrote {png_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
