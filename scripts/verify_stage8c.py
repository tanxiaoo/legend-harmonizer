"""Stage 8c verification: the alpha sweep and the disagreement report.

Runs both Stage 8c scripts on the cached ``hrlc30_africa`` x ``worldcover_2020``
pair and checks their outputs (docs/PIPELINE.md, Stage 8c -> Verification):

1. The sweep writes one table per (alpha, direction).
2. The alpha = 0 sweep table equals the Stage 8b regression output -- the same
   matching table ``compute_affinity(..., alpha=0)`` produces. This is what ties
   the sweep to the verified pipeline rather than to a reimplementation of it.
3. The quadrant counts sum to M x N, so every class pair is classified exactly
   once and none is silently dropped.
4. The disagreement CSV's own rows agree with a recomputation of the quadrant
   rule, and the PNG is written when matplotlib is available.
5. The sweep's scoring metrics behave: a crosswalk built from a table's own
   output scores perfectly against that table, which exercises the Jaccard and
   top-1 paths including the empty-set (orphan) case.

Everything runs in a temp directory except the cached GMMs, so the real
``cache/`` artifacts are not overwritten.

Run:  python scripts/verify_stage8c.py

Pure CPU, no GEE. Stages 2-3 must have run for the pair.
"""

from __future__ import annotations

import csv
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import disagreement_report as dr  # noqa: E402
import semantic_sweep as sweep  # noqa: E402

from harmonizer.affinity import compute_affinity  # noqa: E402
from harmonizer.config import CONFIG  # noqa: E402
from harmonizer.decision import (  # noqa: E402
    absent_decisions,
    build_matching_table,
    classify_rows,
    save_matching_table_csv,
)

REFERENCE_ID = "hrlc30_africa"
COMPARE_ID = "worldcover_2020"
ALPHAS = (0.0, 0.25, 0.5, 0.75, 1.0)


def _rule(title: str) -> None:
    print(f"\n{title}\n{'-' * len(title)}")


def check(ok: bool, message: str, detail: str = "") -> bool:
    print(f"  {'PASS' if ok else 'FAIL'}  {message}" + (f" [{detail}]" if detail else ""))
    return bool(ok)


def _read_csv(path: Path) -> list[list[str]]:
    with path.open(encoding="utf-8", newline="") as fh:
        return list(csv.reader(fh))


def main() -> int:
    ok = True

    from harmonizer.modeling import gmm_cache_path

    for pid in (REFERENCE_ID, COMPARE_ID):
        if not gmm_cache_path(pid).exists():
            print(f"FAILED: no cached GMM for {pid}; run Stages 2-3 first.")
            return 1

    tmp = Path(tempfile.mkdtemp(prefix="verify8c_"))
    print(f"working in {tmp}")

    # ---------------------------------------------------------------- 1. #
    _rule("1. Sweep writes a table per (alpha, direction)")
    sweep_dir = tmp / "sweep"
    rc = sweep.main(
        [
            "--reference-id", REFERENCE_ID,
            "--compare-id", COMPARE_ID,
            "--alphas", ",".join(f"{a:g}" for a in ALPHAS),
            "--out-dir", str(sweep_dir),
        ]
    )
    ok &= check(rc == 0, "semantic_sweep.py exited cleanly", f"rc={rc}")

    written = sorted(p.name for p in sweep_dir.glob("matching_table_alpha*.csv"))
    expected_n = len(ALPHAS) * len(sweep.DIRECTIONS)
    ok &= check(
        len(written) == expected_n,
        f"sweep wrote {expected_n} tables (one per alpha x direction)",
        f"{len(written)} files",
    )

    # ---------------------------------------------------------------- 2. #
    _rule("2. alpha=0 sweep table == the Stage 8b regression output")

    aff0 = compute_affinity(REFERENCE_ID, COMPARE_ID, alpha=0.0)
    decisions0, _ = classify_rows(aff0)
    rows0 = build_matching_table(
        aff0, decisions0, include_absent=absent_decisions(REFERENCE_ID, aff0)
    )
    direct = save_matching_table_csv(rows0, tmp / "direct_alpha0.csv")

    swept = sweep_dir / "matching_table_alpha0_ref2cmp.csv"
    ok &= check(swept.exists(), "alpha=0 reference_to_compare table exists", swept.name)
    if swept.exists():
        ok &= check(
            _read_csv(direct) == _read_csv(swept),
            "sweep's alpha=0 table is identical to the direct Stage 8b table",
        )

    # A non-zero alpha must actually differ, or the sweep is not sweeping.
    swept1 = sweep_dir / "matching_table_alpha1_ref2cmp.csv"
    if swept1.exists():
        ok &= check(
            _read_csv(swept) != _read_csv(swept1),
            "alpha=1 table differs from alpha=0 (the sweep has an effect)",
        )

    # ---------------------------------------------------------------- 3. #
    _rule("3. Disagreement quadrants sum to M x N")

    report_dir = tmp / "report"
    rc = dr.main(
        [
            "--reference-id", REFERENCE_ID,
            "--compare-id", COMPARE_ID,
            "--out-dir", str(report_dir),
        ]
    )
    ok &= check(rc == 0, "disagreement_report.py exited cleanly", f"rc={rc}")

    csv_path = report_dir / "disagreement.csv"
    ok &= check(csv_path.exists(), "disagreement.csv written")

    m, n = len(aff0.reference_classes), len(aff0.compare_classes)
    data = _read_csv(csv_path)
    body = data[1:]
    ok &= check(
        len(body) == m * n,
        "one CSV row per class pair",
        f"{len(body)} rows vs {m} x {n} = {m * n}",
    )

    header = data[0]
    qi = header.index("quadrant")
    counts: dict[str, int] = {}
    for row in body:
        counts[row[qi]] = counts.get(row[qi], 0) + 1
    ok &= check(
        sum(counts.values()) == m * n,
        "quadrant counts sum to M x N",
        ", ".join(f"{k}={v}" for k, v in sorted(counts.items())),
    )
    ok &= check(
        set(counts) <= set(dr.QUADRANTS),
        "every row carries a known quadrant name",
        f"{sorted(set(counts))}",
    )

    # ---------------------------------------------------------------- 4. #
    _rule("4. Quadrant labels match a recomputation of the rule")

    aef_floor = CONFIG.affinity.absolute_affinity_floor
    sem_floor = CONFIG.affinity.semantic_orphan_floor
    ai, si = header.index("s_aef"), header.index("s_sem")
    mismatched = [
        row
        for row in body
        if dr.quadrant(float(row[ai]), float(row[si]), aef_floor, sem_floor) != row[qi]
    ]
    ok &= check(
        not mismatched,
        "every CSV row's quadrant matches the rule applied to its own values",
        f"{len(mismatched)} mismatch(es)",
    )

    png = report_dir / "disagreement.png"
    try:
        import matplotlib  # noqa: F401

        ok &= check(png.exists() and png.stat().st_size > 5000, "scatter PNG written",
                    f"{png.stat().st_size // 1024} KB" if png.exists() else "missing")
    except ImportError:
        print("  SKIP  matplotlib not installed, no PNG expected")

    # ---------------------------------------------------------------- 5. #
    _rule("5. Sweep scoring metrics")

    # Build a crosswalk from the alpha=1 table's own candidates: scoring that
    # table against it must be perfect. This exercises Jaccard, top-1, and the
    # empty-set path (a row the pipeline calls orphan is expected to have none).
    aff1 = compute_affinity(REFERENCE_ID, COMPARE_ID, alpha=1.0)
    d1, _ = classify_rows(aff1)
    rows1 = build_matching_table(aff1, d1)

    xw = tmp / "self_crosswalk.csv"
    with xw.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["reference_value", "compare_values"])
        for r in rows1:
            if r.side != "reference":
                continue
            expected = [] if r.status == "orphan" else r.compare_values
            w.writerow([r.reference_value, "|".join(str(v) for v in expected)])

    loaded = sweep.load_reference(xw)
    ok &= check(
        len(loaded) == len([r for r in rows1 if r.side == "reference"]),
        "crosswalk loader reads every reference row",
        f"{len(loaded)} classes",
    )
    empties = [k for k, v in loaded.items() if not v]
    print(f"  note  classes with an empty expected set: {empties or 'none'}")

    score = sweep.score_rows(rows1, loaded)
    ok &= check(score is not None, "scoring returned a result")
    if score is not None:
        ok &= check(
            abs(score.mean_jaccard - 1.0) < 1e-9 and abs(score.top1_hit_rate - 1.0) < 1e-9,
            "a table scored against its own output is perfect",
            f"Jaccard {score.mean_jaccard:.3f}, top-1 {score.top1_hit_rate:.3f}, "
            f"n={score.n_scored}",
        )

    # A deliberately wrong crosswalk must NOT score perfectly, or the metric is
    # not measuring anything.
    wrong = tmp / "wrong_crosswalk.csv"
    with wrong.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["reference_value", "compare_values"])
        for r in rows1:
            if r.side == "reference":
                w.writerow([r.reference_value, "999"])
    bad = sweep.score_rows(rows1, sweep.load_reference(wrong))
    ok &= check(
        bad is not None and bad.mean_jaccard == 0.0 and bad.top1_hit_rate == 0.0,
        "a wrong crosswalk scores zero (the metric discriminates)",
        f"Jaccard {bad.mean_jaccard:.3f}, top-1 {bad.top1_hit_rate:.3f}" if bad else "",
    )

    print(f"\n  artifacts left in {tmp}")
    print(f"\n{'OK' if ok else 'FAILED'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
