"""Stage 7a verification: symmetric absence in the output.

Before Stage 7 the matching table listed only the classes **observed** in the AOI:
Stage 2 builds its class list from ``present_classes(...)``, so a legend class with
no pixels there never entered sampling and never appeared in the table -- not even
as ``absent``. A reader could not tell a class that was *never considered* from one
that was considered and matched. Absent **compare** classes were worse: they simply
lost their matrix column with no trace at all.

This script proves both are fixed, reading only the caches already on disk (no GEE,
no network):

  1. Prints each map's **declared** legend size vs how many classes were modelled,
     and lists every absent class with its reason (``not_in_aoi`` / ``too_rare``).
  2. Rebuilds the matching table and asserts it now carries **one row per declared
     reference class** -- 11 for WorldCover, not the 8 that were observed.
  3. Asserts the absent **compare** classes appear as unmatched-target rows
     (``side="compare"``) rather than vanishing.
  4. Writes ``matching_table.csv`` with the new ``side`` / ``absence_reason``
     columns and prints it.
  5. Checks the **review page's crosswalk export** (``/api/review/export``) marks
     absent classes too -- it is a separate path through ``review.py``, and an
     absent class there used to export as a blank cell, indistinguishable from a
     class the expert simply had not reviewed yet.
  6. Checks an absent class is **confirmable** (Stage 7.5): an expert may declare a
     correspondence the AOI cannot evidence, and that decision must persist and
     reach the export. This is the only way an absent class ever gets a mapping.

Expected on the Sahel test AOI (docs/PIPELINE.md, Stage 7 -> Verification):
WorldCover declares 11 classes, 8 were modelled, so Snow and ice (70), Mangroves
(95), and Moss and lichen (100) come back ``absent`` + ``not_in_aoi``. Dynamic World
declares 9, 8 were modelled, so Snow and ice (8) is reported as an unmatched
compare-side target.

Run:  python scripts/verify_stage7a.py

Stages 2-3 must have run first (``cache/gmm_*.json`` and ``cache/samples_*.json``
must exist). Pure CPU -- no network.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from harmonizer.absence import (  # noqa: E402
    absent_classes,
    declared_classes,
    undeclared_codes,
)
from harmonizer.affinity import compute_affinity  # noqa: E402
from harmonizer.decision import (  # noqa: E402
    absent_decisions,
    build_matching_table,
    classify_rows,
    save_matching_table_csv,
)
from harmonizer.registry.legends import legend_classes  # noqa: E402

REFERENCE_ID = "worldcover"
COMPARE_ID = "dynamicworld"


def _rule(title: str) -> None:
    print(f"\n{title}\n{'-' * len(title)}")


def report_side(product_id: str, label: str) -> list:
    """Print one map's declared-vs-modelled accounting; return its absent classes."""
    declared = declared_classes(product_id)
    absent = absent_classes(product_id)
    modelled = len(declared) - len(absent)

    print(f"\n{label} ({product_id})")
    print(f"  declared in registry legend : {len(declared)}")
    print(f"  modelled (fitted GMM)       : {modelled}")
    print(f"  absent                      : {len(absent)}")
    for a in absent:
        seen = "observed" if a.observed else "not observed"
        pts = f", {a.n_points} pts" if a.observed else ""
        print(f"    - {a.class_value:>3}  {a.class_name:<24} {a.reason:<11} ({seen}{pts})")

    undeclared = undeclared_codes(product_id)
    if undeclared:
        print(f"  !! UNDECLARED CODES (registry YAML incomplete): {undeclared}")
    return absent


def main() -> int:
    _rule("Stage 7a - symmetric absence accounting")
    ref_absent = report_side(REFERENCE_ID, "REFERENCE")
    cmp_absent = report_side(COMPARE_ID, "COMPARE")

    _rule("Matching table (every declared reference class accounted for)")
    aff = compute_affinity(REFERENCE_ID, COMPARE_ID)
    decisions, floor_calibrated = classify_rows(aff)
    absent = absent_decisions(REFERENCE_ID, aff)
    rows = build_matching_table(aff, decisions, include_absent=absent)

    if not floor_calibrated:
        print("  (floor uncalibrated - statuses are 'unassigned'; see Stage 4)")

    ref_rows = [r for r in rows if r.side == "reference"]
    cmp_rows = [r for r in rows if r.side == "compare"]

    print(f"\n  {'side':<10} {'class':<28} {'status':<10} {'reason':<11} candidates")
    for r in rows:
        if r.side == "reference":
            cls = f"{r.reference_value}:{r.reference_name}"
            cands = (
                ", ".join(
                    f"{n} {p:.2f}" for n, p in zip(r.compare_names, r.probabilities)
                )
                or "-"
            )
        else:
            cls = f"{r.compare_values[0]}:{r.compare_names[0]}"
            cands = "(unmatched target)"
        print(
            f"  {r.side:<10} {cls:<28} {r.status:<10} "
            f"{r.absence_reason or '-':<11} {cands}"
        )

    _rule("Checks")
    n_declared_ref = len(declared_classes(REFERENCE_ID))
    ok = True

    # 1. One row per declared reference class -- the headline fix.
    if len(ref_rows) == n_declared_ref:
        print(f"  PASS  reference rows == declared classes ({n_declared_ref})")
    else:
        ok = False
        print(
            f"  FAIL  reference rows {len(ref_rows)} != declared "
            f"{n_declared_ref} - classes are still being dropped"
        )

    # 2. Every absent reference class carries a reason.
    missing_reason = [
        r.reference_value
        for r in ref_rows
        if r.status == "absent" and not r.absence_reason
    ]
    if missing_reason:
        ok = False
        print(f"  FAIL  absent rows with no reason: {missing_reason}")
    else:
        print(f"  PASS  every absent reference row carries a reason ({len(ref_absent)})")

    # 3. Absent compare classes are reported, not dropped.
    if len(cmp_rows) == len(cmp_absent):
        print(f"  PASS  absent compare classes reported as targets ({len(cmp_rows)})")
    else:
        ok = False
        print(
            f"  FAIL  compare-side rows {len(cmp_rows)} != absent compare "
            f"{len(cmp_absent)}"
        )

    # 4. No absent class was force-fit (absent => no probabilities).
    forced = [r.reference_value for r in ref_rows if r.status == "absent" and r.probabilities]
    if forced:
        ok = False
        print(f"  FAIL  absent rows carry probabilities (force-fit?): {forced}")
    else:
        print("  PASS  no absent class was force-fit")

    path = save_matching_table_csv(rows)
    print(f"\n  wrote {path}")

    ok = check_review_export(ref_absent, cmp_absent) and ok
    ok = check_absent_is_confirmable(ref_absent) and ok

    print(f"\n{'OK' if ok else 'FAILED'}")
    return 0 if ok else 1


def check_absent_is_confirmable(ref_absent: list) -> bool:
    """An absent class must stay confirmable, and the decision must reach the export.

    Stage 7.5: a class absent from every AOI is resolvable *only* by an expert --
    the AOI cannot evidence it, so the algorithm never will. Runs against a **copy**
    of the cache in a temp dir so the real feedback store is untouched.
    """
    _rule("Absent classes stay confirmable (Stage 7.5)")
    if not ref_absent:
        print("  SKIP  no absent reference classes in this run")
        return True
    try:
        from fastapi.testclient import TestClient
    except ImportError as exc:
        print(f"  SKIP  fastapi/httpx not available ({exc})")
        return True

    import shutil
    import tempfile

    import harmonizer.config as cfgmod

    target = ref_absent[0]
    real_cache = cfgmod.CONFIG.cache_dir
    tmp = Path(tempfile.mkdtemp(prefix="verify7a_"))
    try:
        # Copy everything except the feedback store, so the confirm below starts
        # clean and never touches the user's real decisions.
        for f in real_cache.glob("*"):
            if f.is_file() and not f.name.startswith("feedback_"):
                shutil.copy(f, tmp / f.name)
        object.__setattr__(cfgmod.CONFIG, "cache_dir", tmp)

        from harmonizer.api import app

        client = TestClient(app)
        pair = {"reference_id": REFERENCE_ID, "compare_id": COMPARE_ID}

        # Confirm an expert edge on the absent class to the first compare class.
        cmp_value = legend_classes(COMPARE_ID)[0].code
        resp = client.post(
            "/api/review/confirm",
            json={**pair, "reference_value": target.class_value,
                  "compare_values": [cmp_value]},
        )
        if resp.status_code != 200:
            print(
                f"  FAIL  confirming an edge on absent {target.class_name!r} "
                f"returned {resp.status_code}: {resp.text[:160]}"
            )
            return False
        print(f"  PASS  edge confirmed on absent {target.class_name!r}")

        # It must survive the round trip as expert-confirmed...
        rows = client.get("/api/review/table", params=pair).json()["rows"]
        row = next(
            (r for r in rows if r["reference_value"] == target.class_value), None
        )
        if not row or not row["edges"]:
            print(f"  FAIL  confirmed edge on {target.class_name!r} was dropped")
            return False
        print(f"  PASS  confirmed edge survives the round trip ({row['edges'][0]['compare_name']})")

        # ...and reach the export as a real mapping, not "Absent".
        csv_text = client.get("/api/review/export", params=pair).text
        line = next(
            (ln for ln in csv_text.splitlines() if target.class_name in ln), ""
        )
        if "Absent" in line:
            print(f"  FAIL  export still reads absent after confirm: {line!r}")
            return False
        print(f"  PASS  export shows the expert mapping: {line!r}")
        return True
    finally:
        object.__setattr__(cfgmod.CONFIG, "cache_dir", real_cache)
        shutil.rmtree(tmp, ignore_errors=True)


def check_review_export(ref_absent: list, cmp_absent: list) -> bool:
    """The review page's crosswalk export must mark absent classes on both sides.

    This is a separate path from the Stage 4 table (``review.py`` +
    ``/api/review/export``), and absent classes used to fall through it as a blank
    cell -- reading exactly like a class the expert had not got to yet.
    """
    _rule("Review page crosswalk export (/api/review/export)")
    try:
        from fastapi.testclient import TestClient

        from harmonizer.api import app
    except ImportError as exc:
        print(f"  SKIP  fastapi/httpx not available ({exc})")
        return True

    resp = TestClient(app).get(
        "/api/review/export",
        params={"reference_id": REFERENCE_ID, "compare_id": COMPARE_ID},
    )
    if resp.status_code != 200:
        print(f"  FAIL  export returned {resp.status_code}: {resp.text[:200]}")
        return False

    print(resp.text)
    lines = [ln for ln in resp.text.strip().splitlines()[1:] if ln]
    ok = True

    # Every absent class must be named on its own line and marked "Absent",
    # never left blank.
    for a in ref_absent + cmp_absent:
        hit = [ln for ln in lines if a.class_name in ln and "Absent" in ln]
        if hit:
            print(f"  PASS  {a.class_name!r} marked absent in the export")
        else:
            ok = False
            print(f"  FAIL  {a.class_name!r} not marked absent in the export")

    # A blank mapping must not be how an absent class reads.
    blank = [ln for ln in lines if ln.endswith(",,") or ln.endswith(',"",""')]
    if blank:
        print(f"  note  {len(blank)} row(s) exported blank (unreviewed, not absent)")
    return ok


if __name__ == "__main__":
    raise SystemExit(main())
