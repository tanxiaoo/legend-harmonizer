"""Stage 7c verification: targeted auxiliary AOIs and the merged table.

Runs the auxiliary-AOI flow end to end against the existing primary caches
(docs/PIPELINE.md, Stage 7 -> Verification):

  1. Prints what the run still cannot model (net of any existing auxiliaries).
  2. Samples an auxiliary AOI **targeted at those classes only**: of each side's
     still-absent classes, the ones actually present in the auxiliary AOI are
     sampled with the full Stage 2 machinery, and the OTHER map's classes
     co-present at those points are sampled there too -- so every distance in
     the auxiliary sub-matrix is a within-AOI comparison.
  3. Asserts the **primary caches were not touched** (reused, not re-sampled):
     adding an auxiliary must never re-pay the primary's GEE sampling.
  4. Asserts the auxiliary sample cache holds **only** the targeted + co-present
     classes -- targeted means targeted.
  5. Rebuilds the **merged matching table** (union of every AOI's rows) and
     asserts the recovered class's row is tagged ``evidence_aoi=<aux>``, its
     candidates were all fitted in that same auxiliary AOI, and classes still
     absent from every AOI remain reported as ``absent``.
  6. Writes ``matching_table.csv`` with the ``evidence_aoi`` column.

The default auxiliary AOI is a small box over the Saloum Delta mangrove coast
(Senegal), chosen to recover WorldCover's Mangroves (95) after the Sahel primary
run. Pass ``--aoi`` to point elsewhere.

Sampling params default to the primary run's effective values (from the stored
run signature), so a coarse-scale test primary gets a coarse-scale auxiliary
automatically -- keep the auxiliary AOI small regardless: sampling cost scales
with AOI area / scale^2.

Run:  python scripts/verify_stage7c.py [--aoi MINLON,MINLAT,MAXLON,MAXLAT]
                                       [--name NAME] [--scale M] [--force]

Needs the primary caches (run the pipeline / verify_stage2+3 first) and an
authenticated Earth Engine account -- the auxiliary sampling is a real, if
small, GEE pass. Re-running with unchanged inputs reuses the auxiliary caches
and costs nothing.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from harmonizer.auxiliary import (  # noqa: E402
    absence_report_all_aois,
    aux_gmm_cache_path,
    aux_sample_cache_path,
    merged_matching_table,
    sample_auxiliary,
    still_absent_classes,
)
from harmonizer.decision import save_matching_table_csv  # noqa: E402
from harmonizer.modeling import gmm_cache_path  # noqa: E402
from harmonizer.sampling import cache_path as sample_cache_path  # noqa: E402

REFERENCE_ID = "worldcover"
COMPARE_ID = "dynamicworld"

# Saloum Delta, Senegal: a small mangrove coast box (~20 x 20 km).
DEFAULT_AUX_AOI = (-16.75, 13.55, -16.55, 13.75)
DEFAULT_NAME = "mangrove-coast"


def _rule(title: str) -> None:
    print(f"\n{title}\n{'-' * len(title)}")


def _print_still_absent(label: str) -> dict[str, list[int]]:
    out: dict[str, list[int]] = {}
    for pid in (REFERENCE_ID, COMPARE_ID):
        absent = still_absent_classes(pid)
        out[pid] = [a.class_value for a in absent]
        names = ", ".join(f"{a.class_value}:{a.class_name} ({a.reason})" for a in absent)
        print(f"  {label} {pid:<13}: {names or 'nothing absent'}")
    return out


def _primary_cache_files() -> list[Path]:
    files = []
    for pid in (REFERENCE_ID, COMPARE_ID):
        npz = sample_cache_path(pid)
        files += [npz, npz.with_suffix(".json"), gmm_cache_path(pid)]
    return [f for f in files if f.exists()]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--aoi",
        default=",".join(str(v) for v in DEFAULT_AUX_AOI),
        help="auxiliary AOI as min_lon,min_lat,max_lon,max_lat "
        f"(default: Saloum Delta mangroves {DEFAULT_AUX_AOI})",
    )
    ap.add_argument("--name", default=DEFAULT_NAME, help="auxiliary AOI name")
    ap.add_argument(
        "--scale", type=float, default=None,
        help="sample scale in metres (default: the primary run's effective scale)",
    )
    ap.add_argument(
        "--force", action="store_true",
        help="re-sample even if this auxiliary's signature is unchanged",
    )
    args = ap.parse_args()
    aoi = tuple(float(v) for v in args.aoi.split(","))
    if len(aoi) != 4:
        ap.error("--aoi needs exactly four comma-separated numbers")

    _rule("Stage 7c - before: what the run still cannot model")
    before = _print_still_absent("still absent for")
    if not before[REFERENCE_ID] and not before[COMPARE_ID]:
        print("\n  Nothing is absent - there is nothing for an auxiliary to target.")
        return 0

    # Record the primary caches' mtimes: the auxiliary must not touch them.
    primary_files = _primary_cache_files()
    mtimes = {f: f.stat().st_mtime_ns for f in primary_files}

    _rule(f"Sampling auxiliary AOI '{args.name}' at {aoi}")
    result = sample_auxiliary(
        REFERENCE_ID,
        COMPARE_ID,
        aoi,  # type: ignore[arg-type]
        args.name,
        sample_scale_m=args.scale,
        force_refresh=args.force,
        progress=lambda f, msg: print(f"  [{f * 100:5.1f}%] {msg}"),
    )
    if result.reused:
        print("  (signature unchanged - auxiliary caches reused, no GEE)")

    for side, label in ((result.reference, "REFERENCE"), (result.compare, "COMPARE")):
        print(f"\n  {label} ({side.product_id})")
        print(f"    targeted (still absent) : {side.targeted}")
        print(f"    found in this AOI       : {side.found}")
        print(f"    co-present (other side's targets occur with these): {side.co_present}")
        print(f"    modelled here           : {side.modelled}")
        print(f"    absent here too         : {side.absent_in_aux}")

    _rule("Checks")
    ok = True

    # 1. The primary caches were reused, not re-sampled (docs 7.4).
    touched = [f.name for f in primary_files if f.stat().st_mtime_ns != mtimes[f]]
    if touched:
        ok = False
        print(f"  FAIL  primary caches modified by the auxiliary run: {touched}")
    else:
        print(f"  PASS  primary caches untouched ({len(primary_files)} files)")

    # 2. The auxiliary sampled ONLY the targeted + co-present classes.
    import json

    for side in (result.reference, result.compare):
        pid = side.product_id
        sidecar = aux_sample_cache_path(pid, result.name).with_suffix(".json")
        if not sidecar.exists():
            if side.found or side.co_present:
                ok = False
                print(f"  FAIL  {pid}: no auxiliary sample cache written")
            else:
                print(f"  PASS  {pid}: nothing to sample here, no cache written")
            continue
        sampled = {
            int(cv)
            for cv in json.loads(sidecar.read_text(encoding="utf-8"))["classes"]
        }
        expected = set(side.found) | set(side.co_present)
        if sampled == expected:
            print(f"  PASS  {pid}: sampled exactly targeted+co-present {sorted(sampled)}")
        else:
            ok = False
            print(
                f"  FAIL  {pid}: sampled {sorted(sampled)} != "
                f"targeted+co-present {sorted(expected)}"
            )

    # 3. Merged table: recovered classes ride in tagged with the auxiliary's name,
    #    and their candidates were fitted in that same AOI.
    rows, info = merged_matching_table(REFERENCE_ID, COMPARE_ID)
    aux_rows = [r for r in rows if r.evidence_aoi == result.name]
    print(
        f"\n  merged table: {len(rows)} rows total; auxiliaries contributing: "
        f"{info['auxiliaries']}; rows from '{result.name}': {len(aux_rows)}"
    )
    print(f"\n  {'evidence_aoi':<16} {'side':<10} {'class':<28} {'status':<10} candidates")
    for r in rows:
        cls = (
            f"{r.reference_value}:{r.reference_name}"
            if r.side == "reference"
            else f"{r.compare_values[0]}:{r.compare_names[0]}"
        )
        cands = ", ".join(
            f"{n} {p:.2f}" for n, p in zip(r.compare_names, r.probabilities)
        ) or ("(unmatched target)" if r.side == "compare" else "-")
        print(
            f"  {r.evidence_aoi or '-':<16} {r.side:<10} {cls:<28} "
            f"{r.status:<10} {cands}"
        )

    if result.reference.modelled or result.compare.modelled:
        aux_ref_rows = {r.reference_value for r in aux_rows if r.side == "reference"}
        missing = [cv for cv in result.reference.modelled if cv not in aux_ref_rows]
        # Reference classes modelled in the aux (targets AND co-present recoveries)
        # must each have a row tagged with the auxiliary's name.
        if missing:
            ok = False
            print(f"  FAIL  aux-modelled reference classes with no tagged row: {missing}")
        elif aux_rows:
            print("  PASS  every aux-modelled reference class has an evidence_aoi row")

        # Candidates of aux rows must be classes fitted in the SAME auxiliary
        # (within-AOI comparisons only - the Stage 7 invariant).
        gmm_path = aux_gmm_cache_path(COMPARE_ID, result.name)
        fitted_cmp = set()
        if gmm_path.exists():
            payload = json.loads(gmm_path.read_text(encoding="utf-8"))
            fitted_cmp = {
                int(cv)
                for cv, cs in payload["classes"].items()
                if cs.get("fitted") and not cs.get("absent")
            }
        stray = [
            (r.reference_value, cv)
            for r in aux_rows
            if r.side == "reference"
            for cv in r.compare_values
            if cv not in fitted_cmp
        ]
        if stray:
            ok = False
            print(f"  FAIL  aux-row candidates not fitted in the auxiliary AOI: {stray}")
        else:
            print("  PASS  every aux-row candidate was fitted in the same auxiliary AOI")
    else:
        print("  note  the auxiliary modelled nothing (wrong AOI for these classes?)")

    # 4. Absence is now net of the auxiliary: covered classes gone, others remain.
    report = absence_report_all_aois(REFERENCE_ID, COMPARE_ID)
    for key, pid in (("reference", REFERENCE_ID), ("compare", COMPARE_ID)):
        absent_now = {a["class_value"] for a in report[key]["absent"]}
        side = result.reference if pid == REFERENCE_ID else result.compare
        wrongly_listed = [cv for cv in side.modelled if cv in absent_now]
        if wrongly_listed:
            ok = False
            print(f"  FAIL  {pid}: aux-covered classes still reported absent: {wrongly_listed}")
        else:
            print(
                f"  PASS  {pid}: absence report net of auxiliary "
                f"(still absent: {sorted(absent_now) or 'none'})"
            )
    # Still-absent classes must still be visible as absent rows (never dropped).
    still_ref = {a.class_value for a in still_absent_classes(REFERENCE_ID)}
    absent_row_values = {
        r.reference_value for r in rows if r.side == "reference" and r.status == "absent"
    }
    if still_ref - absent_row_values:
        ok = False
        print(f"  FAIL  still-absent classes missing from the merged table: {sorted(still_ref - absent_row_values)}")
    else:
        print("  PASS  still-absent classes remain reported in the merged table")

    path = save_matching_table_csv(rows)
    print(f"\n  wrote {path} (evidence_aoi column included)")

    print(f"\n{'OK' if ok else 'FAILED'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
