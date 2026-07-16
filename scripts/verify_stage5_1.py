"""Stage 5.1 verification: API and minimal run UI.

Drives the FastAPI backend end to end with FastAPI's TestClient and confirms the
whole API path works and its results match what the Stage 4 CLI
(``scripts/verify_stage4.py``) produces from the same cached GMMs
(docs/PIPELINE.md, Stage 5.1 -> Verification).

To keep the verification offline and fast, the heavy Stage 2 (server-side GEE
sampling) and Stage 3 (GMM fitting) steps are stubbed so the run **reuses the
GMMs already cached by Stage 3** (``cache/gmm_worldcover.json`` and
``cache/gmm_dynamicworld.json``): the stubs assert those caches exist and are
otherwise no-ops. The real Stage 4 (affinity, decision, matching table, CSVs), the
real background-job runner, the real progress reporting, and the real export
endpoints all run unchanged. This proves the API + minimal UI backbone without a
live GEE run; the human then does the browser pass described at the end.

Checks:
  1. GET /api/products lists the label products with roles + defaults + live
     calibration values.
  2. POST /api/overlap returns the derived overlap bbox and raises the
     slow-combination warning for a large AOI x fine scale (and not for a small
     one).
  3. POST /api/run launches a background job that progresses to `done`.
  4. GET /api/jobs/{id}/results matches a direct Stage 4 computation
     (compute_affinity + classify_rows) cell for cell.
  5. The three CSV exports are byte-identical to the CSVs Stage 4 writes directly.

Run:  python scripts/verify_stage5_1.py
(Stage 3 must have run first so the GMM caches exist. No network.)
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from fastapi.testclient import TestClient

import harmonizer.pipeline as pipeline
from harmonizer.affinity import compute_affinity
from harmonizer.config import CONFIG
from harmonizer.decision import (
    absent_decisions,
    build_matching_table,
    classify_rows,
    save_matching_table_csv,
    save_normalized_affinity_csv,
    save_raw_similarity_csv,
)
from harmonizer.modeling import gmm_cache_path

REFERENCE_ID = "worldcover"
COMPARE_ID = "dynamicworld"


# --------------------------------------------------------------------------- #
# Stubs: reuse the cached Stage-3 GMMs instead of re-sampling/re-fitting on GEE.
# --------------------------------------------------------------------------- #


class _NoopSample:
    def __init__(self, product_id: str) -> None:
        self.product_id = product_id


# Counts stubbed Stage-2 sampling calls, so the verification can tell a fresh
# (GEE) run from a cache-reuse run.
SAMPLE_CALLS = {"n": 0}


def _stub_sample_map(product_id, overlap=None, aoi=None, **kwargs):
    # Stage 5.1 verification does not touch GEE; require the Stage-3 cache instead.
    if not gmm_cache_path(product_id).exists():
        raise FileNotFoundError(
            f"cache/gmm_{product_id}.json missing; run scripts/verify_stage3.py first."
        )
    SAMPLE_CALLS["n"] += 1
    return _NoopSample(product_id)


def _stub_save_map_sample(ms):
    return Path(f"<stub samples_{ms.product_id}.npz>")


def _stub_fit_map(product_id):
    # The real gmm_<product>.json is already on disk; reuse it as-is.
    return _NoopSample(product_id)


def _stub_save_map_gmm(mg):
    return gmm_cache_path(mg.product_id)


def _install_stubs() -> None:
    pipeline.sample_map = _stub_sample_map
    pipeline.save_map_sample = _stub_save_map_sample
    pipeline.fit_map = _stub_fit_map
    pipeline.save_map_gmm = _stub_save_map_gmm


# --------------------------------------------------------------------------- #
# Verification
# --------------------------------------------------------------------------- #


def _run_to_done(client, body):
    """POST /api/run and poll to a terminal state; return (id, state, saw, last)."""
    job_id = client.post("/api/run", json=body).json()["job_id"]
    saw_progress = False
    state = None
    s = {}
    for _ in range(600):  # up to ~60s; the stubbed run is CPU-only and quick
        s = client.get(f"/api/jobs/{job_id}").json()
        state = s["state"]
        if 0.0 < s["progress"] < 1.0:
            saw_progress = True
        if state in ("done", "failed"):
            break
        time.sleep(0.1)
    return job_id, state, saw_progress, s


def main() -> None:
    print("=" * 88)
    print("Stage 5.1 verification - API and minimal run UI")
    print(f"reference={REFERENCE_ID}  compare={COMPARE_ID}")
    print("(Stage 2/3 stubbed to reuse cached GMMs; Stage 4 + API run for real)")
    print("=" * 88)

    for pid in (REFERENCE_ID, COMPARE_ID):
        if not gmm_cache_path(pid).exists():
            print(f"MISSING cache/gmm_{pid}.json - run scripts/verify_stage3.py first.")
            print("ALL CHECKS PASSED: NO")
            return

    _install_stubs()

    # Import the app AFTER stubbing so its handlers call the stubbed pipeline.
    from harmonizer.api import app

    client = TestClient(app)
    all_ok = True

    # 1. products
    r = client.get("/api/products")
    prods = r.json()
    ids = {p["id"] for p in prods["products"]}
    ok1 = (
        r.status_code == 200
        and {REFERENCE_ID, COMPARE_ID} <= ids
        and "defaults" in prods
        and "calibration" in prods
    )
    all_ok &= ok1
    print(f"\n[1] /api/products  -> {len(prods['products'])} products, "
          f"year {prods['working_year']}, calibration {prods['calibration']}")
    print(f"    products/roles/defaults/calibration present: {'YES' if ok1 else 'NO'}")

    # A bounded AOI keeps the (stubbed) runs legitimate; a blank AOI over this
    # global pair is a global region and must be refused (see [2]).
    BOUNDED_AOI = [32.0, 14.0, 33.0, 15.0]

    # 2. overlap: slow warning for a large bounded AOI, hard blocker for global.
    small = client.post("/api/overlap", json={
        "reference_id": REFERENCE_ID, "compare_id": COMPARE_ID,
        "aoi": BOUNDED_AOI, "sample_scale_m": 60.0,
    }).json()
    big_scale = client.post("/api/overlap", json={
        "reference_id": REFERENCE_ID, "compare_id": COMPARE_ID,
        "aoi": [10.0, 0.0, 40.0, 20.0], "sample_scale_m": 10.0,
    }).json()
    glob = client.post("/api/overlap", json={
        "reference_id": REFERENCE_ID, "compare_id": COMPARE_ID,
        "aoi": None, "sample_scale_m": 10.0,
    }).json()
    ok2 = (
        small["slow_warning"] is None
        and big_scale["slow_warning"] is not None
        and glob["runnable"] is False
        and glob["blocker"] is not None
    )
    all_ok &= ok2
    print(f"\n[2] /api/overlap")
    print(f"    bounded 1deg @ 60m: warn={small['slow_warning']}")
    print(f"    bounded 30x20deg @ 10m: warn set={big_scale['slow_warning'] is not None}")
    print(f"    global (blank AOI): runnable={glob['runnable']}  blocker set={glob['blocker'] is not None}")
    print(f"    slow-warning + global-blocker behave correctly: {'YES' if ok2 else 'NO'}")

    # 2b. A blank-AOI global run must be refused early (400), not time out on GEE.
    refuse = client.post("/api/run", json={
        "reference_id": REFERENCE_ID, "compare_id": COMPARE_ID, "aoi": None,
        "force_refresh": True,  # bypass any cache so it hits the global guard
    })
    ok2b = refuse.status_code == 400 and "globe" in refuse.json()["detail"].lower()
    all_ok &= ok2b
    print(f"\n[2b] /api/run with blank AOI (global): HTTP {refuse.status_code} "
          f"-> refused early: {'YES' if ok2b else 'NO'}")

    # 3. run job -> progress -> done. Clear any stale signature so this first run
    #    samples fresh (exercises the full Stage 2 path, stubbed). Uses a bounded
    #    AOI so the run is legitimately runnable.
    pipeline._signature_path().unlink(missing_ok=True)
    SAMPLE_CALLS["n"] = 0
    job_id, state, saw_progress, s = _run_to_done(client, {
        "reference_id": REFERENCE_ID, "compare_id": COMPARE_ID, "aoi": BOUNDED_AOI,
    })
    ok3 = state == "done"
    all_ok &= ok3
    print(f"\n[3] /api/run -> job {job_id[:8]}  final state={state}  saw progress={saw_progress}")
    print(f"    background job reached 'done': {'YES' if ok3 else 'NO'}")
    print(f"    Stage-2 sample calls this run: {SAMPLE_CALLS['n']} (fresh run samples both maps)")
    if state == "failed":
        print(f"    error: {s.get('error')}")

    # 4. results match a direct Stage 4 computation
    res = client.get(f"/api/jobs/{job_id}/results").json()
    aff = compute_affinity(REFERENCE_ID, COMPARE_ID)
    decisions, _ = classify_rows(aff)
    api_norm = np.asarray(res["normalized_affinity"], dtype=float)
    api_raw = np.asarray(res["raw_similarity"], dtype=float)
    norm_match = np.allclose(api_norm, aff.normalized_affinity, atol=1e-9)
    raw_match = np.allclose(api_raw, aff.raw_similarity, atol=1e-9)

    api_status = {row["reference_name"]: row["status"] for row in res["matching_table"]}
    cli_status = {d.reference_name: d.status for d in decisions}
    status_match = api_status == cli_status
    ok4 = norm_match and raw_match and status_match
    all_ok &= ok4
    print(f"\n[4] /api/jobs/{{id}}/results vs Stage 4 CLI")
    print(f"    normalized affinity matches: {norm_match}")
    print(f"    raw similarity matches:      {raw_match}")
    print(f"    per-class statuses match:    {status_match}")
    print(f"    results match the CLI: {'YES' if ok4 else 'NO'}")

    # Print the matching table exactly as the UI would show it.
    print("\n    Matching table (as served to the UI):")
    for row in res["matching_table"]:
        cand = ", ".join(
            f"{c['name']} ({c['probability']:.2f})" for c in row["compare"]
        )
        lc = " [low-confidence]" if row["reference_low_confidence"] else ""
        print(f"      {row['reference_name']:>22} -> {row['status']:>7}: {cand}{lc}")

    # 5. CSV exports byte-identical to Stage 4's direct writes
    absent = absent_decisions(REFERENCE_ID, aff)
    rows = build_matching_table(aff, decisions, include_absent=absent)
    ref_paths = {
        "raw_similarity": save_raw_similarity_csv(
            aff, CONFIG.cache_dir / "_ref_raw.csv"),
        "normalized_affinity": save_normalized_affinity_csv(
            aff, CONFIG.cache_dir / "_ref_norm.csv"),
        "matching_table": save_matching_table_csv(
            rows, CONFIG.cache_dir / "_ref_table.csv"),
    }
    ok5 = True
    print(f"\n[5] CSV exports vs Stage 4 direct writes")
    for which, ref_path in ref_paths.items():
        got = client.get(f"/api/jobs/{job_id}/export/{which}")
        same = got.status_code == 200 and got.content == ref_path.read_bytes()
        ok5 &= same
        print(f"    {which:>20}: byte-identical={same}")
        ref_path.unlink(missing_ok=True)
    all_ok &= ok5
    print(f"    all three CSV exports match: {'YES' if ok5 else 'NO'}")

    # 6. Cache reuse. The run in [3] wrote a signature (with BOUNDED_AOI); an
    #    identical run must reuse the cache (no Stage-2 sampling), while
    #    force_refresh and a changed AOI must re-sample.
    print(f"\n[6] Cache reuse (skip GEE when inputs unchanged)")
    same_body = {
        "reference_id": REFERENCE_ID, "compare_id": COMPARE_ID, "aoi": BOUNDED_AOI,
    }

    SAMPLE_CALLS["n"] = 0
    jid2, st2, _, _ = _run_to_done(client, same_body)
    reuse_calls = SAMPLE_CALLS["n"]
    reuse_flag = client.get(f"/api/jobs/{jid2}/results").json().get("reused_cache")
    reuse_ok = st2 == "done" and reuse_calls == 0 and reuse_flag is True
    print(f"    identical re-run: sample calls={reuse_calls}  reused_cache={reuse_flag}  "
          f"-> reused: {'YES' if reuse_ok else 'NO'}")

    SAMPLE_CALLS["n"] = 0
    jid3, st3, _, _ = _run_to_done(client, {**same_body, "force_refresh": True})
    force_calls = SAMPLE_CALLS["n"]
    force_flag = client.get(f"/api/jobs/{jid3}/results").json().get("reused_cache")
    force_ok = st3 == "done" and force_calls == 2 and force_flag is False
    print(f"    force_refresh re-run: sample calls={force_calls}  reused_cache={force_flag}  "
          f"-> re-sampled: {'YES' if force_ok else 'NO'}")

    SAMPLE_CALLS["n"] = 0
    jid4, st4, _, _ = _run_to_done(
        client, {**same_body, "aoi": [10.0, 5.0, 20.0, 15.0]})
    changed_calls = SAMPLE_CALLS["n"]
    changed_flag = client.get(f"/api/jobs/{jid4}/results").json().get("reused_cache")
    changed_ok = st4 == "done" and changed_calls == 2 and changed_flag is False
    print(f"    changed-AOI re-run: sample calls={changed_calls}  reused_cache={changed_flag}  "
          f"-> re-sampled: {'YES' if changed_ok else 'NO'}")

    ok6 = reuse_ok and force_ok and changed_ok
    all_ok &= ok6
    print(f"    cache reuse behaves correctly: {'YES' if ok6 else 'NO'}")

    print("\n" + "=" * 88)
    print(f"ALL AUTOMATED CHECKS PASSED: {'YES' if all_ok else 'NO'}")
    print(
        "\nBrowser pass (do this once, with GEE authenticated):\n"
        "  python run.py  ->  choose WorldCover x Dynamic World and ENTER A\n"
        "  BOUNDED AOI (e.g. 32,14,33,15) -- both maps are global, so a blank AOI\n"
        "  is refused to avoid a whole-globe GEE timeout. Run it, and confirm the\n"
        "  heatmap + matching table appear and the CSVs download and match the\n"
        "  Stage 4 CLI output."
    )


if __name__ == "__main__":
    main()
