"""Stage 6c verification: the Review UI (frontend).

Stage 6c is the **frontend** half of Stage 6: a Harmonize/Review mode switch, the
table-to-review handoff, the two large synchronized inspector maps, the left rail
(class-pair dropdowns + patch thumbnail index), the right rail (candidate edges +
multi-select confirm + per-edge provenance), the basemap/year switcher, and live
(file-backed) shared state -- consuming only the 6b ``/api/review/*`` endpoints, no
new backend (docs/PIPELINE.md, Stage 6.7 / 6c).

The UI itself is a browser concern (Leaflet, the mode switch, the rails), so the
automated part here checks the two things that must hold for that UI to work:

  1. **The 6b endpoints the Review UI calls are all registered and shape-correct**
     -- ``/api/review/table``, ``/api/review/explore``, ``/api/review/confirm``,
     ``/api/review/unconfirm`` -- driven with FastAPI's TestClient offline. The
     evidence-explorer engine needs live Earth Engine, so ``explore_evidence`` is
     **stubbed**: the query's mode dispatch and payload shape are exercised without a
     GEE call. ``review.py`` (recompute/confirm/unconfirm) is pure on-disk state, so
     it runs for real against a tiny synthetic feedback+GMM cache in a temp dir.

  2. **The shipped frontend actually wires those endpoints and preserves the 5.4
     ids** -- ``web/index.html`` / ``web/app.js`` reference each review endpoint, the
     mode switch and the two rails exist, and every 5.4 element id app.js relies on
     is still present (6c layers on top; it must not break Harmonize).

Run:  python scripts/verify_stage6c.py   (no network; explorer stubbed)

Then the browser pass at the end (with GEE authenticated), per docs/PIPELINE.md 6.7.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi.testclient import TestClient

REFERENCE_ID = "worldcover"
COMPARE_ID = "dynamicworld"

WEB = Path(__file__).resolve().parent.parent / "web"


# --------------------------------------------------------------------------- #
# Stub the live-GEE evidence explorer so /api/review/explore is testable offline.
# The stub records the mode + values it was dispatched with (proving the three-mode
# frontend dispatch maps to the backend call) and returns a fixed set of locations.
# --------------------------------------------------------------------------- #

EXPLORE_CALLS: list[dict] = []


def _stub_explore_evidence(
    reference_id, compare_id, *, mode, reference_value=None,
    compare_value=None, aoi=None, n=None,
):
    from harmonizer.explorer import EvidenceLocation, EvidenceResult

    EXPLORE_CALLS.append(
        {
            "mode": mode,
            "reference_value": reference_value,
            "compare_value": compare_value,
            "aoi": aoi,
        }
    )
    # Two synthetic, spread locations with both sides' labels populated.
    locs = [
        EvidenceLocation(
            lon=32.10 + i * 0.2, lat=14.10 + i * 0.2,
            reference_label=(reference_value if reference_value is not None else 30),
            reference_label_name="Ref class",
            compare_label=(compare_value if compare_value is not None else 5),
            compare_label_name="Cmp class",
            patch_window_px=256,
        )
        for i in range(2)
    ]
    return EvidenceResult(
        mode=mode,
        reference_id=reference_id,
        compare_id=compare_id,
        reference_value=reference_value,
        compare_value=compare_value,
        patch_window_px=256,
        patch_window_m=2560.0,
        reference_pixel_m=10.0,
        compare_pixel_m=10.0,
        locations=locs,
    )


def main() -> None:
    print("=" * 88)
    print("Stage 6c verification - Review UI (frontend over the 6b endpoints)")
    print(f"reference={REFERENCE_ID}  compare={COMPARE_ID}")
    print("(evidence explorer stubbed; review.py on-disk state run for real)")
    print("=" * 88)

    all_ok = True

    # Patch the explorer engine the API imports lazily inside the route.
    import harmonizer.explorer as explorer_mod
    explorer_mod.explore_evidence = _stub_explore_evidence

    from harmonizer.api import app

    client = TestClient(app)

    # ----------------------------------------------------------------------- #
    # 1. The review endpoints the UI calls: registered + shape-correct.
    # ----------------------------------------------------------------------- #
    print("\n[1] Review endpoints the UI consumes")

    routes = {getattr(r, "path", None) for r in app.routes}
    endpoints = [
        "/api/review/table",
        "/api/review/explore",
        "/api/review/confirm",
        "/api/review/unconfirm",
    ]
    registered = {e: (e in routes) for e in endpoints}
    ok_registered = all(registered.values())
    all_ok &= ok_registered
    for e, present in registered.items():
        print(f"    {e:<26} registered: {'YES' if present else 'NO'}")

    # /api/review/explore: the three-mode dispatch reaches the engine with the right
    # mode + values, and the payload carries what the patch index needs.
    print("\n    /api/review/explore (three modes -> engine dispatch):")
    ok_explore = True
    cases = [
        ("both", {"reference_value": 30, "compare_value": 5}),
        ("reference", {"reference_value": 30}),
        ("compare", {"compare_value": 5}),
    ]
    for mode, extra in cases:
        EXPLORE_CALLS.clear()
        body = {
            "reference_id": REFERENCE_ID, "compare_id": COMPARE_ID,
            "mode": mode, "aoi": [32.0, 14.0, 33.0, 15.0], **extra,
        }
        r = client.post("/api/review/explore", json=body)
        body_ok = r.status_code == 200
        payload = r.json() if body_ok else {}
        dispatched = EXPLORE_CALLS and EXPLORE_CALLS[-1]["mode"] == mode
        # Every location must carry a coordinate + both sides' labels + the window
        # (the data the client draws the patch outline / fly-to from).
        locs = payload.get("locations", [])
        loc_shape = bool(locs) and all(
            {"lon", "lat", "reference_label", "compare_label"} <= set(loc)
            for loc in locs
        )
        window_ok = payload.get("patch_window_px") == 256 and bool(payload.get("patch_window_m"))
        # Fix 1: the per-side native pixel size the client draws the one-pixel box from.
        pixel_ok = payload.get("reference_pixel_m") == 10.0 and payload.get("compare_pixel_m") == 10.0
        good = bool(body_ok and dispatched and loc_shape and window_ok and pixel_ok)
        ok_explore &= good
        print(f"      mode={mode:<10} HTTP {r.status_code}, dispatched={bool(dispatched)}, "
              f"locations={len(locs)}, window_ok={bool(window_ok)} -> {'OK' if good else 'BAD'}")
    all_ok &= ok_explore

    # ----------------------------------------------------------------------- #
    # 2. review.py on-disk state (table / confirm / unconfirm) runs for real on a
    #    tiny synthetic cache -- the file-backed "shared live state" the UI reads.
    # ----------------------------------------------------------------------- #
    print("\n[2] File-backed review state (confirm -> freeze -> unconfirm round-trip)")
    ok_state = _check_review_state_roundtrip(client)
    all_ok &= ok_state
    print(f"    confirm/unconfirm round-trip over the API: {'YES' if ok_state else 'NO'}")

    # ----------------------------------------------------------------------- #
    # 3. The shipped frontend wires the endpoints and preserves the 5.4 ids.
    # ----------------------------------------------------------------------- #
    print("\n[3] Frontend wiring (web/index.html + web/app.js)")
    ok_front = _check_frontend()
    all_ok &= ok_front

    print("\n" + "=" * 88)
    print(f"ALL AUTOMATED CHECKS PASSED: {'YES' if all_ok else 'NO'}")
    print(
        "\nBrowser pass (do this once, with GEE authenticated), per docs/PIPELINE.md 6.7:\n"
        "  python run.py  ->  run WorldCover x Dynamic World over a small AOI, then:\n"
        "    - in the matching table, click a `mixed` (or `orphan`) row: the app\n"
        "      switches to Review focused on that reference class;\n"
        "    - Review shows two synced SATELLITE maps fitted to the AOI; each map's\n"
        "      title bar carries the product name | basemap picker | year; pick a\n"
        "      reference class from the dropdown under the map (row mode) and\n"
        "      optionally a compare class (cell mode); its description shows below;\n"
        "    - the bottom band has 3 columns: evidence patches | decision | Sankey;\n"
        "      'Find evidence' lists patches in column 1;\n"
        "    - click a patch: BOTH synced maps fly to it and outline the queried\n"
        "      PIXEL (a ~10 m box per side, not a 2.5 km window) on the imagery;\n"
        "    - in the decision column, multi-select candidate edges (or add one via\n"
        "      '+ more classes') and Confirm: the edges flip to `expert-confirmed`/\n"
        "      frozen; the reviewed-table Sankey (column 3) updates;\n"
        "    - switch back to Harmonize and confirm the decision landed (the row's\n"
        "      confirmed edges are frozen); an *unconfirmed* edge may change after a\n"
        "      refit while the *confirmed* edge does not."
    )


def _check_review_state_roundtrip(client: TestClient) -> bool:
    """Confirm -> freeze -> unconfirm over the API against a synthetic cache.

    Stubs the affinity + reviewed-table recompute so review.py's on-disk feedback
    logic (the file-backed shared state the UI reads back) runs for real without GEE
    or fitted GMMs. Redirects the feedback-store path into a temp dir so the real
    cache is untouched.
    """
    import tempfile

    import numpy as np

    from harmonizer import affinity as affinity_mod
    from harmonizer import review as review_mod

    # A tiny stubbed affinity for one reference class over two compare classes, so
    # confirm_edges has a probability row to freeze from. We stub compute_affinity
    # rather than fit real GMMs (Stage 3/4 have their own verifications).
    from dataclasses import dataclass

    @dataclass
    class _Aff:
        reference_id: str
        compare_id: str
        reference_classes: list
        compare_classes: list
        normalized_affinity: object

    cmp_classes = [5, 8]
    aff = _Aff(
        reference_id=REFERENCE_ID, compare_id=COMPARE_ID,
        reference_classes=[30], compare_classes=cmp_classes,
        normalized_affinity=np.array([[0.7, 0.3]]),
    )

    orig_compute = review_mod.recompute_reviewed_table
    orig_affinity = affinity_mod.compute_affinity
    orig_cached = review_mod.cached_affinity
    orig_path = review_mod.feedback_cache_path
    tmp = Path(tempfile.mkdtemp(prefix="verify6c_"))

    def _stub_recompute(reference_id, compare_id, store=None, aff=None):
        # Reflect the store's confirmed edges into a ReviewedRow so the UI-facing
        # table shows provenance. Open edges carry the algorithm probability.
        from harmonizer.review import (
            EXPERT_CONFIRMED,
            ALGORITHM_PROPOSED,
            ReviewedEdge,
            ReviewedRow,
            load_feedback,
        )

        st = store if store is not None else load_feedback(reference_id, compare_id)
        confirmed = {e.compare_value: e.retained_probability
                     for e in st.confirmed_edges(30)}
        edges = []
        for cv, p in zip(cmp_classes, [0.7, 0.3]):
            if cv in confirmed:
                edges.append(ReviewedEdge(30, "ref30", cv, f"cmp{cv}",
                                          confirmed[cv], EXPERT_CONFIRMED))
            else:
                edges.append(ReviewedEdge(30, "ref30", cv, f"cmp{cv}",
                                          p, ALGORITHM_PROPOSED))
        row = ReviewedRow(reference_value=30, reference_name="ref30",
                          status="mixed", edges=edges)
        return aff, [row]

    try:
        # Persist the feedback store into the temp dir, not the real cache.
        review_mod.feedback_cache_path = (
            lambda r, c: tmp / f"feedback_{r}__{c}.json"
        )
        # The confirm route uses review.cached_affinity (memoised on the GMM
        # caches' mtimes), which would stat missing caches here — stub it, plus
        # compute_affinity for any path that still reaches it.
        affinity_mod.compute_affinity = lambda r, c: aff  # type: ignore[attr-defined]
        review_mod.cached_affinity = lambda r, c: aff
        review_mod.recompute_reviewed_table = _stub_recompute

        # Confirm compare class 5 for reference class 30.
        r_conf = client.post("/api/review/confirm", json={
            "reference_id": REFERENCE_ID, "compare_id": COMPARE_ID,
            "reference_value": 30, "compare_values": [5], "refit": False,
        })
        conf_rows = r_conf.json().get("rows", []) if r_conf.status_code == 200 else []
        edge5 = _find_edge(conf_rows, 30, 5)
        froze = edge5 is not None and edge5["provenance"] == "expert-confirmed"

        # The table endpoint reads the same frozen state back.
        r_tab = client.get("/api/review/table", params={
            "reference_id": REFERENCE_ID, "compare_id": COMPARE_ID})
        tab_rows = r_tab.json().get("rows", []) if r_tab.status_code == 200 else []
        edge5_tab = _find_edge(tab_rows, 30, 5)
        persisted = edge5_tab is not None and edge5_tab["provenance"] == "expert-confirmed"

        # Unconfirm reopens it.
        r_un = client.post("/api/review/unconfirm", json={
            "reference_id": REFERENCE_ID, "compare_id": COMPARE_ID,
            "reference_value": 30, "compare_value": 5,
        })
        un_rows = r_un.json().get("rows", []) if r_un.status_code == 200 else []
        edge5_un = _find_edge(un_rows, 30, 5)
        reopened = edge5_un is not None and edge5_un["provenance"] == "algorithm-proposed"

        print(f"      confirm freezes edge 30->5:      {froze}")
        print(f"      table reads the frozen edge back: {persisted}")
        print(f"      unconfirm reopens it:             {reopened}")
        return bool(froze and persisted and reopened)
    finally:
        review_mod.recompute_reviewed_table = orig_compute
        affinity_mod.compute_affinity = orig_affinity
        review_mod.cached_affinity = orig_cached
        review_mod.feedback_cache_path = orig_path
        import shutil
        shutil.rmtree(tmp, ignore_errors=True)


def _find_edge(rows, rv, cv):
    for row in rows:
        if row.get("reference_value") == rv:
            for e in row.get("edges", []):
                if e.get("compare_value") == cv:
                    return e
    return None


def _check_frontend() -> bool:
    """The shipped web/ wires the review endpoints and preserves the 5.4 ids."""
    html = (WEB / "index.html").read_text(encoding="utf-8")
    js = (WEB / "app.js").read_text(encoding="utf-8")

    # 6c calls only the /api/review/* endpoints (no new backend). Unconfirms now
    # ride inside the confirm request (unconfirm_values); retraining is its own
    # explicit /api/review/refit action.
    calls_ok = all(ep in js for ep in (
        "/api/review/table", "/api/review/explore",
        "/api/review/confirm", "/api/review/refit",
    ))
    # The mode switch + Review workspace shell exist (maps row + 3-col band with
    # title-bar basemap pickers, class dropdowns, and per-side descriptions).
    shell_ok = all(x in html for x in (
        'id="mode-review"', 'id="mode-harmonize"', 'id="review-workspace"',
        'id="rev-maps-row"', 'id="rev-patches"', 'id="rev-edges"',
        'id="rev-class-ref"', 'id="rev-class-cmp"',
        'id="rev-desc-ref"', 'id="rev-desc-cmp"', 'id="rev-sankey"',
        'id="rev-basemap-ref"', 'id="rev-basemap-cmp"',
        'id="rev-patches-panel"', 'id="rev-decision"', 'id="rev-sankey-panel"',
    ))
    # Class-dropdown focus, per-side descriptions, satellite basemaps, AOI fit,
    # reviewed Sankey, and the three-mode dispatch are present in app.js.
    features_ok = all(x in js for x in (
        "fillClassSelect", "showClassDescription",
        "BASEMAPS", "applyBasemap", "fitReviewToAoi",
        "drawReviewSankey", "makeMoreClasses",
        "enterReview", "runExplore", "confirmEdges", "flyToPatch",
    ))
    # 6c must NOT break Harmonize: every 5.4 id app.js drives must still exist.
    preserved_ids = [
        "reference", "compare", "min_lon", "min_lat", "max_lon", "max_lat",
        "sample_scale_m", "n_components", "points_floor", "points_target",
        "run", "force_refresh", "progress", "check-overlap", "clear-aoi",
        "aoi-file", "csv-pick", "csv-download", "csv-view", "sankey",
        "map-ref", "map-cmp", "legend-ref", "legend-cmp", "resize-rows",
    ]
    ids_ok = all(f'id="{i}"' in html for i in preserved_ids)

    print(f"    app.js calls all four review endpoints: {calls_ok}")
    print(f"    mode switch + Review workspace present: {shell_ok}")
    print(f"    basemap switcher + 3-mode dispatch:     {features_ok}")
    print(f"    all 5.4 element ids preserved:          {ids_ok}")
    missing = [i for i in preserved_ids if f'id="{i}"' not in html]
    if missing:
        print(f"      MISSING ids: {missing}")
    return calls_ok and shell_ok and features_ok and ids_ok


if __name__ == "__main__":
    main()
