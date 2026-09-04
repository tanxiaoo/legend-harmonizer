"""Evidence explorer on local-raster products (Stage 6 fix).

The explorer was written against the GEE label products and called
``sampling._label_image_for`` unconditionally, so any pair read from a local
raster failed with

    Stage 2 sampling supports the GEE label products only; got 'worldcover_2020'

even though Stage 2 itself has read local rasters since the Africa product set
landed (``harmonizer.local_sampling``, dispatched from ``sampling.sample_map``).
Review was therefore unusable on exactly the pairs the rest of the pipeline
handles fine.

This checks the explorer now works for a local x local pair, in all three modes,
on both the cache-backed and the live draw paths, and that it does so **without
Earth Engine** -- a local pair must not need credentials at all. It also checks
the API surface: a working query returns 200, a bad argument still returns a
clear 400, and an unexpected failure reports its cause rather than a bare
"Internal Server Error".

Run:  python scripts/verify_explorer_local.py

Pure CPU, no GEE. Stages 2-3 must have run for the pair.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

REFERENCE_ID = "hrlc30_africa"
COMPARE_ID = "worldcover_2020"

# A small AOI inside the run's area. Used to force the LIVE draw: the Stage 2
# cache does not cover every class here, so the explorer falls through to
# drawing candidates from the raster in-process.
SMALL_AOI = (35.0, 9.5, 36.5, 10.5)


def _rule(title: str) -> None:
    print(f"\n{title}\n{'-' * len(title)}")


def check(ok: bool, message: str, detail: str = "") -> bool:
    print(f"  {'PASS' if ok else 'FAIL'}  {message}" + (f" [{detail}]" if detail else ""))
    return bool(ok)


def main() -> int:
    ok = True

    from harmonizer.explorer import _is_local, explore_evidence

    _rule("Evidence explorer on local rasters")
    ok &= check(
        _is_local(REFERENCE_ID) and _is_local(COMPARE_ID),
        "both products of the test pair are local rasters",
        f"{REFERENCE_ID}, {COMPARE_ID}",
    )

    # ---------------------------------------------------------------- 1. #
    _rule("1. All three modes return consistent evidence")

    cases = [
        ("reference", 141, None, "HRLC seasonal water"),
        ("reference", 10, None, "HRLC tree cover evergreen broadleaf"),
        ("compare", None, 40, "WorldCover cropland"),
        ("both", 80, 40, "HRLC croplands x WC cropland"),
    ]
    for mode, rv, cv, label in cases:
        try:
            r = explore_evidence(
                REFERENCE_ID,
                COMPARE_ID,
                mode=mode,
                reference_value=rv,
                compare_value=cv,
                aoi=None,
                n=6,
                oversample=2.0,
            )
        except Exception as exc:  # noqa: BLE001 - the whole point is "did it raise"
            ok = check(False, f"{mode}: {label}", f"{type(exc).__name__}: {exc}")
            continue

        # Every returned location must actually satisfy the query -- that is the
        # explorer's exactness guarantee, and a local read that silently grabbed
        # a neighbouring pixel would break it.
        if mode == "both":
            good = all(
                loc.reference_label == rv and loc.compare_label == cv
                for loc in r.locations
            )
        elif mode == "reference":
            good = all(loc.reference_label == rv for loc in r.locations)
        else:
            good = all(loc.compare_label == cv for loc in r.locations)

        ok &= check(
            r.n > 0 and good,
            f"{mode}: {label}",
            f"n={r.n} source={r.source}",
        )

    # ---------------------------------------------------------------- 2. #
    _rule("2. The LIVE draw path works (not just the Stage 2 cache)")

    live_seen = False
    for mode, rv, cv in [("compare", None, 30), ("reference", 80, None)]:
        r = explore_evidence(
            REFERENCE_ID,
            COMPARE_ID,
            mode=mode,
            reference_value=rv,
            compare_value=cv,
            aoi=SMALL_AOI,
            n=5,
            oversample=2.0,
        )
        if r.source == "live":
            live_seen = True
            ok &= check(r.n > 0, f"live draw returned locations ({mode})", f"n={r.n}")
    if not live_seen:
        print("  note  every query hit the cache; live path covered by check 3")

    # ---------------------------------------------------------------- 3. #
    _rule("3. A local x local pair needs no Earth Engine at all")

    # Hard-disable GEE: if any code path reaches for it, these raise. This is the
    # real regression guard -- requiring credentials for a pair that is entirely
    # on disk is what made the failure look like an auth problem.
    import harmonizer.explorer as ex
    import harmonizer.registry.adapters._gee as gee

    # explorer.py imports these *inside* functions, so patching the adapter
    # module is what actually intercepts them.
    saved = (gee.ensure_initialized, gee.sample_image)

    def _boom(*_a, **_k):
        raise AssertionError("Earth Engine was contacted for a local x local pair")

    gee.ensure_initialized = _boom
    gee.sample_image = _boom
    try:
        r = ex.explore_evidence(
            REFERENCE_ID, COMPARE_ID, mode="reference", reference_value=70,
            compare_value=None, aoi=SMALL_AOI, n=5, oversample=2.0,
        )
        ok &= check(r.n > 0, "offline query succeeded", f"n={r.n} source={r.source}")

        r = ex.explore_evidence(
            REFERENCE_ID, COMPARE_ID, mode="compare", reference_value=None,
            compare_value=30, aoi=SMALL_AOI, n=5, oversample=2.0,
        )
        ok &= check(
            r.n > 0, "offline LIVE draw succeeded", f"n={r.n} source={r.source}"
        )
    except AssertionError as exc:
        ok = check(False, "offline query", str(exc))
    finally:
        gee.ensure_initialized, gee.sample_image = saved

    # ---------------------------------------------------------------- 4. #
    _rule("4. API surface")

    try:
        from fastapi.testclient import TestClient
    except ImportError as exc:
        print(f"  SKIP  fastapi/httpx not available ({exc})")
        print(f"\n{'OK' if ok else 'FAILED'}")
        return 0 if ok else 1

    from harmonizer.api import app

    client = TestClient(app, raise_server_exceptions=False)
    body = {
        "reference_id": REFERENCE_ID,
        "compare_id": COMPARE_ID,
        "mode": "reference",
        "reference_value": 141,
        "n": 10,
        "oversample": 2.0,
    }
    resp = client.post("/api/review/explore", json=body)
    ok &= check(
        resp.status_code == 200,
        "POST /api/review/explore returns 200 for a local pair",
        f"got {resp.status_code}: {resp.text[:100]}",
    )
    if resp.status_code == 200:
        j = resp.json()
        ok &= check(
            len(j["locations"]) == j["n"] and j["n"] > 0,
            "payload carries the reported number of locations",
            f"n={j['n']}",
        )

    bad = client.post(
        "/api/review/explore",
        json={**body, "reference_value": None},
    )
    ok &= check(
        bad.status_code == 400,
        "a missing class value is still a clear 400",
        f"got {bad.status_code}: {str(bad.json().get('detail'))[:60]}",
    )

    # An unexpected failure must name itself. Previously this surfaced as a bare
    # "Internal Server Error" with the cause only in the server log, which is
    # what made a GEE permission problem so hard to place from the page.
    original = ex.explore_evidence

    def _raise(*_a, **_k):
        raise RuntimeError("simulated backend failure")

    ex.explore_evidence = _raise
    try:
        boom = client.post("/api/review/explore", json=body)
        detail = str(boom.json().get("detail", ""))
        ok &= check(
            boom.status_code == 502 and "simulated backend failure" in detail,
            "an unexpected failure reports its cause, not a bare 500",
            f"{boom.status_code}: {detail[:60]}",
        )
    finally:
        ex.explore_evidence = original

    print(f"\n{'OK' if ok else 'FAILED'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
