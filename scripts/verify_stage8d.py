"""Stage 8d verification: the alpha control in the UI.

Alpha is a *display* control, not a run parameter: it sits outside the run
signature, so changing it re-decides Stage 4 from the cached models with no GEE
call and no re-sampling. This checks the backend that makes that possible, and
the frontend wiring that exposes it.

Checks:

1. ``/api/affinity`` at a given alpha returns exactly what
   ``compute_affinity(..., alpha=...)`` produces -- the endpoint must not be a
   second implementation of the fusion.
2. The endpoint's guards: a negative alpha is refused, a pair with no cached
   models is refused with 409 rather than a stack trace, and an omitted alpha
   falls back to the calibrated config value.
3. ``include_aef`` returns the alpha = 0 table alongside, and that table matches
   a direct alpha = 0 computation (statuses included, not just the matrix).
4. The export path honours ``alpha`` and does **not** overwrite the run's cached
   CSVs, which belong to the run rather than to a what-if view.
5. The products payload carries ``semantic_prior_alpha`` so the slider can
   anchor on the calibrated value.
6. Frontend wiring: the control exists in the HTML, is styled, and app.js keeps
   alpha in session state rather than posting it as a run parameter.

Run:  python scripts/verify_stage8d.py

Pure CPU, no GEE. Stages 2-3 must have run for the pair.
"""

from __future__ import annotations

import csv
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402

from harmonizer.affinity import compute_affinity  # noqa: E402
from harmonizer.config import CONFIG, REPO_ROOT  # noqa: E402
from harmonizer.decision import (  # noqa: E402
    absent_decisions,
    build_matching_table,
    classify_rows,
)

REFERENCE_ID = "hrlc30_africa"
COMPARE_ID = "worldcover_2020"


def _rule(title: str) -> None:
    print(f"\n{title}\n{'-' * len(title)}")


def check(ok: bool, message: str, detail: str = "") -> bool:
    print(f"  {'PASS' if ok else 'FAIL'}  {message}" + (f" [{detail}]" if detail else ""))
    return bool(ok)


def _direct(alpha: float):
    """The matching table a direct Stage 4 computation gives at one alpha."""
    aff = compute_affinity(REFERENCE_ID, COMPARE_ID, alpha=alpha)
    decisions, _ = classify_rows(aff)
    rows = build_matching_table(
        aff, decisions, include_absent=absent_decisions(REFERENCE_ID, aff)
    )
    return aff, rows


def _summarise(payload_rows) -> list[tuple]:
    """Comparable summary of an API matching table: identity, status, candidates.

    A compare-side absent row names its class but has no probability, which the
    API payload renders as ``None``. Dropping those entries (rather than
    coercing them to 0.0) is what makes this comparable to the dataclass rows,
    where such a row simply has an empty ``probabilities`` list.
    """
    return [
        (
            r["side"],
            r["reference_value"],
            r["status"],
            tuple(
                (c["value"], round(c["probability"], 9))
                for c in r["compare"]
                if c["probability"] is not None
            ),
        )
        for r in payload_rows
    ]


def _summarise_rows(rows) -> list[tuple]:
    return [
        (
            r.side,
            r.reference_value,
            r.status,
            tuple(zip(r.compare_values, [round(p, 9) for p in r.probabilities])),
        )
        for r in rows
    ]


def main() -> int:
    ok = True

    from harmonizer.modeling import gmm_cache_path

    for pid in (REFERENCE_ID, COMPARE_ID):
        if not gmm_cache_path(pid).exists():
            print(f"FAILED: no cached GMM for {pid}; run Stages 2-3 first.")
            return 1

    try:
        from fastapi.testclient import TestClient
    except ImportError as exc:
        print(f"FAILED: fastapi/httpx needed for this verification ({exc})")
        return 1

    from harmonizer.api import app

    client = TestClient(app)
    pair = {"reference_id": REFERENCE_ID, "compare_id": COMPARE_ID}

    # ---------------------------------------------------------------- 1. #
    _rule("1. /api/affinity matches a direct compute_affinity at each alpha")

    for alpha in (0.0, 0.5, 1.0, 2.0):
        resp = client.get("/api/affinity", params={**pair, "alpha": alpha})
        if resp.status_code != 200:
            ok = check(False, f"alpha={alpha} returned {resp.status_code}", resp.text[:120])
            continue
        payload = resp.json()
        aff, rows = _direct(alpha)

        same_matrix = np.allclose(
            np.array(payload["normalized_affinity"]), aff.normalized_affinity, atol=0
        )
        same_table = _summarise(payload["matching_table"]) == _summarise_rows(rows)
        ok &= check(
            same_matrix and same_table and abs(payload["alpha"] - alpha) < 1e-12,
            f"alpha={alpha}: matrix, statuses and candidates all match",
            f"strong={sum(1 for r in payload['matching_table'] if r['status'] == 'strong')}",
        )

    # ---------------------------------------------------------------- 2. #
    _rule("2. Guards")

    r = client.get("/api/affinity", params={**pair, "alpha": -0.5})
    ok &= check(r.status_code == 400, "negative alpha refused with 400", f"got {r.status_code}")

    r = client.get(
        "/api/affinity",
        params={"reference_id": "dynamicworld", "compare_id": COMPARE_ID},
    )
    ok &= check(
        r.status_code == 409,
        "pair with no cached models refused with 409",
        f"got {r.status_code}",
    )

    r = client.get("/api/affinity", params=pair)
    ok &= check(
        r.status_code == 200
        and abs(r.json()["alpha"] - CONFIG.affinity.semantic_prior_alpha) < 1e-12,
        "omitted alpha falls back to the calibrated config value",
        f"{r.json().get('alpha')} vs config {CONFIG.affinity.semantic_prior_alpha}",
    )

    # ---------------------------------------------------------------- 3. #
    _rule("3. include_aef returns the observational-only table")

    r = client.get(
        "/api/affinity", params={**pair, "alpha": 1.0, "include_aef": "true"}
    )
    payload = r.json()
    ok &= check("matching_table_aef" in payload, "matching_table_aef present")
    if "matching_table_aef" in payload:
        _, aef_rows = _direct(0.0)
        ok &= check(
            _summarise(payload["matching_table_aef"]) == _summarise_rows(aef_rows),
            "the aef table equals a direct alpha=0 computation",
        )
        # It must actually differ from the fused table, or the toggle shows nothing.
        ok &= check(
            _summarise(payload["matching_table_aef"])
            != _summarise(payload["matching_table"]),
            "the aef table differs from the fused table at alpha=1",
        )

    r = client.get("/api/affinity", params={**pair, "alpha": 1.0})
    ok &= check(
        "matching_table_aef" not in r.json(),
        "aef table omitted by default (comparison is opt-in)",
    )

    # ---------------------------------------------------------------- 4. #
    _rule("4. Export honours alpha without clobbering the run's CSVs")

    cached = CONFIG.cache_dir / "matching_table.csv"

    # A job is needed for the export route; run the cheap cached path.
    from harmonizer.pipeline import RunParams, can_reuse_cache

    params = RunParams(
        reference_id=REFERENCE_ID,
        compare_id=COMPARE_ID,
        aoi=(26.0637, 3.52308, 43.2877, 16.28083),
        sample_scale_m=100.0,
    )
    if not can_reuse_cache(params):
        print("  SKIP  no matching run signature; export check needs a cached run")
    else:
        start = client.post(
            "/api/run",
            json={
                "reference_id": REFERENCE_ID,
                "compare_id": COMPARE_ID,
                "aoi": [26.0637, 3.52308, 43.2877, 16.28083],
                "sample_scale_m": 100.0,
            },
        )
        if start.status_code != 200:
            ok = check(False, "could not start a run for the export check",
                       start.text[:160])
        else:
            import time

            job_id = start.json()["job_id"]
            # The job runs as a background task, so poll with a real pause --
            # a tight loop never yields and the state would never advance.
            st = {"state": "queued"}
            for _ in range(120):
                st = client.get(f"/api/jobs/{job_id}").json()
                if st["state"] in ("done", "failed"):
                    break
                time.sleep(0.5)
            ok &= check(
                st["state"] == "done",
                "cached run finished",
                f"{st['state']}: {st.get('message', '')[:80]}",
            )

            got = {}
            for alpha in (0.0, 1.0):
                e = client.get(
                    f"/api/jobs/{job_id}/export/matching_table", params={"alpha": alpha}
                )
                got[alpha] = e.text
                ok &= check(
                    e.status_code == 200, f"export at alpha={alpha} returned 200",
                    f"got {e.status_code}"
                )
            ok &= check(
                got.get(0.0) != got.get(1.0),
                "exports at alpha 0 and 1 differ (the parameter reaches the file)",
            )
            # The alpha=1 export must equal a direct alpha=1 table.
            _, direct_rows = _direct(1.0)
            rows_in_csv = list(csv.DictReader(got[1.0].splitlines()))
            ref_rows = [r for r in rows_in_csv if r["side"] == "reference"]
            ok &= check(
                len(ref_rows) == len([r for r in direct_rows if r.side == "reference"]),
                "exported row count matches the direct computation",
                f"{len(ref_rows)} rows",
            )

            # The what-if export must not rewrite the run's own cached CSV: the
            # cache belongs to the run, and a download at another alpha is a
            # view. Snapshot AFTER the run (which legitimately writes it), then
            # export at a different alpha and confirm the file is untouched.
            snapshot = cached.read_bytes() if cached.exists() else None
            off_alpha = 0.0 if CONFIG.affinity.semantic_prior_alpha else 1.0
            client.get(
                f"/api/jobs/{job_id}/export/matching_table",
                params={"alpha": off_alpha},
            )
            ok &= check(
                snapshot == (cached.read_bytes() if cached.exists() else None),
                "a what-if export leaves the run's cached CSV untouched",
                f"exported at alpha={off_alpha}",
            )

    # ---------------------------------------------------------------- 4b. #
    _rule("4b. The MERGED table honours alpha too")

    # The merged table replaces the primary-only one whenever a pair has
    # auxiliaries, and the UI calls it right after applying alpha. If it ignored
    # alpha it would silently revert the user's choice on exactly those runs --
    # which is not visible in a pair that happens to have no auxiliaries, so it
    # is asserted directly here.
    merged = {}
    for alpha in (0.0, 1.0):
        m = client.get("/api/merged/table", params={**pair, "alpha": alpha})
        ok &= check(
            m.status_code == 200, f"merged table at alpha={alpha} returned 200",
            f"got {m.status_code}"
        )
        if m.status_code == 200:
            merged[alpha] = _summarise(m.json()["matching_table"])
    ok &= check(
        len(merged) == 2 and merged[0.0] != merged[1.0],
        "merged table differs between alpha 0 and 1 (alpha reaches every AOI)",
    )
    ok &= check(
        client.get("/api/merged/table", params={**pair, "alpha": -1}).status_code
        == 400,
        "merged table refuses a negative alpha",
    )

    # The auxiliary sub-matrices resolve their scoped ids back to base products
    # for the prior; otherwise auxiliary rows stay observational while primary
    # rows are fused, inside one table.
    from harmonizer.semantics import base_product_id

    ok &= check(
        base_product_id("worldcover_2020__aux_coast") == "worldcover_2020"
        and base_product_id("worldcover_2020") == "worldcover_2020",
        "an auxiliary-scoped id resolves to its base product for the prior",
    )

    # ---------------------------------------------------------------- 5. #
    _rule("5. Products payload carries the calibrated alpha")

    prod = client.get("/api/products").json()
    ok &= check(
        "semantic_prior_alpha" in prod.get("calibration", {}),
        "calibration block exposes semantic_prior_alpha",
        str(prod.get("calibration", {}).get("semantic_prior_alpha")),
    )

    # ---------------------------------------------------------------- 6. #
    _rule("6. Frontend wiring")

    html = (REPO_ROOT / "web" / "index.html").read_text(encoding="utf-8")
    js = (REPO_ROOT / "web" / "app.js").read_text(encoding="utf-8")
    css = (REPO_ROOT / "web" / "style.css").read_text(encoding="utf-8")

    for ident in ("alpha-panel", 'id="alpha"', "alpha-compare", "alpha-value"):
        ok &= check(ident in html, f"index.html carries {ident}")
    ok &= check("#alpha-panel" in css, "style.css styles the alpha panel")
    for fn in ("applyAlpha", "currentAlpha", "setAlphaEnabled", "wireAlpha"):
        ok &= check(f"function {fn}" in js, f"app.js defines {fn}()")

    # The control must be a VIEW control: alpha must not be posted as a run
    # parameter, or it would enter the run signature and force a re-sample.
    run_post = re.search(r"sample_scale_m: Number\(\$\(\"sample_scale_m\"\)", js)
    ok &= check(run_post is not None, "run request builder found in app.js")
    posts_alpha = re.search(r"alpha:\s*(Number\(\$\(\"alpha\"\)|currentAlpha\(\))\s*,[^}]*force_refresh", js)
    ok &= check(
        posts_alpha is None,
        "alpha is NOT sent as a run parameter (it stays a display control)",
    )
    ok &= check(
        "let ALPHA = null" in js,
        "alpha kept in session state, not written back to config",
    )

    print(f"\n{'OK' if ok else 'FAILED'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
