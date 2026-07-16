"""Stage 5.2 verification: comparison map (footprints, legends, label tiles).

Stage 5.2 layers the split comparison map onto the working 5.1 app: two label
maps side by side with per-class show/hide toggles, both products' footprints and
their overlap drawn, and AOI by draw/upload/type/full-overlap default. It adds
three *display-only* backend endpoints and reuses the 5.1 run/results/export path
unchanged (docs/PIPELINE.md, Stage 5.2 -> Verification).

The map itself is a browser concern (Leaflet), so the automated part here checks
the backend the map depends on, driving the FastAPI app with TestClient. To stay
offline and fast, the one endpoint that needs live Earth Engine -- the label-tile
renderer (``getMapId``) -- is **stubbed** so the tiles route's plumbing and the
per-class class-subset filtering are exercised without a GEE call. The legend and
footprint endpoints are pure (no network) and run for real.

Checks:
  1. GET /api/legend/{product} returns each map's class legend (value/name/colour)
     for both label maps, ordered by value, with the published palette; and 404s
     for a product with no drawable legend (the embedding).
  2. GET /api/footprints returns each selected product's derived footprint (global
     -> null box) plus the overlap outline, and the overlap geometry matches
     compute_overlap for the same AOI -- so the outline the map draws is correct.
  3. GET /api/tiles/{product} returns a tile template, passes the toggled class
     subset through to the tile builder, and 404s for an undrawable product.
  4. The 5.1 run/results/export path is untouched (products endpoint still shape-
     compatible; the three exports still registered).

Run:  python scripts/verify_stage5_2.py   (no network; tile builder stubbed)

Then the browser pass described at the end (with GEE authenticated): confirm both
maps display side by side, the per-class toggles work independently per map, and
the overlap outline on the map matches the /api/footprints geometry.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi.testclient import TestClient

import harmonizer.tiles as tiles
from harmonizer.affinity import class_name

REFERENCE_ID = "worldcover"
COMPARE_ID = "dynamicworld"
EMBEDDING_ID = "alphaearth"


# --------------------------------------------------------------------------- #
# Stub the live-GEE tile renderer so the tiles route is testable offline. The
# stub records the class subset it was asked to render (the toggle state) and
# returns a fake XYZ template, so we verify the wiring, not Earth Engine.
# --------------------------------------------------------------------------- #

TILE_CALLS: list[tuple[str, tuple[int, ...] | None]] = []


def _stub_tile_template(product_id, visible_values=None):
    subset = None if visible_values is None else tuple(int(v) for v in visible_values)
    TILE_CALLS.append((product_id, subset))
    if product_id not in ("worldcover", "dynamicworld"):
        raise KeyError(product_id)
    return f"https://example.test/{product_id}/{{z}}/{{x}}/{{y}}.png"


def main() -> None:
    print("=" * 88)
    print("Stage 5.2 verification - comparison map (footprints, legends, tiles)")
    print(f"reference={REFERENCE_ID}  compare={COMPARE_ID}")
    print("(label-tile renderer stubbed; legend + footprint endpoints run for real)")
    print("=" * 88)

    # Patch the tile renderer the API imported (api.py does `from harmonizer import
    # tiles` and calls tiles.tile_template), so patch it on the module.
    tiles.tile_template = _stub_tile_template

    from harmonizer.api import app

    client = TestClient(app)
    all_ok = True

    # 1. Legends -----------------------------------------------------------------
    print("\n[1] /api/legend/{product}")
    ok1 = True
    for pid in (REFERENCE_ID, COMPARE_ID):
        r = client.get(f"/api/legend/{pid}")
        body = r.json()
        classes = body.get("classes", []) if r.status_code == 200 else []
        values = [c["value"] for c in classes]
        ordered = values == sorted(values)
        # Names come from the single source (the product registry YAML via
        # class_name); colours are 6-digit hex. Legend order follows the YAML's
        # declared order, which for the MVP maps is ascending by code.
        names_ok = all(c["name"] == class_name(pid, c["value"]) for c in classes)
        colors_ok = all(
            isinstance(c["color"], str) and len(c["color"]) == 6 for c in classes
        )
        good = r.status_code == 200 and classes and ordered and names_ok and colors_ok
        ok1 &= good
        print(f"    {pid:>13}: {len(classes)} classes, ordered={ordered}, "
              f"names_ok={names_ok}, colours_ok={colors_ok} -> {'OK' if good else 'BAD'}")
        if good:
            preview = ", ".join(
                f"{c['value']}:{c['name']}#{c['color']}" for c in classes[:3]
            )
            print(f"                  e.g. {preview} ...")

    # The embedding product has no drawable legend -> 404.
    r404 = client.get(f"/api/legend/{EMBEDDING_ID}")
    embed_404 = r404.status_code == 404
    ok1 &= embed_404
    print(f"    {EMBEDDING_ID:>13}: HTTP {r404.status_code} (no drawable legend) "
          f"-> 404 as expected: {'YES' if embed_404 else 'NO'}")
    all_ok &= ok1
    print(f"    legends correct: {'YES' if ok1 else 'NO'}")

    # 2. Footprints + overlap ----------------------------------------------------
    print("\n[2] /api/footprints (footprints + overlap outline)")
    AOI = [32.0, 14.0, 33.0, 15.0]
    params = {
        "reference_id": REFERENCE_ID, "compare_id": COMPARE_ID,
        "min_lon": AOI[0], "min_lat": AOI[1], "max_lon": AOI[2], "max_lat": AOI[3],
    }
    fp = client.get("/api/footprints", params=params).json()

    # Both MVP label maps are global -> null footprint box (nothing to draw).
    prod_boxes = {p["id"]: p["bbox"] for p in fp["products"]}
    globals_ok = prod_boxes.get(REFERENCE_ID) is None and prod_boxes.get(COMPARE_ID) is None

    # The overlap the endpoint reports must match compute_overlap for this AOI, so
    # the outline the map draws is the same geometry the run will use.
    from harmonizer.pipeline import RunParams, compute_overlap
    ov = compute_overlap(RunParams(
        reference_id=REFERENCE_ID, compare_id=COMPARE_ID, aoi=tuple(AOI)))
    overlap_bbox_match = [round(x, 6) for x in fp["overlap"]["bbox"]] == \
        [round(x, 6) for x in ov.bbox]
    # The reported polygon must be the AOI (both maps global -> overlap == AOI).
    ring = fp["overlap"]["geometry"]["coordinates"][0]
    lons = [pt[0] for pt in ring]
    lats = [pt[1] for pt in ring]
    poly_match = (
        round(min(lons), 6) == AOI[0] and round(min(lats), 6) == AOI[1]
        and round(max(lons), 6) == AOI[2] and round(max(lats), 6) == AOI[3]
    )
    ok2 = globals_ok and overlap_bbox_match and poly_match
    all_ok &= ok2
    print(f"    both label maps global (null box): {globals_ok}")
    print(f"    overlap bbox == compute_overlap:   {overlap_bbox_match} "
          f"(endpoint {fp['overlap']['bbox']} vs {list(ov.bbox)})")
    print(f"    overlap polygon == AOI:            {poly_match}")
    print(f"    footprints/overlap correct: {'YES' if ok2 else 'NO'}")

    # 3. Tiles + class-subset filtering -----------------------------------------
    print("\n[3] /api/tiles/{product} (label tiles + per-class toggles)")
    TILE_CALLS.clear()
    # All classes (no ?classes) -> renderer called with None (render everything).
    r_all = client.get(f"/api/tiles/{REFERENCE_ID}")
    all_template = r_all.json().get("template", "")
    # A toggled subset must pass exactly those values through to the renderer.
    subset = [10, 80]  # Tree cover + Permanent water
    r_sub = client.get(f"/api/tiles/{REFERENCE_ID}", params={"classes": "10,80"})
    sub_template = r_sub.json().get("template", "")

    passed_none = (REFERENCE_ID, None) in TILE_CALLS
    passed_subset = (REFERENCE_ID, tuple(subset)) in TILE_CALLS
    templates_ok = "{z}" in all_template and "{z}" in sub_template

    # An undrawable product (the embedding) -> 404.
    r_bad = client.get(f"/api/tiles/{EMBEDDING_ID}")
    bad_404 = r_bad.status_code == 404

    ok3 = passed_none and passed_subset and templates_ok and bad_404
    all_ok &= ok3
    print(f"    no ?classes -> render all (renderer got None): {passed_none}")
    print(f"    ?classes=10,80 -> subset passed through:       {passed_subset}")
    print(f"    tile templates are XYZ ({{z}}/{{x}}/{{y}}):        {templates_ok}")
    print(f"    undrawable product (embedding) -> 404:         {bad_404}")
    print(f"    tiles + toggles wired correctly: {'YES' if ok3 else 'NO'}")

    # 4. 5.1 path untouched ------------------------------------------------------
    print("\n[4] 5.1 run/results/export path unchanged")
    prods = client.get("/api/products").json()
    products_ok = (
        "products" in prods and "defaults" in prods and "calibration" in prods
    )
    # The three exports are still registered (checked via the export map on the app).
    from harmonizer.api import _EXPORTS
    exports_ok = set(_EXPORTS) == {
        "raw_similarity", "normalized_affinity", "matching_table"
    }
    ok4 = products_ok and exports_ok
    all_ok &= ok4
    print(f"    /api/products still shape-compatible: {products_ok}")
    print(f"    three CSV exports still registered:   {exports_ok}")
    print(f"    5.1 path intact: {'YES' if ok4 else 'NO'}")

    print("\n" + "=" * 88)
    print(f"ALL AUTOMATED CHECKS PASSED: {'YES' if all_ok else 'NO'}")
    print(
        "\nBrowser pass (do this once, with GEE authenticated):\n"
        "  python run.py  ->  choose WorldCover x Dynamic World, and confirm:\n"
        "    - both selected maps display side by side (label tiles under your\n"
        "      GEE account), synced as you pan/zoom;\n"
        "    - the per-class show/hide toggles work independently per map;\n"
        "    - both products' footprints and their overlap outline are drawn, and\n"
        "      the overlap matches /api/footprints for the selected AOI;\n"
        "    - the AOI can be set by drawing a rectangle, uploading a GeoJSON\n"
        "      boundary, typing a bbox, or clearing it back to the full overlap;\n"
        "    - the 5.1 run -> heatmap -> matching table -> CSV flow still works."
    )


if __name__ == "__main__":
    main()
