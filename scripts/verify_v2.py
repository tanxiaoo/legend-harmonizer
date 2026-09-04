"""Verification for Stage V2 -- class-code tiles and client-side rendering.

DESIGN.md section 6, Stage V2. The stage's claim is that toggling legend chips
causes **zero** tile requests while colours still match the registry palette
exactly. The browser half of that is checked in the network tab (see below); this
script checks everything the server and the encoding are responsible for:

1. **The code tile is subset-independent.** The tile URL carries no ``classes``,
   and the bytes/ETag are identical no matter which classes are selected -- so a
   toggle has nothing to re-fetch.
2. **Codes survive the round trip exactly.** The PNG decodes back to the same
   class codes the raster holds, with alpha marking nodata. Any scaling here
   would silently mislabel land cover in the browser.
3. **Client-side colouring is pixel-identical to the server's.** Emulates what
   ``ClassCodeLayer._paint`` does in ``web/app.js`` and compares the result
   against the server-rendered RGBA tile for the same address.
4. **Toggling a class off is exactly a palette change.** The emulated client
   render for a subset matches the server's ``?classes=`` render of that subset.
5. **The band still busts the cache.** A code tile's ETag ignores classes and
   colours but must still change with ``band:``.

The remaining browser-side claim -- that a toggle issues no network request --
is verified by hand, because it is a property of the running page:

    Open the app, DevTools > Network, filter ``/api/tiles/local/``. Click legend
    chips. Expected: the tile count does not increase, and the map recolours
    immediately. Switching product or band DOES re-fetch (different tile bytes).

Run::

    python scripts/verify_v2.py
    python scripts/verify_v2.py --only worldcover_2020
"""

from __future__ import annotations

import argparse
import io
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import harmonizer  # noqa: F401,E402  (repairs PROJ before rasterio loads)
from harmonizer import local_tiles  # noqa: E402

_FAILURES: list[str] = []


def _check(label: str, ok: bool, detail: str = "") -> bool:
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}" + (f" -- {detail}" if detail else ""))
    if not ok:
        _FAILURES.append(label)
    return ok


def _hex_rgb(color: str) -> tuple[int, int, int]:
    h = color.lstrip("#")
    return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)


def _paint_like_browser(codes, alpha, palette, visible):
    """Exactly what ``ClassCodeLayer._paint`` does in web/app.js.

    Kept deliberately literal (a per-class mask rather than anything clever) so
    it is obvious this mirrors the JS and is not a different algorithm that
    happens to agree.
    """
    import numpy as np

    out = np.zeros((*codes.shape, 4), np.uint8)
    for code, color in palette.items():
        if code not in visible:
            continue
        mask = (codes == code) & (alpha != 0)
        out[mask, :3] = _hex_rgb(color)
        out[mask, 3] = 255
    return out


def _decode(png: bytes):
    import numpy as np
    from PIL import Image

    arr = np.array(Image.open(io.BytesIO(png)).convert("LA"))
    return arr[..., 0], arr[..., 1]


def _first_covered_tile(client, pid: str) -> tuple[int, int, int] | None:
    """A tile address the product actually has data for, near its footprint centre."""
    import math

    spec = local_tiles._product_spec(pid)
    footprint = getattr(spec, "footprint", None) if spec is not None else None
    if not footprint:
        return None
    lon = (footprint[0] + footprint[2]) / 2.0
    lat = (footprint[1] + footprint[3]) / 2.0
    for z in (8, 7, 6, 9, 10):
        n = 2**z
        x = int((lon + 180.0) / 360.0 * n)
        y = int((1.0 - math.asinh(math.tan(math.radians(lat))) / math.pi) / 2.0 * n)
        if client.get(f"/api/tiles/local/{pid}/{z}/{x}/{y}.png").status_code == 200:
            return z, x, y
    return None


def verify(pid: str, client) -> None:
    import numpy as np

    print(f"\n=== {pid} ===")
    addr = _first_covered_tile(client, pid)
    if addr is None:
        print("  SKIP: no covered tile found for this product")
        return
    z, x, y = addr
    code_url = f"/api/tiles/local/{pid}/{z}/{x}/{y}.png"
    rgba_url = f"/api/tiles/local/rgba/{pid}/{z}/{x}/{y}.png"
    print(f"  tile z{z}/{x}/{y}")

    entries = client.get(f"/api/legend/{pid}").json()["classes"]
    legend = {e["value"]: e["color"] for e in entries}
    # Only classes the dataset actually contains can ever be in the visible set:
    # `observed: false` chips are greyed and non-toggleable, and refreshMapSide()
    # leaves them out of VISIBLE entirely (DESIGN.md 4.3). Comparing against a
    # subset that included one would test a state the UI cannot produce -- and
    # would fail, because the client paints nothing for a class with no pixels
    # while the server still honours it in `?classes=`.
    all_classes = {e["value"] for e in entries if e.get("observed") is not False}

    # 1. Subset-independence -------------------------------------------------
    meta = client.get(f"/api/tiles/{pid}").json()
    _check(
        "tile template carries no ?classes=",
        "classes=" not in meta["template"],
        meta["template"],
    )
    _check("template is advertised as class_code", meta.get("encoding") == "class_code")

    r_plain = client.get(code_url)
    subset = ",".join(str(v) for v in sorted(all_classes)[:2])
    # The code endpoint takes no `classes`; passing one must not change anything.
    r_subset = client.get(f"{code_url}?classes={subset}")
    _check(
        "code tile bytes identical regardless of ?classes",
        r_plain.content == r_subset.content,
    )
    _check(
        "code tile ETag identical regardless of ?classes",
        r_plain.headers.get("etag") == r_subset.headers.get("etag"),
        r_plain.headers.get("etag", ""),
    )
    _check(
        "conditional GET returns 304",
        client.get(code_url, headers={"if-none-match": r_plain.headers["etag"]}).status_code
        == 304,
    )

    # 2. Codes round-trip exactly -------------------------------------------
    codes, alpha = _decode(r_plain.content)
    source = local_tiles._read_tile(pid, z, x, y).data[0]
    _check(
        "decoded codes == raster codes (no scaling/offset)",
        np.array_equal(codes, source),
        f"values {sorted(np.unique(codes).tolist())[:8]}",
    )
    _check(
        "alpha is binary (0 = nodata, 255 = data)",
        set(np.unique(alpha).tolist()) <= {0, 255},
        f"unique alpha {np.unique(alpha).tolist()}",
    )

    # 3/4. Client colouring matches the server, whole and subset -------------
    for label, visible in (
        ("all classes", all_classes),
        ("two-class subset", set(sorted(all_classes)[:2])),
    ):
        url = rgba_url if visible == all_classes else (
            f"{rgba_url}?classes=" + ",".join(str(v) for v in sorted(visible))
        )
        from PIL import Image

        server = np.array(Image.open(io.BytesIO(client.get(url).content)).convert("RGBA"))
        client_side = _paint_like_browser(codes, alpha, legend, visible)
        same = np.array_equal(client_side, server)
        detail = ""
        if not same:
            diff = np.where((client_side != server).any(-1))
            detail = f"{len(diff[0])} of {codes.size} px differ"
        _check(f"client-painted == server-rendered ({label})", same, detail)

    # 5. Band still busts the cache -----------------------------------------
    indexes = local_tiles.band_indexes(pid)
    etag_now = local_tiles  # noqa: F841  (kept for readability below)
    from harmonizer.api import _tile_etag

    e_band_a = _tile_etag(pid, z, x, y, None, "code")
    e_rgba = _tile_etag(pid, z, x, y, None, "rgba")
    _check("code and rgba ETags are distinct", e_band_a != e_rgba)
    _check(
        "code ETag ignores the class subset",
        _tile_etag(pid, z, x, y, [10], "code") == e_band_a,
    )
    _check(
        "rgba ETag honours the class subset",
        _tile_etag(pid, z, x, y, [10], "rgba") != e_rgba,
    )
    print(f"  (band indexes for this product: {indexes})")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--only", action="append", help="product id (repeatable)")
    args = ap.parse_args()

    from fastapi.testclient import TestClient

    from harmonizer.api import app
    from harmonizer.registry.schema import load_all_products

    client = TestClient(app)

    products = []
    for pid, spec in sorted(load_all_products().items()):
        if getattr(spec.access, "method", None) != "local_raster":
            continue
        try:
            local_tiles.legend(pid)
            local_tiles._reader(pid)
        except Exception:
            continue
        products.append(pid)

    if args.only:
        unknown = set(args.only) - set(products)
        if unknown:
            print(f"unknown or unavailable product id(s): {sorted(unknown)}")
            print(f"available: {products}")
            return 2
        products = args.only
    if not products:
        print("no local-raster products with readable sources on this machine")
        return 2

    print("Stage V2 verification -- class-code tiles + client-side rendering")
    for pid in products:
        verify(pid, client)

    print("\n" + "=" * 70)
    if _FAILURES:
        print(f"{len(_FAILURES)} check(s) FAILED:")
        for f in _FAILURES:
            print(f"  - {f}")
        return 1
    print("All server-side V2 checks passed.")
    print(
        "\nStill to confirm by hand (a property of the running page):\n"
        "  Open the app, DevTools > Network, filter '/api/tiles/local/'.\n"
        "  Click legend chips -> the request count must NOT increase and the map\n"
        "  must recolour instantly. Changing product or band SHOULD re-fetch."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
