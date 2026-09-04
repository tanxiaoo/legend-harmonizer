"""Verification for Stage W1 -- drop-in registration, picker, greyed legend.

DESIGN.md section 6, Stage W1. The stage's promise is: *drop a dataset folder
and its legend CSV into ``data/``, start the app, and the map is choosable* --
with the UI telling you what state everything is in rather than silently
omitting what is not ready.

Checks:

1. **Detection.** Every folder under ``data/`` holding rasters is a candidate,
   with a state of ``ready`` / ``needs-legend`` / ``needs-conversion`` /
   ``indexing`` / ``converting`` / ``error``. A dataset is never silently absent.
2. **Legend location.** A dataset's legend is ``data/<Dataset>/legend.csv``,
   beside its rasters -- one name, one place, nothing matched or guessed. A
   folder without one reports ``needs-legend`` naming the exact path it wants.
   Its ``IsClass`` column decides which rows are land cover: rows marked FALSE
   reach neither the map legend nor the sampler. Deleting a dataset folder is
   detected (``missing``), the product is listed but not selectable, and its
   derived files can be removed explicitly -- never automatically, and never
   from ``data/``.
3. **The API surfaces state.** ``/api/products`` carries per-product ``state``,
   resolution and years so the picker can group, badge and disable; only
   ``ready`` products are selectable.
4. **Greyed legend classes.** A class declared in the legend CSV but not found
   in the dataset's pixels is kept in the registry YAML with ``observed:
   false``, and the legend API reports it -- so the UI can grey it out instead
   of dropping it.
5. **A registration actually runs** (with ``--register <folder>``): index ->
   convert -> ready, with progress advancing and failures surfacing as an
   ``error`` state carrying a usable message.

Run::

    python scripts/verify_w1.py
    python scripts/verify_w1.py --register JAXA_HRLULC_SEA_2023
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Keep the startup auto-registration from firing when this script builds a
# TestClient: the checks below want to observe state, not start conversions.
os.environ.setdefault("HARMONIZER_NO_AUTOREGISTER", "1")

import harmonizer  # noqa: F401,E402  (repairs PROJ before rasterio loads)
from harmonizer import registration as reg  # noqa: E402
from harmonizer.config import CONFIG  # noqa: E402

_FAILURES: list[str] = []

_VALID_STATES = {
    reg.READY,
    reg.INDEXING,
    reg.CONVERTING,
    reg.NEEDS_LEGEND,
    reg.NEEDS_CONVERSION,
    reg.ERROR,
    reg.MISSING,
}


def _check(label: str, ok: bool, detail: str = "") -> bool:
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}" + (f" -- {detail}" if detail else ""))
    if not ok:
        _FAILURES.append(label)
    return ok


def check_detection() -> list[reg.DatasetState]:
    print("\n=== 1. Detection: every data/ folder is accounted for ===")
    states = reg.scan_datasets()

    folders_with_rasters = sorted(
        p.name
        for p in CONFIG.data_dir.iterdir()
        if p.is_dir()
        and p.name != "legend"
        and any(
            f.suffix == ".tif" and not f.name.endswith(".aux.xml") for f in p.glob("*.tif")
        )
    )
    seen = sorted(s.folder for s in states)
    _check(
        "every folder holding rasters is a candidate",
        seen == folders_with_rasters,
        f"found {seen}",
    )

    print(f"  {'folder':34s} {'product id':28s} {'state':17s} files")
    for s in states:
        print(f"  {s.folder:34s} {s.product_id:28s} {s.state:17s} {s.n_files:5d}")
        if s.detail:
            print(f"      {s.detail[:110]}")
    _check(
        "every candidate has a valid state",
        all(s.state in _VALID_STATES for s in states),
        f"states {sorted({s.state for s in states})}",
    )
    return states


def check_legend_convention(states: list[reg.DatasetState]) -> None:
    """The rule is one file name in one place: data/<Dataset>/legend.csv."""
    print("\n=== 2. Legend discovery: data/<Dataset>/legend.csv ===")

    for s in states:
        resolved = "found" if s.state != reg.NEEDS_LEGEND else "NOT FOUND"
        print(f"  {s.folder:34s} -> {s.legend}  [{resolved}]")

    # Every dataset that has a legend at all should be using the in-folder one,
    # unless it is on the legacy path or named explicitly in the manifest.
    off_convention = []
    for s in states:
        if s.state == reg.NEEDS_LEGEND or s.legend == reg.LEGEND_FILENAME:
            continue
        settings = reg._manifest().get(s.folder) or {}
        if not settings.get("legend"):
            off_convention.append((s.folder, s.legend))
    _check(
        f"datasets use the in-folder {reg.LEGEND_FILENAME} (or an explicit manifest entry)",
        not off_convention,
        f"legacy/other: {off_convention}" if off_convention else "",
    )

    # The in-folder legend must win over anything in the legacy directory, so a
    # migrated dataset does not silently keep reading the old file.
    for s in states:
        inside = CONFIG.data_dir / s.folder / reg.LEGEND_FILENAME
        if inside.is_file():
            _check(
                f"{s.folder}: uses its own {reg.LEGEND_FILENAME}",
                s.legend == reg.LEGEND_FILENAME,
                f"resolved to {s.legend}",
            )

    needs = [s for s in states if s.state == reg.NEEDS_LEGEND]
    _check(
        "a dataset without a legend is told exactly where to put it",
        all(reg.LEGEND_FILENAME in (s.detail or "") for s in needs) if needs else True,
        f"{len(needs)} dataset(s) awaiting a legend",
    )


def check_is_class() -> None:
    """`IsClass = FALSE` rows must not reach the registry, the map, or sampling."""
    print("\n=== 2b. IsClass: user-declared fill rows are excluded ===")
    from harmonizer.indexer import LEGEND_FILENAME, load_legend_csv, parse_is_class

    # The parser must accept the spellings a hand-edited CSV really contains,
    # and must default to "class" so an un-annotated legend loses nothing.
    cases = {
        "TRUE": True, "true": True, "Yes": True, "1": True,
        "FALSE": False, "false": False, "no": False, "0": False,
        "": True, None: True, "maybe": True,
    }
    bad = {k: parse_is_class(k) for k, v in cases.items() if parse_is_class(k) != v}
    _check("IsClass parsing accepts the usual spellings, defaults to True", not bad, str(bad))

    from harmonizer.registry.schema import load_all_products
    products = load_all_products()

    for state in reg.scan_datasets():
        csv_path = CONFIG.data_dir / state.folder / LEGEND_FILENAME
        if not csv_path.is_file():
            continue
        try:
            rows = load_legend_csv(csv_path)
        except Exception:
            continue
        declared_fill = sorted(c for c, r in rows.items() if not r[3])
        if not declared_fill:
            print(f"  {state.folder:34s} no IsClass=FALSE rows")
            continue
        print(f"  {state.folder:34s} IsClass=FALSE -> {declared_fill}")

        spec = products.get(state.product_id)
        if spec is None:
            continue
        in_registry = [c.code for c in (spec.legend or ()) if c.code in declared_fill]
        _check(
            f"{state.product_id}: IsClass=FALSE rows are absent from the registry legend",
            not in_registry,
            f"leaked {in_registry}" if in_registry else "",
        )

        # And the sampler's allowlist must not contain them either -- the legend
        # is what feeds it, so this is the property that actually protects the
        # matching table.
        from harmonizer.chunked_sampling import _class_codes

        allow = _class_codes(state.product_id)
        leaked = sorted(c for c in declared_fill if allow is not None and c in allow)
        _check(
            f"{state.product_id}: IsClass=FALSE rows are not samplable",
            not leaked,
            f"leaked {leaked}" if leaked else "",
        )


def check_inputs_come_from_the_dataset_folder() -> None:
    """Everything a dataset produces must derive from files inside its own folder.

    A dataset is self-contained: its rasters and its ``legend.csv`` live in one
    folder, and nothing outside it may influence what is generated. That was
    violated in a way nothing in the output revealed: legend resolution used
    ``Path(name).exists()``, which is relative to the **current working
    directory**, so a stray ``legend.csv`` in the repo root won every lookup.
    Two CCI HRLC datasets were indexed against a different product's legend and
    produced YAML that looked entirely plausible while naming every class
    wrongly -- the header even recorded the correct path it had not read.

    Checked by comparing each product's generated legend against the CSV in its
    own folder, which is the property that actually matters, rather than by
    asserting anything about how paths are resolved.
    """
    print("\n=== 2f. Inputs come from the dataset's own folder ===")
    from harmonizer.indexer import LEGEND_FILENAME, load_legend_csv
    from harmonizer.registry.schema import load_all_products

    products = load_all_products()
    checked = 0
    for state in reg.scan_datasets():
        if state.state == reg.MISSING:
            continue
        csv_path = CONFIG.data_dir / state.folder / LEGEND_FILENAME
        spec = products.get(state.product_id)
        if not csv_path.is_file() or spec is None or not spec.legend:
            continue
        checked += 1

        try:
            rows = load_legend_csv(csv_path)
        except Exception as exc:
            _check(f"{state.folder}: its legend.csv parses", False, str(exc)[:80])
            continue

        mismatched = [
            (c.code, c.name, rows[c.code][0])
            for c in spec.legend
            if c.code in rows and c.name != rows[c.code][0]
        ]
        _check(
            f"{state.product_id}: legend names match data/{state.folder}/{LEGEND_FILENAME}",
            not mismatched,
            f"e.g. code {mismatched[0][0]}: YAML {mismatched[0][1]!r} vs CSV "
            f"{mismatched[0][2]!r}" if mismatched else "",
        )

        # Every generated class must exist in that folder's CSV -- a class from
        # anywhere else means some other file was read.
        foreign = [c.code for c in spec.legend if c.code not in rows]
        _check(
            f"{state.product_id}: no class comes from outside its folder",
            not foreign,
            f"codes {foreign[:5]}" if foreign else "",
        )

        # And the raster it points at must be inside the dataset folder (or the
        # VRT built from it), never another dataset's file.
        path = Path(spec.access.path)
        if not path.is_absolute():
            path = CONFIG.data_dir.parent / path
        in_folder = (CONFIG.data_dir / state.folder) in path.parents
        is_own_vrt = path.name == f"{state.product_id}.vrt"
        _check(
            f"{state.product_id}: access.path is its own raster or VRT",
            in_folder or is_own_vrt,
            str(path),
        )

        _check_tiles_come_from_this_dataset(state, spec, rows)

    # Every recorded artifact must name its own product: a manifest entry
    # pointing at another product's file would let one dataset's cleanup delete
    # another's data.
    from harmonizer import manifest

    foreign_artifacts = [
        (pid, a.path)
        for pid, m in manifest.all_manifests().items()
        for a in m.artifacts
        if pid not in a.path
    ]
    _check(
        "every manifest artifact names its own product",
        not foreign_artifacts,
        str(foreign_artifacts[:3]) if foreign_artifacts else "",
    )

    _check("at least one dataset was checked", checked > 0, f"{checked} checked")


def _check_tiles_come_from_this_dataset(state, spec, legend_rows) -> None:
    """The pixels served must come from this dataset's own raster.

    Three links, each verified against the dataset folder rather than assumed:

    1. tiles are served from **this product's** COG tree (or its own VRT);
    2. that COG is geometrically the same raster as the file in ``data/`` --
       identical bounds and shape, so it cannot be a copy of another dataset;
    3. the codes a served tile actually contains all appear in **this folder's**
       ``legend.csv``.

    (3) is the one that catches a mixed-up dataset in practice: two products can
    share a code range and still be entirely different maps, so comparing the
    served pixels against the folder's own legend is what proves they belong
    together.
    """
    import io
    import math

    import numpy as np
    from PIL import Image
    from fastapi.testclient import TestClient

    from harmonizer import local_tiles
    from harmonizer.api import app

    pid = state.product_id

    source = local_tiles.cog_source(pid)
    if source is not None:
        served = Path(source[0])
        owns = pid in served.parts or served.name == f"{pid}.vrt"
        _check(f"{pid}: tiles are served from its own converted tree", owns, str(served))

        # A single converted COG must match its source raster exactly. (A mosaic
        # is many files over the same folder; its bounds are the union, so the
        # per-file comparison does not apply.)
        if not source[1] and served.suffix == ".tif":
            import rasterio

            folder = CONFIG.data_dir / state.folder
            tifs = sorted(p for p in folder.glob("*.tif") if not p.name.endswith(".aux.xml"))
            if len(tifs) == 1:
                try:
                    with rasterio.open(tifs[0]) as a, rasterio.open(served) as b:
                        same = (
                            [round(v, 4) for v in a.bounds] == [round(v, 4) for v in b.bounds]
                            and a.shape == b.shape
                        )
                    _check(f"{pid}: its COG is the same raster as data/{state.folder}/", same)
                except Exception as exc:
                    _check(f"{pid}: its COG is the same raster as data/{state.folder}/",
                           False, str(exc)[:80])

    # Served pixels must only carry codes this folder's legend declares.
    fp = getattr(spec, "footprint", None)
    if not fp:
        return
    client = TestClient(app)
    lon, lat = (fp[0] + fp[2]) / 2, (fp[1] + fp[3]) / 2
    foreign: set[int] = set()
    for z in (5, 8, 11):
        n = 2**z
        x = int((lon + 180.0) / 360.0 * n)
        y = int((1.0 - math.asinh(math.tan(math.radians(lat))) / math.pi) / 2.0 * n)
        r = client.get(f"/api/tiles/local/{pid}/{z}/{x}/{y}.png")
        if r.status_code != 200:
            continue
        arr = np.array(Image.open(io.BytesIO(r.content)).convert("LA"))
        codes = set(np.unique(arr[..., 0][arr[..., 1] > 0]).tolist()) - {0}
        foreign |= codes - set(legend_rows)
    _check(
        f"{pid}: served tiles carry only codes from its own legend.csv",
        not foreign,
        f"foreign codes {sorted(foreign)[:8]}" if foreign else "",
    )


def check_registry_invalidation() -> None:
    """A newly written registry YAML must be visible without a restart.

    Registration writes the YAML and then immediately converts, which reads the
    registry back. When the cache reset silently did nothing, that second read
    saw a stale registry and conversion failed with
    ``not a local-raster product: <id>`` while the YAML sat on disk, correct --
    the failure surfaced two steps away from its cause.

    The reset was ineffective because it *probed* for ``cache_clear`` on names
    that are plain functions. This checks the behaviour (is the new entry
    visible?) rather than the mechanism, so it stays honest if the caching
    changes again.
    """
    print("\n=== 2e. Registry cache: a new YAML is visible immediately ===")
    from harmonizer.registry import legends
    from harmonizer.registry.schema import PRODUCTS_DIR

    probe = PRODUCTS_DIR / "__invalidation_probe__.yaml"
    try:
        legends._specs()  # prime the cache so a stale read is possible
        probe.write_text(
            "id: __invalidation_probe__\n"
            "display_name: probe\nprovider: x\nrole: reference\nkind: label\n"
            "access:\n  method: local_raster\n  path: cache/nope.tif\n"
            "band: 1\nlegend:\n  - code: 1\n    name: a\n    color: \"#000000\"\n",
            encoding="utf-8",
        )
        stale = "__invalidation_probe__" in legends._specs()
        reg._invalidate_registry()
        fresh = "__invalidation_probe__" in legends._specs()
        _check(
            "a YAML written after a lookup is invisible until invalidation",
            not stale,
            "cache was not primed -- the check proves nothing" if stale else "",
        )
        _check("_invalidate_registry() makes it visible", fresh)
    finally:
        probe.unlink(missing_ok=True)
        reg._invalidate_registry()


def check_manifests() -> None:
    """Cleanup is driven by a recorded manifest, not by guessing from names."""
    print("\n=== 2d. Artifact manifests: ownership is recorded, not inferred ===")
    from harmonizer import manifest

    manifests = manifest.all_manifests()
    for pid, m in sorted(manifests.items()):
        kinds = ", ".join(sorted({a.kind for a in m.artifacts}))
        print(f"  {pid:28s} {len(m.artifacts)} artifact(s)  [{kinds}]")

    _check("registered products have a manifest", bool(manifests))

    # The single most important property: a manifest may never name a source
    # file. It is the input to a delete, so one bad entry destroys a download.
    leaking = [
        (pid, a.path)
        for pid, m in manifests.items()
        for a in m.artifacts
        if manifest._is_under_data(a.resolve())
    ]
    _check(
        "no manifest records anything under data/",
        not leaking,
        str(leaking[:3]) if leaking else "",
    )

    # And recording one must be refused outright, not merely absent by luck.
    try:
        manifest.record("__probe__", CONFIG.data_dir / "anything.tif")
        refused = False
    except ValueError:
        refused = True
    finally:
        manifest.forget("__probe__")
    _check("recording a data/ path is refused", refused)

    # A product with no manifest owns nothing: no manifest, no evidence this app
    # created anything, so nothing may be deleted for it.
    from harmonizer.registry.schema import load_all_products

    unmanifested = [
        pid
        for pid in load_all_products()
        if pid not in manifests and reg.product_artifacts(pid)
    ]
    _check(
        "a product without a manifest reports no artifacts",
        not unmanifested,
        str(unmanifested) if unmanifested else "",
    )

    # Sibling products whose names are prefixes of one another must not claim
    # each other's files -- the exact failure a name-pattern approach invites.
    pairs = [
        (a, b)
        for a in manifests
        for b in manifests
        if a != b and (b.startswith(a) or a.startswith(b))
    ]
    bad = []
    for a, b in pairs:
        shared = set(reg.product_artifacts(a)) & set(reg.product_artifacts(b))
        if shared:
            bad.append((a, b, [p.name for p in shared]))
    _check(
        "prefix-sibling products share no artifacts",
        not bad,
        str(bad[:2]) if bad else f"{len(pairs)} prefix pair(s) checked",
    )


def check_missing_detection() -> None:
    """A deleted data/ folder is detected, listed disabled, and cleanable."""
    print("\n=== 2c. Deleted dataset: detected, not selectable, removable ===")

    # Never flag the repo's hand-written cluster-only entries (hrlc, wsf,
    # worldcover_local, ...): they have a registry YAML but no derived files, and
    # inviting the user to "clean up" curated repo files would be worse than the
    # problem this detection solves.
    from harmonizer.registry.schema import load_all_products

    flagged = {s.product_id for s in reg.scan_datasets() if s.state == reg.MISSING}
    repo_only = {
        pid
        for pid, spec in load_all_products().items()
        if getattr(spec.access, "method", None) == "local_raster"
        and not reg.product_artifacts(pid)
    }
    print(f"  registry entries with no derived files: {sorted(repo_only)}")
    _check(
        "hand-written cluster-only entries are NOT flagged as deleted datasets",
        not (flagged & repo_only),
        f"false positives: {sorted(flagged & repo_only)}",
    )

    # product_artifacts must never name anything under data/ -- that is the
    # user's download, and removal would otherwise destroy it.
    leaking = []
    for pid in load_all_products():
        for p in reg.product_artifacts(pid):
            if CONFIG.data_dir in p.parents or p == CONFIG.data_dir:
                leaking.append(str(p))
    _check(
        "removal never targets anything under data/",
        not leaking,
        str(leaking[:3]) if leaking else "",
    )

    # Removing a product whose folder still exists must be refused: that is not
    # a cleanup, it is discarding a working product's converted data.
    live = [s for s in reg.scan_datasets() if s.state == reg.READY]
    if live:
        pid = live[0].product_id
        try:
            reg.remove_product(pid)
            _check(f"refuses to remove {pid} while its data folder exists", False)
        except ValueError:
            _check(f"refuses to remove {pid} while its data folder exists", True)


def check_api() -> None:
    print("\n=== 3. API: picker data (state, resolution, years) ===")
    from fastapi.testclient import TestClient

    from harmonizer.api import app

    client = TestClient(app)

    payload = client.get("/api/datasets").json()
    _check("/api/datasets responds with a dataset list", "datasets" in payload)

    products = client.get("/api/products").json()["products"]
    local = [p for p in products if p["source"] == "local_raster"]
    _check("/api/products returns local products", bool(local))
    missing = [p["id"] for p in local if "state" not in p]
    _check("every local product carries a ready state", not missing, str(missing))

    print(f"  {'product':28s} {'state':17s} {'res':>7s}  years")
    for p in sorted(products, key=lambda x: (x["source"] != "local_raster", x["id"])):
        res = f"{p['resolution_m']:.0f}" if p.get("resolution_m") else "-"
        print(
            f"  {p['id']:28s} {p.get('state', '?'):17s} {res:>7s}  {p.get('years') or []}"
        )
    _check(
        "GEE products are always ready (nothing to index or convert)",
        all(p.get("state") == "ready" for p in products if p["source"] != "local_raster"),
    )


def check_observed_classes() -> None:
    print("\n=== 4. Greyed legend: classes absent from the data ===")
    from fastapi.testclient import TestClient

    from harmonizer.api import app
    from harmonizer.registry.schema import load_all_products

    client = TestClient(app)

    checked = 0
    for pid, spec in sorted(load_all_products().items()):
        if getattr(spec.access, "method", None) != "local_raster":
            continue
        legend = getattr(spec, "legend", None) or ()
        flagged = [c for c in legend if c.observed is not None]
        if not flagged:
            continue
        checked += 1
        absent = [c.code for c in legend if c.observed is False]
        present = [c.code for c in legend if c.observed is True]
        print(
            f"  {pid:28s} {len(present)} observed, {len(absent)} declared-but-absent"
            + (f" {absent}" if absent else "")
        )

        # The API must pass the flag through, or the UI cannot grey anything.
        resp = client.get(f"/api/legend/{pid}")
        if resp.status_code != 200:
            continue
        classes = resp.json()["classes"]
        _check(
            f"{pid}: /api/legend reports `observed` per class",
            all("observed" in c for c in classes),
        )
        api_absent = sorted(c["value"] for c in classes if c["observed"] is False)
        _check(
            f"{pid}: API absent-class list matches the registry",
            api_absent == sorted(absent),
            f"api {api_absent} vs yaml {sorted(absent)}",
        )

    if checked == 0:
        print(
            "  (no product carries observed flags yet -- they are written when a\n"
            "   dataset is indexed by this stage; run with --register to create one)"
        )


def check_registration(folder: str, convert: bool) -> None:
    print(f"\n=== 5. Registration run: {folder} ===")
    states = {s.folder: s for s in reg.scan_datasets()}
    if folder not in states:
        _check(f"{folder} is a known dataset folder", False, "not found under data/")
        return

    product_id = states[folder].product_id
    job = reg.register_dataset(folder, convert=convert)
    if job is None:
        print("  a registration is already running for this product; watching it")

    last = None
    t0 = time.time()
    while time.time() - t0 < 7200:
        j = reg.REGISTRATIONS.get(product_id)
        if j is None:
            break
        if (j.state, j.stage) != last:
            print(f"    {j.progress * 100:5.1f}%  {j.state:8s} {j.stage}")
            last = (j.state, j.stage)
        if j.state in ("done", "failed"):
            break
        time.sleep(2)

    j = reg.REGISTRATIONS.get(product_id)
    _check(f"{folder}: registration finished", j is not None and j.state == "done",
           (j.error or "")[:200] if j else "no job")
    if j is not None and j.state == "done":
        final = {s.product_id: s for s in reg.scan_datasets()}.get(product_id)
        _check(
            f"{folder}: product reports ready after registration",
            final is not None and final.state == reg.READY,
            final.state if final else "missing",
        )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--register", help="actually register this data/ folder")
    ap.add_argument(
        "--no-convert",
        action="store_true",
        help="with --register: index only, skip COG conversion",
    )
    args = ap.parse_args()

    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")

    print("Stage W1 verification -- drop-in registration, picker, greyed legend")
    print(f"  data dir: {CONFIG.data_dir}")

    states = check_detection()
    check_legend_convention(states)
    check_is_class()
    check_missing_detection()
    check_inputs_come_from_the_dataset_folder()
    check_registry_invalidation()
    check_manifests()
    check_api()
    check_observed_classes()
    if args.register:
        check_registration(args.register, convert=not args.no_convert)

    print("\n" + "=" * 70)
    if _FAILURES:
        print(f"{len(_FAILURES)} check(s) FAILED:")
        for f in _FAILURES:
            print(f"  - {f}")
        return 1
    print("All Stage W1 checks passed.")
    print(
        "\nStill to confirm by hand (browser):\n"
        "  - the picker groups Local / GEE, badges non-ready products and disables them\n"
        "  - '↻ datasets' rescans data/ and a newly dropped folder becomes selectable\n"
        "  - a declared-but-absent legend class shows as a greyed, non-clickable chip"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
