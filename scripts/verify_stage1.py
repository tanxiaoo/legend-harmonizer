"""Stage 1 verification: data access.

Feeds five hardcoded coordinates inside the Eastern Sahel box through the three
Stage 1 adapters and prints, for each point:

  * the WorldCover label (test-swap reference, annual static 2021, from GEE),
  * the Dynamic World label (annual modal, from GEE), and
  * the AlphaEarth embedding vector length and a short preview.

For the current test swap the reference is ESA WorldCover (GEE-native, so no
local download is needed); HRLC remains the eventual real reference. This proves
GEE access before anything else is built (docs/PIPELINE.md, Stage 1 ->
Verification).

Run:  python scripts/verify_stage1.py

Each of the three sources is exercised independently, so an un-authenticated GEE
session degrades to a clear message for that source rather than aborting the
whole run.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Allow running from a fresh checkout without `pip install -e .` by putting the
# repository root on the import path.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from harmonizer.config import CONFIG
from harmonizer.registry.products import default_registry

# Five global sample coordinates (lon, lat). WorldCover and Dynamic World are
# both global, so these are just probe points for exercising the adapters; they
# are not bounds-checked against any footprint. Product footprints and their
# overlap (the real valid AOI) are a Stage 5 concern, not Stage 1's. These points
# happen to span plausible land-cover types across the Eastern Sahel.
SAMPLE_COORDS = [
    (15.0, 12.0),   # central Sahel, Chad
    (32.5, 15.6),   # near Khartoum / Nile confluence, Sudan
    (14.4, 13.1),   # Lake Chad vicinity
    (38.7, 9.0),    # Ethiopian highlands edge
    (20.0, 6.5),    # southern vegetated belt, CAR/Sudan border area
]


def _worldcover_labels():
    reg = default_registry()
    try:
        adapter = reg.get("worldcover").adapter_factory()
        return adapter.sample_labels(SAMPLE_COORDS)
    except Exception as exc:  # noqa: BLE001 - report and continue
        print(f"WorldCover  : unavailable -> {type(exc).__name__}: {exc}")
        return None


def _dynamicworld_labels():
    reg = default_registry()
    try:
        adapter = reg.get("dynamicworld").adapter_factory()
        return adapter.sample_labels(SAMPLE_COORDS)
    except Exception as exc:  # noqa: BLE001
        print(f"DynamicWorld: unavailable -> {type(exc).__name__}: {exc}")
        return None


def _alphaearth_vectors():
    reg = default_registry()
    try:
        adapter = reg.get("alphaearth").adapter_factory()
        return adapter.sample_embeddings(SAMPLE_COORDS)
    except Exception as exc:  # noqa: BLE001
        print(f"AlphaEarth  : unavailable -> {type(exc).__name__}: {exc}")
        return None


def _fmt_label(res) -> str:
    if res is None:
        return "n/a"
    return "no-data" if res.masked else str(res.label)


def _fmt_vector(res) -> str:
    if res is None:
        return "n/a"
    if res.masked or res.vector is None:
        return "no-data"
    v = np.asarray(res.vector, dtype=float)
    preview = ", ".join(f"{x:+.3f}" for x in v[:4])
    return f"len={v.size}  [{preview}, ...]"


def main() -> None:
    print("=" * 72)
    print("Stage 1 verification - data access")
    print("=" * 72)

    wc = _worldcover_labels()
    dw = _dynamicworld_labels()
    ae = _alphaearth_vectors()
    print()

    header = f"{'#':>2}  {'lon':>7}  {'lat':>6}  {'WCov':>7}  {'DW':>7}  embedding"
    print(header)
    print("-" * len(header))
    for i, (lon, lat) in enumerate(SAMPLE_COORDS):
        wc_res = wc[i] if wc is not None else None
        dw_res = dw[i] if dw is not None else None
        ae_res = ae[i] if ae is not None else None
        print(
            f"{i:>2}  {lon:>7.3f}  {lat:>6.3f}  "
            f"{_fmt_label(wc_res):>7}  {_fmt_label(dw_res):>7}  "
            f"{_fmt_vector(ae_res)}"
        )

    # Explicit confirmation of the key contract: 64-value vectors.
    if ae is not None:
        good = [r for r in ae if not r.masked and r.vector is not None]
        if good:
            dims = {r.vector.size for r in good}
            ok = dims == {CONFIG.maps.embedding_dims}
            print()
            print(
                f"Embedding dims across {len(good)} unmasked point(s): {sorted(dims)} "
                f"-> {'OK' if ok else 'UNEXPECTED'} (expected "
                f"{CONFIG.maps.embedding_dims})"
            )

    print()
    print("Confirm: WorldCover/DW labels are plausible for each location and every")
    print("unmasked embedding has 64 values.")


if __name__ == "__main__":
    main()
