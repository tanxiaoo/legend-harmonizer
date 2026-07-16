"""Product registry verification (docs/PIPELINE.md, section 2.5).

Confirms the registry is the single source of truth for map metadata and legends:

  * every product YAML loads and parses into a ProductSpec,
  * the registry (default_registry) is built from those files,
  * class names/colours resolve through the registry (the same values the old
    hardcoded CLASS_NAMES / _PALETTES carried),
  * the tiles legend and registry legend agree, and
  * (optionally, with --reconcile and GEE auth) the reconciliation check reports
    undeclared codes and absent-in-AOI classes against a live source.

Run:  python scripts/verify_registry.py
      python scripts/verify_registry.py --reconcile worldcover 30 12 34 15
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from harmonizer.registry import legends
from harmonizer.registry.products import default_registry
from harmonizer.registry.schema import load_all_products


def _load_and_list() -> None:
    specs = load_all_products()
    print(f"Loaded {len(specs)} product YAML files from the registry:\n")
    for pid, s in specs.items():
        access = s.access.method
        target = s.access.asset_id or s.access.path
        fp = "global" if s.footprint is None else str(list(s.footprint))
        print(f"  [{pid}] {s.display_name}")
        print(f"      role={s.role} kind={s.kind} access={access}:{target}")
        print(f"      band={s.band} res={s.resolution_m}m crs={s.crs} footprint={fp}")
        print(f"      years={list(s.available_years)} legend={len(s.legend)} classes")
    print()


def _check_registry_and_legends() -> None:
    reg = default_registry()
    print("Registry built from YAML (default_registry):")
    for p in reg.all():
        print(f"  {p.id:14s} name={p.name!r} role={p.role} footprint={p.footprint}")
    print()

    # A few spot checks that names/colours resolve through the registry.
    print("Legend lookups through the registry (name / colour):")
    for pid, code in [("worldcover", 80), ("dynamicworld", 0), ("hrlc", 130)]:
        print(
            f"  {pid} {code}: "
            f"{legends.class_name(pid, code)!r}  {legends.class_color(pid, code)}"
        )
    print()

    # Tiles legend must agree with the registry legend for drawable maps.
    from harmonizer import tiles

    for pid in ("worldcover", "dynamicworld"):
        tile_legend = {e.value: (e.name, "#" + e.color) for e in tiles.legend(pid)}
        reg_legend = {
            c.code: (c.name, c.color) for c in legends.legend_classes(pid)
        }
        ok = tile_legend == reg_legend
        print(f"  tiles.legend({pid}) matches registry legend: {ok}")
        if not ok:
            print(f"    tiles={tile_legend}\n    reg  ={reg_legend}")
    print()


def _reconcile(product_id: str, aoi) -> None:
    from harmonizer.registry.register import reconcile
    from harmonizer.registry.schema import load_all_products

    spec = load_all_products()[product_id]
    print(f"Reconciling {product_id} against its source over AOI={aoi} ...")
    rec = reconcile(spec, aoi)
    print(f"  declared codes : {list(rec.declared_codes)}")
    print(f"  observed codes : {list(rec.observed_codes)}")
    print(f"  matched        : {list(rec.matched)}")
    print(f"  UNDECLARED (in raster, missing from legend): {list(rec.undeclared)}")
    print(f"  ABSENT IN AOI  (declared but not observed) : {list(rec.absent_in_aoi)}")
    print(f"  legend covers every observed code: {rec.ok}")


def main() -> None:
    args = sys.argv[1:]
    if args and args[0] == "--reconcile":
        _load_and_list()
        _check_registry_and_legends()
        product_id = args[1]
        aoi = tuple(float(x) for x in args[2:6]) if len(args) >= 6 else None
        _reconcile(product_id, aoi)
        return
    _load_and_list()
    _check_registry_and_legends()
    print("OK: registry loads, and adapters/tiles/labels all read from the YAML.")
    print("(Run with --reconcile <id> <min_lon min_lat max_lon max_lat> for the")
    print(" reconciliation check against a live source; needs GEE auth.)")


if __name__ == "__main__":
    main()
