"""Register the tiled local-raster products under data/Africa/2019/ (one-off).

Each subfolder of data/Africa/2019/ (docs/PIPELINE.md section 2.5's "existing
products" comparison set) holds a map's tiles as many separate, non-overlapping
GeoTIFFs -- not the single file the registry's local_raster access method expects.
This script builds a GDAL VRT (a small XML index over the tiles; no pixel data is
copied or resampled) per product into cache/vrt/, then runs the registry's own
registration flow (harmonizer.registry.register) against that VRT to auto-detect
CRS, footprint, resolution, and the class codes actually present, writing a
pre-filled registry YAML into harmonizer/registry/products/.

Scope: only products where every tile shares one CRS (a VRT mosaic needs that).
Three products in this folder do not and are skipped -- see the printed report:
  - ESRI_Annual_LULC_2019, GL30_2019: each tile is in its own UTM zone.
  - GLC_FCS30D_2019: 23 bands per file (a multi-year time series, not one label
    band) -- needs a band choice, not just mosaicking.

Legends: where the source GeoTIFF carries an embedded colour table (WorldCover,
Copernicus_LCFM), names stay TODO but colours are pulled automatically. Where it
doesn't, every observed code gets a placeholder "Class N" name and a colour from
a generated qualitative palette, so the product is immediately selectable and
tileable; correct the names/colours in its YAML by hand afterwards.

Run:  python scripts/register_africa_products.py
      python scripts/register_africa_products.py --only worldcover_local
"""

from __future__ import annotations

import argparse
import colorsys
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from harmonizer.registry.register import Detected, build_registration_yaml, detect_local_raster
from harmonizer.registry.schema import PRODUCTS_DIR

DATA_DIR = REPO_ROOT / "data" / "Africa" / "2019"
VRT_DIR = REPO_ROOT / "cache" / "vrt"

# gdalbuildvrt is a standalone CLI (no PROJ/GDAL Python bindings needed in this
# venv). Prefer PATH; fall back to the Leonardo `gdal` environment module's
# known install if PATH doesn't have it (module-loading in this shell would
# also set PROJ_LIB to a version rasterio's bundled PROJ database rejects, so
# we call the binary directly rather than `module load gdal` in this process).
import shutil

_MODULE_GDALBUILDVRT = (
    "/leonardo/prod/spack/06/install/0.22/linux-rhel8-icelake/gcc-12.2.0/"
    "gdal-3.8.5-7h4jhaedtmovo2u3zpi3qd3dvtjuszt5/bin/gdalbuildvrt"
)
GDALBUILDVRT = shutil.which("gdalbuildvrt") or _MODULE_GDALBUILDVRT

# folder name -> (product id, display name, provider). Only the folders whose
# tiles share one CRS and one band (see module docstring for what's excluded).
PRODUCTS: dict[str, tuple[str, str, str]] = {
    "WorldCover_2020": ("worldcover_local", "ESA WorldCover v100 (Africa tiles, local)", "ESA"),
    "DynamicWorld_2019": ("dynamicworld_local", "Dynamic World V1 (Africa tiles, local, 2019)", "Google"),
    "FROM_GLC_2017": ("from_glc", "FROM-GLC10 (Africa tiles, local, 2017)", "Tsinghua/FROM-GLC"),
    "GFC_2020": ("gfc", "JRC Global Forest Cover 2020 (Africa tiles, local)", "JRC"),
    "GSW_Yearly_2019": ("gsw_yearly", "JRC Global Surface Water, yearly classification (Africa tiles, local, 2019)", "JRC"),
    "GWL_FCS30D_2019": ("gwl_fcs30d", "GWL_FCS30D wetland maps (Africa tiles, local, 2019)", "AIR/CAS"),
    "WSF_2019": ("wsf", "World Settlement Footprint 2019 (Africa tiles, local)", "DLR"),
    "FNF4_2019": ("fnf4", "Forest/Non-Forest 4-class (Africa tiles, local, 2019)", "JAXA"),
    "Copernicus_LCFM_LCM-10_2020": ("copernicus_lcfm", "Copernicus LCFM LCM-10 (Africa tiles, local, 2020)", "Copernicus"),
}

# Explicitly out of scope -- printed as a skip report, not silently dropped.
SKIPPED = {
    "ESRI_Annual_LULC_2019": "each tile is in its own UTM zone (per-tile CRS, not one shared CRS)",
    "GL30_2019": "each tile is in its own UTM zone (per-tile CRS, not one shared CRS)",
    "GLC_FCS30D_2019": "23 bands per file (multi-year time series) -- needs a band choice first",
}


def build_vrt(folder: Path, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tifs = sorted(str(p) for p in folder.glob("*.tif"))
    if not tifs:
        raise FileNotFoundError(f"no .tif files in {folder}")
    subprocess.run(
        [GDALBUILDVRT, "-q", str(out_path), *tifs],
        check=True,
    )


def _placeholder_palette(n: int) -> list[str]:
    """n visually-distinct hex colours (evenly spaced hue, fixed sat/light)."""
    colors = []
    for i in range(n):
        h = i / max(n, 1)
        r, g, b = colorsys.hls_to_rgb(h, 0.5, 0.55)
        colors.append(f"#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}")
    return colors


def fill_placeholders(det: Detected) -> None:
    """Give every observed code without a name/colour a generic placeholder."""
    missing = [c for c in det.class_codes if c not in det.class_names]
    if not missing:
        return
    palette = _placeholder_palette(len(det.class_codes))
    color_by_code = dict(zip(det.class_codes, palette))
    for code in missing:
        det.class_names[code] = f"Class {code}"
        det.class_colors.setdefault(code, color_by_code[code])


def register_one(folder_name: str, product_id: str, display_name: str, provider: str) -> None:
    folder = DATA_DIR / folder_name
    vrt_path = VRT_DIR / f"{product_id}.vrt"
    yaml_path = PRODUCTS_DIR / f"{product_id}.yaml"

    print(f"[{product_id}] building VRT from {folder} ...")
    build_vrt(folder, vrt_path)

    print(f"[{product_id}] detecting CRS/footprint/codes from {vrt_path.name} ...")
    det = detect_local_raster(vrt_path)
    fill_placeholders(det)

    yaml_text = build_registration_yaml(
        product_id,
        detected=det,
        role="reference",
        kind="label",
        access={"method": "local_raster", "path": f"cache/vrt/{product_id}.vrt"},
    )
    # Patch in display_name/provider (build_registration_yaml leaves them TODO).
    yaml_text = yaml_text.replace(
        'display_name: "TODO: complete or confirm by hand"',
        f"display_name: {display_name}",
        1,
    )
    yaml_text = yaml_text.replace(
        'provider: "TODO: complete or confirm by hand"',
        f"provider: {provider}",
        1,
    )
    yaml_path.write_text(yaml_text, encoding="utf-8")
    print(f"[{product_id}] wrote {yaml_path} ({len(det.class_codes)} classes)")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", help="register only this product id")
    args = parser.parse_args()

    for folder_name, reason in SKIPPED.items():
        print(f"[skip] {folder_name}: {reason}")
    print()

    for folder_name, (product_id, display_name, provider) in PRODUCTS.items():
        if args.only and args.only != product_id:
            continue
        register_one(folder_name, product_id, display_name, provider)
        print()


if __name__ == "__main__":
    main()
