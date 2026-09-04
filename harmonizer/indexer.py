"""Drop-in dataset indexer: data/ folder -> registry YAML -> layer in the web UI.

Turns "download a map, put it in ``data/``, see it on the web page" into one
command. For each entry in ``data/datasets.yaml`` this:

1. finds the dataset's GeoTIFF(s) under ``data/<folder>/``;
2. builds a GDAL VRT when the folder holds many tiles (a small XML index; no
   pixel data is copied or resampled), or uses the file directly when it holds
   one;
3. auto-detects CRS, footprint, resolution and the class codes actually present,
   via ``registry.register.detect_local_raster`` -- the same detection the
   hand-run Africa registration script uses;
4. attaches the legend from ``data/<folder>/legend.csv``; and
5. writes ``harmonizer/registry/products/<id>.yaml``.

**The legend comes from the CSV, the codes come from the raster, and the two are
reconciled.** A code in the raster with no legend row is reported rather than
given a placeholder colour: an invented colour is indistinguishable from a real
one once it is on the map, and silently mislabelling land cover is worse than
refusing to draw it. Legend rows with no matching raster code are kept (a class
legitimately absent from this AOI is normal) but counted in the report.

Existing registry YAML is **never overwritten** -- hand-verified entries such as
``hrlc30.yaml`` carry notes that regeneration would discard. Delete the YAML
first if you want it rebuilt.

Run:  python -m harmonizer.indexer            # index everything in the manifest
      python -m harmonizer.indexer --list     # show what would happen, write nothing
      python -m harmonizer.indexer --only GLC_FCS30D_2019
      python -m harmonizer.indexer --force    # overwrite existing registry YAML
"""

from __future__ import annotations

import argparse
import csv
import re
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from harmonizer.config import CONFIG, REPO_ROOT
from harmonizer.registry.register import Detected, detect_local_raster
from harmonizer.registry.schema import PRODUCTS_DIR

__all__ = ["index_all", "index_one", "load_manifest", "load_legend_csv", "Report"]

MANIFEST_PATH = CONFIG.data_dir / "datasets.yaml"
#: A dataset's legend lives in its own folder under this name. One file, one
#: place, nothing to match -- see ``registration._legend_for``.
LEGEND_FILENAME = "legend.csv"
VRT_DIR = REPO_ROOT / "cache" / "vrt"

#: CSV column names. The legend CSVs use these headers verbatim; anything else is
#: rejected loudly rather than guessed at, because a mis-read legend produces a
#: map that looks right and is wrong.
_COL_CODE = "Class Code"
_COL_COLOR = "Color code"
_COL_LABEL = "Label"
_COL_DESC = "Description"

#: Optional column declaring whether a row is a real land-cover class.
#:
#: Published legends list fill values alongside the classes -- Copernicus LCM-10
#: declares 254 "Unclassifiable" and 255 "No Data"; several products here declare
#: 0 "No data". Those are not land cover, and treating them as classes is not a
#: cosmetic error: the sampler would erode them, fetch embeddings for them, fit a
#: GMM to "pixels nobody could classify", and emit crosswalk rows matching that
#: non-class against real land cover in the other map.
#:
#: **The user declares this, per row.** An earlier version inferred it from the
#: label text ("no data", "unclassifiable", ...), which is guesswork: it depends
#: on the producer's wording, silently misses anything phrased differently
#: ("Sin datos", "Fill", "Cloud"), and could in principle drop a real class whose
#: name happens to contain a matching word. A column is unambiguous, visible in
#: the file the user already edits, and wrong only if the user says something
#: wrong.
_COL_IS_CLASS = "IsClass"

#: Accepted spellings for the boolean, so a legend written by hand in Excel or
#: exported from another tool is not rejected over capitalisation.
_TRUE_WORDS = {"true", "t", "yes", "y", "1"}
_FALSE_WORDS = {"false", "f", "no", "n", "0"}


def parse_is_class(raw: str | None, *, default: bool = True) -> bool:
    """Parse the ``IsClass`` cell.

    Defaults to **True** for a blank cell or a missing column: a legend that
    predates this column, or a row the user did not annotate, keeps every row as
    a class -- the same behaviour as before. Only an explicit falsey value
    excludes a row, so adding the column can never silently drop land cover.
    """
    if raw is None:
        return default
    text = str(raw).strip().lower()
    if text == "":
        return default
    if text in _FALSE_WORDS:
        return False
    if text in _TRUE_WORDS:
        return True
    # Anything unrecognised is treated as "yes, a class": refusing to draw a
    # class because its flag was misspelled would lose real data, whereas
    # including a fill value is caught by the observed-codes reconciliation.
    return default


@dataclass
class Report:
    """What happened to one dataset, for the printed summary."""

    folder: str
    product_id: str
    status: str  # "written" | "exists" | "skipped" | "failed"
    detail: str = ""
    n_files: int = 0
    n_classes: int = 0
    #: Codes in the raster with no row in the legend CSV -- these cannot be drawn.
    unmapped_codes: list[int] = field(default_factory=list)
    #: Legend rows with no pixels here. Normal for a regional subset; informational.
    absent_codes: list[int] = field(default_factory=list)


def _slugify(folder_name: str) -> str:
    """A registry product id derived from the folder name."""
    slug = re.sub(r"[^0-9a-zA-Z]+", "_", folder_name).strip("_").lower()
    return slug or "dataset"


def load_manifest(path: Path | None = None) -> dict[str, dict]:
    """Parse ``data/datasets.yaml``."""
    path = path or MANIFEST_PATH
    if not path.is_file():
        raise FileNotFoundError(
            f"no dataset manifest at {path} -- create it to declare which legend "
            f"each folder under {CONFIG.data_dir} uses"
        )
    doc = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(doc, dict):
        raise ValueError(f"{path}: expected a mapping of folder -> settings")
    return doc


#: One parsed legend row: ``(label, "#RRGGBB", description, is_class)``.
LegendRow = tuple[str, str, "str | None", bool]


def load_legend_csv(name: str | Path) -> dict[int, LegendRow]:
    """Parse a legend CSV into ``{code: (label, "#RRGGBB", description, is_class)}``.

    Takes a path to the file -- callers resolve it against the dataset's own
    folder (``data/<Dataset>/legend.csv``). Uses ``utf-8-sig`` because these
    files are exported with a BOM, which would otherwise corrupt the first header
    name and make the code column unfindable.

    ``is_class`` comes from the optional ``IsClass`` column and defaults to True
    (see :func:`parse_is_class`), so a legend without the column behaves exactly
    as it did before.
    """
    path = Path(name)
    if not path.is_file():
        raise FileNotFoundError(f"legend CSV not found: {path}")

    out: dict[int, LegendRow] = {}
    with path.open(encoding="utf-8-sig", newline="") as fh:
        reader = csv.DictReader(fh)
        missing = [
            c for c in (_COL_CODE, _COL_COLOR, _COL_LABEL) if c not in (reader.fieldnames or [])
        ]
        if missing:
            raise ValueError(
                f"{path.name}: missing column(s) {missing}; "
                f"found {reader.fieldnames}"
            )
        for row in reader:
            raw = (row.get(_COL_CODE) or "").strip()
            if not raw:
                continue
            try:
                code = int(raw)
            except ValueError:
                raise ValueError(f"{path.name}: non-integer class code {raw!r}") from None
            color = (row.get(_COL_COLOR) or "").strip()
            if not color.startswith("#"):
                color = "#" + color
            label = (row.get(_COL_LABEL) or "").strip() or f"Class {code}"
            desc = (row.get(_COL_DESC) or "").strip() or None
            is_class = parse_is_class(row.get(_COL_IS_CLASS))
            out[code] = (label, color, desc, is_class)
    if not out:
        raise ValueError(f"{path.name}: no legend rows parsed")
    return out


def _build_vrt(tifs: list[Path], out_path: Path) -> Path:
    """Index many tiles into one VRT (a small XML index; no pixels are copied).

    Prefers rasterio's own ``WarpedVRT``-free path via ``gdal.BuildVRT`` when the
    GDAL Python bindings are importable, then falls back to the ``gdalbuildvrt``
    CLI. Either is fine -- but neither is guaranteed to be present, and
    ``shutil.which`` can resolve a binary belonging to an unrelated environment,
    so a clear error beats a mysterious one.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    paths = [str(p) for p in tifs]

    try:
        from osgeo import gdal
    except ImportError:
        gdal = None

    if gdal is not None:
        gdal.UseExceptions()
        vrt = gdal.BuildVRT(str(out_path), paths)
        if vrt is None:
            raise RuntimeError(f"gdal.BuildVRT produced no output for {out_path.name}")
        vrt.FlushCache()
        del vrt  # closing the dataset is what writes the XML to disk
        return out_path

    exe = shutil.which("gdalbuildvrt")
    if exe is None:
        raise RuntimeError(
            "cannot build a VRT: neither the GDAL Python bindings (osgeo) nor the "
            "gdalbuildvrt CLI are available. Install GDAL (`pip install gdal`) to "
            "index a folder of many tiles as one layer."
        )

    # Pass the file list through a temporary text file rather than argv.
    #
    # A tile set is routinely hundreds of files: the JAXA SEA 2023 dataset is 619
    # rasters whose absolute paths total ~59_000 characters, well past Windows'
    # ~32_768 command-line limit. Spelling them out on argv fails there with a
    # bare ``FileNotFoundError: [WinError 206] The filename or extension is too
    # long``, which points at nothing useful. ``-input_file_list`` is GDAL's own
    # answer and has no such bound, so it is used unconditionally -- the small
    # datasets behave identically and the large ones stop being a special case.
    import tempfile

    with tempfile.NamedTemporaryFile(
        "w", suffix=".txt", delete=False, encoding="utf-8"
    ) as fh:
        fh.write("\n".join(paths))
        list_path = fh.name
    try:
        subprocess.run(
            [exe, "-q", "-input_file_list", list_path, str(out_path)], check=True
        )
    finally:
        try:
            os.unlink(list_path)
        except OSError:
            pass
    return out_path


def _source_for(folder: Path, product_id: str) -> tuple[Path, int]:
    """The single readable path for a dataset, plus how many files it came from.

    One ``.tif`` is used directly; many are indexed into a VRT. A VRT mosaic
    requires every tile to share one CRS, so a folder whose tiles are each in
    their own UTM zone raises here rather than producing a silently broken layer.
    """
    tifs = sorted(p for p in folder.glob("*.tif") if not p.name.endswith(".aux.xml"))
    if not tifs:
        raise FileNotFoundError(f"no .tif files in {folder}")
    if len(tifs) == 1:
        return tifs[0], 1

    import rasterio

    crs_seen = set()
    counts = set()
    for p in tifs:
        with rasterio.open(p) as ds:
            crs_seen.add(str(ds.crs))
            counts.add(ds.count)
    if len(crs_seen) > 1:
        raise ValueError(
            f"{folder.name}: tiles do not share one CRS ({len(crs_seen)} different, "
            f"e.g. {sorted(crs_seen)[:2]}) -- a VRT mosaic needs a single CRS. "
            f"Reproject the tiles to a common CRS first."
        )
    if len(counts) > 1:
        raise ValueError(
            f"{folder.name}: tiles have differing band counts {sorted(counts)}"
        )
    return _build_vrt(tifs, VRT_DIR / f"{product_id}.vrt"), len(tifs)


def _epsg_or_wkt(crs_text: str | None) -> str | None:
    """``EPSG:4326`` where the CRS has an EPSG code, else ``None``.

    ``detect_local_raster`` reports whatever ``str(ds.crs)`` gives, which for
    these WorldCover tiles is a full multi-line ``GEOGCS[...]`` WKT blob. Written
    unquoted into YAML that is fragile (brackets and commas are YAML syntax) and
    unreadable. An EPSG code is the useful form; if the CRS has none, emit
    ``null`` rather than a WKT wall, since the authoritative CRS is read from the
    source at run time anyway.
    """
    if not crs_text:
        return None
    try:
        from rasterio.crs import CRS

        crs = CRS.from_string(crs_text)
    except Exception:
        return None

    epsg = crs.to_epsg()
    if epsg:
        return f"EPSG:{epsg}"

    # Some WKT carries no authority code that GDAL will match -- these
    # WorldCover tiles declare plain WGS 84 with no AUTHORITY on the GEOGCS, so
    # ``to_epsg()`` gives None even though the CRS is exactly EPSG:4326.
    #
    # Deliberately does not compare against ``CRS.from_epsg(4326)``: that call
    # needs a working PROJ database, and a stale ``proj.db`` from an unrelated
    # install earlier on PATH makes it raise (seen here with a PostgreSQL/PostGIS
    # copy). Matching on the WKT's own datum/ellipsoid text needs no database, so
    # detection does not depend on the environment's PROJ being sane.
    text = crs_text.replace(" ", "").upper()
    is_wgs84 = ('"WGS84"' in text or "WGS_1984" in text) and "6378137" in text
    if is_wgs84 and "GEOGCS" in text and "PROJCS" not in text:
        return "EPSG:4326"
    return None


def _resolution_m(detected: Detected, source: Path) -> float | None:
    """Ground resolution in **metres**, converting from degrees when needed.

    ``detect_local_raster`` returns the raw pixel size in the raster's own units.
    For a geographic CRS that is degrees (8.3e-05 for WorldCover's 10 m), and
    this field is consumed as metres -- ``explorer`` uses it directly as a
    sampling scale, where a degrees value is silently treated as sub-millimetre.
    Converts at ~111320 m per degree of latitude.
    """
    if detected.resolution_m is None:
        return None
    try:
        import rasterio

        with rasterio.open(source) as ds:
            geographic = bool(ds.crs and ds.crs.is_geographic)
    except Exception:
        geographic = False
    value = float(detected.resolution_m) * (111_320.0 if geographic else 1.0)
    return round(value, 3)


def _relative(path: Path) -> str:
    """Repo-relative POSIX path where possible, so the YAML stays portable."""
    try:
        return path.resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def _render_yaml(
    product_id: str,
    *,
    folder: str,
    settings: dict,
    detected: Detected,
    legend: dict[int, tuple[str, str, str | None]],
    source: Path,
    n_files: int,
    drawable: list[int],
) -> str:
    """The registry YAML for one indexed dataset."""
    name = settings.get("name") or folder
    provider = settings.get("provider") or "unknown"
    role = settings.get("role") or "reference"
    band = settings.get("band")
    year = settings.get("year")
    # Falls back to the standard name, which is what index_one actually loaded
    # when the manifest names no legend (the usual case).
    legend_file = settings.get("legend") or LEGEND_FILENAME

    def q(value: str) -> str:
        return '"' + str(value).replace('"', "'") + '"'

    lines: list[str] = [
        "# Generated by `python -m harmonizer.indexer` from data/datasets.yaml.",
        f"# Source: data/{folder}/ ({n_files} file(s));"
        f" legend: data/{folder}/{legend_file}",
        "#",
        "# Regenerating will not overwrite this file -- delete it first if you want",
        "# it rebuilt. Hand edits below are therefore safe.",
        "",
        f"id: {product_id}",
        f"display_name: {name}",
        f"provider: {provider}",
        f"role: {role}",
        "kind: label",
        "",
        "access:",
        "  method: local_raster",
        f"  path: {_relative(source)}",
        "",
    ]
    if band is not None:
        lines.append(
            f"band: {int(band)}                # of {detected_band_count(source)} "
            "bands in the source"
        )
    else:
        lines.append("band: 1")
    res_m = _resolution_m(detected, source)
    lines.append(
        f"resolution_m: {res_m}" if res_m is not None else "resolution_m: null"
    )
    lines.append(f"available_years: [{year}]" if year is not None else "available_years: []")
    crs = _epsg_or_wkt(detected.crs)
    lines.append(f"crs: {crs}" if crs else "crs: null")
    if detected.footprint is not None:
        lines.append(f"footprint: {[round(v, 6) for v in detected.footprint]}")
    else:
        lines.append("footprint: null")
    lines.append("licence: null")
    lines.append("citation: null")
    lines.append("")
    lines.append(
        "# Every class named in the legend CSV, with `observed:` recording whether"
    )
    lines.append(
        "# it was actually found in this dataset's pixels. Classes declared by the"
    )
    lines.append(
        "# published legend but absent here are KEPT and marked rather than dropped:"
    )
    lines.append(
        "# a regional subset legitimately lacks some classes of a global legend, and"
    )
    lines.append(
        "# the map legend greys those out instead of pretending they never existed"
    )
    lines.append("# (DESIGN.md 4.3).")
    lines.append("legend:")
    drawable_set = set(drawable)
    for code in sorted(legend):
        label, color, desc, is_class = legend[code]
        # Rows the legend marks `IsClass = FALSE` are fill/no-data values, not
        # land cover. They are excluded from the `observed` accounting above for
        # that reason, so emitting them here would put a "No data" chip in the
        # map legend and, worse, offer them to the sampler as a class to model.
        # Dropped here too, so the legend and the accounting agree.
        if code == 0 or not is_class:
            continue
        lines.append(f"  - code: {code}")
        lines.append(f"    name: {q(label)}")
        lines.append(f"    color: {q(color)}")
        lines.append(f"    observed: {'true' if code in drawable_set else 'false'}")
        if desc:
            lines.append(f"    description: {q(desc)}")
    return "\n".join(lines) + "\n"


def detected_band_count(source: Path) -> int:
    import rasterio

    with rasterio.open(source) as ds:
        return ds.count


#: First-line marker every machine-written registry YAML carries (_render_yaml).
_GENERATED_MARKER = "# Generated by `python -m harmonizer.indexer`"


def _is_generated(yaml_path: Path) -> bool:
    """True when this registry YAML was written by the indexer, not by a human."""
    try:
        with yaml_path.open(encoding="utf-8") as fh:
            return fh.readline().startswith(_GENERATED_MARKER)
    except OSError:
        return False


def index_one(
    folder_name: str, settings: dict, *, force: bool = False, dry_run: bool = False
) -> Report:
    """Index a single dataset folder into a registry YAML."""
    product_id = settings.get("id") or _slugify(folder_name)
    folder = CONFIG.data_dir / folder_name
    yaml_path = PRODUCTS_DIR / f"{product_id}.yaml"

    if not folder.is_dir():
        return Report(folder_name, product_id, "failed", f"no such folder: {folder}")

    if yaml_path.exists() and not force:
        return Report(
            folder_name,
            product_id,
            "exists",
            f"{yaml_path.name} already registered (use --force to overwrite)",
        )

    # A hand-written entry is never regenerated, even with --force.
    #
    # ``force`` exists to refresh *generated* YAML (a new legend, a changed
    # folder). Some entries are not generated: hrlc30.yaml carries a legend
    # verified against the raster in QGIS, plus notes recording which observed
    # codes are resampling artifacts rather than classes. Regenerating it
    # silently replaced those verified colours with the ones from whatever CSV
    # was to hand -- the map still drew, in the wrong colours, which is exactly
    # the "looks right and is wrong" failure this module refuses elsewhere.
    #
    # Generated files announce themselves on line 1 (see _render_yaml); anything
    # else is treated as human-authored and left alone.
    if yaml_path.exists() and force and not _is_generated(yaml_path):
        return Report(
            folder_name,
            product_id,
            "skipped",
            f"{yaml_path.name} is hand-written (no 'Generated by' header) -- "
            f"refusing to overwrite it, even with --force. Delete the file first "
            f"if you really want it rebuilt from {folder / LEGEND_FILENAME}.",
        )

    # The legend lives beside the rasters: data/<Dataset>/legend.csv. A manifest
    # `legend:` overrides that, resolved relative to the same folder.
    # The legend is resolved against the DATASET'S OWN FOLDER, always.
    #
    # This used to be `Path(name)` with a fallback to `folder / name` only when
    # the bare path did not exist -- and `Path("legend.csv").exists()` is
    # relative to the *current working directory*. A stray legend.csv in the repo
    # root therefore won every lookup: two HRLC datasets were indexed against a
    # completely different product's legend, producing YAML that looked
    # plausible and named every class wrongly. Nothing in the output revealed it;
    # the header even recorded the correct path it had not read.
    #
    # An absolute path in `datasets.yaml` is still honoured, since that is an
    # explicit instruction rather than an accident of the working directory.
    legend_name = settings.get("legend") or LEGEND_FILENAME
    legend_path = Path(legend_name)
    if not legend_path.is_absolute():
        legend_path = folder / legend_name
    if not legend_path.is_file():
        return Report(
            folder_name,
            product_id,
            "failed",
            f"no legend found -- save this dataset's legend as "
            f"{folder / LEGEND_FILENAME} "
            f"(columns: {_COL_CODE}, {_COL_COLOR}, {_COL_LABEL}, {_COL_DESC})",
        )

    try:
        legend = load_legend_csv(legend_path)
        source, n_files = _source_for(folder, product_id)
        band = int(settings["band"]) if settings.get("band") is not None else 1
        detected = detect_local_raster(source, band=band)
    except Exception as exc:
        detail = f"{type(exc).__name__}: {exc}"
        if "IReadBlock" in str(exc) or "TIFFReadEncoded" in str(exc):
            detail += (
                "\n    This is a corrupt or incompletely downloaded source file, "
                "not a configuration problem -- re-download it and retry."
            )
        return Report(folder_name, product_id, "failed", detail)

    # Rows the user marked `IsClass = FALSE` are fill/no-data values, not
    # land-cover classes, and must not reach the map legend, the
    # observed/absent accounting, or the sampler (see _COL_IS_CLASS). Dropped
    # once, here, so every downstream set below is already clean.
    fill_codes = {c for c, row in legend.items() if not row[3]}  # IsClass = FALSE
    fill_codes.add(0)  # 0 is the fill value in these products even when unnamed
    classes = {c for c in legend if c not in fill_codes}

    observed = set(detected.class_codes) - fill_codes
    unmapped = sorted(observed - set(legend))
    drawable = sorted(observed & classes)
    absent = sorted(classes - observed)

    if not drawable:
        return Report(
            folder_name,
            product_id,
            "failed",
            "no raster code matches any legend row -- wrong legend CSV, or wrong band?",
            n_files=n_files,
            unmapped_codes=unmapped,
        )

    if not dry_run:
        text = _render_yaml(
            product_id,
            folder=folder_name,
            settings=settings,
            detected=detected,
            legend=legend,
            source=source,
            n_files=n_files,
            drawable=drawable,
        )
        yaml_path.parent.mkdir(parents=True, exist_ok=True)
        yaml_path.write_text(text, encoding="utf-8")

    return Report(
        folder_name,
        product_id,
        "would write" if dry_run else "written",
        f"{yaml_path.name}",
        n_files=n_files,
        n_classes=len(drawable),
        unmapped_codes=unmapped,
        absent_codes=absent,
    )


def index_all(
    *, only: str | None = None, force: bool = False, dry_run: bool = False
) -> list[Report]:
    manifest = load_manifest()
    reports = []
    for folder_name, settings in manifest.items():
        if only and only not in (folder_name, (settings or {}).get("id"), _slugify(folder_name)):
            continue
        reports.append(
            index_one(folder_name, settings or {}, force=force, dry_run=dry_run)
        )
    return reports


def check_readable(folder_name: str, settings: dict, step: int = 4096) -> list[tuple[str, int, int]]:
    """Probe every source file for unreadable blocks.

    A partially downloaded GeoTIFF opens fine and reports correct metadata --
    the damage only surfaces when a tile read hits a truncated strip, which in
    the web app looks like a layer that renders in some places and errors in
    others. This reads a coarse grid of windows from each file and counts the
    failures, so a bad download is identified as such instead of being mistaken
    for a bug.

    Returns ``[(filename, n_failed_windows, n_total_windows)]``.
    """
    import rasterio
    from rasterio.windows import Window

    folder = CONFIG.data_dir / folder_name
    band = int(settings["band"]) if settings.get("band") is not None else 1
    results = []
    for path in sorted(p for p in folder.glob("*.tif") if not p.name.endswith(".aux.xml")):
        try:
            with rasterio.open(path) as ds:
                use = band if band <= ds.count else 1
                bad = total = 0
                for top in range(0, ds.height, step):
                    for left in range(0, ds.width, step):
                        total += 1
                        window = Window(
                            left, top,
                            min(step, ds.width - left),
                            min(step, ds.height - top),
                        )
                        try:
                            ds.read(use, window=window)
                        except Exception:
                            bad += 1
                results.append((path.name, bad, total))
        except Exception:
            results.append((path.name, -1, 0))  # could not even open
    return results


def _print_reports(reports: list[Report]) -> None:
    for r in reports:
        head = f"[{r.status}] {r.folder} -> {r.product_id}"
        print(head if not r.detail else f"{head}: {r.detail}")
        if r.n_files:
            print(f"    {r.n_files} file(s), {r.n_classes} drawable class(es)")
        if r.unmapped_codes:
            print(
                f"    !! {len(r.unmapped_codes)} code(s) in the raster are missing "
                f"from the legend CSV and will NOT be drawn: {r.unmapped_codes}"
            )
        if r.absent_codes:
            print(
                f"    (i) {len(r.absent_codes)} legend class(es) not present in this "
                f"raster -- normal for a regional subset"
            )
    if not reports:
        print("nothing matched")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--only", help="index just this folder / product id")
    ap.add_argument("--force", action="store_true", help="overwrite existing registry YAML")
    ap.add_argument(
        "--list",
        action="store_true",
        dest="dry_run",
        help="report what would happen, write nothing",
    )
    ap.add_argument(
        "--check",
        action="store_true",
        help="probe every source file for unreadable blocks (bad downloads)",
    )
    args = ap.parse_args()

    if args.check:
        manifest = load_manifest()
        bad_total = 0
        for folder_name, settings in manifest.items():
            if args.only and args.only not in (folder_name, _slugify(folder_name)):
                continue
            print(f"{folder_name}:")
            for name, bad, total in check_readable(folder_name, settings or {}):
                if bad < 0:
                    print(f"    {name}: CANNOT OPEN")
                    bad_total += 1
                elif bad:
                    pct = 100.0 * bad / total if total else 0.0
                    print(f"    {name}: {bad}/{total} windows unreadable ({pct:.0f}%)")
                    bad_total += 1
                else:
                    print(f"    {name}: ok")
        if bad_total:
            print(f"\n{bad_total} damaged file(s) -- re-download these before indexing.")
            return 1
        print("\nall files readable")
        return 0

    try:
        reports = index_all(only=args.only, force=args.force, dry_run=args.dry_run)
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    _print_reports(reports)
    if any(r.status == "failed" for r in reports):
        return 1
    if not args.dry_run and any(r.status == "written" for r in reports):
        print(
            "\nDone. Restart the server (python run.py) to pick up new layers.\n"
            "Large tiled datasets should also be converted for speed:\n"
            "  python tools/to_cog.py --only <product_id>"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
