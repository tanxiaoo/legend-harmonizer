"""legend-harmonizer: embedding + GMM legend harmonization pipeline.

See docs/PIPELINE.md for the staged build specification. Constants live in
harmonizer.config (mirroring section 2 of that document).
"""

import os as _os
from pathlib import Path as _Path


def _protect_source_data() -> None:
    """Forbid GDAL from writing anything alongside the user's downloaded rasters.

    ``data/`` holds files the **user** downloaded; this project never produces
    them and must never alter them. Derived data goes to ``cache/`` (see
    ``config.COG_DIR``), and every code path already respects that -- the only
    thing that does not is GDAL itself, which by default may drop a ``.aux.xml``
    PAM sidecar next to any raster it opens (to memoise statistics, histograms
    or a guessed CRS) and may create ``.ovr`` overviews on request.

    Those sidecars do not touch the ``.tif``'s bytes, but they are still writes
    into a directory that should be read-only from this app's point of view, and
    on a read-only or full volume they surface as confusing I/O errors during an
    ordinary read. Disabling PAM removes the last way this process could write
    into ``data/`` at all, which turns "the app does not modify your downloads"
    from a claim into a property.

    Set ``HARMONIZER_ALLOW_PAM=1`` to restore GDAL's default if some future
    workflow genuinely needs sidecars.
    """
    if _os.environ.get("HARMONIZER_ALLOW_PAM"):
        return
    # setdefault so an explicitly-set environment still wins.
    _os.environ.setdefault("GDAL_PAM_ENABLED", "NO")


_protect_source_data()


def _repair_proj_env() -> None:
    """Point PROJ at rasterio's own database when the inherited one is unusable.

    An unrelated GDAL install elsewhere on the machine (PostgreSQL/PostGIS, QGIS,
    an OSGeo4W shell) commonly exports a machine-wide ``PROJ_LIB``/``GDAL_DATA``.
    When that database is older than the PROJ inside rasterio's wheel, *every*
    CRS lookup fails with "contains DATABASE.LAYOUT.VERSION.MINOR = 2 whereas a
    number >= 5 is expected", which takes down ``import rio_tiler`` before the
    app starts -- it builds ``CRS.from_epsg(3857)`` at module scope.

    rasterio wheels ship a matching ``proj_data``, so the fix is to prefer it
    whenever the inherited value does not look valid. Only ever *replaces* a
    broken setting; a working PROJ environment is left untouched, and the
    variables are not cleared (empty means "look nowhere", not "use the
    default"). Runs at package import, before rasterio/rio-tiler load.
    """
    inherited = _os.environ.get("PROJ_LIB") or _os.environ.get("PROJ_DATA")
    if inherited and (_Path(inherited) / "proj.db").is_file():
        try:
            # Cheap version probe: the layout version lives in the metadata
            # table. If this database is too old, PROJ refuses it outright.
            import sqlite3

            with sqlite3.connect(
                f"file:{(_Path(inherited) / 'proj.db').as_posix()}?mode=ro", uri=True
            ) as conn:
                row = conn.execute(
                    "SELECT value FROM metadata WHERE key = 'DATABASE.LAYOUT.VERSION.MINOR'"
                ).fetchone()
            if row is not None and int(row[0]) >= 5:
                return  # inherited database is fine -- leave the environment alone
        except Exception:
            pass  # unreadable/unexpected: fall through and prefer the bundled one

    # Locate rasterio's bundled proj_data *without importing rasterio*: PROJ
    # reads these variables when its library initialises, so importing rasterio
    # here would lock in the broken path before the fix could apply.
    import importlib.util

    spec = importlib.util.find_spec("rasterio")
    if spec is None or not spec.origin:
        return
    bundled = _Path(spec.origin).parent / "proj_data"
    if (bundled / "proj.db").is_file():
        _os.environ["PROJ_LIB"] = str(bundled)
        _os.environ["PROJ_DATA"] = str(bundled)


_repair_proj_env()

from harmonizer.config import CONFIG, Config

__all__ = ["CONFIG", "Config"]
