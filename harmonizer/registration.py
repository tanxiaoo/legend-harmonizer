"""Drop-in dataset registration (DESIGN.md section 4.1).

Turns *"download a land-cover map and its legend into ``data/``, run the app, and
the map is choosable in the window"* into the actual behaviour: no CLI step.

``harmonizer.indexer`` stays the engine -- this module is the automatic flow
around it. It scans ``data/``, decides what state each candidate dataset is in,
and runs registration as a background job through the same progress reporting
the run jobs use:

    index (CRS/bands, VRT for tile sets, legend reconciliation, registry YAML)
      -> COG conversion + mosaic (tools/to_cog.py)
      -> invalidate the cached COG/footprint lookups
      -> the product appears in the picker

**Ready state is the product of this module.** A dataset is never silently
absent: a folder with rasters but no legend CSV appears as ``needs-legend``
naming the file it expects, a half-converted product appears as
``needs-conversion``, and a failure appears as ``error`` carrying its message.
The picker renders those states directly (section 4.2).

**Never overwrites a human-edited registry YAML.** The indexer's
no-overwrite-without-``--force`` rule is kept: auto-registration only *creates*
missing entries and *completes* missing COG trees.
"""

from __future__ import annotations

import logging
import re
import threading
import uuid
from dataclasses import dataclass, field
from pathlib import Path

from harmonizer.config import CONFIG
# Shared with the indexer so the scanner and the indexer cannot disagree
# about where a dataset's legend lives.
from harmonizer.indexer import LEGEND_FILENAME

__all__ = [
    "DatasetState",
    "scan_datasets",
    "product_states",
    "register_dataset",
    "REGISTRATIONS",
]

_LOG = logging.getLogger(__name__)

#: Ready states a dataset/product can be in, in the order the UI should treat as
#: "increasingly usable". Only ``ready`` is selectable for viewing and runs.
READY = "ready"
INDEXING = "indexing"
CONVERTING = "converting"
NEEDS_LEGEND = "needs-legend"
NEEDS_CONVERSION = "needs-conversion"
ERROR = "error"
#: Registered, but its data/ folder has been deleted -- derived files are stranded.
MISSING = "missing"


def drop_in_rules() -> dict:
    """The naming convention, in a form the UI can show the user.

    Served by ``/api/datasets`` so the rules live in **one** place and the help
    text cannot drift from the code that enforces them. A user who hits
    ``needs-legend`` should be able to learn the convention from the app, not
    from a comment in a YAML file they have never opened.
    """
    return {
        "data_dir": str(CONFIG.data_dir),
        "manifest": str(CONFIG.data_dir / "datasets.yaml"),
        "layout": (
            "data/\n"
            "  WorldCover_2020/\n"
            "    *.tif\n"
            "    legend.csv\n"
            "  JAXA_HRLULC_SEA_2023/\n"
            "    *.tif\n"
            "    legend.csv"
        ),
        "steps": [
            "Make a folder for the dataset: data/<AnyName>/",
            "Put its rasters in it: *.tif",
            f"Put its legend beside them, named exactly {LEGEND_FILENAME}",
        ],
        "legend_columns": [
            "Class Code",
            "Color code",
            "Label",
            "IsClass",
            "Description",
        ],
        "note": (
            f"The folder name is yours to choose -- it becomes the map's name in "
            f"the picker. Only the legend has a fixed name ({LEGEND_FILENAME}), "
            f"so nothing has to be matched or guessed: the legend is simply the "
            f"file sitting next to the rasters it describes."
        ),
        "fill_rows": (
            "Mark a row IsClass = FALSE when it is a fill/no-data value rather "
            "than land cover (e.g. 'No Data', 'Unclassifiable', 'Cloud'). Those "
            "rows are dropped from the map legend and never sampled, so they "
            "cannot end up in the matching table. Leave the column blank, or "
            "omit it entirely, and every row counts as a class."
        ),
    }


@dataclass
class DatasetState:
    """What the app knows about one candidate dataset under ``data/``."""

    #: Folder name under ``data/`` (the drop-in unit).
    folder: str
    #: Registry product id it maps to (from the manifest, else slugified folder).
    product_id: str
    state: str
    detail: str = ""
    #: Legend CSV this dataset uses, or the path it is waiting for.
    legend: str | None = None
    n_files: int = 0
    registered: bool = False
    #: Set while a registration job is running, so the UI can show progress.
    progress: float = 0.0

    def as_dict(self) -> dict:
        return {
            "folder": self.folder,
            "product_id": self.product_id,
            "state": self.state,
            "detail": self.detail,
            "legend": self.legend,
            "n_files": self.n_files,
            "registered": self.registered,
            "progress": self.progress,
        }


# --------------------------------------------------------------------------- #
# Registration jobs
# --------------------------------------------------------------------------- #


@dataclass
class RegistrationJob:
    id: str
    product_id: str
    folder: str
    state: str = "queued"  # queued | running | done | failed
    stage: str = "queued"
    progress: float = 0.0
    error: str | None = None

    def as_dict(self) -> dict:
        return {
            "id": self.id,
            "product_id": self.product_id,
            "folder": self.folder,
            "state": self.state,
            "stage": self.stage,
            "progress": self.progress,
            "error": self.error,
        }


class RegistrationStore:
    """Thread-safe store of registration jobs, keyed by product id.

    Keyed by product rather than by job id because the question the UI asks is
    always "what is happening to *this* product", and only one registration per
    product may run at a time.
    """

    def __init__(self) -> None:
        self._jobs: dict[str, RegistrationJob] = {}
        self._lock = threading.Lock()

    def start(self, product_id: str, folder: str) -> RegistrationJob | None:
        """Create a job unless one is already in flight for this product."""
        with self._lock:
            existing = self._jobs.get(product_id)
            if existing is not None and existing.state in ("queued", "running"):
                return None
            job = RegistrationJob(
                id=uuid.uuid4().hex, product_id=product_id, folder=folder
            )
            self._jobs[product_id] = job
            return job

    def update(self, product_id: str, **fields) -> None:
        with self._lock:
            job = self._jobs.get(product_id)
            if job is None:
                return
            for k, v in fields.items():
                setattr(job, k, v)

    def get(self, product_id: str) -> RegistrationJob | None:
        with self._lock:
            return self._jobs.get(product_id)

    def all(self) -> dict[str, RegistrationJob]:
        with self._lock:
            return dict(self._jobs)

    def active(self) -> bool:
        with self._lock:
            return any(
                j.state in ("queued", "running") for j in self._jobs.values()
            )

    def forget(self, product_id: str) -> None:
        """Drop a product's recorded job, after its files have been removed."""
        with self._lock:
            self._jobs.pop(product_id, None)


REGISTRATIONS = RegistrationStore()


# --------------------------------------------------------------------------- #
# Detection
# --------------------------------------------------------------------------- #


def _slugify(folder_name: str) -> str:
    """Same rule the indexer uses, so ids agree between the two paths."""
    from harmonizer.indexer import _slugify as _indexer_slugify

    return _indexer_slugify(folder_name)


def _manifest() -> dict[str, dict]:
    """``data/datasets.yaml`` if present, else an empty manifest.

    The manifest is an *override*, not a requirement (DESIGN.md 4.1): it supplies
    display names, band selection, year and irregular legend pairings. A folder
    absent from it is still a candidate, discovered by convention.
    """
    try:
        from harmonizer.indexer import load_manifest

        return load_manifest() or {}
    except Exception:
        return {}


def _legend_for(folder: Path, settings: dict) -> tuple[str | None, Path | None]:
    """The legend CSV a dataset should use: ``(name, resolved_path)``.

    **The whole rule:** ``data/<Dataset>/legend.csv``, beside that dataset's
    rasters. Nothing is matched, ranked or guessed -- either the file is there or
    it is not. A user who can see the folder can see whether it is set up
    correctly, and a wrong legend cannot be attached to a map even in principle.

    An explicit ``legend:`` in ``data/datasets.yaml`` still overrides it, for a
    genuinely irregular layout (e.g. one legend shared by two datasets); it is
    resolved relative to the dataset's own folder.
    """
    named = settings.get("legend")
    if named:
        path = Path(named)
        # Relative names resolve against the dataset folder, never the current
        # working directory -- a stray legend.csv in the repo root must not be
        # able to win this lookup (see indexer.index_one for what that cost).
        if not path.is_absolute():
            path = folder / named
        return str(named), (path if path.is_file() else None)

    inside = folder / LEGEND_FILENAME
    return LEGEND_FILENAME, (inside if inside.is_file() else None)


def _registry_yaml(product_id: str) -> Path:
    from harmonizer.registry.schema import PRODUCTS_DIR

    return PRODUCTS_DIR / f"{product_id}.yaml"


def scan_datasets() -> list[DatasetState]:
    """Every candidate dataset under ``data/`` and the state it is in.

    A candidate is a folder holding at least one ``.tif``; its legend sits
    inside it as ``legend.csv``.
    """
    states: list[DatasetState] = []
    manifest = _manifest()
    data_dir = CONFIG.data_dir
    if not data_dir.is_dir():
        return states

    # Manifest entries are keyed by folder name; build a reverse lookup so an
    # explicit `id:` (e.g. WorldCover_2020 -> worldcover_2020) is honoured.
    for folder in sorted(p for p in data_dir.iterdir() if p.is_dir()):
        if folder.name == "legend":
            continue
        tifs = [p for p in folder.glob("*.tif") if not p.name.endswith(".aux.xml")]
        if not tifs:
            continue

        settings = manifest.get(folder.name) or {}
        product_id = settings.get("id") or _slugify(folder.name)
        legend_name, legend_path = _legend_for(folder, settings)

        # One-time migration for products registered before artifact manifests
        # existed: without a manifest they would report no artifacts and could
        # never be cleaned up. No-ops once a manifest exists, and records nothing
        # for a product that has no derived files.
        _backfill_manifest(product_id, folder.name)

        state = _state_for(product_id, legend_path, folder.name)
        states.append(
            DatasetState(
                folder=folder.name,
                product_id=product_id,
                state=state[0],
                detail=state[1],
                legend=legend_name,
                n_files=len(tifs),
                registered=_registry_yaml(product_id).exists(),
                progress=state[2],
            )
        )

    states.extend(_missing_datasets({s.product_id for s in states}))
    return states


def _backfill_manifest(product_id: str, folder: str) -> None:
    """Record a manifest for a product that predates them. Best-effort.

    Imported inside the function because ``scan_datasets`` binds a local named
    ``manifest`` (the parsed ``datasets.yaml``), which would shadow the module.
    """
    from harmonizer import manifest as manifest_mod

    try:
        manifest_mod.backfill(product_id, folder)
    except Exception:  # pragma: no cover - never let bookkeeping break a scan
        _LOG.debug("could not backfill manifest for %s", product_id, exc_info=True)


def _missing_datasets(seen: set[str]) -> list[DatasetState]:
    """Registered local products whose ``data/`` folder is gone.

    Deleting a folder removes the rasters but nothing else: the registry YAML,
    the VRT, the converted COG tree (often several GB) and the tile cache all
    stay. Worse, for a **tile set** the registry's ``access.path`` points at the
    VRT under ``cache/``, not into ``data/`` -- so the usual "does the file
    exist?" readiness check still passes and the product keeps appearing in the
    picker as if it were fine, failing only when a tile is actually read.

    Reporting it as ``missing`` makes that visible and gives the UI something to
    hang a cleanup action on. Nothing is deleted here: removing gigabytes as a
    side effect of a rescan is not this function's decision to make (see
    :func:`remove_product`).
    """
    from harmonizer.registry.schema import load_all_products

    try:
        products = load_all_products()
    except Exception:
        return []

    out: list[DatasetState] = []
    for pid, spec in sorted(products.items()):
        if getattr(spec.access, "method", None) != "local_raster":
            continue
        if pid in seen:
            continue
        # A manifest is the evidence that *this app* registered the product from
        # a local folder. A hand-written registry entry for data that has not
        # arrived yet has none, so it is never mistaken for a deleted dataset --
        # which is what the old "does it look derived?" heuristic was guarding
        # against, now answered by a record instead of an inference.
        leftovers = product_artifacts(pid)
        if not leftovers:
            continue
        out.append(
            DatasetState(
                folder=_folder_of(pid),
                product_id=pid,
                state=MISSING,
                detail=(
                    f"dataset folder is gone, but {_describe_size(leftovers)} of "
                    f"derived files remain (COGs, registry entry, VRT, caches). "
                    f"Remove them, or restore the folder under data/ and refresh."
                ),
                legend=None,
                n_files=0,
                registered=True,
                progress=0.0,
            )
        )
    return out


def _folder_of(product_id: str) -> str:
    """The data/ folder a product came from, for display.

    Recovered from the manifest where possible so the name matches what the user
    actually deleted; otherwise the product id is the best label available.
    """
    for folder, settings in (_manifest() or {}).items():
        if ((settings or {}).get("id") or _slugify(folder)) == product_id:
            return folder
    return product_id


def _describe_size(paths: list[Path]) -> str:
    """Human-readable total size of a set of files/directories."""
    total = 0
    for p in paths:
        try:
            if p.is_dir():
                total += sum(f.stat().st_size for f in p.rglob("*") if f.is_file())
            elif p.is_file():
                total += p.stat().st_size
        except OSError:
            pass
    if total >= 1e9:
        return f"{total / 1e9:.1f} GB"
    if total >= 1e6:
        return f"{total / 1e6:.0f} MB"
    return f"{max(1, round(total / 1e3))} KB"


def product_artifacts(product_id: str) -> list[Path]:
    """Every derived file a product owns, from its recorded manifest.

    Ownership is **recorded at creation** (``harmonizer.manifest``), not inferred
    from names: registration declares each artifact as it writes it, and this
    reads that list back. A product with no manifest returns nothing -- there is
    no evidence this app created anything for it, and deleting on a name match
    alone is the guesswork the manifest replaced.

    Never includes anything under ``data/``: those are the user's own downloads,
    and ``manifest.record`` refuses such a path outright, so a bad entry cannot
    reach a delete. Enforced from the other side by
    ``scripts/verify_data_readonly.py``.
    """
    from harmonizer import manifest

    return manifest.existing_artifacts(product_id)


def _state_for(
    product_id: str, legend_path: Path | None, folder: str
) -> tuple[str, str, float]:
    """``(state, detail, progress)`` for one candidate."""
    job = REGISTRATIONS.get(product_id)
    if job is not None and job.state in ("queued", "running"):
        # A live job's own stage is the most specific thing we can report.
        stage = INDEXING if job.stage.startswith("index") else CONVERTING
        return stage, job.stage, job.progress
    if job is not None and job.state == "failed":
        return ERROR, job.error or "registration failed", 0.0

    if legend_path is None:
        # One instruction, naming the exact path. Nothing to choose between.
        return (
            NEEDS_LEGEND,
            f"no legend found. Save this dataset's legend as "
            f"data/{folder}/{LEGEND_FILENAME} "
            f"(columns: Class Code, Color code, Label, Description), "
            f"then press ↻ datasets.",
            0.0,
        )

    if not _registry_yaml(product_id).exists():
        return NEEDS_CONVERSION, "not indexed yet", 0.0

    # Registered: the remaining question is whether its COG tree is complete.
    from harmonizer import local_tiles

    try:
        source_state = local_tiles.source_state(product_id)
    except Exception:
        source_state = NEEDS_CONVERSION
    if source_state in ("mosaic", "single"):
        return READY, "", 1.0
    return (
        NEEDS_CONVERSION,
        "indexed, but no converted COGs -- tiles and sampling use the slow raw "
        "source and, if its overviews are averaged, may show class codes that "
        "are not in the legend",
        0.0,
    )


def product_states() -> dict[str, DatasetState]:
    """Candidate states keyed by product id, for the products endpoint."""
    return {s.product_id: s for s in scan_datasets()}


# --------------------------------------------------------------------------- #
# Registration
# --------------------------------------------------------------------------- #


def register_dataset(
    folder: str, *, convert: bool = True, force: bool = False
) -> RegistrationJob | None:
    """Index a dataset and convert it to COGs, in a background thread.

    Returns the job, or ``None`` when one is already running for this product.
    """
    manifest = _manifest()
    settings = manifest.get(folder) or {}
    product_id = settings.get("id") or _slugify(folder)

    job = REGISTRATIONS.start(product_id, folder)
    if job is None:
        return None

    thread = threading.Thread(
        target=_run_registration,
        args=(job, folder, settings, convert, force),
        daemon=True,
        name=f"register-{product_id}",
    )
    thread.start()
    return job


def _run_registration(
    job: RegistrationJob,
    folder: str,
    settings: dict,
    convert: bool,
    force: bool,
) -> None:
    """Worker: index, then convert, reporting progress as it goes."""
    pid = job.product_id

    def report(progress: float, stage: str) -> None:
        REGISTRATIONS.update(pid, state="running", progress=progress, stage=stage)

    try:
        report(0.02, "indexing: reading rasters and legend")

        # -- index ---------------------------------------------------------- #
        # Only when the YAML is missing: a human may have edited it, and the
        # indexer's own rule is never to overwrite without --force.
        if force or not _registry_yaml(pid).exists():
            from harmonizer.indexer import index_one

            report(0.05, "indexing: building VRT / detecting CRS and classes")
            # No legend pre-resolution needed: index_one defaults to the
            # dataset's own legend.csv, the same rule _legend_for applies.
            report_ = index_one(folder, settings, force=force)
            if report_.status == "failed":
                raise RuntimeError(_explain_index_failure(folder, report_.detail))
            _LOG.info("registered %s: %s", pid, report_.detail)
            _record_index_artifacts(pid, folder)
            # The YAML was just (re)written, so any tile rendered from the old
            # legend now shows the wrong colours.
            _drop_rendered_tiles(pid)
        else:
            report(0.20, "indexing: already registered, keeping the existing YAML")
            # Still record: an earlier run may predate the manifest, and cleanup
            # can only remove what it knows about.
            _record_index_artifacts(pid, folder)

        # The registry caches parsed specs; drop them so the new product is
        # visible to everything downstream without a restart.
        _invalidate_registry()
        report(0.25, "indexed")

        # -- convert -------------------------------------------------------- #
        if convert:
            report(0.30, "converting: writing Cloud-Optimized GeoTIFFs")
            _convert(pid, report)

        _invalidate_caches(pid)
        REGISTRATIONS.update(
            pid, state="done", progress=1.0, stage="ready", error=None
        )
    except Exception as exc:
        _LOG.exception("registration failed for %s", pid)
        REGISTRATIONS.update(
            pid,
            state="failed",
            stage="failed",
            error=f"{type(exc).__name__}: {exc}",
        )


#: GDAL/rasterio wording that means "this file's bytes are bad", not
#: "this dataset is configured wrong". Kept literal because these strings are
#: what actually surfaces from a truncated GeoTIFF.
_CORRUPTION_MARKERS = (
    "IReadBlock",
    "TIFFReadEncoded",
    "TIFFFillTile",
    "Read failed",
    "premature end",
)


def _explain_index_failure(folder: str, detail: str) -> str:
    """Turn a raw rasterio error into something the user can act on.

    A partially downloaded GeoTIFF opens fine and reports correct metadata; the
    damage only appears when a read hits a truncated tile, and the message that
    escapes is an opaque ``RasterioIOError: Read failed``. That names neither the
    file nor the remedy. When the failure looks like corruption, identify the
    damaged files here -- ``check_readable`` is exactly the probe for it -- and
    list them, so the badge says *which* files to re-download.
    """
    if not any(m in detail for m in _CORRUPTION_MARKERS):
        return detail

    # Header-only truncation check first: it reads the TIFF's tile offset tables
    # rather than any pixels, so it identifies incomplete downloads across
    # hundreds of files in well under a second. ``indexer.check_readable``
    # answers a subtly different question (which *regions* fail to decode) and
    # costs minutes, so it is not what belongs on an error badge.
    try:
        import sys

        sys.path.insert(0, str(CONFIG.data_dir.parent / "scripts"))
        from check_downloads import declared_extent

        folder_path = CONFIG.data_dir / folder
        truncated = []
        total = 0
        for path in sorted(folder_path.glob("*.tif")):
            if path.name.endswith(".aux.xml"):
                continue
            total += 1
            try:
                size = path.stat().st_size
                if declared_extent(path) > size:
                    truncated.append((path.name, size))
            except Exception:
                truncated.append((path.name, -1))
    except Exception:
        return detail

    if not truncated:
        return detail

    listed = ", ".join(name for name, _ in truncated[:5])
    more = f" and {len(truncated) - 5} more" if len(truncated) > 5 else ""
    return (
        f"{len(truncated)} of {total} source file(s) are INCOMPLETE DOWNLOADS -- "
        f"the file ends before the pixel data its own header promises. This is a "
        f"transfer that was cut off, not a corrupt dataset and not a "
        f"configuration problem. Delete and re-fetch these, then refresh: "
        f"{listed}{more}. "
        f"(Full report: python scripts/check_downloads.py data/{folder})"
    )


def _drop_rendered_tiles(product_id: str) -> None:
    """Discard PNGs rendered from a previous version of this product's legend.

    Re-indexing can change class names and colours (a corrected legend.csv, a
    different CSV having been read), but the RGBA tile cache is keyed only by
    product/z/x/y/band -- so a recoloured legend would keep serving the old
    palette until those files aged out. Seen for real: two datasets re-indexed
    against the correct legend still painted the previous product's colours.

    Code tiles are unaffected (they carry raw class codes, not colours), but
    clearing both is simpler than reasoning about which is which, and the cache
    is regenerated on demand.
    """
    try:
        from harmonizer import local_tiles

        n = local_tiles.clear_tile_cache(product_id)
        if n:
            _LOG.info("%s: dropped %d cached tile(s) after re-indexing", product_id, n)
    except Exception:  # pragma: no cover - a stale cache must not fail a run
        _LOG.debug("could not clear tiles for %s", product_id, exc_info=True)


def _record_index_artifacts(product_id: str, folder: str) -> None:
    """Declare what the indexing step produced: the registry YAML and any VRT.

    Called after ``index_one`` rather than inside it, because the indexer is also
    a standalone CLI and has no business knowing about the app's cleanup
    bookkeeping.
    """
    from harmonizer import manifest

    yaml_path = _registry_yaml(product_id)
    if yaml_path.exists():
        manifest.record(
            product_id, yaml_path, kind="registry", stage="index", folder=folder
        )
    # A single-file dataset has no VRT; a tile set's VRT is what access.path
    # points at, and it survives deleting data/ -- which is exactly why it must
    # be recorded rather than inferred.
    vrt = CONFIG.cache_dir / "vrt" / f"{product_id}.vrt"
    if vrt.exists():
        manifest.record(product_id, vrt, kind="vrt", stage="index", folder=folder)


def _convert(product_id: str, report) -> None:
    """Run ``tools/to_cog.py`` for one product, reporting per-file progress."""
    import sys

    sys.path.insert(0, str(CONFIG.data_dir.parent))
    from tools import to_cog

    try:
        tiles = to_cog.source_tiles(product_id)
    except Exception as exc:
        raise RuntimeError(f"cannot locate source files: {exc}") from exc

    # Record the output directory BEFORE writing into it, not after.
    #
    # Conversion of a large tile set runs for many minutes and can be interrupted
    # (a killed server, a full disk, a cancelled job). Recording only on success
    # left whatever had been written with nothing owning it: 1.9 GB of partial
    # COGs survived a cleanup that reported freeing 42 KB, because the manifest
    # had never heard of them. Declaring the directory up front makes a partial
    # conversion cleanable by exactly the same path as a complete one.
    from harmonizer import manifest

    manifest.record(
        product_id, CONFIG.cog_dir / product_id, kind="dir", stage="convert"
    )

    total = max(1, len(tiles))
    done = 0
    cogs: list[Path] = []
    for src in tiles:
        dst, status, detail = to_cog._convert_one(product_id, src, force=False)
        done += 1
        if status != "failed":
            cogs.append(dst)
        else:
            _LOG.warning("%s: %s failed to convert (%s)", product_id, src.name, detail)
        # Conversion is the long pole: give it 0.30 -> 0.95 of the bar.
        report(
            0.30 + 0.65 * (done / total),
            f"converting: {done}/{total} files",
        )

    if not cogs:
        raise RuntimeError("no source file converted successfully")

    if len(cogs) >= 2:
        report(0.96, "converting: building the mosaic index")
        to_cog.build_mosaic(product_id, sorted(set(cogs)))

    # The COG tree itself was recorded before conversion began (see above), as
    # one directory rather than hundreds of files: it is created and removed as
    # a unit, and 619 entries for JAXA would make both the manifest and the
    # confirmation dialog unreadable. Only the tile cache is added here, since it
    # may not have existed earlier. Both live under CONFIG.cog_dir, never data/.
    tile_cache = CONFIG.cog_dir / "_tilecache" / product_id
    if tile_cache.exists():
        manifest.record(product_id, tile_cache, kind="dir", stage="convert")


def _invalidate_registry() -> None:
    """Drop cached registry/legend lookups so a new YAML is picked up live.

    Each module is asked through its **own** documented reset call, not by
    probing attributes for a ``cache_clear``. The probing version looked
    thorough and cleared nothing: it tried ``legends.spec`` and
    ``legends.legend_classes``, which are plain functions -- the cache is
    ``legends._specs``, reset by ``legends.reload()``. A freshly written YAML was
    therefore invisible to everything reading through ``legends``, and
    registering a renamed folder failed at the conversion step with
    ``not a local-raster product: hrlc30_africa`` while its YAML sat on disk,
    correct.

    Kept individually guarded so one module's reset failing does not skip the
    rest, but no longer silent about *which* reset was missing: an unexpected
    failure here means a stale registry, which surfaces far from its cause.
    """
    from harmonizer.registry import legends

    # ``legends._specs`` is the only memoised read of the products directory --
    # ``schema.load_all_products`` and ``products.default_registry`` re-read the
    # files on every call, so this one reset covers the whole registry. Verified
    # rather than assumed: a YAML written after a lookup is invisible until this
    # runs, and visible immediately after.
    try:
        legends.reload()
    except Exception:  # pragma: no cover - never break registration
        _LOG.warning("could not reset the registry cache", exc_info=True)


def _invalidate_caches(product_id: str) -> None:
    """Forget COG-source and footprint lookups after conversion writes files."""
    _invalidate_registry()
    try:
        from harmonizer import local_tiles

        local_tiles.invalidate_cog_source(product_id)
    except Exception:  # pragma: no cover
        pass
    try:
        from harmonizer import footprints

        footprints.invalidate(product_id)
    except Exception:  # pragma: no cover
        pass


def remove_product(product_id: str, *, force: bool = False) -> dict:
    """Delete a product's derived files. **Never touches ``data/``.**

    The exact counterpart of registration: registration declares each artifact as
    it creates it, and this removes precisely that recorded list -- COG tree,
    tile cache, VRT, registry entry, sample/GMM caches -- all regenerable from
    the source folder. Nothing is matched by name, so a product can never claim
    a sibling's files (``worldcover`` vs ``worldcover_2020``) nor silently
    orphan one whose naming scheme the patterns had not been taught.

    Refuses by default while the dataset's folder still exists, because then this
    is not a cleanup: it would delete a working product's converted data and the
    next refresh would spend hours rebuilding it. ``force=True`` is the explicit
    "yes, reconvert from scratch" path.

    Returns what was removed and how much space it freed, so the caller can
    report it rather than deleting silently.
    """
    folder = CONFIG.data_dir / _folder_of(product_id)
    if folder.is_dir() and not force:
        raise ValueError(
            f"{product_id}: its data folder {folder.name} still exists, so this "
            f"would delete a working product's derived files (they would have to "
            f"be rebuilt). Delete the folder first, or pass force=True to "
            f"reconvert from scratch."
        )

    import shutil

    targets = product_artifacts(product_id)
    freed = _describe_size(targets)
    removed: list[str] = []
    failed: list[str] = []
    for path in targets:
        try:
            if path.is_dir():
                shutil.rmtree(path)
            else:
                path.unlink()
            removed.append(str(path))
        except OSError as exc:
            failed.append(f"{path}: {exc}")

    # The manifest itself goes last, and only when everything it listed is
    # actually gone: while it exists the product stays cleanable, so an
    # interrupted or partly-failed removal can simply be retried. Dropping it
    # early would strand whatever survived, with nothing recording who owned it.
    from harmonizer import manifest

    if not failed:
        manifest.forget(product_id)

    # Drop the cached registry/COG/footprint lookups so the product disappears
    # from the picker without a restart.
    _invalidate_caches(product_id)
    REGISTRATIONS.forget(product_id)

    _LOG.info("removed %s: %d artifact(s), %s freed", product_id, len(removed), freed)
    return {
        "product_id": product_id,
        "removed": removed,
        "failed": failed,
        "freed": freed,
    }


def register_all_pending(convert: bool = True) -> list[RegistrationJob]:
    """Start registration for every candidate that is not ready yet.

    Called on server startup and by the UI's "refresh datasets" action. Datasets
    that only need conversion are included; ones missing a legend are not (there
    is nothing to do until the user drops the CSV in).
    """
    jobs = []
    for state in scan_datasets():
        if state.state in (READY, INDEXING, CONVERTING):
            continue
        if state.state == NEEDS_LEGEND:
            continue
        job = register_dataset(state.folder, convert=convert)
        if job is not None:
            jobs.append(job)
    return jobs
