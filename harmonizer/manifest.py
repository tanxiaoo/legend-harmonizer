"""Per-product artifact manifests: what registration created, recorded as it goes.

Cleanup used to *infer* ownership by name -- glob ``samples_<pid>*``, assume
``cache/cog/<pid>/`` belongs to ``<pid>``, and so on. That works until it
doesn't, and both failure directions are bad:

* **Guessing too much.** ``worldcover`` and ``worldcover_2020`` are different
  products whose names are prefixes of one another; a careless pattern deletes
  one while cleaning the other. Patterns also cannot know that a file was
  written by a *different* product's run.
* **Guessing too little.** Auxiliary-AOI caches use a scoped id
  (``<pid>__aux_<name>``), cross-label caches name *both* products of a pair,
  and a future stage will invent another naming scheme. Every one of those is a
  file the pattern list has to be taught about, and until someone remembers, it
  is silently orphaned.

So ownership is **recorded, not deduced**: every step that writes a derived file
declares it here, and cleanup reads back exactly that list. A manifest is the
difference between "these files look like they belong to X" and "X created these
files".

Layout -- one JSON per product under ``cache/manifests/``::

    {
      "product_id": "worldcover_2020",
      "folder": "WorldCover_2020",
      "created": "2026-08-05T14:22:31Z",
      "updated": "2026-08-05T14:39:02Z",
      "artifacts": [
        {"path": "cache/vrt/worldcover_2020.vrt", "kind": "vrt", "stage": "index"},
        {"path": "cache/cog/worldcover_2020", "kind": "dir", "stage": "convert"},
        ...
      ]
    }

Paths are stored **repo-relative** so a manifest survives the checkout moving,
and are resolved against ``REPO_ROOT`` on read.

**Nothing under ``data/`` may ever be recorded.** :func:`record` rejects such a
path outright rather than trusting callers: the manifest is the input to a
delete, so a bad entry here is the one bug that could destroy a user's download.
"""

from __future__ import annotations

import json
import logging
import os
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from harmonizer.config import CONFIG, REPO_ROOT

__all__ = [
    "Artifact",
    "Manifest",
    "manifest_path",
    "load",
    "record",
    "record_many",
    "forget",
    "existing_artifacts",
    "all_manifests",
]

_LOG = logging.getLogger(__name__)

#: One lock for all manifest writes. Registration runs one job per product in its
#: own thread, but two jobs can finish at the same moment and the read-modify-
#: write below is not atomic on its own.
_LOCK = threading.Lock()


def manifests_dir() -> Path:
    return CONFIG.cache_dir / "manifests"


def manifest_path(product_id: str) -> Path:
    return manifests_dir() / f"{product_id}.json"


@dataclass(frozen=True)
class Artifact:
    """One derived file or directory a product owns.

    ``kind`` is advisory ("file" | "dir" | "vrt" | "cog" | "registry" | "cache")
    and exists so the UI can group what it is about to delete. ``stage`` records
    which step produced it, which is what makes a partial cleanup possible later
    (e.g. "drop the COGs, keep the registry entry") without re-deriving anything.
    """

    path: str  # repo-relative, POSIX separators
    kind: str = "file"
    stage: str = "unknown"

    def resolve(self) -> Path:
        return REPO_ROOT / self.path

    def as_dict(self) -> dict:
        return {"path": self.path, "kind": self.kind, "stage": self.stage}


@dataclass
class Manifest:
    product_id: str
    folder: str | None = None
    created: str | None = None
    updated: str | None = None
    artifacts: list[Artifact] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "product_id": self.product_id,
            "folder": self.folder,
            "created": self.created,
            "updated": self.updated,
            "artifacts": [a.as_dict() for a in self.artifacts],
        }

    def paths(self) -> list[Path]:
        return [a.resolve() for a in self.artifacts]


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _relative(path: Path) -> str:
    """A repo-relative POSIX path, or the absolute one if it lies outside."""
    p = Path(path)
    if not p.is_absolute():
        p = REPO_ROOT / p
    try:
        return p.resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return p.resolve().as_posix()


def _is_under_data(path: Path) -> bool:
    """True when a path lies inside ``data/`` -- the user's own downloads."""
    p = Path(path)
    if not p.is_absolute():
        p = REPO_ROOT / p
    try:
        data = CONFIG.data_dir.resolve()
    except OSError:
        data = CONFIG.data_dir
    # Compare resolved strings rather than Path.parents so a symlinked data/ (as
    # on the cluster, data/Africa -> a Lustre volume) is still recognised.
    try:
        resolved = p.resolve()
    except OSError:
        resolved = p
    return resolved == data or str(resolved).startswith(str(data) + os.sep)


def load(product_id: str) -> Manifest | None:
    """The recorded manifest for a product, or ``None`` if it has none."""
    path = manifest_path(product_id)
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return Manifest(
        product_id=doc.get("product_id", product_id),
        folder=doc.get("folder"),
        created=doc.get("created"),
        updated=doc.get("updated"),
        artifacts=[
            Artifact(
                path=a["path"],
                kind=a.get("kind", "file"),
                stage=a.get("stage", "unknown"),
            )
            for a in doc.get("artifacts", [])
            if a.get("path")
        ],
    )


def record_many(
    product_id: str,
    paths,
    *,
    kind: str = "file",
    stage: str = "unknown",
    folder: str | None = None,
) -> Manifest:
    """Record several artifacts for a product in one write.

    Idempotent: re-registering a product re-declares the same paths and the
    manifest does not grow. Order is preserved so a manifest reads in the order
    the pipeline produced things.
    """
    with _LOCK:
        m = load(product_id) or Manifest(product_id=product_id, created=_now())
        if folder and not m.folder:
            m.folder = folder

        seen = {a.path for a in m.artifacts}
        for raw in paths:
            path = Path(raw)
            if _is_under_data(path):
                # Refused rather than skipped quietly: a caller trying to record
                # a source file has a bug, and the manifest is what a delete
                # reads back.
                raise ValueError(
                    f"refusing to record {path} in {product_id}'s manifest: it is "
                    f"under data/, which holds the user's downloads and is never "
                    f"deleted by this app"
                )
            rel = _relative(path)
            if rel in seen:
                continue
            seen.add(rel)
            m.artifacts.append(Artifact(path=rel, kind=kind, stage=stage))

        m.updated = _now()
        _write(m)
        return m


def record(
    product_id: str,
    path,
    *,
    kind: str = "file",
    stage: str = "unknown",
    folder: str | None = None,
) -> Manifest:
    """Record a single artifact. See :func:`record_many`."""
    return record_many(product_id, [path], kind=kind, stage=stage, folder=folder)


def _write(m: Manifest) -> None:
    """Persist a manifest atomically.

    Written to a temp file and renamed: a half-written manifest would either
    under-report (leaving files orphaned) or, worse, be unparseable and make the
    product uncleanable.
    """
    path = manifest_path(m.product_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.stem}.{os.getpid()}.tmp.json")
    tmp.write_text(json.dumps(m.as_dict(), indent=2), encoding="utf-8")
    os.replace(tmp, path)


def forget(product_id: str) -> bool:
    """Delete the manifest itself, after its artifacts have been removed."""
    try:
        manifest_path(product_id).unlink()
        return True
    except OSError:
        return False


def _lazy_artifacts(product_id: str) -> list[Path]:
    """Derived locations written *outside* registration, by ordinary use.

    Two things are produced lazily rather than by a registration step, so there
    is no single moment at which they could be recorded:

    * the rendered PNG tile cache, written per tile as the user browses;
    * Stage 2/3 sample and GMM caches, written by a run -- including the
      per-auxiliary-AOI ones, whose ids are scoped (``<pid>__aux_<name>``).

    Recording every tile write would make the manifest enormous and hot; instead
    these are *derived by construction* -- their names are built from the product
    id by this app's own cache-path helpers, so ownership is not a guess. They
    are folded in at read time so cleanup still removes them.

    The ``__``-scoped forms are matched explicitly rather than with a bare
    ``<pid>*`` prefix: that would make ``worldcover`` claim
    ``worldcover_2020``'s files.
    """
    out: list[Path] = [CONFIG.cog_dir / "_tilecache" / product_id]
    for pattern in (
        f"samples_{product_id}.*",
        f"samples_{product_id}__*",
        f"gmm_{product_id}.*",
        f"gmm_{product_id}__*",
        f"crosslabels_{product_id}__*",
        f"crosslabels_*__{product_id}.*",
        f"feedback_{product_id}__*",
        f"feedback_*__{product_id}.*",
    ):
        out.extend(sorted(CONFIG.cache_dir.glob(pattern)))
    return out


def existing_artifacts(product_id: str) -> list[Path]:
    """Everything this product owns that is still on disk, in safe delete order.

    The recorded manifest (authoritative -- what registration declared) plus the
    lazily-written caches (:func:`_lazy_artifacts`). Returns nothing at all when
    the product has no manifest: without one there is no evidence this app
    created anything, and deleting on a name match alone is exactly the guessing
    this module replaced.

    Ordered so an interrupted delete leaves the product *more* cleanable, not
    less: plain files first, then directories (contents before parents), and the
    registry YAML **last** -- while it exists the product is still listed, still
    has a manifest, and can still be cleaned up. Removing it first would strand
    the rest as an invisible orphan.
    """
    m = load(product_id)
    if m is None:
        return []

    seen: set[Path] = set()
    out: list[Path] = []
    for p in list(m.paths()) + _lazy_artifacts(product_id):
        if p in seen or not p.exists():
            continue
        seen.add(p)
        out.append(p)

    out.sort(key=lambda p: (p.suffix == ".yaml", p.is_dir(), str(p)))
    return out


def backfill(product_id: str, folder: str | None = None) -> Manifest | None:
    """Write a manifest for a product registered before manifests existed.

    A one-time migration, not a fallback: it records the two artifacts whose
    location is fixed by construction -- ``cache/cog/<pid>/`` and
    ``cache/vrt/<pid>.vrt`` -- plus the registry YAML, and only when they are
    actually on disk. Everything lazily written (tile cache, sample/GMM caches)
    is folded in at read time anyway.

    Returns ``None`` when there is nothing to record, which is the correct answer
    for a hand-written registry entry whose data has never been here: it keeps
    such a product out of the deleted-dataset detection, exactly as before.
    """
    from harmonizer.registry.schema import PRODUCTS_DIR

    if load(product_id) is not None:
        return load(product_id)

    known = [
        (CONFIG.cog_dir / product_id, "dir", "convert"),
        (CONFIG.cache_dir / "vrt" / f"{product_id}.vrt", "vrt", "index"),
    ]
    present = [(p, k, s) for p, k, s in known if p.exists()]
    if not present:
        # No derived data: nothing this app created, so nothing it may delete.
        return None

    yaml_path = PRODUCTS_DIR / f"{product_id}.yaml"
    if yaml_path.exists():
        present.append((yaml_path, "registry", "index"))

    m = None
    for path, kind, stage in present:
        m = record(product_id, path, kind=kind, stage=stage, folder=folder)
    _LOG.info("backfilled manifest for %s (%d artifact(s))", product_id, len(present))
    return m


def all_manifests() -> dict[str, Manifest]:
    """Every recorded manifest, keyed by product id."""
    d = manifests_dir()
    if not d.is_dir():
        return {}
    out: dict[str, Manifest] = {}
    for path in sorted(d.glob("*.json")):
        m = load(path.stem)
        if m is not None:
            out[m.product_id] = m
    return out
