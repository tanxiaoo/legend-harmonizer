"""FastAPI backend (Stage 5.1).

Wraps Stages 1-4 behind a small local, single-user API (docs/PIPELINE.md, Stage
5.1). Endpoints mirror the run flow:

  GET  /api/products            list registry products (id/name/kind/role)
  POST /api/overlap             derive footprints, intersect with the AOI, and
                                return the overlap bbox + a slow-combination
                                warning; refuse early on an empty overlap
  POST /api/run                 launch a harmonization run as a background job
  GET  /api/jobs/{id}           poll a job's state and progress
  GET  /api/jobs/{id}/results   the affinity matrix + decorated matching table
  GET  /api/jobs/{id}/export/{which}   download one of the three Stage-4 CSVs

The job runner is FastAPI background tasks over a simple in-process job store --
local and single-user, so no external queue. GEE calls run under the user's
authenticated account; no credentials pass through the API. The static frontend in
``web/`` is served at ``/``.

Stage 5.2 layers the comparison map on top of 5.1 without changing the run path.
It adds three display-only endpoints the split map needs -- reusing the 5.1
run/results/export flow unchanged:

  GET  /api/footprints          each selected product's derived footprint + the
                                overlap outline, to draw before any run
  GET  /api/legend/{product}    a product's class legend (value/name/colour) for
                                the per-class toggles
  GET  /api/tiles/{product}     an XYZ tile-URL template for the product's label
                                map, optionally restricted to a subset of classes

Stage 6b adds the **review backend** over HTTP -- both the evidence-explorer engine
(``harmonizer/explorer.py``) and the Stage 6a feedback logic
(``harmonizer/review.py``) -- so the Review UI (6c) has endpoints to call:

  POST /api/review/explore      three-mode co-located-pixel query: N declustered
                                evidence locations with both maps' labels
  GET  /api/review/table        the reviewed matching table (per-edge provenance),
                                recomputed from the on-disk GMM caches + feedback
  POST /api/review/confirm      confirm/freeze + reopen edges for a reference
                                class in one request (save-only, no retraining)
  POST /api/review/unconfirm    reopen a previously confirmed edge
  POST /api/review/refit        explicit retrain: warm-start-refit the confirmed
                                classes' GMMs and re-propose the open edges

These follow the file-backed model (docs/PIPELINE.md, Stage 6.7): a confirm
persists the feedback store to ``cache/`` and the reviewed table is recomputed from
the on-disk GMM caches -- there is no shared in-memory review state.

Stage 7c adds the **auxiliary AOI** endpoints (``harmonizer/auxiliary.py``), so
the 7b dialog's "Add another AOI" button has a backend: targeted sampling of the
still-absent classes (plus the other map's co-present classes) into per-AOI
caches, and the merged matching table whose rows carry ``evidence_aoi``:

  POST /api/aoi/auxiliary       sample an auxiliary AOI as a background job
                                (same job store/polling as /api/run)
  GET  /api/aoi/list            the recorded AOI list (active auxiliaries + cap)
  GET  /api/merged/table        the merged matching table (union of every AOI's
                                rows, evidence_aoi-tagged) + the net absence report
  GET  /api/merged/export       the merged matching table as matching_table.csv
"""

from __future__ import annotations

import hashlib
import logging
import math
import os
import threading
import traceback
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from harmonizer.affinity import class_name
from harmonizer.config import CONFIG
from harmonizer.overlap import GLOBAL_BBOX, Overlap
from harmonizer.pipeline import (
    RunParams,
    RunResult,
    can_reuse_cache,
    can_reuse_samples,
    compute_overlap,
    run_pipeline,
    run_sampling,
)
from harmonizer.registry.products import default_registry
from harmonizer import local_tiles, tiles

WEB_DIR = Path(__file__).resolve().parent.parent / "web"


# --------------------------------------------------------------------------- #
# Slow-combination estimate (AOI x sample scale)
# --------------------------------------------------------------------------- #
# A rough proxy for how much server-side raster work Stage 2 will do: the AOI area
# in square degrees divided by the square of the sample scale (finer scale -> many
# more pixels). This is a heuristic guard, not a precise cost model; its only job
# is to warn before a costly AOI x scale can hang Stage 2 (docs/PIPELINE.md,
# section 2 -> slow-combination warning).

# ~pixels-per-cell threshold above which we warn. Tuned to flag a global/full-scale
# run while letting a small test AOI at a coarse scale through.
_SLOW_SCORE_THRESHOLD = 4.0e9


def _aoi_area_deg2(bbox: tuple[float, float, float, float]) -> float:
    min_lon, min_lat, max_lon, max_lat = bbox
    return max(0.0, max_lon - min_lon) * max(0.0, max_lat - min_lat)


def _slow_estimate(
    bbox: tuple[float, float, float, float], scale_m: float
) -> tuple[float, bool]:
    """Return (score, is_slow) for an AOI bbox at a sample scale in metres.

    ~1 degree is ~111 km, so degrees-to-metres is ~111_000; area in m^2 over
    scale^2 approximates the pixel count the erosion/sampling touches.
    """
    area_deg2 = _aoi_area_deg2(bbox)
    metres_per_deg = 111_000.0
    area_m2 = area_deg2 * metres_per_deg * metres_per_deg
    score = area_m2 / (scale_m * scale_m) if scale_m > 0 else float("inf")
    return score, score > _SLOW_SCORE_THRESHOLD


# Sample scales the auto-suggestion may choose from, finest first. These are
# ordinary label-map resolutions rather than arbitrary numbers, so a suggested
# scale is always one a user would recognise.
_SCALE_LADDER = (10.0, 20.0, 30.0, 50.0, 100.0, 200.0, 500.0, 1000.0)

# Local chunked sampling is bounded by pixels read, not by GEE round trips, so it
# tolerates far more than the GEE threshold above.
#
# Calibrated against a measured full-overlap run rather than guessed: hrlc30 x
# worldcover_2020 (17.2 deg x 12.8 deg) at 30 m is 2.98e9 pixels and took 570 s
# per map, i.e. **~190 s per gigapixel** on this machine. A run samples *both*
# maps, so wall time is roughly 2 x that.
#
# The budget is therefore set so an auto-suggested scale keeps the whole run in
# the minutes range, which DESIGN.md 3.3 asks for: 1.0e9 px/map is ~3 min/map,
# ~6 min for the pair. (The previous 4.0e9 implied ~13 min/map / ~25 min a run --
# defensible as a *ceiling*, but not what "auto-suggest something that finishes
# in minutes" means.) A user who wants finer detail can still override the
# suggestion; the estimate is always shown alongside it.
_SECONDS_PER_GIGAPIXEL = 190.0
_LOCAL_PIXEL_BUDGET = 1.0e9


def _estimated_seconds(pixels: float) -> float:
    """Rough wall-clock seconds to sample **both** maps over ``pixels`` each."""
    return 2.0 * (pixels / 1e9) * _SECONDS_PER_GIGAPIXEL


def _local_pixel_estimate(
    bbox: tuple[float, float, float, float], scale_m: float
) -> float:
    """Approximate pixels the chunked sampler reads for an AOI at a scale.

    ``pixels ~= overlap_area_m2 / sample_scale_m^2`` (DESIGN.md 3.3). Latitude is
    accounted for, unlike the flat GEE heuristic above: a degree of longitude
    shrinks with ``cos(lat)``, and these AOIs span tens of degrees.
    """
    min_lon, min_lat, max_lon, max_lat = bbox
    if scale_m <= 0:
        return float("inf")
    mid_lat = math.radians(0.5 * (min_lat + max_lat))
    m_per_deg_lat = 111_320.0
    m_per_deg_lon = 111_320.0 * max(0.05, math.cos(mid_lat))
    area_m2 = (
        max(0.0, max_lon - min_lon)
        * m_per_deg_lon
        * max(0.0, max_lat - min_lat)
        * m_per_deg_lat
    )
    return area_m2 / (scale_m * scale_m)


def _suggest_scale(bbox: tuple[float, float, float, float]) -> float:
    """The finest scale from the ladder that keeps an AOI within the pixel budget.

    Returns the coarsest ladder entry when even that exceeds the budget -- the
    caller still shows the estimate, so the user can decide to proceed or narrow
    the AOI rather than being blocked (DESIGN.md 3.3: suggest, show the cost, and
    let the user override).
    """
    for scale in _SCALE_LADDER:
        if _local_pixel_estimate(bbox, scale) <= _LOCAL_PIXEL_BUDGET:
            return scale
    return _SCALE_LADDER[-1]


# --------------------------------------------------------------------------- #
# Job store
# --------------------------------------------------------------------------- #


@dataclass
class Job:
    id: str
    params: RunParams
    state: Literal["queued", "running", "done", "failed"] = "queued"
    progress: float = 0.0
    stage: str = "queued"
    error: str | None = None
    result: RunResult | None = None
    # Stage 7c: "run" for a harmonization run, "aux" for an auxiliary-AOI
    # sampling job; an aux job stores its outcome summary here (its "result"
    # lives on disk as per-AOI caches -- the merged table is fetched separately).
    kind: str = "run"
    aux: dict | None = None


class JobStore:
    """A tiny thread-safe in-process job store (local, single-user)."""

    def __init__(self) -> None:
        self._jobs: dict[str, Job] = {}
        self._lock = threading.Lock()

    def create(self, params: RunParams) -> Job:
        job = Job(id=uuid.uuid4().hex, params=params)
        with self._lock:
            self._jobs[job.id] = job
        return job

    def get(self, job_id: str) -> Job | None:
        with self._lock:
            return self._jobs.get(job_id)

    def update(self, job_id: str, **fields) -> None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return
            for k, v in fields.items():
                setattr(job, k, v)


JOBS = JobStore()


# --------------------------------------------------------------------------- #
# Request / response models
# --------------------------------------------------------------------------- #


class OverlapRequest(BaseModel):
    reference_id: str
    compare_id: str
    aoi: tuple[float, float, float, float] | None = None
    sample_scale_m: float | None = None


class RunRequest(BaseModel):
    reference_id: str
    compare_id: str
    aoi: tuple[float, float, float, float] | None = None
    sample_scale_m: float | None = None
    n_components: int | None = None
    points_floor: int | None = None
    points_target: int | None = None
    # Ignore any cached GMMs and re-sample from GEE even if inputs are unchanged.
    force_refresh: bool = False
    # "sample": Stage 2 only — collect and cache the points (the per-AOI card
    # button). "full": the whole pipeline; reuses cached points when they match,
    # so "Run all" after sampling fits the GMMs without GEE.
    mode: str = "full"


class AuxAoiRequest(BaseModel):
    """Sample an auxiliary AOI for the still-absent classes (Stage 7c).

    The AOI is required (an auxiliary exists to cover specific ground).
    Sampling params omitted here default to the primary run's effective values,
    so the auxiliary GMMs sit comparably beside the primary's.
    """

    reference_id: str
    compare_id: str
    aoi: tuple[float, float, float, float]
    name: str | None = None
    sample_scale_m: float | None = None
    n_components: int | None = None
    points_floor: int | None = None
    points_target: int | None = None
    force_refresh: bool = False
    # Whose still-absent classes this AOI targets: "both" (samples every class
    # present in the AOI, like a primary run), "reference", or "compare" (that
    # side's absents + the other map's co-present classes).
    target_side: str = "both"
    # False: sample-only pass (cache the points; no GMM fitting). True: fit —
    # from the cached points when the signature matches, else sampling first.
    fit_models: bool = True


# -- Stage 6 review / evidence-explorer models ------------------------------ #


class ExploreRequest(BaseModel):
    """A three-mode evidence-explorer query (docs/PIPELINE.md, Stage 6.2)."""

    reference_id: str
    compare_id: str
    mode: Literal["both", "reference", "compare"]
    reference_value: int | None = None
    compare_value: int | None = None
    aoi: tuple[float, float, float, float] | None = None
    n: int | None = None
    # Live-path candidate oversample factor (>= 1); default from config.
    oversample: float | None = None


class ConfirmRequest(BaseModel):
    """Confirm (freeze) one or more edges for a reference class (Stage 6.5).

    One request carries the whole decision for the class: ``compare_values`` to
    confirm/freeze and ``unconfirm_values`` to reopen -- so the UI's save is a
    single round trip with one affinity recompute, not one POST per edge.
    """

    reference_id: str
    compare_id: str
    reference_value: int
    compare_values: list[int]
    unconfirm_values: list[int] = []
    # Option A -- exclusive confirmation: the confirmed edges are this row's ONLY
    # mapping (renormalised to sum 1; open edges zeroed). False keeps the default
    # quantitative freeze + open re-balance.
    complete: bool = False
    # After confirming, warm-start-refit the reference class's GMM from its samples
    # so the model re-proposes the open edges (Stage 6.6). Off by default so a
    # confirm is a cheap save; retraining is its own explicit action (see
    # /api/review/refit).
    refit: bool = False


class RefitRequest(BaseModel):
    """Explicitly retrain: warm-start-refit confirmed classes' GMMs (Stage 6.6).

    ``reference_value`` limits the refit to one reference class; omit it to refit
    every reference class that has confirmed edges.
    """

    reference_id: str
    compare_id: str
    reference_value: int | None = None


class UnconfirmRequest(BaseModel):
    """Reopen a previously confirmed edge (an expert changing their mind)."""

    reference_id: str
    compare_id: str
    reference_value: int
    compare_value: int


# --------------------------------------------------------------------------- #
# App
# --------------------------------------------------------------------------- #

_LOG = logging.getLogger(__name__)

app = FastAPI(title="Legend Harmonizer", version="0.1.0")


def _local_raster_ready(spec) -> bool:
    """True if a local_raster product's declared file actually exists.

    Filters out placeholder registry entries (``access.path`` still a TODO/
    directory, no file dropped yet) from what the UI offers, so a product only
    appears once it's actually usable.
    """
    if spec.access.method != "local_raster":
        return True
    from pathlib import Path

    p = spec.access.path
    return bool(p) and Path(p).is_file()


@app.on_event("startup")
def _autoregister_datasets() -> None:
    """Register drop-in datasets found under ``data/`` (DESIGN.md 4.1).

    Runs in background threads, so the server answers requests immediately and
    the picker shows each product's state (``indexing…`` / ``converting…`` /
    ``ready``) as it progresses. Failures land on the product as an ``error``
    badge rather than making it silently absent.

    Set ``HARMONIZER_NO_AUTOREGISTER=1`` to skip it -- useful for a quick server
    start when the COG conversion of a large new dataset would otherwise begin
    immediately.
    """
    if os.environ.get("HARMONIZER_NO_AUTOREGISTER"):
        return
    try:
        from harmonizer import registration

        jobs = registration.register_all_pending()
        if jobs:
            _LOG.info(
                "auto-registering %d dataset(s): %s",
                len(jobs),
                ", ".join(j.product_id for j in jobs),
            )
    except Exception:  # never let this stop the server coming up
        _LOG.exception("dataset auto-registration failed to start")


@app.get("/api/datasets")
def list_datasets() -> dict:
    """Drop-in datasets under ``data/`` and their registration state.

    The picker's source of truth for ready-state badges (DESIGN.md 4.2), and
    what the "refresh datasets" action polls while registration runs.
    """
    from harmonizer import registration

    states = [s.as_dict() for s in registration.scan_datasets()]
    return {
        "datasets": states,
        "active": registration.REGISTRATIONS.active(),
        "jobs": [j.as_dict() for j in registration.REGISTRATIONS.all().values()],
        # The drop-in naming convention, so the UI can teach it at the point of
        # failure rather than leaving it in a YAML comment.
        "rules": registration.drop_in_rules(),
    }


@app.post("/api/datasets/refresh")
def refresh_datasets() -> dict:
    """Rescan ``data/`` and start registering anything not ready yet."""
    from harmonizer import registration

    jobs = registration.register_all_pending()
    return {
        "started": [j.as_dict() for j in jobs],
        "datasets": [s.as_dict() for s in registration.scan_datasets()],
    }


@app.get("/api/datasets/{product_id}/artifacts")
def dataset_artifacts(product_id: str) -> dict:
    """The derived files a product owns, so the UI can say what removal deletes.

    Read-only: this is what a confirmation prompt shows before anything is
    touched.
    """
    from harmonizer import registration

    paths = registration.product_artifacts(product_id)
    return {
        "product_id": product_id,
        "artifacts": [str(p) for p in paths],
        "size": registration._describe_size(paths),
        "data_folder_present": (
            CONFIG.data_dir / registration._folder_of(product_id)
        ).is_dir(),
    }


@app.delete("/api/datasets/{product_id}")
def delete_dataset(product_id: str, force: bool = False) -> dict:
    """Remove a product's **derived** files (COGs, VRT, registry entry, caches).

    Deliberately an explicit action rather than something a rescan does on its
    own: deleting a data folder can strand several GB, but silently deleting
    several GB because a folder went missing is the kind of surprise that loses
    work. ``data/`` is never touched.

    Refuses while the dataset's folder still exists unless ``force=true`` -- that
    would not be a cleanup, it would be discarding a working product's converted
    data.
    """
    from harmonizer import registration

    try:
        return registration.remove_product(product_id, force=force)
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@app.get("/api/products")
def list_products() -> dict:
    """The registry's products, so the UI offers only valid pairings.

    Local-raster products carry their drop-in registration state (``ready`` /
    ``indexing`` / ``converting`` / ``needs-legend`` / ``needs-conversion`` /
    ``error``) so the picker can group and badge them, and disable the ones that
    are not usable yet. GEE products are always ``ready``: there is nothing to
    index or convert.
    """
    from harmonizer import registration

    reg = default_registry()
    states = registration.product_states()
    products = []
    for p in reg.all():
        is_local = p.spec.access.method == "local_raster"
        state = states.get(p.id)
        # A product whose data/ folder has been deleted is listed but DISABLED,
        # rather than hidden. Hiding it would leave several GB of derived files
        # on disk with nothing in the UI referring to them; showing it as
        # `missing` is what gives the user something to clean up.
        #
        # This must come before the readiness check below, because for a tile set
        # `access.path` is the VRT under cache/ -- which survives deleting the
        # folder, so the file-exists test still passes and the product would
        # otherwise appear perfectly selectable while every tile read fails.
        if is_local and state is not None and state.state == registration.MISSING:
            products.append(
                {
                    "id": p.id,
                    "name": p.name,
                    "kind": p.kind,
                    "role": p.role,
                    "source": p.spec.access.method,
                    "state": registration.MISSING,
                    "state_detail": state.detail,
                    "progress": 0.0,
                    "resolution_m": getattr(p.spec, "resolution_m", None),
                    "years": list(getattr(p.spec, "available_years", None) or []),
                }
            )
            continue
        if not _local_raster_ready(p.spec):
            continue
        products.append(
            {
                "id": p.id,
                "name": p.name,
                "kind": p.kind,
                "role": p.role,
                "source": p.spec.access.method,  # "local_raster" | "gee"
                "state": (state.state if state else "ready") if is_local else "ready",
                "state_detail": state.detail if (is_local and state) else "",
                "progress": state.progress if (is_local and state) else 1.0,
                "resolution_m": getattr(p.spec, "resolution_m", None),
                "years": list(getattr(p.spec, "available_years", None) or []),
            }
        )

    # Datasets under data/ that are NOT in the registry yet -- a folder waiting
    # for its legend.csv, one still indexing, or one whose registration failed.
    #
    # The loop above walks the *registry*, so an unregistered dataset has nothing
    # to iterate over and was simply absent from the picker: dropping a folder in
    # without a legend produced no response at all, which reads as "the app did
    # not notice my folder" when in fact it had, and was waiting for a file. The
    # whole point of the ready-state badges is that a dataset is never silently
    # missing -- that has to include the ones that never got a registry entry.
    listed = {p["id"] for p in products}
    for state in registration.scan_datasets():
        if state.product_id in listed or state.state == registration.READY:
            continue
        products.append(
            {
                "id": state.product_id,
                "name": state.folder,  # no registry entry yet: the folder is the name
                "kind": "label",
                "role": "reference",
                "source": "local_raster",
                "state": state.state,
                "state_detail": state.detail,
                "progress": state.progress,
                "resolution_m": None,
                "years": [],
            }
        )

    return {
        "products": products,
        "working_year": CONFIG.maps.working_year,
        "defaults": {
            "sample_scale_m": CONFIG.sampling.sample_scale_m,
            "n_components": CONFIG.gmm.n_components,
            "points_floor": CONFIG.sampling.points_floor,
            "points_target": CONFIG.sampling.points_target,
        },
        "calibration": {
            "softmax_temperature": CONFIG.affinity.softmax_temperature,
            "absolute_affinity_floor": CONFIG.affinity.absolute_affinity_floor,
            "margin_threshold": CONFIG.affinity.margin_threshold,
            # Stage 8d: the calibrated alpha the UI slider anchors on.
            "semantic_prior_alpha": CONFIG.affinity.semantic_prior_alpha,
        },
        "review": {
            "patches_per_pair": CONFIG.review.patches_per_pair,
            "live_oversample": CONFIG.review.live_oversample,
        },
    }


@app.post("/api/overlap")
def overlap(req: OverlapRequest) -> dict:
    """Derive footprints, intersect with the AOI, and warn on a slow combination.

    Refuses early (400) if the selected products and AOI do not overlap.
    """
    params = RunParams(
        reference_id=req.reference_id,
        compare_id=req.compare_id,
        aoi=req.aoi,
    )
    try:
        ov: Overlap = compute_overlap(params)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except KeyError as exc:
        raise HTTPException(
            status_code=400, detail=f"unknown product id: {exc}"
        ) from exc

    scale = req.sample_scale_m or CONFIG.sampling.sample_scale_m
    # Warn on the region that will actually be sampled.
    score, is_slow = _slow_estimate(ov.bbox, scale)

    # Both maps local => the run never touches GEE for labels, and the chunked
    # sampler streams the region cell by cell in bounded memory. The cost model
    # and the global blocker are therefore different in kind (DESIGN.md 3.3).
    both_local = _is_local_raster(req.reference_id) and _is_local_raster(req.compare_id)

    blocker = None
    warning = None
    suggested = None
    estimated_pixels = None

    if both_local:
        suggested = _suggest_scale(ov.bbox)
        estimated_pixels = _local_pixel_estimate(ov.bbox, scale)
        estimated_seconds = _estimated_seconds(estimated_pixels)
        # The full overlap is the default and is expected to be runnable here, so
        # a large region is informational rather than a warning to be feared.
        if estimated_pixels > _LOCAL_PIXEL_BUDGET:
            warning = (
                f"This AOI at {scale:.0f} m is about {estimated_pixels / 1e9:.1f} "
                f"billion pixels per map -- roughly {estimated_seconds / 60:.0f} "
                f"minutes to sample both. {suggested:.0f} m brings that to about "
                f"{_estimated_seconds(_local_pixel_estimate(ov.bbox, suggested)) / 60:.0f} "
                "minutes; a narrower AOI also works. You can run this anyway if "
                "the finer scale is what you want."
            )
    elif ov.is_global:
        # A global region is not runnable *on the GEE path*: Stage 2 would sample
        # the whole globe server-side and time out. This is a hard blocker (the
        # user must supply an AOI), distinct from the soft slow-combination
        # warning for a merely large-but-bounded AOI. It deliberately does not
        # apply to a local x local run, which samples in this process.
        blocker = (
            "Both maps are global, so the full overlap is the entire globe -- "
            "sampling it will time out on Earth Engine. Enter a bounding-box AOI "
            "to bound the run."
        )
    elif is_slow:
        warning = (
            "This AOI at the chosen sample scale is large; Stage 2 sampling may be "
            "very slow or hang. Consider a smaller AOI or a coarser sample scale "
            f"(current {scale:.0f} m)."
        )

    return {
        "bbox": list(ov.bbox),
        "is_global": ov.is_global,
        "runnable": blocker is None,
        "sample_scale_m": scale,
        "blocker": blocker,
        "slow_warning": warning,
        "slow_score": score,
        # Local-path cost model: what this AOI costs at the chosen scale, and the
        # finest scale that stays within budget. Null for runs involving GEE.
        "both_local": both_local,
        "suggested_scale_m": suggested,
        "estimated_pixels": estimated_pixels,
        "estimated_seconds": (
            _estimated_seconds(estimated_pixels)
            if estimated_pixels is not None
            else None
        ),
        # What the *suggested* scale would cost, so the UI can offer the
        # trade-off without recomputing the cost model client-side.
        "suggested_estimated_seconds": (
            _estimated_seconds(_local_pixel_estimate(ov.bbox, suggested))
            if suggested is not None
            else None
        ),
    }


# --------------------------------------------------------------------------- #
# Stage 5.2 -- comparison map: footprints, legends, and label-map tiles
# --------------------------------------------------------------------------- #


def _bbox_to_geojson(bbox: tuple[float, float, float, float]) -> dict:
    """A (min_lon, min_lat, max_lon, max_lat) box as a GeoJSON polygon."""
    min_lon, min_lat, max_lon, max_lat = bbox
    return {
        "type": "Polygon",
        "coordinates": [
            [
                [min_lon, min_lat],
                [max_lon, min_lat],
                [max_lon, max_lat],
                [min_lon, max_lat],
                [min_lon, min_lat],
            ]
        ],
    }


@app.get("/api/footprints")
def footprints(
    reference_id: str,
    compare_id: str,
    min_lon: float | None = None,
    min_lat: float | None = None,
    max_lon: float | None = None,
    max_lat: float | None = None,
) -> dict:
    """Each selected product's derived footprint plus the overlap outline.

    A product's footprint is a property of its source (rasterio bounds for HRLC,
    global for GEE maps), so a global product reports ``bbox: null`` (the whole
    world -- the frontend draws no box). The overlap is the intersection with the
    optional AOI, so the map can draw both footprints and the overlap before any
    run. Display-only; drawing happens in the frontend (docs/PIPELINE.md 5.2).
    """
    reg = default_registry()
    aoi = None
    aoi_vals = (min_lon, min_lat, max_lon, max_lat)
    if all(v is not None for v in aoi_vals):
        aoi = tuple(float(v) for v in aoi_vals)  # type: ignore[arg-type]

    out_products = []
    for pid in (reference_id, compare_id):
        try:
            prod = reg.get(pid)
        except KeyError as exc:
            raise HTTPException(
                status_code=400, detail=f"unknown product id: {pid}"
            ) from exc
        # The *operational* footprint -- rasterio bounds of the real source for a
        # local product (DESIGN.md 3.1) -- so the box drawn here is the same
        # extent the tile layer is confined to. The registry's declared box is
        # advisory and can disagree.
        from harmonizer.footprints import operational_footprint

        fp = operational_footprint(prod.id)
        out_products.append(
            {
                "id": prod.id,
                "name": prod.name,
                "bbox": list(fp) if fp is not None else None,
                "geometry": _bbox_to_geojson(fp) if fp is not None else None,
                "global": fp is None,
            }
        )

    params = RunParams(
        reference_id=reference_id, compare_id=compare_id, aoi=aoi
    )
    # A pair with NO overlap is a legitimate thing to look at -- an Africa map
    # beside a Southeast Asia one -- and the footprints are exactly what the user
    # needs to see to understand why. Previously this raised 400, the frontend's
    # fetch threw, and it drew nothing at all: no boxes, and no view fit, so the
    # maps stayed wherever they were showing the *previous* product's tiles.
    # Report the empty overlap as data instead of as an error; /api/overlap and
    # the run endpoint still refuse the run itself.
    overlap_payload = None
    overlap_error = None
    try:
        ov: Overlap = compute_overlap(params)
        overlap_payload = {
            "bbox": list(ov.bbox),
            "geometry": _bbox_to_geojson(ov.bbox),
            "is_global": ov.is_global,
        }
    except ValueError as exc:
        overlap_error = str(exc)

    return {
        "products": out_products,
        "aoi": list(aoi) if aoi is not None else None,
        "overlap": overlap_payload,
        "overlap_error": overlap_error,
    }


def _is_local_raster(product_id: str) -> bool:
    reg = default_registry()
    try:
        return reg.spec(product_id).access.method == "local_raster"
    except KeyError:
        return False


@app.get("/api/legend/{product_id}")
def legend(product_id: str) -> dict:
    """A product's class legend (value/name/colour) for the per-class toggles."""
    renderer = local_tiles if _is_local_raster(product_id) else tiles
    try:
        entries = renderer.legend(product_id)
    except KeyError as exc:
        raise HTTPException(
            status_code=404, detail=f"no legend for product: {product_id}"
        ) from exc
    return {
        "product_id": product_id,
        "classes": [
            {
                "value": e.value,
                "name": e.name,
                "color": e.color,
                "description": e.description,
                # None = not determined; False = declared by the legend but not
                # present in this dataset's pixels (DESIGN.md 4.3), which the UI
                # renders as a greyed, non-toggleable chip.
                "observed": e.observed,
            }
            for e in entries
        ],
    }


def _parse_classes(classes: str | None) -> list[int] | None:
    if classes is None or classes.strip() == "":
        return None
    try:
        return [int(v) for v in classes.split(",") if v.strip() != ""]
    except ValueError as exc:
        raise HTTPException(
            status_code=400, detail=f"invalid classes list: {classes}"
        ) from exc


@app.get("/api/tiles/{product_id}")
def label_tiles(product_id: str, classes: str | None = None) -> dict:
    """An XYZ tile-URL template for the product's label map.

    GEE-backed products stream tiles straight from Earth Engine to the browser
    and get a single ``template``, with ``classes`` baked into it.

    Local-raster products are rendered by this process and answer with
    ``encoding: "class_code"`` plus two templates:

    ``template``
        Class-code tiles (``.png``) -- one greyscale+alpha tile per position
        encoding each pixel's class code, with no palette and no class subset
        applied. The frontend colours these on a canvas, so toggling classes
        costs no request at all (DESIGN.md 2.3). ``classes`` does not appear in
        this URL by design.
    ``template_rgba``
        The server-coloured fallback (``/api/tiles/local/rgba/...``), which does
        honour ``classes``. Every toggle state is a separate render and cache
        entry here; it exists for clients that cannot decode codes themselves.
    """
    visible = _parse_classes(classes)
    if _is_local_raster(product_id):
        try:
            local_tiles.legend(product_id)  # validates the product up front
        except KeyError as exc:
            raise HTTPException(
                status_code=404, detail=f"no tiles for product: {product_id}"
            ) from exc
        # The class-code template (DESIGN.md 2.3) -- no ``?classes=``, because
        # one tile per position serves every toggle state and the browser applies
        # the palette. ``classes`` is still accepted on this endpoint and still
        # drives ``template_rgba`` below, for clients that want a ready-coloured
        # tile.
        # ``rgba`` is its own path segment rather than a ``.rgba.png`` suffix:
        # with a suffix, ``{y}.png`` matches first and the tile row fails to
        # parse as an int (a 422, not a fallback).
        template = f"/api/tiles/local/{product_id}/{{z}}/{{x}}/{{y}}.png"
        template_rgba = f"/api/tiles/local/rgba/{product_id}/{{z}}/{{x}}/{{y}}.png"
        if visible is not None:
            template_rgba += "?classes=" + ",".join(str(v) for v in visible)
        # Let the frontend upscale in the browser past the data's real
        # resolution instead of requesting finer tiles the raster cannot fill.
        #
        # ``bounds`` is the product's footprint, so Leaflet can be told the
        # layer's real extent and simply not request tiles outside it. Without
        # it the browser asks for every tile in the viewport, and a regional
        # product answers 404 for most of them -- hundreds of console errors
        # that are correct behaviour but drown out real failures.
        spec = default_registry().spec(product_id)
        footprint = getattr(spec, "footprint", None) if spec is not None else None
        return {
            "product_id": product_id,
            "template": template,
            "template_rgba": template_rgba,
            # Tells the frontend this template serves class *codes*, so it should
            # colour them itself instead of drawing the PNG directly.
            "encoding": "class_code",
            "max_native_zoom": local_tiles.max_native_zoom(product_id),
            "bounds": list(footprint) if footprint else None,
        }

    try:
        template = tiles.tile_template(product_id, visible_values=visible)
    except KeyError as exc:
        raise HTTPException(
            status_code=404, detail=f"no tiles for product: {product_id}"
        ) from exc
    return {"product_id": product_id, "template": template}


def _tile_response(
    request: Request,
    product_id: str,
    z: int,
    x: int,
    y: int,
    visible: list[int] | None,
    kind: str,
) -> Response:
    """Shared conditional-GET plumbing for both local tile encodings."""
    try:
        etag = _tile_etag(product_id, z, x, y, visible, kind)
    except KeyError as exc:
        raise HTTPException(
            status_code=404, detail=f"no tiles for product: {product_id}"
        ) from exc

    headers = {
        "Cache-Control": "public, max-age=604800",  # one week
        "ETag": etag,
    }
    # The browser already holds this exact tile: skip rendering entirely.
    if request.headers.get("if-none-match") == etag:
        return Response(status_code=304, headers=headers)

    try:
        if kind == "code":
            png = local_tiles.code_tile_png(product_id, z, x, y)
        else:
            png = local_tiles.tile_png(product_id, z, x, y, visible_values=visible)
    except KeyError as exc:
        raise HTTPException(
            status_code=404, detail=f"no tiles for product: {product_id}"
        ) from exc
    except local_tiles.TileOutsideBounds as exc:
        # Leaflet requests every tile in the viewport regardless of a regional
        # product's actual extent; a tile outside it simply doesn't exist here.
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return Response(content=png, media_type="image/png", headers=headers)


@app.get("/api/tiles/local/{product_id}/{z}/{x}/{y}.png")
def local_tile_png(
    request: Request, product_id: str, z: int, x: int, y: int
) -> Response:
    """A **class-code** tile for a local-raster product's label map.

    The counterpart of GEE tile streaming for products this process reads
    directly from disk (docs/PIPELINE.md, Stage 5.2) -- see ``local_tiles.py``.

    Each pixel carries its class code (greyscale) plus an alpha channel marking
    nodata; no palette and no class subset are applied here. The browser colours
    the tile on a canvas, which is what makes toggling classes free: **one tile
    per position serves every toggle state**, so a toggle re-runs a local canvas
    pass and issues no request (DESIGN.md 2.3).

    Deliberately takes no ``classes`` parameter -- accepting one would reintroduce
    per-subset cache entries for tiles whose bytes do not depend on the subset.

    Tiles are immutable for a given (product, z/x/y, band): the class codes only
    change when the underlying raster is replaced. So the response carries a long
    ``Cache-Control`` and an ``ETag``. ``_tile_etag`` changes if the product's
    band changes, so a re-banded product is not served stale.
    """
    return _tile_response(request, product_id, z, x, y, None, "code")


@app.get("/api/tiles/local/rgba/{product_id}/{z}/{x}/{y}.png")
def local_tile_rgba_png(
    request: Request,
    product_id: str,
    z: int,
    x: int,
    y: int,
    classes: str | None = None,
) -> Response:
    """A server-coloured RGBA tile: the documented fallback to code tiles.

    Honours ``classes`` (the per-class show/hide toggles) by baking the subset
    into the rendered pixels, which is why every toggle state is a separate
    render, a separate cache entry and a separate ETag -- exactly the cost the
    code-tile endpoint above removes. The frontend does not use this; it is kept
    for clients that want a ready-coloured tile.
    """
    visible = _parse_classes(classes)
    return _tile_response(request, product_id, z, x, y, visible, "rgba")


def _tile_etag(
    product_id: str,
    z: int,
    x: int,
    y: int,
    visible: list[int] | None,
    kind: str = "rgba",
) -> str:
    """A cache validator for one rendered tile.

    Covers everything that changes the PNG's bytes: the tile address, the band
    being rendered, and -- for the ``rgba`` encoding only -- the selected class
    subset and the product's legend colours.

    A ``code`` tile's bytes depend on **neither the subset nor the palette**, so
    both are excluded from its validator: including them would expire tiles that
    are still byte-identical, re-fetching data the browser already holds every
    time a colour changed. Recolouring a legend now updates the map with no tile
    traffic at all, because the colours are applied client-side.

    The band is always included: for a multi-band annual series, changing
    ``band:`` changes the year shown, and without it a browser holding a
    week-long cached tile would keep showing the old one. Raises ``KeyError`` for
    a product that has no drawable legend, so an unknown product still 404s
    before any rendering work.
    """
    entries = local_tiles.legend(product_id)
    indexes = local_tiles.band_indexes(product_id)
    parts = [
        product_id,
        kind,
        f"{z}/{x}/{y}",
        f"band={'d' if indexes is None else indexes[0]}",
    ]
    if kind != "code":
        parts.append(
            ",".join(str(v) for v in sorted(visible))
            if visible is not None
            else "all"
        )
        parts.append(";".join(f"{e.value}:{e.color}" for e in entries))
    return '"' + hashlib.sha1("|".join(parts).encode()).hexdigest()[:16] + '"'


def _execute(job_id: str) -> None:
    """Background worker: run the pipeline and record progress/result/error."""
    job = JOBS.get(job_id)
    if job is None:
        return
    JOBS.update(job_id, state="running", stage="starting", progress=0.0)

    def on_progress(frac: float, stage: str) -> None:
        JOBS.update(job_id, progress=frac, stage=stage)

    try:
        result = run_pipeline(job.params, progress=on_progress)
        JOBS.update(
            job_id, state="done", progress=1.0, stage="done", result=result
        )
        # Pre-warm the Review evidence caches in the background while the user
        # reads the results: read each map's labels at the other map's sample
        # points once (both directions), so every Review evidence query is fully
        # local by the time they switch modes. Best-effort -- on failure Review
        # simply builds the cache on demand per class.
        threading.Thread(
            target=_prewarm_review,
            args=(job.params.reference_id, job.params.compare_id),
            daemon=True,
        ).start()
    except Exception as exc:  # noqa: BLE001 - surface the failure to the client
        JOBS.update(
            job_id,
            state="failed",
            error=f"{type(exc).__name__}: {exc}",
            stage="failed",
        )
        traceback.print_exc()


def _prewarm_review(reference_id: str, compare_id: str) -> None:
    """Best-effort background build of the Review cross-label caches."""
    try:
        from harmonizer.explorer import prewarm_cross_labels

        prewarm_cross_labels(reference_id, compare_id)
    except Exception:  # noqa: BLE001 - never let prewarm break anything
        traceback.print_exc()


def _execute_sample(job_id: str) -> None:
    """Background worker: Stage 2 only — sample and cache the primary points."""
    job = JOBS.get(job_id)
    if job is None:
        return
    JOBS.update(job_id, state="running", stage="starting", progress=0.0)

    def on_progress(frac: float, stage: str) -> None:
        JOBS.update(job_id, progress=frac, stage=stage)

    try:
        summary = run_sampling(job.params, progress=on_progress)
        JOBS.update(
            job_id, state="done", progress=1.0, stage="done",
            aux={"kind": "sample", **summary},
        )
    except Exception as exc:  # noqa: BLE001 - surface the failure to the client
        JOBS.update(
            job_id,
            state="failed",
            error=f"{type(exc).__name__}: {exc}",
            stage="failed",
        )
        traceback.print_exc()


@app.post("/api/run")
def start_run(req: RunRequest) -> dict:
    """Launch a harmonization run as a background job; returns the job id.

    ``mode="sample"`` runs Stage 2 only (collect + cache the points, no GMMs,
    no crosswalk) — the per-AOI card button. ``mode="full"`` runs everything,
    fitting from cached points when they match instead of re-sampling.
    """
    if req.mode not in ("full", "sample"):
        raise HTTPException(
            status_code=400,
            detail=f"mode must be 'full' or 'sample', not {req.mode!r}.",
        )
    params = RunParams(
        reference_id=req.reference_id,
        compare_id=req.compare_id,
        aoi=req.aoi,
        sample_scale_m=req.sample_scale_m,
        n_components=req.n_components,
        points_floor=req.points_floor,
        points_target=req.points_target,
        force_refresh=req.force_refresh,
    )
    # Refuse early on an empty overlap / unknown product, before backgrounding.
    try:
        ov = compute_overlap(params)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except KeyError as exc:
        raise HTTPException(
            status_code=400, detail=f"unknown product id: {exc}"
        ) from exc

    # Refuse a global (unbounded) sampling region: Stage 2 would sample the whole
    # globe server-side and GEE times out. Both maps here are global, so a blank
    # AOI yields a global region -- the user must supply an AOI to bound it. A
    # cache-reuse run does no GEE sampling, so it is exempt.
    #
    # This guards the **GEE** path specifically (DESIGN.md 3.3 keeps it). A
    # local x local run samples in this process via the chunked sampler, in
    # memory bounded by one grid cell, so a large region is a cost to be shown
    # rather than a reason to refuse -- and in any case two local products'
    # derived footprints are never global.
    both_local_run = _is_local_raster(params.reference_id) and _is_local_raster(
        params.compare_id
    )
    if (
        ov.is_global
        and not both_local_run
        and not (can_reuse_cache(params) or can_reuse_samples(params))
    ):
        raise HTTPException(
            status_code=400,
            detail=(
                "The selected maps are both global, so leaving the AOI blank means "
                "sampling the entire globe -- Earth Engine will time out. Enter a "
                "bounding-box AOI (e.g. min lon 32, min lat 14, max lon 33, max "
                "lat 15) to bound the run."
            ),
        )

    job = JOBS.create(params)
    if req.mode == "sample":
        JOBS.update(job.id, kind="sample")
        threading.Thread(target=_execute_sample, args=(job.id,), daemon=True).start()
        return {"job_id": job.id, "state": job.state, "kind": "sample"}
    # A dedicated thread keeps the CPU-bound + blocking-GEE run off the event loop.
    threading.Thread(target=_execute, args=(job.id,), daemon=True).start()
    return {"job_id": job.id, "state": job.state}


@app.get("/api/jobs/{job_id}")
def job_status(job_id: str) -> dict:
    job = JOBS.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="unknown job id")
    return {
        "job_id": job.id,
        "state": job.state,
        "progress": job.progress,
        "stage": job.stage,
        "error": job.error,
        "kind": job.kind,
        # Aux jobs carry their outcome summary inline (targeted/found/co-present
        # /modelled per side); their table lands via GET /api/merged/table.
        "aux": job.aux,
    }


def _matching_table_payload(rows) -> list[dict]:
    """The decorated matching table for the UI (status/margin/entropy/low-conf).

    Carries the Stage 7 ``side`` and ``absence_reason`` so the UI can tell an
    unmatched compare-side target from a reference row, and say *why* an ``absent``
    class could not be modelled -- plus the Stage 7c ``evidence_aoi`` naming the
    AOI whose evidence produced each row.
    """
    out: list[dict] = []
    for r in rows:
        # An absent row lists its class but has no probabilities, so zip on the
        # class values and fill the rest -- zipping on probabilities would silently
        # drop every absent row's compare entry.
        compare = [
            {
                "value": v,
                "name": n,
                "probability": r.probabilities[i] if i < len(r.probabilities) else None,
                "low_confidence": (
                    r.compare_low_confidence[i]
                    if i < len(r.compare_low_confidence)
                    else False
                ),
            }
            for i, (v, n) in enumerate(zip(r.compare_values, r.compare_names))
        ]
        out.append(
            {
                "side": r.side,
                "evidence_aoi": r.evidence_aoi,
                "reference_value": r.reference_value,
                "reference_name": r.reference_name,
                "status": r.status,
                "absence_reason": r.absence_reason,
                "compare": compare,
                "best_raw_similarity": _nan_to_none(r.best_raw_similarity),
                "margin": _nan_to_none(r.margin),
                "entropy": _nan_to_none(r.entropy),
                "reference_low_confidence": r.reference_low_confidence,
            }
        )
    return out


def _nan_to_none(x: float) -> float | None:
    return None if (x is None or np.isnan(x)) else float(x)


@app.get("/api/jobs/{job_id}/results")
def job_results(job_id: str) -> dict:
    """The affinity matrix (for the heatmap) and the decorated matching table."""
    job = JOBS.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="unknown job id")
    if job.state != "done" or job.result is None:
        raise HTTPException(
            status_code=409, detail=f"job not finished (state={job.state})"
        )

    aff = job.result.affinity
    ref_labels = [
        f"{cv}: {class_name(aff.reference_id, cv)}" for cv in aff.reference_classes
    ]
    cmp_labels = [
        f"{cv}: {class_name(aff.compare_id, cv)}" for cv in aff.compare_classes
    ]
    from harmonizer.auxiliary import absence_report_all_aois

    return {
        "reference_id": aff.reference_id,
        "compare_id": aff.compare_id,
        "reused_cache": job.result.reused_cache,
        "reference_labels": ref_labels,
        "compare_labels": cmp_labels,
        # Heatmap uses the row-normalised (softmax) probabilities.
        "normalized_affinity": aff.normalized_affinity.tolist(),
        "raw_similarity": aff.raw_similarity.tolist(),
        "matching_table": _matching_table_payload(job.result.matching_rows),
        # Stage 7b: which declared classes this run could not model, and why. Read
        # from the caches the run just wrote -- no GEE. Drives the Add-AOI dialog
        # the UI raises once the run lands. Net of any active auxiliary AOIs
        # (Stage 7c): a class an auxiliary already covers is no longer reported.
        "absence": absence_report_all_aois(aff.reference_id, aff.compare_id),
        "max_auxiliary_aois": CONFIG.absence.max_auxiliary_aois,
        "calibration": {
            "softmax_temperature": CONFIG.affinity.softmax_temperature,
            "absolute_affinity_floor": CONFIG.affinity.absolute_affinity_floor,
            "margin_threshold": CONFIG.affinity.margin_threshold,
            # Stage 8d: the alpha this payload was computed at, and the
            # calibrated default the slider anchors on. They differ whenever the
            # user has moved the slider, which is why both are reported --
            # the UI must be able to say which value is in force.
            "semantic_prior_alpha": float(aff.alpha),
            "semantic_prior_alpha_default": CONFIG.affinity.semantic_prior_alpha,
        },
    }


# --------------------------------------------------------------------------- #
# Stage 8d -- re-decide at a different semantic-prior alpha (no re-sampling)
# --------------------------------------------------------------------------- #
# Alpha is deliberately NOT part of the run signature (harmonizer/pipeline.py):
# it only affects Stage 4, which is cheap and recomputed from the cached GMMs.
# So changing it is a *view* operation costing no GEE quota and no re-sampling,
# unlike sample scale / K / point floor, which invalidate the cache. That is the
# whole reason this endpoint can exist.


def _affinity_at_alpha(reference_id: str, compare_id: str, alpha: float | None):
    """Recompute Stage 4 for a pair at one alpha, from the cached GMMs."""
    from harmonizer.affinity import compute_affinity
    from harmonizer.decision import (
        absent_decisions,
        build_matching_table,
        classify_rows,
    )

    aff = compute_affinity(reference_id, compare_id, alpha=alpha)
    decisions, _ = classify_rows(aff)
    rows = build_matching_table(
        aff, decisions, include_absent=absent_decisions(reference_id, aff)
    )
    return aff, rows


@app.get("/api/affinity")
def affinity_at_alpha(
    reference_id: str,
    compare_id: str,
    alpha: float | None = None,
    include_aef: bool = False,
) -> dict:
    """The matching table and matrices at a given semantic-prior ``alpha``.

    Reads the **cached GMMs** and recomputes only Stage 4, so it needs a run to
    have happened for this pair but costs no GEE call. ``alpha`` omitted uses the
    calibrated ``CONFIG.affinity.semantic_prior_alpha``.

    ``include_aef`` additionally returns the alpha = 0 (observational-only)
    matching table under ``matching_table_aef``, so the UI can show the fused and
    unfused answers side by side without a second request. It is off by default
    because the comparison view is opt-in.
    """
    from harmonizer.modeling import gmm_cache_path

    for pid in (reference_id, compare_id):
        if not gmm_cache_path(pid).exists():
            raise HTTPException(
                status_code=409,
                detail=(
                    f"no fitted models cached for {pid!r}; run a harmonization "
                    "for this pair first"
                ),
            )

    if alpha is not None and alpha < 0:
        raise HTTPException(status_code=400, detail="alpha must be >= 0")

    aff, rows = _affinity_at_alpha(reference_id, compare_id, alpha)

    payload = {
        "reference_id": aff.reference_id,
        "compare_id": aff.compare_id,
        "reference_labels": [
            f"{cv}: {class_name(aff.reference_id, cv)}" for cv in aff.reference_classes
        ],
        "compare_labels": [
            f"{cv}: {class_name(aff.compare_id, cv)}" for cv in aff.compare_classes
        ],
        "normalized_affinity": aff.normalized_affinity.tolist(),
        "raw_similarity": aff.raw_similarity.tolist(),
        "semantic_prior": (
            aff.semantic_prior.tolist() if aff.semantic_prior is not None else None
        ),
        "matching_table": _matching_table_payload(rows),
        "alpha": float(aff.alpha),
        "alpha_default": CONFIG.affinity.semantic_prior_alpha,
    }

    if include_aef:
        # The observational-only answer, for the comparison toggle. Recomputed
        # rather than read off `normalized_affinity_aef`, because the *statuses*
        # and candidate lists also differ at alpha = 0, not just the matrix.
        _, aef_rows = _affinity_at_alpha(reference_id, compare_id, 0.0)
        payload["matching_table_aef"] = _matching_table_payload(aef_rows)

    return payload


_EXPORTS = {
    "raw_similarity": ("raw_similarity_csv", "raw_similarity.csv"),
    "normalized_affinity": ("normalized_affinity_csv", "normalized_affinity.csv"),
    "matching_table": ("matching_table_csv", "matching_table.csv"),
}


@app.get("/api/jobs/{job_id}/export/{which}")
def export_csv(job_id: str, which: str, alpha: float | None = None) -> FileResponse:
    """Download one of the three Stage-4 CSVs the run wrote to cache/.

    ``alpha`` (Stage 8d) re-exports at a different semantic-prior alpha, so a
    download matches what the slider is currently showing rather than whatever
    alpha the run happened to use. It is written to a **separate temp file**, not
    over the run's cached artifacts: the cache belongs to the run, and a
    what-if download must not overwrite it. Omitted, the run's own file is
    served unchanged.
    """
    job = JOBS.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="unknown job id")
    if job.state != "done" or job.result is None:
        raise HTTPException(
            status_code=409, detail=f"job not finished (state={job.state})"
        )
    if which not in _EXPORTS:
        raise HTTPException(status_code=404, detail=f"unknown export: {which}")

    attr, filename = _EXPORTS[which]

    run_alpha = float(job.result.affinity.alpha)
    if alpha is not None and abs(float(alpha) - run_alpha) > 1e-12:
        return _export_at_alpha(job, which, float(alpha), filename)

    path: Path = getattr(job.result, attr)
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"missing file: {filename}")
    return FileResponse(path, filename=filename, media_type="text/csv")


def _export_at_alpha(job, which: str, alpha: float, filename: str) -> FileResponse:
    """Write one CSV at an off-run alpha into a temp dir and serve it."""
    import tempfile

    from harmonizer.decision import (
        save_matching_table_csv,
        save_normalized_affinity_csv,
        save_raw_similarity_csv,
    )

    if alpha < 0:
        raise HTTPException(status_code=400, detail="alpha must be >= 0")

    aff, rows = _affinity_at_alpha(
        job.result.reference_id, job.result.compare_id, alpha
    )
    out = Path(tempfile.mkdtemp(prefix="export_alpha_")) / filename

    if which == "matching_table":
        save_matching_table_csv(rows, out)
    elif which == "normalized_affinity":
        save_normalized_affinity_csv(aff, out)
    else:
        # raw_similarity does not depend on alpha at all (it is the
        # observational orphan signal), but serving it through the same path
        # keeps the UI's download links uniform.
        save_raw_similarity_csv(aff, out)
    return FileResponse(out, filename=filename, media_type="text/csv")


# --------------------------------------------------------------------------- #
# Stage 7c -- auxiliary AOIs and the merged (multi-AOI) matching table
# --------------------------------------------------------------------------- #
# File-backed like the review endpoints: an auxiliary job writes per-AOI caches
# plus cache/aois.json, and the merged table is recomputed from disk on demand.
# Only the auxiliary's own targeted classes cost GEE; the primary caches are
# reused untouched (docs/PIPELINE.md, Stage 7.3-7.4).


def _execute_aux(job_id: str, req: AuxAoiRequest) -> None:
    """Background worker: sample one auxiliary AOI and record its summary."""
    from harmonizer.auxiliary import sample_auxiliary, _side_dict

    job = JOBS.get(job_id)
    if job is None:
        return
    JOBS.update(job_id, state="running", stage="starting", progress=0.0)

    def on_progress(frac: float, stage: str) -> None:
        JOBS.update(job_id, progress=frac, stage=stage)

    try:
        result = sample_auxiliary(
            req.reference_id,
            req.compare_id,
            tuple(req.aoi),
            req.name,
            sample_scale_m=req.sample_scale_m,
            n_components=req.n_components,
            points_floor=req.points_floor,
            points_target=req.points_target,
            force_refresh=req.force_refresh,
            target_side=req.target_side,
            fit_models=req.fit_models,
            progress=on_progress,
        )
        JOBS.update(
            job_id,
            state="done",
            progress=1.0,
            stage="done",
            aux={
                "name": result.name,
                "bbox": list(result.bbox),
                "reused": result.reused,
                "reference": _side_dict(result.reference),
                "compare": _side_dict(result.compare),
            },
        )
    except Exception as exc:  # noqa: BLE001 - surface the failure to the client
        JOBS.update(
            job_id,
            state="failed",
            error=f"{type(exc).__name__}: {exc}",
            stage="failed",
        )
        traceback.print_exc()


@app.post("/api/aoi/auxiliary")
def add_auxiliary_aoi(req: AuxAoiRequest) -> dict:
    """Sample an auxiliary AOI as a background job (poll via /api/jobs/{id}).

    Refuses early when there is no primary run to top up, when the AOI does not
    overlap the products, when the auxiliary cap is reached, or when nothing is
    absent to target -- all before any GEE work is queued.
    """
    from harmonizer.auxiliary import (
        current_primary_hash,
        sanitize_aux_name,
        still_absent_classes,
        stored_auxiliaries,
    )

    params = RunParams(
        reference_id=req.reference_id,
        compare_id=req.compare_id,
        aoi=tuple(req.aoi),
    )
    try:
        compute_overlap(params)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except KeyError as exc:
        raise HTTPException(
            status_code=400, detail=f"unknown product id: {exc}"
        ) from exc

    if current_primary_hash() is None:
        raise HTTPException(
            status_code=409,
            detail="no primary run found; run a harmonization before adding "
            "auxiliary AOIs (an auxiliary tops up a primary run).",
        )

    # Stored (not just active): an unused auxiliary still owns its name and its
    # slot — re-sampling it by name re-uses it rather than creating a new one.
    stored = stored_auxiliaries()
    names = {e["name"] for e in stored}
    name = sanitize_aux_name(req.name) if req.name else None
    if name not in names and len(stored) >= CONFIG.absence.max_auxiliary_aois:
        raise HTTPException(
            status_code=400,
            detail=(
                f"auxiliary AOI limit reached "
                f"({CONFIG.absence.max_auxiliary_aois} per run)."
            ),
        )

    if req.target_side not in ("both", "reference", "compare"):
        raise HTTPException(
            status_code=400,
            detail=f"target_side must be 'both', 'reference', or 'compare', "
            f"not {req.target_side!r}.",
        )
    # "Nothing to target" is judged against the side(s) this AOI is aimed at.
    # An EXISTING auxiliary is exempt: its own coverage would mask its targets,
    # and re-sampling it either reuses the cached points (same box + params) or
    # recomputes targets excluding itself — the job handles both.
    if name not in names:
        need_ref = req.target_side in ("both", "reference")
        need_cmp = req.target_side in ("both", "compare")
        if not (need_ref and still_absent_classes(req.reference_id)) and not (
            need_cmp and still_absent_classes(req.compare_id)
        ):
            raise HTTPException(
                status_code=400,
                detail="nothing to target: every declared class of the selected "
                "map(s) is already modelled by an existing AOI.",
            )

    job = JOBS.create(params)
    JOBS.update(job.id, kind="aux")
    threading.Thread(target=_execute_aux, args=(job.id, req), daemon=True).start()
    return {"job_id": job.id, "state": job.state, "kind": "aux"}


@app.get("/api/aoi/list")
def list_aois(reference_id: str, compare_id: str) -> dict:
    """The run's AOI inventory, for the AOI manager panel.

    One entry per AOI -- the primary plus every active auxiliary -- each naming
    the classes it evidences (fitted GMMs) per side, so the user sees at a
    glance which AOI supplies which classes. Plus the cap and what is still
    absent from every AOI. Cache-only; no GEE.
    """
    import json as _json

    from harmonizer.absence import covered_classes
    from harmonizer.auxiliary import (
        _signature_path,
        aux_scoped_id,
        still_absent_classes,
        stored_auxiliaries,
    )

    def _named(pid: str, values) -> list[dict]:
        return [{"value": int(v), "name": class_name(pid, int(v))} for v in sorted(values)]

    def _absent_payload(pid: str) -> list[dict]:
        return [
            {"class_value": a.class_value, "class_name": a.class_name, "reason": a.reason}
            for a in still_absent_classes(pid)
        ]

    # The primary AOI: its bbox from the stored run signature, its classes from
    # the primary GMM caches. None before any run has landed.
    primary = None
    sig_path = _signature_path()
    if sig_path.exists():
        try:
            sig = _json.loads(sig_path.read_text(encoding="utf-8")).get("signature", {})
        except (OSError, _json.JSONDecodeError):
            sig = {}
        # "modelled" here means COVERED: fitted GMMs once Stage 3 has run, else
        # (a sample-only pass) the healthily sampled classes awaiting their fit.
        primary = {
            "name": "primary",
            "bbox": sig.get("aoi"),
            "reference": {"modelled": _named(reference_id, covered_classes(reference_id))},
            "compare": {"modelled": _named(compare_id, covered_classes(compare_id))},
        }

    auxes = []
    # Stored (not just active) so unused auxiliaries stay visible in the
    # manager — the model ignores them, but the user can switch them back on.
    for e in stored_auxiliaries():
        name = e["name"]
        auxes.append(
            {
                "name": name,
                "bbox": e["bbox"],
                # Whose absent classes this AOI was added to cover (Stage 7c
                # side targeting); older entries default to "both".
                "target_side": e.get("target_side", "both"),
                # Unused: kept on disk, excluded from the model until re-used.
                "disabled": bool(e.get("disabled", False)),
                # Live from the per-AOI caches (the truth), not the stored
                # summary -- a hand-edited or partial cache shows as it really
                # is. Covered = fitted GMMs, or healthy samples pending a fit.
                "reference": {
                    "modelled": _named(
                        reference_id, covered_classes(aux_scoped_id(reference_id, name))
                    ),
                    "targeted": e.get("reference", {}).get("targeted", []),
                },
                "compare": {
                    "modelled": _named(
                        compare_id, covered_classes(aux_scoped_id(compare_id, name))
                    ),
                    "targeted": e.get("compare", {}).get("targeted", []),
                },
            }
        )

    return {
        "primary": primary,
        "auxiliaries": auxes,
        "max_auxiliary_aois": CONFIG.absence.max_auxiliary_aois,
        "still_absent": {
            "reference": _absent_payload(reference_id),
            "compare": _absent_payload(compare_id),
        },
    }


class RenameAoiRequest(BaseModel):
    reference_id: str
    compare_id: str
    old_name: str
    new_name: str


class UseAoiRequest(BaseModel):
    reference_id: str
    compare_id: str
    name: str
    # False = unuse (set aside: kept on disk, excluded from the model);
    # True = use it again. Sampling an unused auxiliary also re-uses it.
    use: bool


@app.post("/api/aoi/auxiliary/use")
def use_aux_aoi(req: UseAoiRequest) -> dict:
    """Mark an auxiliary AOI unused (kept, but the model ignores it) or in use.

    Unuse is the middle ground between keeping and deleting: the entry and its
    cached points/fits stay on disk, but coverage, absence, targeting, the
    merged table, and the review upgrade all skip it until re-used.
    """
    from harmonizer.auxiliary import set_auxiliary_disabled

    try:
        ok = set_auxiliary_disabled(req.name, disabled=not req.use)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if not ok:
        raise HTTPException(
            status_code=404, detail=f"no auxiliary named {req.name!r}."
        )
    return {"name": req.name, "in_use": req.use}


@app.post("/api/aoi/auxiliary/rename")
def rename_aux_aoi(req: RenameAoiRequest) -> dict:
    """Rename an auxiliary AOI (cache files move; no re-sample -- the signature
    covers bbox/params/targets, not the name)."""
    from harmonizer.auxiliary import rename_auxiliary

    try:
        new = rename_auxiliary(
            req.reference_id, req.compare_id, req.old_name, req.new_name
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"renamed": True, "name": new}


@app.delete("/api/aoi/auxiliary/{name}")
def delete_aux_aoi(name: str, reference_id: str, compare_id: str) -> dict:
    """Delete an auxiliary AOI (entry + its per-AOI caches). The merged table and
    absence report recompute from what remains; the primary is untouched."""
    from harmonizer.auxiliary import delete_auxiliary

    try:
        deleted = delete_auxiliary(reference_id, compare_id, name)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if not deleted:
        raise HTTPException(status_code=404, detail=f"no auxiliary named {name!r}")
    return {"deleted": True, "name": name}


@app.get("/api/merged/table")
def merged_table(
    reference_id: str, compare_id: str, alpha: float | None = None
) -> dict:
    """The merged matching table: the union of every AOI's rows (Stage 7.4).

    Recomputed from the on-disk primary and per-auxiliary caches; each row
    carries ``evidence_aoi``. Also returns the net absence report so the UI's
    dialog and notes reflect auxiliary coverage without a second request.

    ``alpha`` (Stage 8d) must be forwarded: this table REPLACES the primary-only
    one whenever the pair has auxiliaries, so ignoring alpha here would silently
    undo the user's choice on exactly those runs.
    """
    from harmonizer.auxiliary import absence_report_all_aois, merged_matching_table

    if alpha is not None and alpha < 0:
        raise HTTPException(status_code=400, detail="alpha must be >= 0")

    try:
        rows, info = merged_matching_table(reference_id, compare_id, alpha)
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=409,
            detail=(
                f"no cached GMMs for {reference_id!r} x {compare_id!r}; run the "
                "pipeline first."
            ),
        ) from exc

    return {
        "reference_id": reference_id,
        "compare_id": compare_id,
        "auxiliaries": info["auxiliaries"],
        "matching_table": _matching_table_payload(rows),
        "absence": absence_report_all_aois(reference_id, compare_id),
        "max_auxiliary_aois": CONFIG.absence.max_auxiliary_aois,
        "alpha": (
            CONFIG.affinity.semantic_prior_alpha if alpha is None else float(alpha)
        ),
    }


@app.get("/api/merged/export")
def merged_export(
    reference_id: str, compare_id: str, alpha: float | None = None
) -> FileResponse:
    """The merged matching table as ``matching_table.csv`` -- the deliverable.

    The matching table is defined as the union of the AOIs' rows (docs 7.4), so
    this rewrites ``cache/matching_table.csv`` with the current merged rows
    (evidence_aoi column included) and serves it. With no auxiliaries it is the
    primary table unchanged, evidence_aoi reading "primary" throughout.

    ``alpha`` (Stage 8d) fuses at the displayed semantic-prior weight, so the
    downloaded deliverable matches what is on screen.
    """
    from harmonizer.auxiliary import merged_matching_table
    from harmonizer.decision import save_matching_table_csv

    if alpha is not None and alpha < 0:
        raise HTTPException(status_code=400, detail="alpha must be >= 0")

    try:
        rows, _ = merged_matching_table(reference_id, compare_id, alpha)
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=409,
            detail="no cached GMMs; run the pipeline first.",
        ) from exc
    path = save_matching_table_csv(rows)
    return FileResponse(path, filename="matching_table.csv", media_type="text/csv")


# --------------------------------------------------------------------------- #
# Stage 6 -- review / evidence explorer
# --------------------------------------------------------------------------- #
# Wraps the evidence-explorer engine (explorer.py) and the Stage 6a feedback logic
# (review.py) as HTTP endpoints, so the Review UI (6c) has a backend to call. The
# review state is file-backed (feedback store in cache/, GMMs recomputed from the
# cache); there is no shared in-memory review state (docs/PIPELINE.md, Stage 6.7).


def _reviewed_table_payload(rows) -> list[dict]:
    """Serialise reviewed rows (per-edge provenance) for the UI."""
    out: list[dict] = []
    for r in rows:
        out.append(
            {
                "reference_value": r.reference_value,
                "reference_name": r.reference_name,
                "status": r.status,
                # Stage 7: why an absent class could not be modelled, so the
                # review UI can mark it rather than show an empty candidate list.
                "absence_reason": r.absence_reason,
                "has_confirmed": r.has_confirmed,
                "complete": r.complete,
                # Full re-balanced row (every compare class), so the UI can show
                # the algorithm's mass for classes below the display cutoff.
                "all_probabilities": {
                    str(cv): p for cv, p in r.all_probabilities.items()
                },
                "edges": [
                    {
                        "compare_value": e.compare_value,
                        "compare_name": e.compare_name,
                        "probability": e.probability,
                        "provenance": e.provenance,
                    }
                    for e in r.edges
                ],
            }
        )
    return out


@app.post("/api/review/explore")
def review_explore(req: ExploreRequest) -> dict:
    """Three-mode co-located-pixel query: N declustered evidence locations.

    Returns center coordinates + each map's label at each location (no imagery --
    the patch is drawn client-side at the returned window; docs/PIPELINE.md, Stage
    6.3). Co-occurrence here is evidence retrieval only, never scoring (Stage 6.2).
    """
    from harmonizer.explorer import explore_evidence

    try:
        result = explore_evidence(
            req.reference_id,
            req.compare_id,
            mode=req.mode,
            reference_value=req.reference_value,
            compare_value=req.compare_value,
            aoi=req.aoi,
            n=req.n,
            oversample=req.oversample,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return {
        "mode": result.mode,
        "reference_id": result.reference_id,
        "compare_id": result.compare_id,
        "reference_value": result.reference_value,
        "compare_value": result.compare_value,
        "patch_window_px": result.patch_window_px,
        "patch_window_m": result.patch_window_m,
        "reference_pixel_m": result.reference_pixel_m,
        "compare_pixel_m": result.compare_pixel_m,
        # "cache" = Stage 2 training sample points (default); "live" = fresh draw.
        "source": result.source,
        "n": result.n,
        "locations": [
            {
                "lon": loc.lon,
                "lat": loc.lat,
                "reference_label": loc.reference_label,
                "reference_label_name": loc.reference_label_name,
                "compare_label": loc.compare_label,
                "compare_label_name": loc.compare_label_name,
            }
            for loc in result.locations
        ],
    }


@app.get("/api/review/table")
def review_table(reference_id: str, compare_id: str) -> dict:
    """The reviewed matching table with per-edge provenance.

    Recomputes affinity/decisions from the (possibly refit) on-disk GMM caches and
    applies the feedback store's confirmed edges (frozen), re-balancing only the open
    edges (docs/PIPELINE.md, Stage 6.6). This is the file-backed "shared live state":
    the next fetch reflects any confirmation.
    """
    from harmonizer.auxiliary import (
        recompute_reviewed_table_all_aois,
        still_absent_classes,
    )

    try:
        _, rows = recompute_reviewed_table_all_aois(reference_id, compare_id)
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=409,
            detail=(
                f"no cached GMMs for {reference_id!r} x {compare_id!r}; run the "
                "pipeline first."
            ),
        ) from exc

    # Absent classes per side (Stage 7), so the review UI can mark them in both
    # class pickers. Net of auxiliaries: a class an auxiliary evidences is no
    # longer absent -- its upgraded row carries that auxiliary's proposals. They
    # stay selectable -- an expert may declare a correspondence no AOI can
    # evidence -- but must not look like ordinary modelled classes.
    def _absent_payload(pid: str) -> list[dict]:
        return [
            {"class_value": a.class_value, "class_name": a.class_name, "reason": a.reason}
            for a in still_absent_classes(pid)
        ]

    return {
        "reference_id": reference_id,
        "compare_id": compare_id,
        "rows": _reviewed_table_payload(rows),
        "absent": {
            "ref": _absent_payload(reference_id),
            "cmp": _absent_payload(compare_id),
        },
    }


@app.post("/api/review/confirm")
def review_confirm(req: ConfirmRequest) -> dict:
    """Confirm/freeze (and reopen) edges for a reference class in one request.

    Persists the whole decision -- ``compare_values`` confirmed/frozen,
    ``unconfirm_values`` reopened -- with a single affinity recompute (memoised on
    the GMM caches), so the save is fast. Retraining the model is NOT done here:
    it is the explicit ``/api/review/refit`` action (Stage 6.6), unless the caller
    sets the legacy ``refit`` flag. Returns the freshly recomputed reviewed table.
    """
    from harmonizer.auxiliary import (
        aux_affinity_for_class,
        recompute_reviewed_table_all_aois,
    )
    from harmonizer.review import (
        cached_affinity,
        confirm_edges,
        load_feedback,
        save_feedback,
        unconfirm_edge,
        warm_start_refit,
    )

    store = load_feedback(req.reference_id, req.compare_id)
    try:
        aff = cached_affinity(req.reference_id, req.compare_id)
        # A class only an auxiliary evidences has no primary row: confirm
        # against that auxiliary's sub-affinity, so the frozen retained
        # probability is the one that actually proposed the edge (not 0).
        confirm_aff = aff
        if int(req.reference_value) not in aff.reference_classes:
            confirm_aff = (
                aux_affinity_for_class(
                    req.reference_id, req.compare_id, req.reference_value
                )
                or aff
            )
        for cv in req.unconfirm_values:
            unconfirm_edge(store, req.reference_value, cv)
        if req.compare_values:
            confirm_edges(store, confirm_aff, req.reference_value, req.compare_values)
        # The request's complete flag is the truth for this row on every save.
        # complete WITH confirmed edges = "these are the only mapping";
        # complete WITHOUT confirmed edges = "no matching class in the compare
        # map" (the UI's explicit no-match decision).
        if req.complete:
            store.complete.add(int(req.reference_value))
        else:
            store.complete.discard(int(req.reference_value))
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=409, detail="no cached GMMs; run the pipeline first."
        ) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    save_feedback(store)

    refit_info = None
    if req.refit:
        r = warm_start_refit(
            req.reference_id, req.compare_id, req.reference_value, store
        )
        refit_info = {
            "refit": r.refit,
            "n_points_refit": r.n_points_refit,
            "n_components": r.n_components,
            "covariance_type_used": r.covariance_type_used,
        }
        # The refit rewrote the GMM cache; the affinity memo (keyed on the
        # caches' mtimes) recomputes fresh automatically.

    _, rows = recompute_reviewed_table_all_aois(
        req.reference_id, req.compare_id, store
    )
    return {
        "reference_id": req.reference_id,
        "compare_id": req.compare_id,
        "refit": refit_info,
        "rows": _reviewed_table_payload(rows),
    }


@app.post("/api/review/refit")
def review_refit(req: RefitRequest) -> dict:
    """Explicit retrain: warm-start-refit confirmed classes' GMMs (Stage 6.6).

    Decoupled from confirm so saving a decision stays instant; the expert
    retrains once after adjusting as many classes as they like. Refits the given
    reference class, or every class with confirmed edges when none is given, then
    returns the recomputed reviewed table (open edges re-proposed; confirmed
    edges stay frozen).
    """
    from harmonizer.auxiliary import recompute_reviewed_table_all_aois
    from harmonizer.review import (
        load_feedback,
        refit_all_confirmed,
        warm_start_refit,
    )

    store = load_feedback(req.reference_id, req.compare_id)
    try:
        if req.reference_value is not None:
            results = [
                warm_start_refit(
                    req.reference_id, req.compare_id, req.reference_value, store
                )
            ]
        else:
            results = refit_all_confirmed(req.reference_id, req.compare_id, store)
        _, rows = recompute_reviewed_table_all_aois(
            req.reference_id, req.compare_id, store
        )
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=409, detail="no cached GMMs; run the pipeline first."
        ) from exc

    return {
        "reference_id": req.reference_id,
        "compare_id": req.compare_id,
        "refits": [
            {
                "reference_value": r.reference_value,
                "refit": r.refit,
                "n_points_refit": r.n_points_refit,
                "n_components": r.n_components,
            }
            for r in results
        ],
        "rows": _reviewed_table_payload(rows),
    }


@app.get("/api/review/export")
def review_export(reference_id: str, compare_id: str) -> Response:
    """The reviewed matching table as CSV -- the crosswalk deliverable.

    Mirrors the review UI's table exactly: four columns (code and class name
    for each of the two maps, named in the header), one line per mapping edge.
    """
    import csv
    import io

    from harmonizer.absence import compare_absent_classes
    from harmonizer.auxiliary import (
        aux_modelled_classes,
        recompute_reviewed_table_all_aois,
    )

    try:
        _, rows = recompute_reviewed_table_all_aois(reference_id, compare_id)
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=409, detail="no cached GMMs; run the pipeline first."
        ) from exc

    import re

    def short_name(pid: str) -> str:
        # Product names carry long qualifiers, e.g. "ESA WorldCover v200
        # (global static 2021) [test-swap reference]"; headers only need the
        # product itself.
        reg = default_registry()
        return re.sub(r"\s*[\(\[].*$", "", reg.get(pid).name).strip()

    ref_name = short_name(reference_id)
    cmp_name = short_name(compare_id)

    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(
        [f"{ref_name} code", f"{ref_name} class",
         f"{cmp_name} code", f"{cmp_name} class"]
    )
    for r in rows:
        # Same rule as the UI table/Sankey: once the expert has confirmed
        # edges for a class, the decision IS the mapping -- open algorithm
        # proposals are not exported alongside it. One CSV line per reference
        # class: a one-to-many mapping joins its codes and names in the cell
        # (e.g. "1,2,3" / "Trees, Grass, Flooded vegetation").
        confirmed = [e for e in r.edges if e.provenance == "expert-confirmed"]
        edges = confirmed or ([] if r.complete else r.edges)
        if not edges:
            # Three different no-edge cases, which must not read alike (Stage 7):
            #   absent   - the class could not be modelled here at all, so there was
            #              never anything to review. Says so, with the reason. (An
            #              absent class the expert HAS mapped by hand has edges and
            #              exports normally below -- the expert's call stands.)
            #   complete - the expert's explicit "no matching class" decision.
            #   neither  - simply not reviewed yet; stays blank.
            if r.status == "absent":
                w.writerow(
                    [
                        r.reference_value,
                        r.reference_name,
                        "-",
                        f"Absent ({r.absence_reason})" if r.absence_reason else "Absent",
                    ]
                )
            elif r.complete:
                w.writerow([r.reference_value, r.reference_name, "-", "No matching class"])
            else:
                w.writerow([r.reference_value, r.reference_name, "", ""])
            continue
        w.writerow(
            [r.reference_value, r.reference_name,
             ",".join(str(e.compare_value) for e in edges),
             ", ".join(e.compare_name for e in edges)]
        )

    # Compare classes that could not be modelled here have no reference row to
    # appear in, so they are listed from the compare side (Stage 7): without this
    # the crosswalk silently omits part of the compare legend. Net of
    # auxiliaries: a compare class an auxiliary evidences is not absent.
    aux_cmp = aux_modelled_classes(compare_id)
    for a in compare_absent_classes(compare_id):
        if a.class_value in aux_cmp:
            continue
        w.writerow(
            [
                "-",
                f"Absent ({a.reason})" if a.reason else "Absent",
                a.class_value,
                a.class_name,
            ]
        )

    return Response(
        content=buf.getvalue(),
        media_type="text/csv",
        headers={
            "Content-Disposition": (
                f'attachment; filename="reviewed_table_{reference_id}__'
                f'{compare_id}.csv"'
            )
        },
    )


@app.post("/api/review/unconfirm")
def review_unconfirm(req: UnconfirmRequest) -> dict:
    """Reopen a previously confirmed edge and return the recomputed table."""
    from harmonizer.auxiliary import recompute_reviewed_table_all_aois
    from harmonizer.review import (
        load_feedback,
        save_feedback,
        unconfirm_edge,
    )

    store = load_feedback(req.reference_id, req.compare_id)
    unconfirm_edge(store, req.reference_value, req.compare_value)
    save_feedback(store)
    try:
        _, rows = recompute_reviewed_table_all_aois(
            req.reference_id, req.compare_id, store
        )
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=409, detail="no cached GMMs; run the pipeline first."
        ) from exc
    return {
        "reference_id": req.reference_id,
        "compare_id": req.compare_id,
        "rows": _reviewed_table_payload(rows),
    }


# The static one-page frontend. Mounted last so /api/* wins.
#
# Served with caching DISABLED. This is a local, single-user app whose frontend
# is edited in place: the default StaticFiles ETag/Last-Modified handling makes
# a browser reuse its cached app.js after the file has changed, so a UI fix
# appears not to work and the only clue is stale behaviour with no error. The
# files are a few hundred KB served from localhost, so re-sending them costs
# nothing next to that confusion.
class _NoCacheStaticFiles(StaticFiles):
    def is_not_modified(self, response_headers, request_headers) -> bool:
        # Never answer 304: always send the current bytes.
        return False

    def file_response(self, *args, **kwargs):
        resp = super().file_response(*args, **kwargs)
        resp.headers["Cache-Control"] = "no-store, must-revalidate"
        # Starlette's MutableHeaders has no .pop(); delete only if present, or
        # __delitem__ raises KeyError.
        for h in ("etag", "last-modified"):
            if h in resp.headers:
                del resp.headers[h]
        return resp


if WEB_DIR.exists():
    app.mount("/", _NoCacheStaticFiles(directory=str(WEB_DIR), html=True), name="web")
