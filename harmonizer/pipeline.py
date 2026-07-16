"""End-to-end run orchestration (Stage 5.1).

Drives Stages 1-4 for one product pair over an AOI and returns the paths to the
three Stage-4 CSVs plus the in-memory affinity/decision result, reporting coarse
progress through a callback so the API can surface it on a background job
(docs/PIPELINE.md, Stage 5.1).

This module *composes* the existing stage modules; it does not reimplement any of
them. Per-run parameters (sample scale, GMM K, point floor/target) are applied by
building a run-scoped :class:`~harmonizer.config.Config` and swapping it onto the
stage module singletons for the duration of the run, so nothing in config.py or
the stages is edited. The calibration constants (temperature, floor, margin) are
read live from ``CONFIG.affinity`` by Stage 4, so overriding them here is optional
and requires no code change.

The heavy Stage 2 sampling issues server-side GEE calls; a run therefore needs
Earth Engine authenticated, exactly as the CLI verifications do.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from harmonizer import affinity as _affinity_mod
from harmonizer import decision as _decision_mod
from harmonizer import modeling as _modeling_mod
from harmonizer import sampling as _sampling_mod
from harmonizer.affinity import AffinityResult, compute_affinity
from harmonizer.config import CONFIG, Config
from harmonizer.decision import (
    absent_decisions,
    build_matching_table,
    classify_rows,
    raw_similarity_summary,
    save_matching_table_csv,
    save_normalized_affinity_csv,
    save_raw_similarity_csv,
)
from harmonizer.modeling import fit_map, gmm_cache_path, save_map_gmm
from harmonizer.overlap import Overlap, overlap_for_products
from harmonizer.sampling import sample_map, save_map_sample

# A progress callback receives (fraction_complete in [0,1], human-readable stage).
ProgressCb = Callable[[float, str], None]

# The embedding product pulled in alongside the two label products.
_EMBEDDING_ID = "alphaearth"


@dataclass
class RunParams:
    """The user-settable parameters for one harmonization run.

    ``reference_id``/``compare_id`` name the two label products (which is
    reference matters for the matrix orientation). ``aoi`` is the lon/lat box
    (min_lon, min_lat, max_lon, max_lat) narrowing the footprint-derived region,
    or ``None`` for the full overlap. The rest default from config when left
    ``None`` and are applied for this run only.
    """

    reference_id: str
    compare_id: str
    aoi: tuple[float, float, float, float] | None = None
    sample_scale_m: float | None = None
    n_components: int | None = None
    points_floor: int | None = None
    points_target: int | None = None
    # When True, ignore any cached GMMs and re-sample from GEE even if the run
    # signature matches (see cache reuse below).
    force_refresh: bool = False


@dataclass
class RunResult:
    """What a finished run exposes to the API."""

    reference_id: str
    compare_id: str
    affinity: AffinityResult
    raw_similarity_csv: Path
    normalized_affinity_csv: Path
    matching_table_csv: Path
    # Decorated matching rows and the raw-similarity summary, ready for the UI.
    matching_rows: list = field(default_factory=list)
    raw_summary: object = None
    # True when Stage 2/3 were skipped and the cached GMMs were reused.
    reused_cache: bool = False


def _run_config(params: RunParams) -> Config:
    """Build a run-scoped Config from the defaults, overriding only what's set."""
    sampling = CONFIG.sampling
    replace_sampling: dict = {}
    if params.sample_scale_m is not None:
        replace_sampling["sample_scale_m"] = float(params.sample_scale_m)
    if params.points_floor is not None:
        replace_sampling["points_floor"] = int(params.points_floor)
    if params.points_target is not None:
        replace_sampling["points_target"] = int(params.points_target)
    if replace_sampling:
        sampling = dataclasses.replace(sampling, **replace_sampling)

    gmm = CONFIG.gmm
    if params.n_components is not None:
        gmm = dataclasses.replace(gmm, n_components=int(params.n_components))

    return dataclasses.replace(CONFIG, sampling=sampling, gmm=gmm)


def _apply_config(cfg: Config) -> None:
    """Swap a run-scoped Config onto every stage module singleton.

    The stage modules read ``CONFIG`` at call time from their own module global,
    so pointing each at ``cfg`` scopes the overrides to this run without editing
    config.py (mirrors what verify_stage2.py does for the sample scale).
    """
    _sampling_mod.CONFIG = cfg
    _modeling_mod.CONFIG = cfg
    _affinity_mod.CONFIG = cfg
    _decision_mod.CONFIG = cfg


def _restore_config() -> None:
    """Restore the default singleton on every stage module."""
    _sampling_mod.CONFIG = CONFIG
    _modeling_mod.CONFIG = CONFIG
    _affinity_mod.CONFIG = CONFIG
    _decision_mod.CONFIG = CONFIG


def compute_overlap(params: RunParams) -> Overlap:
    """The working region for a run: selected products' footprints ∩ AOI.

    Raises ``ValueError`` (from :func:`overlap_for_products`) if the selected
    products and AOI do not overlap, so the API can refuse early.
    """
    return overlap_for_products(
        [params.reference_id, params.compare_id, _EMBEDDING_ID], aoi=params.aoi
    )


# --------------------------------------------------------------------------- #
# Cache reuse: skip Stage 2/3 when the same inputs already have cached GMMs.
# --------------------------------------------------------------------------- #
# Stage 2 (server-side GEE sampling) is the slow, quota-consuming part. A run
# hashes the inputs that determine the sampled points and fitted GMMs into a "run
# signature" stored beside the GMM cache; if that signature matches and both maps'
# gmm_<product>.json exist, Stage 2/3 are skipped and the cached GMMs reused. The
# CALIBRATION constants (temperature/floor/margin) are deliberately NOT part of the
# signature: they only affect the cheap Stage 4, which always recomputes and re-
# reads them live -- so changing a calibration value re-decides on reuse without a
# re-sample. force_refresh bypasses reuse entirely.


def _signature_path() -> Path:
    return CONFIG.cache_dir / "run_signature.json"


def run_signature(params: RunParams, cfg: Config) -> dict:
    """The inputs that determine the sampled points / fitted GMMs, as a dict.

    Uses the *effective* values (from the run-scoped ``cfg``) so an omitted param
    and its explicit default hash the same. Ordering the product pair by role
    (reference, compare) keeps the signature orientation-aware.
    """
    return {
        "reference_id": params.reference_id,
        "compare_id": params.compare_id,
        "working_year": cfg.maps.working_year,
        "aoi": list(params.aoi) if params.aoi is not None else None,
        "sample_scale_m": cfg.sampling.sample_scale_m,
        "n_components": cfg.gmm.n_components,
        "points_floor": cfg.sampling.points_floor,
        "points_target": cfg.sampling.points_target,
    }


def _signature_hash(sig: dict) -> str:
    payload = json.dumps(sig, sort_keys=True).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _cached_gmms_present(params: RunParams) -> bool:
    return (
        gmm_cache_path(params.reference_id).exists()
        and gmm_cache_path(params.compare_id).exists()
    )


def _cached_samples_present(params: RunParams) -> bool:
    from harmonizer.sampling import cache_path as sample_cache_path

    return (
        sample_cache_path(params.reference_id).exists()
        and sample_cache_path(params.compare_id).exists()
    )


def _signature_matches(params: RunParams, cfg: Config) -> bool:
    sig_path = _signature_path()
    if not sig_path.exists():
        return False
    try:
        stored = json.loads(sig_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return stored.get("hash") == _signature_hash(run_signature(params, cfg))


def can_reuse_cache(params: RunParams, cfg: Config | None = None) -> bool:
    """True if a prior run with the same signature left reusable GMM caches.

    ``cfg`` defaults to the run-scoped config derived from ``params`` so callers
    (e.g. the API's pre-flight guard) need not build it themselves.
    """
    if cfg is None:
        cfg = _run_config(params)
    if params.force_refresh:
        return False
    if not _cached_gmms_present(params):
        return False
    return _signature_matches(params, cfg)


def can_reuse_samples(params: RunParams, cfg: Config | None = None) -> bool:
    """True if a prior sampling pass with the same signature left point caches.

    The split-run path: the per-AOI "Sample points" button runs Stage 2 only,
    so a later full run (Run all) finds matching sample caches and fits the
    GMMs from them without touching GEE.
    """
    if cfg is None:
        cfg = _run_config(params)
    if params.force_refresh:
        return False
    if not _cached_samples_present(params):
        return False
    return _signature_matches(params, cfg)


def _write_signature(params: RunParams, cfg: Config) -> None:
    sig = run_signature(params, cfg)
    CONFIG.cache_dir.mkdir(parents=True, exist_ok=True)
    _signature_path().write_text(
        json.dumps({"hash": _signature_hash(sig), "signature": sig}, indent=2),
        encoding="utf-8",
    )


def run_sampling(
    params: RunParams,
    progress: ProgressCb | None = None,
) -> dict:
    """Run Stage 2 ONLY: sample both maps' points over the AOI and cache them.

    The split-run path (docs Stage 7c UI): the per-AOI "Sample points" button
    collects the expensive GEE points; "Run all" later fits the GMMs and
    computes the crosswalk from the caches, without re-sampling. Writes the run
    signature so a subsequent :func:`run_pipeline` (and auxiliary AOIs, which
    key on it) recognise the cached points. When the signature already matches
    and the sample caches exist (and ``force_refresh`` is off), GEE is skipped.

    Returns a summary dict: per side, how many classes were sampled healthily.
    """
    def report(frac: float, stage: str) -> None:
        if progress is not None:
            progress(max(0.0, min(1.0, frac)), stage)

    from harmonizer.absence import _sampled_ok_classes

    cfg = _run_config(params)
    _apply_config(cfg)
    try:
        overlap = compute_overlap(params)

        reused = can_reuse_samples(params, cfg) or can_reuse_cache(params, cfg)
        if reused:
            report(1.0, "reusing cached points (same inputs); no GEE sampling")
        else:
            report(0.02, "overlap computed; sampling reference map points")
            save_map_sample(sample_map(params.reference_id, overlap=overlap))
            report(0.50, "sampling compare map points")
            save_map_sample(sample_map(params.compare_id, overlap=overlap))
            # A fresh sampling pass supersedes any GMMs fitted from older
            # points; drop them so nothing downstream mixes the two.
            gmm_cache_path(params.reference_id).unlink(missing_ok=True)
            gmm_cache_path(params.compare_id).unlink(missing_ok=True)
            _write_signature(params, cfg)
            report(1.0, "points sampled and cached")

        return {
            "reused": reused,
            "reference": {
                "product_id": params.reference_id,
                "sampled_ok": sorted(_sampled_ok_classes(params.reference_id)),
            },
            "compare": {
                "product_id": params.compare_id,
                "sampled_ok": sorted(_sampled_ok_classes(params.compare_id)),
            },
        }
    finally:
        _restore_config()


def run_pipeline(
    params: RunParams,
    progress: ProgressCb | None = None,
) -> RunResult:
    """Run Stages 1-4 end to end for a product pair and write the three CSVs.

    ``progress(fraction, stage)`` is called at coarse checkpoints. The bulk of the
    time is Stage 2 (server-side GEE sampling of both maps); the remaining stages
    are CPU-only and quick. If the cache already holds GMMs for the same run
    signature (and ``force_refresh`` is not set), Stage 2/3 are skipped and only
    the cheap Stage 4 runs.
    """
    def report(frac: float, stage: str) -> None:
        if progress is not None:
            progress(max(0.0, min(1.0, frac)), stage)

    cfg = _run_config(params)
    _apply_config(cfg)
    try:
        overlap = compute_overlap(params)

        reused = can_reuse_cache(params, cfg)
        if reused:
            # Same inputs already sampled/fitted: reuse the cached GMMs, skip GEE.
            report(0.85, "reusing cached GMMs (same inputs); skipping GEE sampling")
        elif can_reuse_samples(params, cfg):
            # Split-run path: the points were collected by a "Sample points"
            # pass with the same signature; fit the GMMs from the caches
            # without touching GEE (Stage 3 only).
            report(0.70, "fitting reference GMMs from cached points")
            save_map_gmm(fit_map(params.reference_id))
            report(0.80, "fitting compare GMMs from cached points")
            save_map_gmm(fit_map(params.compare_id))
        else:
            # Stage 2 - sample both label maps (the expensive, networked part).
            report(0.02, "overlap computed; sampling reference map")
            ref_sample = sample_map(params.reference_id, overlap=overlap)
            save_map_sample(ref_sample)

            report(0.35, "sampling compare map")
            cmp_sample = sample_map(params.compare_id, overlap=overlap)
            save_map_sample(cmp_sample)

            # Stage 3 - fit GMMs per class for both maps.
            report(0.70, "fitting reference GMMs")
            save_map_gmm(fit_map(params.reference_id))
            report(0.80, "fitting compare GMMs")
            save_map_gmm(fit_map(params.compare_id))

            # Record the signature so the next identical run can reuse the cache.
            _write_signature(params, cfg)

        # Stage 4 - affinity, decision, matching table, CSVs.
        report(0.88, "computing affinity")
        aff = compute_affinity(params.reference_id, params.compare_id)
        decisions, _ = classify_rows(aff)
        # Stage 7a: absent classes come from the declared registry legend (both
        # sides), so a class with no pixels in the AOI is reported rather than
        # silently missing from the table.
        absent = absent_decisions(params.reference_id, aff)
        rows = build_matching_table(aff, decisions, include_absent=absent)

        report(0.95, "writing results")
        raw_path = save_raw_similarity_csv(aff)
        norm_path = save_normalized_affinity_csv(aff)
        table_path = save_matching_table_csv(rows)

        report(1.0, "done")
        return RunResult(
            reference_id=params.reference_id,
            compare_id=params.compare_id,
            affinity=aff,
            raw_similarity_csv=raw_path,
            normalized_affinity_csv=norm_path,
            matching_table_csv=table_path,
            matching_rows=rows,
            raw_summary=raw_similarity_summary(aff),
            reused_cache=reused,
        )
    finally:
        _restore_config()
