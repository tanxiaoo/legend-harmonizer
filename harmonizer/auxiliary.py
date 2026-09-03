"""Auxiliary AOIs -- targeted sampling and the merged table (Stage 7c).

A run's AOI becomes a **list**: one primary AOI (the Stage 2-4 run, whose caches
keep their existing names) plus up to ``CONFIG.absence.max_auxiliary_aois``
auxiliary AOIs the user adds to cover classes the primary could not evidence
(docs/PIPELINE.md, sections 7.3-7.4).

**Targeted, one GMM per class, never pooled.** An auxiliary AOI samples only:

  * the still-``absent`` classes it was added to cover (per side), and
  * the **other** map's classes co-present at those classes' sampled locations --
    which is what makes the auxiliary edges meaningful: an absent class must be
    compared against the other map's classes *as they look in the same AOI*
    (Mangroves-on-the-coast against Dynamic World's coast classes, never against
    its Sahel fits). Every distance is computed **within one AOI** -- the
    invariant Stage 7 exists to preserve.

A class's points are never pooled across AOIs (AlphaEarth distributions shift
with biome; pooling smears the distribution). A co-present class may therefore
have a *second* GMM fitted in the auxiliary AOI -- that fit exists only inside
the auxiliary's own sub-matrix and never replaces the class's home (primary)
GMM or its primary matching-table row.

**Caching and provenance (7.4).** Per-auxiliary caches live beside the primary
ones (``samples_<pid>__aux_<name>.npz`` / ``gmm_<pid>__aux_<name>.json``), so
adding an auxiliary never invalidates the primary's expensive GEE sampling. The
AOI list is recorded in ``cache/aois.json`` with each auxiliary's own signature
(bbox + params + targeted classes) and the primary run-signature hash it tops
up: a fresh primary run (new signature) makes the old auxiliaries inactive,
because *which* classes needed covering came from that primary's absence
report. The merged matching table is the union of each AOI's rows, tagged with
``evidence_aoi``; row probabilities normalise within their AOI's sub-matrix.

This module **composes** the Stage 2/3/4 modules (``sampling``, ``modeling``,
``affinity``, ``decision``) and Stage 7a's ``absence``; it modifies none of
them. The only earlier-stage change 7c makes is the ``evidence_aoi`` field on
``decision.MatchingRow`` (declared there).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from harmonizer.absence import (
    AbsentClass,
    _modelled_classes,
    absent_classes,
    compare_absent_classes,
)
from harmonizer.affinity import AffinityResult, class_name, compute_affinity
from harmonizer.config import CONFIG
from harmonizer.decision import MatchingRow, build_matching_table, classify_rows
from harmonizer.modeling import fit_map, gmm_cache_path, save_map_gmm
from harmonizer.overlap import BBox, Overlap, overlap_for_products
from harmonizer.pipeline import (
    RunParams,
    _apply_config,
    _restore_config,
    _run_config,
    _signature_hash,
    _signature_path,
)
from harmonizer.sampling import (
    MapSample,
    _count_candidates_both,
    _label_image_for,
    _stratified_candidates,
    drawable_classes,
    cache_path as sample_cache_path,
    present_classes,
    sample_class,
    save_map_sample,
)

ProgressCb = Callable[[float, str], None]

_EMBEDDING_ID = "alphaearth"


def _sample_gee_class(
    label_image, class_value: int, overlap, label_adapter, embedding_adapter
):
    """``sample_class`` for a GEE label image, building its two closures.

    ``sample_class`` takes ``count_candidates_both`` / ``draw_candidates``
    callables rather than an image (so the GEE and local paths can share the
    absent-vs-buffered-away rule). This mirrors how ``sampling._sample_map_gee``
    builds them; auxiliary AOIs still sample GEE-side, so they need the same two.
    """
    return sample_class(
        class_value,
        label_adapter,
        embedding_adapter,
        count_candidates_both=lambda erode_pixels: _count_candidates_both(
            label_image, class_value, overlap, erode_pixels=erode_pixels
        ),
        draw_candidates=lambda erode_pixels: _stratified_candidates(
            label_image,
            class_value,
            overlap,
            erode_pixels=erode_pixels,
            target=CONFIG.sampling.points_target,
        ),
    )


# --------------------------------------------------------------------------- #
# The AOI list (cache/aois.json)
# --------------------------------------------------------------------------- #


def aois_path() -> Path:
    return CONFIG.cache_dir / "aois.json"


def load_aois() -> dict:
    """The recorded AOI list: primary signature hash + auxiliary entries."""
    path = aois_path()
    if not path.exists():
        return {"primary_hash": None, "auxiliaries": []}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"primary_hash": None, "auxiliaries": []}


def _save_aois(payload: dict) -> Path:
    CONFIG.cache_dir.mkdir(parents=True, exist_ok=True)
    path = aois_path()
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def current_primary_hash() -> str | None:
    """The primary run's signature hash (Stage 5.1), or None before any run."""
    path = _signature_path()
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8")).get("hash")
    except (OSError, json.JSONDecodeError):
        return None


def stored_auxiliaries() -> list[dict]:
    """Every recorded auxiliary of the current primary run with caches on disk,
    INCLUDING the unused (disabled) ones — the AOI manager shows those so the
    user can switch them back on.

    An entry counts when it was added against the current primary signature
    and at least one side's cache exists. A sample-only pass (points collected,
    GMMs fitted later by Run all) counts: the AOI is real and its points
    already answer what it covers. A fresh primary run with a new signature
    leaves old auxiliaries out rather than silently mixing evidence gathered
    for a different primary AOI.
    """
    payload = load_aois()
    primary = current_primary_hash()
    if primary is None or payload.get("primary_hash") != primary:
        return []
    out = []
    for entry in payload.get("auxiliaries", []):
        name = entry.get("name", "")
        if not name:
            continue
        if any(
            aux_gmm_cache_path(pid, name).exists()
            or aux_sample_cache_path(pid, name).exists()
            for pid in (entry["reference_id"], entry["compare_id"])
        ):
            out.append(entry)
    return out


def active_auxiliaries() -> list[dict]:
    """The auxiliaries the MODEL uses: stored ones not marked unused.

    An unused ("disabled") auxiliary keeps its entry and caches — delete is for
    throwing evidence away, unuse is for setting it aside — but contributes
    nothing anywhere this function feeds: coverage, absence, targeting, the
    merged table, and the review upgrade all ignore it until re-enabled.
    """
    return [e for e in stored_auxiliaries() if not e.get("disabled", False)]


def set_auxiliary_disabled(name: str, disabled: bool) -> bool:
    """Mark an auxiliary unused (True) or back in use (False). Caches untouched.

    Returns False when no such auxiliary is recorded.
    """
    name = sanitize_aux_name(name)
    payload = load_aois()
    entry = next(
        (e for e in payload.get("auxiliaries", []) if e.get("name") == name), None
    )
    if entry is None:
        return False
    entry["disabled"] = bool(disabled)
    _save_aois(payload)
    return True


# --------------------------------------------------------------------------- #
# Per-AOI cache naming
# --------------------------------------------------------------------------- #
# The primary AOI keeps the existing cache names (samples_<pid>.npz,
# gmm_<pid>.json) so Stages 2-6 read on unchanged; an auxiliary scopes the
# product id, which flows through the existing path helpers untouched.


def sanitize_aux_name(name: str) -> str:
    """A filesystem- and id-safe auxiliary name (kebab-case)."""
    slug = re.sub(r"[^a-z0-9-]+", "-", name.strip().lower()).strip("-")
    if not slug:
        raise ValueError(f"auxiliary AOI name {name!r} has no usable characters")
    return slug


def aux_scoped_id(product_id: str, aux_name: str) -> str:
    """The product id scoped to one auxiliary AOI, e.g. ``worldcover__aux_coast``.

    Existing cache-path helpers (``sampling.cache_path``,
    ``modeling.gmm_cache_path``) take any id string, so the scoped id routes the
    Stage 2/3 machinery to per-AOI files without modifying those modules.
    """
    return f"{product_id}__aux_{sanitize_aux_name(aux_name)}"


def aux_sample_cache_path(product_id: str, aux_name: str) -> Path:
    return sample_cache_path(aux_scoped_id(product_id, aux_name))


def aux_gmm_cache_path(product_id: str, aux_name: str) -> Path:
    return gmm_cache_path(aux_scoped_id(product_id, aux_name))


def primary_modelled_classes(product_id: str) -> list[int]:
    """Classes with a fitted GMM in the primary caches (for the AOI manager)."""
    return sorted(_modelled_classes(product_id))


def _aux_cache_files(product_id: str, aux_name: str) -> list[Path]:
    npz = aux_sample_cache_path(product_id, aux_name)
    return [npz, npz.with_suffix(".json"), aux_gmm_cache_path(product_id, aux_name)]


def delete_auxiliary(reference_id: str, compare_id: str, name: str) -> bool:
    """Remove an auxiliary AOI: its entry in the AOI list and its cache files.

    The merged table and absence report recompute from what remains; the primary
    is untouched (it is not an auxiliary and cannot be deleted here). Returns
    False when no such auxiliary exists.
    """
    name = sanitize_aux_name(name)
    payload = load_aois()
    entries = payload.get("auxiliaries", [])
    kept = [e for e in entries if e.get("name") != name]
    if len(kept) == len(entries):
        return False
    payload["auxiliaries"] = kept
    _save_aois(payload)
    for pid in (reference_id, compare_id):
        for f in _aux_cache_files(pid, name):
            f.unlink(missing_ok=True)
    return True


def rename_auxiliary(
    reference_id: str, compare_id: str, old_name: str, new_name: str
) -> str:
    """Rename an auxiliary AOI, moving its cache files to the new scoped names.

    The auxiliary's signature covers bbox/params/targets -- not the name -- so a
    rename never triggers a re-sample. Returns the sanitised new name.
    """
    old = sanitize_aux_name(old_name)
    new = sanitize_aux_name(new_name)
    if new == "primary":
        raise ValueError('"primary" is reserved for the primary AOI.')
    payload = load_aois()
    entries = payload.get("auxiliaries", [])
    if any(e.get("name") == new for e in entries):
        raise ValueError(f"an auxiliary named {new!r} already exists.")
    entry = next((e for e in entries if e.get("name") == old), None)
    if entry is None:
        raise ValueError(f"no auxiliary named {old!r}.")
    if new == old:
        return new

    for pid in (reference_id, compare_id):
        for src, dst in zip(_aux_cache_files(pid, old), _aux_cache_files(pid, new)):
            if not src.exists():
                continue
            src.rename(dst)
            # Keep the scoped product id recorded inside the JSON caches in step
            # with the file name (nothing keys on it, but stale ids mislead).
            if dst.suffix == ".json":
                try:
                    data = json.loads(dst.read_text(encoding="utf-8"))
                    data["product_id"] = aux_scoped_id(pid, new)
                    dst.write_text(json.dumps(data, indent=2), encoding="utf-8")
                except (OSError, json.JSONDecodeError):
                    pass

    entry["name"] = new
    _save_aois(payload)
    return new


def aux_modelled_classes(product_id: str) -> dict[int, str]:
    """Class value -> auxiliary AOI name, for classes fitted in any active auxiliary.

    Only classes with a real fitted GMM count (a class targeted by an auxiliary
    but too rare *there* as well stays absent). First auxiliary wins if two
    somehow modelled the same class.
    """
    out: dict[int, str] = {}
    for entry in active_auxiliaries():
        name = entry["name"]
        for cv in _modelled_classes(aux_scoped_id(product_id, name)):
            out.setdefault(cv, name)
    return out


def aux_covered_classes(
    product_id: str, exclude_aux: str | None = None
) -> dict[int, str]:
    """Class value -> auxiliary AOI name, for classes an auxiliary accounts for.

    Like :func:`aux_modelled_classes` but sample-aware: in a split run the
    points are collected first ("Sample points") and the GMMs fitted later
    ("Run all"), so a class healthily sampled in an auxiliary counts as covered
    before its fit exists — absence and further targeting are judged from the
    points (see :func:`harmonizer.absence.covered_classes`).

    ``exclude_aux`` ignores one auxiliary's own caches — used when re-sampling
    that auxiliary, whose targets must not be judged "covered" by itself.
    """
    from harmonizer.absence import covered_classes

    out: dict[int, str] = {}
    for entry in active_auxiliaries():
        name = entry["name"]
        if exclude_aux is not None and name == exclude_aux:
            continue
        for cv in covered_classes(aux_scoped_id(product_id, name)):
            out.setdefault(cv, name)
    return out


# --------------------------------------------------------------------------- #
# Absence, aggregated over the AOI list
# --------------------------------------------------------------------------- #


def still_absent_classes(
    product_id: str, exclude_aux: str | None = None
) -> list[AbsentClass]:
    """Declared classes with no fitted GMM in the primary *or any active auxiliary*.

    This is the Stage 7 meaning of ``absent``: 'could not be modelled from any
    of the run's AOIs'. The reason carried is the primary's (an auxiliary that
    also failed to model the class does not change why the user was told it was
    missing in the first place). Sample-aware: a class whose points are already
    collected (sample-only pass, GMMs pending) is not absent. ``exclude_aux``
    ignores one auxiliary's own coverage (see :func:`aux_covered_classes`).
    """
    covered = aux_covered_classes(product_id, exclude_aux=exclude_aux)
    return [a for a in absent_classes(product_id) if a.class_value not in covered]


def absence_report_all_aois(reference_id: str, compare_id: str) -> dict:
    """The Stage 7b absence report, net of auxiliary coverage.

    Same shape as :func:`harmonizer.absence.absence_report` (the dialog renders
    it unchanged) but a class modelled by an active auxiliary no longer counts
    as absent, and each side lists which classes its auxiliaries covered. Still
    cache-only -- no GEE.
    """
    from harmonizer.absence import absence_report

    report = absence_report(reference_id, compare_id)
    for key, pid in (("reference", reference_id), ("compare", compare_id)):
        covered = aux_modelled_classes(pid)
        part = report[key]
        part["covered_by_auxiliary"] = [
            {
                "class_value": cv,
                "class_name": class_name(pid, cv),
                "aoi": covered[cv],
            }
            for cv in sorted(covered)
            if any(a["class_value"] == cv for a in part["absent"])
        ]
        part["absent"] = [
            a for a in part["absent"] if a["class_value"] not in covered
        ]
    report["any_absent"] = bool(
        report["reference"]["absent"] or report["compare"]["absent"]
    )
    report["auxiliaries"] = [
        {"name": e["name"], "bbox": e["bbox"]} for e in active_auxiliaries()
    ]
    return report


# --------------------------------------------------------------------------- #
# Targeted sampling of one auxiliary AOI
# --------------------------------------------------------------------------- #


@dataclass
class AuxSideOutcome:
    """What one side (one map) got out of an auxiliary AOI."""

    product_id: str
    targeted: list[int] = field(default_factory=list)   # still-absent classes to cover
    found: list[int] = field(default_factory=list)      # targeted and present in the AOI
    co_present: list[int] = field(default_factory=list) # sampled because the OTHER side's
                                                        # targets co-occur with them here
    modelled: list[int] = field(default_factory=list)   # fitted GMM in this auxiliary
    absent_in_aux: list[int] = field(default_factory=list)  # sampled but starved here too


@dataclass
class AuxiliaryResult:
    """Outcome of sampling one auxiliary AOI (both sides)."""

    name: str
    bbox: BBox
    reference: AuxSideOutcome
    compare: AuxSideOutcome
    reused: bool = False  # signature matched: caches reused, no GEE


def _aux_signature(
    reference_id: str,
    compare_id: str,
    bbox: BBox,
    cfg,
    targets_ref: list[int],
    targets_cmp: list[int],
) -> dict:
    """The inputs that determine an auxiliary's sampled points and fitted GMMs.

    Includes the targeted class lists: they came from the primary's absence
    report, so a different primary run (different absences) naturally changes
    the signature and forces a re-sample instead of reusing mistargeted caches.
    """
    return {
        "reference_id": reference_id,
        "compare_id": compare_id,
        "bbox": list(bbox),
        "working_year": cfg.maps.working_year,
        "sample_scale_m": cfg.sampling.sample_scale_m,
        "n_components": cfg.gmm.n_components,
        "points_floor": cfg.sampling.points_floor,
        "points_target": cfg.sampling.points_target,
        "targets_reference": sorted(targets_ref),
        "targets_compare": sorted(targets_cmp),
    }


def _primary_signature_params() -> dict:
    """The primary run's effective sampling params, to default an auxiliary's.

    An auxiliary's GMMs sit in the same deliverable as the primary's, so unless
    the caller overrides them the auxiliary samples with the same effective
    scale/floor/target/K the primary used (read from the stored run signature).
    """
    path = _signature_path()
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8")).get("signature", {})
    except (OSError, json.JSONDecodeError):
        return {}


def _valid_labels(label_results) -> list[int]:
    return [
        int(r.label)
        for r in label_results
        if not r.masked and r.label is not None
    ]


def sample_auxiliary(
    reference_id: str,
    compare_id: str,
    aoi: BBox,
    name: str | None = None,
    *,
    sample_scale_m: float | None = None,
    n_components: int | None = None,
    points_floor: int | None = None,
    points_target: int | None = None,
    force_refresh: bool = False,
    target_side: str = "both",
    fit_models: bool = True,
    progress: ProgressCb | None = None,
) -> AuxiliaryResult:
    """Sample one auxiliary AOI for the classes the run still lacks (Stage 7.3).

    ``target_side`` says WHOSE absent classes this AOI was added to cover and
    sets the sampling scope:

      * ``"reference"`` / ``"compare"`` -- sample that side's still-absent
        classes plus the OTHER map's co-present classes at those points (the
        minimum that makes the auxiliary edges within-AOI comparisons).
      * ``"both"`` (default) -- sample EVERY declared class present in this
        AOI on both maps, like a primary run; the merged table later keeps
        only the absent-class results from it (docs 7.4).

    The target lists feed the auxiliary's signature, so changing the side
    re-samples. ``fit_models=False`` is the split-run path: collect and cache
    the points only (no GMM fitting) -- a later call with ``fit_models=True``
    and an unchanged signature fits from the cached points without GEE.

    Per side, targets = the still-absent declared classes; of those, the ones
    ``present_classes`` finds in this AOI are sampled with the full Stage 2
    machinery (erosion, homogeneity, declustering, floor rule). Then, at the
    surviving points of each side's targets, the **other** map's labels are read
    (a small point table -- no rasters cross the network) and every distinct
    co-present class of the other map is sampled here too, so the auxiliary
    sub-matrix compares the recovered class against the other legend *as it
    looks in this AOI*. Fitted GMMs go to per-AOI caches; the primary's caches
    are not touched, let alone re-sampled.

    Sampling params default to the primary run's effective values (from the
    stored run signature) so the auxiliary's GMMs are comparable; pass overrides
    to deviate. If this auxiliary's signature matches its stored entry and the
    caches exist, the whole GEE pass is skipped (``reused=True``).
    """
    def report(frac: float, msg: str) -> None:
        if progress is not None:
            progress(max(0.0, min(1.0, frac)), msg)

    primary = current_primary_hash()
    if primary is None:
        raise ValueError(
            "no primary run signature found -- run the pipeline once before "
            "adding auxiliary AOIs (an auxiliary tops up a primary run)."
        )

    # Default the sampling params to the primary run's effective values.
    prim = _primary_signature_params()
    params = RunParams(
        reference_id=reference_id,
        compare_id=compare_id,
        sample_scale_m=(
            sample_scale_m if sample_scale_m is not None
            else prim.get("sample_scale_m")
        ),
        n_components=(
            n_components if n_components is not None else prim.get("n_components")
        ),
        points_floor=(
            points_floor if points_floor is not None else prim.get("points_floor")
        ),
        points_target=(
            points_target if points_target is not None else prim.get("points_target")
        ),
    )
    cfg = _run_config(params)

    payload = load_aois()
    if payload.get("primary_hash") != primary:
        # The old auxiliaries targeted a different primary's absences; start a
        # fresh list for this primary (their cache files are simply superseded).
        payload = {"primary_hash": primary, "auxiliaries": []}

    existing_names = [e["name"] for e in payload["auxiliaries"]]
    if name is None:
        i = 1
        while f"aux{i}" in existing_names:
            i += 1
        name = f"aux{i}"
    name = sanitize_aux_name(name)

    if name not in existing_names and (
        len(existing_names) >= CONFIG.absence.max_auxiliary_aois
    ):
        raise ValueError(
            f"auxiliary AOI limit reached ({CONFIG.absence.max_auxiliary_aois}); "
            "remove or reuse an existing auxiliary name."
        )

    if target_side not in ("both", "reference", "compare"):
        raise ValueError(
            f"target_side must be 'both', 'reference', or 'compare', "
            f"not {target_side!r}."
        )

    stored = next((e for e in payload["auxiliaries"] if e["name"] == name), None)

    # Sampling an unused auxiliary is the act of using it again.
    if stored is not None and stored.get("disabled"):
        stored["disabled"] = False
        _save_aois(payload)

    # Reuse is judged on what actually defines the cached points -- the AOI
    # box, the effective sampling params, the working year, the pair, and the
    # target side -- NOT on the run's still-absent lists. Those move the moment
    # this auxiliary itself covers its targets (or another AOI changes), and
    # must never force a re-sample of identical inputs: same box + params =
    # same cached points, reused.
    def _non_target(s: dict) -> dict:
        return {k: v for k, v in s.items() if not k.startswith("targets_")}

    base_sig = _aux_signature(reference_id, compare_id, aoi, cfg, [], [])
    sig_match = (
        not force_refresh
        and stored is not None
        and stored.get("target_side", "both") == target_side
        and _non_target(stored.get("signature", {})) == _non_target(base_sig)
    )
    gmms_exist = (
        aux_gmm_cache_path(reference_id, name).exists()
        or aux_gmm_cache_path(compare_id, name).exists()
    )
    samples_exist = (
        aux_sample_cache_path(reference_id, name).exists()
        or aux_sample_cache_path(compare_id, name).exists()
    )

    # Same points already on disk, and either the fits exist too or none are
    # asked for: nothing to do.
    if sig_match and (gmms_exist if fit_models else (samples_exist or gmms_exist)):
        report(1.0, "auxiliary signature unchanged; reusing cached auxiliary data")
        return AuxiliaryResult(
            name=name,
            bbox=aoi,
            reference=AuxSideOutcome(product_id=reference_id, **stored["reference"]),
            compare=AuxSideOutcome(product_id=compare_id, **stored["compare"]),
            reused=True,
        )

    # Split-run path: the points were collected by a sample-only pass with the
    # same signature; fit the GMMs from the per-AOI caches -- no GEE.
    if sig_match and fit_models and samples_exist and not gmms_exist:
        report(0.10, "fitting auxiliary GMMs from cached points")
        _apply_config(cfg)
        try:
            def _side_from_stored(pid: str, key: str) -> AuxSideOutcome:
                d = dict(stored.get(key) or {})
                d["modelled"], d["absent_in_aux"] = [], []
                return AuxSideOutcome(product_id=pid, **d)

            sides = {
                reference_id: _side_from_stored(reference_id, "reference"),
                compare_id: _side_from_stored(compare_id, "compare"),
            }
            for pid in (reference_id, compare_id):
                if not aux_sample_cache_path(pid, name).exists():
                    continue
                mg = fit_map(aux_scoped_id(pid, name))
                save_map_gmm(mg)
                for cv, cg in sorted(mg.classes.items()):
                    if cg.fitted and not cg.absent:
                        sides[pid].modelled.append(cv)
                    else:
                        sides[pid].absent_in_aux.append(cv)
        finally:
            _restore_config()
        stored["reference"] = _side_dict(sides[reference_id])
        stored["compare"] = _side_dict(sides[compare_id])
        _save_aois(payload)
        report(1.0, "auxiliary GMMs fitted from cached points")
        return AuxiliaryResult(
            name=name,
            bbox=aoi,
            reference=sides[reference_id],
            compare=sides[compare_id],
            reused=False,
        )

    # Fresh sampling: targets = what the run still cannot model, ignoring this
    # auxiliary's own caches (its points are about to be replaced).
    targets_ref = (
        [a.class_value for a in still_absent_classes(reference_id, exclude_aux=name)]
        if target_side in ("both", "reference")
        else []
    )
    targets_cmp = (
        [a.class_value for a in still_absent_classes(compare_id, exclude_aux=name)]
        if target_side in ("both", "compare")
        else []
    )
    if not targets_ref and not targets_cmp:
        raise ValueError(
            "nothing to target: every declared class of the selected map(s) is "
            "already modelled by the primary or an existing auxiliary AOI."
        )

    sig = _aux_signature(reference_id, compare_id, aoi, cfg, targets_ref, targets_cmp)
    sig_hash = _signature_hash(sig)

    from harmonizer.registry.adapters._gee import ensure_initialized
    from harmonizer.registry.products import default_registry

    ensure_initialized()
    _apply_config(cfg)
    try:
        overlap: Overlap = overlap_for_products(
            [reference_id, compare_id, _EMBEDDING_ID], aoi=aoi
        )
        region = overlap.ee_geometry()
        reg = default_registry()
        embedding_adapter = reg.get(_EMBEDDING_ID).adapter_factory()

        sides = {
            reference_id: AuxSideOutcome(product_id=reference_id, targeted=targets_ref),
            compare_id: AuxSideOutcome(product_id=compare_id, targeted=targets_cmp),
        }
        label_images: dict[str, object] = {}
        label_adapters: dict[str, object] = {}
        samples: dict[str, dict[int, object]] = {reference_id: {}, compare_id: {}}

        # Which classes does this AOI actually hold? (one coarse histogram per
        # map, skipped for a side that samples nothing here.)
        from harmonizer.absence import declared_classes

        report(0.05, "checking which classes this AOI holds")
        present: dict[str, set[int]] = {}
        for pid in (reference_id, compare_id):
            label_images[pid] = _label_image_for(pid).clip(region)
            label_adapters[pid] = reg.get(pid).adapter_factory()
            need = target_side == "both" or bool(sides[pid].targeted)
            # Filtered to the legend's real classes, so a fill value observed in
            # this AOI (0 / "No Data" / "Unclassifiable") is never sampled here
            # either -- same rule as the primary run.
            present[pid] = (
                set(
                    drawable_classes(
                        pid, present_classes(label_images[pid], region)
                    )
                )
                if need
                else set()
            )
            sides[pid].found = [
                cv for cv in sides[pid].targeted if cv in present[pid]
            ]

        # The sampling plan per side. target_side="both" samples every declared
        # class present here on both maps, like a primary run (the merged table
        # keeps only the absent-class results); a single-side AOI samples just
        # its found targets, plus the other map's co-present classes below.
        plan: dict[str, list[int]] = {}
        for pid in (reference_id, compare_id):
            if target_side == "both":
                plan[pid] = sorted(set(declared_classes(pid)) & present[pid])
            else:
                plan[pid] = list(sides[pid].found)

        n_plan = sum(len(plan[pid]) for pid in plan) or 1
        done = 0
        for pid in (reference_id, compare_id):
            for cv in plan[pid]:
                report(
                    0.10 + 0.45 * (done / n_plan),
                    f"sampling class {cv} of {pid}",
                )
                samples[pid][cv] = _sample_gee_class(
                    label_images[pid], cv, overlap,
                    label_adapters[pid], embedding_adapter,
                )
                if cv not in sides[pid].found:
                    sides[pid].co_present.append(cv)
                done += 1

        if target_side != "both":
            # Co-present classes: at each side's targets' surviving points, read
            # the OTHER map's labels (a small point table over the network) and
            # sample every distinct class found there on the other map -- at its
            # own stratified locations in this AOI. Not optional: this is what
            # makes the auxiliary edges within-AOI comparisons (docs 7.3).
            # (target_side="both" already samples everything present.)
            report(0.55, "discovering co-present classes of the other map")
            co: dict[str, set[int]] = {reference_id: set(), compare_id: set()}
            for pid, other in ((reference_id, compare_id), (compare_id, reference_id)):
                coords = [
                    c for cv in sides[pid].found for c in samples[pid][cv].coords
                ]
                if not coords:
                    continue
                labels = _valid_labels(label_adapters[other].sample_labels(coords))
                co[other] |= set(labels)
            for pid in (reference_id, compare_id):
                co[pid] -= set(samples[pid])  # already sampled as a target here

            n_co = sum(len(co[pid]) for pid in co) or 1
            done = 0
            for pid in (reference_id, compare_id):
                for cv in sorted(co[pid]):
                    report(
                        0.60 + 0.30 * (done / n_co),
                        f"sampling co-present class {cv} of {pid}",
                    )
                    samples[pid][cv] = _sample_gee_class(
                        label_images[pid], cv, overlap,
                        label_adapters[pid], embedding_adapter,
                    )
                    sides[pid].co_present.append(cv)
                    done += 1

        # Persist per-AOI sample caches; fit per-AOI GMMs (Stage 3, reused)
        # unless this is a sample-only pass ("Run all" fits later from these
        # caches). Fresh points supersede any GMMs fitted from older ones.
        report(0.92, "fitting auxiliary GMMs" if fit_models else "caching sampled points")
        for pid in (reference_id, compare_id):
            if not samples[pid]:
                continue
            scoped = aux_scoped_id(pid, name)
            ms = MapSample(
                product_id=scoped,
                working_year=cfg.maps.working_year,
                floor=cfg.sampling.points_floor,
                target=cfg.sampling.points_target,
                classes=dict(sorted(samples[pid].items())),
            )
            save_map_sample(ms)
            if not fit_models:
                aux_gmm_cache_path(pid, name).unlink(missing_ok=True)
                continue
            mg = fit_map(scoped)
            save_map_gmm(mg)
            for cv, cg in sorted(mg.classes.items()):
                if cg.fitted and not cg.absent:
                    sides[pid].modelled.append(cv)
                else:
                    sides[pid].absent_in_aux.append(cv)
    finally:
        _restore_config()

    # Record the auxiliary in the AOI list.
    entry = {
        "name": name,
        "reference_id": reference_id,
        "compare_id": compare_id,
        "bbox": list(aoi),
        "target_side": target_side,
        "hash": sig_hash,
        "signature": sig,
        "reference": _side_dict(sides[reference_id]),
        "compare": _side_dict(sides[compare_id]),
    }
    payload["auxiliaries"] = [
        e for e in payload["auxiliaries"] if e["name"] != name
    ] + [entry]
    _save_aois(payload)

    report(1.0, "auxiliary AOI done")
    return AuxiliaryResult(
        name=name,
        bbox=aoi,
        reference=sides[reference_id],
        compare=sides[compare_id],
        reused=False,
    )


def _side_dict(s: AuxSideOutcome) -> dict:
    return {
        "targeted": s.targeted,
        "found": s.found,
        "co_present": s.co_present,
        "modelled": s.modelled,
        "absent_in_aux": s.absent_in_aux,
    }


# --------------------------------------------------------------------------- #
# Per-AOI sub-matrix and the merged matching table (7.4)
# --------------------------------------------------------------------------- #


def aux_affinity(
    reference_id: str, compare_id: str, aux_name: str, alpha: float | None = None
) -> AffinityResult:
    """The self-consistent affinity sub-matrix of one auxiliary AOI.

    Both sides' GMMs were fitted in that AOI, so every distance is within-AOI.
    Computed from the per-AOI caches via the scoped ids, then re-labelled with
    the base product ids so class names resolve from the registry as usual.

    ``alpha`` (Stage 8d) fuses this sub-matrix at the same semantic-prior weight
    as the primary, so a merged table does not mix fused and unfused rows. The
    prior itself resolves the scoped ids back to their base products
    (``semantics.base_product_id``), since the legend belongs to the product
    rather than to the AOI it was sampled in.
    """
    aff = compute_affinity(
        aux_scoped_id(reference_id, aux_name),
        aux_scoped_id(compare_id, aux_name),
        alpha=alpha,
    )
    aff.reference_id = reference_id
    aff.compare_id = compare_id
    return aff


def merged_matching_table(
    reference_id: str, compare_id: str, alpha: float | None = None
) -> tuple[list[MatchingRow], dict]:
    """The union of every AOI's matching-table rows, tagged with ``evidence_aoi``.

    Row order: primary rows first (``evidence_aoi="primary"``), then each active
    auxiliary's rows (its own sub-matrix, so probabilities normalise within that
    AOI), then the classes still absent from *every* AOI -- reference side as
    candidate-less rows, compare side as unmatched targets. A reference class
    can appear in more than one AOI's rows (a co-present class fitted in an
    auxiliary as well as at home); the ``evidence_aoi`` tag is what tells the
    reader the two rows rest on different ground (docs 7.4).

    A still-absent class with expert-confirmed edges (Stage 7.5) keeps those
    edges in its row with ``evidence_aoi="none"`` -- the expert, not an AOI, is
    its evidence. Returns ``(rows, info)`` where ``info`` names the auxiliaries
    that contributed.

    ``alpha`` (Stage 8d) sets the semantic-prior weight for every AOI's
    sub-matrix, so the merged deliverable is fused at the same weight the UI is
    displaying. Omitted, it is the calibrated config value. This must be
    threaded through: the merged table REPLACES the primary-only one whenever a
    pair has auxiliaries, so leaving it at the default would silently undo the
    user's alpha choice on exactly those runs.
    """
    from harmonizer.absence import covered_classes
    from harmonizer.review import cached_affinity, load_feedback

    aff = cached_affinity(reference_id, compare_id, alpha)
    decisions, _ = classify_rows(aff)
    rows = build_matching_table(aff, decisions, include_compare_absent=False)

    # Coverage accumulates AOI by AOI: the primary first, then each auxiliary
    # in the order it was added. An auxiliary contributes rows only for what
    # every AOI BEFORE it still lacked, so two auxiliaries that both hold a
    # recovered class do not both add rows for it — the earlier one is its
    # evidence. Accumulation uses each auxiliary's FITTED classes (rows can
    # only come from fits); an aux that is merely sampled blocks nothing.
    covered_so_far = {
        reference_id: set(covered_classes(reference_id)),
        compare_id: set(covered_classes(compare_id)),
    }

    aux_used: list[str] = []
    for entry in active_auxiliaries():
        aux_name = entry["name"]
        try:
            aaff = aux_affinity(reference_id, compare_id, aux_name, alpha)
        except FileNotFoundError:
            continue
        if not aaff.reference_classes or not aaff.compare_classes:
            # One side has no fitted class in this auxiliary; no within-AOI
            # comparison exists, so it contributes no rows.
            continue
        adec, _ = classify_rows(aaff)
        arows = build_matching_table(aaff, adec, include_compare_absent=False)
        # Only the ABSENT-CLASS results enter the deliverable: keep a row when
        # no earlier AOI covers its reference class (this auxiliary is the
        # class's evidence), or when it offers a candidate no earlier AOI had
        # on the compare side (the row is that recovered class's evidence).
        # Judged against accumulated coverage, not the auxiliary's stored
        # target lists -- those are a historical artifact of when it was
        # sampled and can omit a class another pass already covered, which
        # would silently drop the class's only row. The other classes are
        # sampled to make the within-AOI comparison possible (with
        # target_side="both", the whole AOI is sampled like a primary run) --
        # they must not duplicate rows already evidenced.
        ref_seen = covered_so_far[reference_id]
        cmp_seen = covered_so_far[compare_id]
        arows = [
            r
            for r in arows
            if r.reference_value not in ref_seen
            or any(cv not in cmp_seen for cv in r.compare_values)
        ]
        for r in arows:
            r.evidence_aoi = aux_name
        rows.extend(arows)
        aux_used.append(aux_name)
        # This auxiliary's fitted classes now count as seen for later ones.
        for pid in (reference_id, compare_id):
            covered_so_far[pid] |= _modelled_classes(aux_scoped_id(pid, aux_name))

    # Classes absent from every AOI. Reference side: candidate-less rows -- or
    # the expert's hand-declared edges (7.5), tagged evidence_aoi="none".
    store = load_feedback(reference_id, compare_id)
    for a in still_absent_classes(reference_id):
        confirmed = store.confirmed_edges(a.class_value)
        rows.append(
            MatchingRow(
                reference_value=a.class_value,
                reference_name=a.class_name,
                status="absent",
                compare_values=[e.compare_value for e in confirmed],
                compare_names=[
                    class_name(compare_id, e.compare_value) for e in confirmed
                ],
                probabilities=[e.retained_probability for e in confirmed],
                best_raw_similarity=float("nan"),
                margin=float("nan"),
                entropy=float("nan"),
                reference_low_confidence=True,
                compare_low_confidence=[False] * len(confirmed),
                absence_reason=a.reason,
                evidence_aoi="none" if confirmed else "",
            )
        )

    # Compare side: unmatched targets (as in Stage 7a, but net of auxiliaries).
    for a in compare_absent_classes(compare_id):
        if a.class_value in aux_modelled_classes(compare_id):
            continue
        rows.append(
            MatchingRow(
                reference_value=-1,
                reference_name="",
                status="absent",
                compare_values=[a.class_value],
                compare_names=[a.class_name],
                probabilities=[],
                best_raw_similarity=float("nan"),
                margin=float("nan"),
                entropy=float("nan"),
                reference_low_confidence=False,
                compare_low_confidence=[True],
                absence_reason=a.reason,
                side="compare",
                evidence_aoi="",
            )
        )

    return rows, {"auxiliaries": aux_used}


# --------------------------------------------------------------------------- #
# Review integration (Stage 7c -> Stage 6): auxiliary evidence in the reviewed
# table. Only rows the primary could NOT model are upgraded -- everything else,
# including every expert-confirmed edge, is left exactly as the feedback store
# says. The expert's manual fixes are never overwritten by new AOI evidence.
# --------------------------------------------------------------------------- #


def upgrade_reviewed_rows(
    reference_id: str, compare_id: str, rows: list, store=None
) -> list:
    """Swap each previously-ABSENT reviewed row an auxiliary now evidences for
    that auxiliary's proposals (first auxiliary in AOI order wins).

    Touches nothing else: rows the primary models keep their primary proposals,
    and confirmed edges stay frozen -- the upgraded rows are built through the
    same :func:`harmonizer.review.build_reviewed_table` with the same feedback
    store, so a hand-declared edge on a formerly absent class survives the
    upgrade with its retained probability, now surrounded by the auxiliary's
    algorithm proposals instead of an empty absent row.
    """
    from harmonizer.review import build_reviewed_table, load_feedback

    store = store or load_feedback(reference_id, compare_id)
    index = {r.reference_value: i for i, r in enumerate(rows)}
    done: set[int] = set()
    for entry in active_auxiliaries():
        try:
            aaff = aux_affinity(reference_id, compare_id, entry["name"])
        except FileNotFoundError:
            continue
        if not aaff.reference_classes or not aaff.compare_classes:
            continue
        adec, _ = classify_rows(aaff)
        for ar in build_reviewed_table(aaff, adec, store):
            rv = ar.reference_value
            i = index.get(rv)
            if i is None or rv in done:
                continue
            if rows[i].status != "absent":
                continue  # only classes the primary could not model
            rows[i] = ar
            done.add(rv)
    return rows


def recompute_reviewed_table_all_aois(
    reference_id: str, compare_id: str, store=None
):
    """:func:`harmonizer.review.recompute_reviewed_table` + the auxiliary
    upgrade: the reviewed table the API serves, with previously-absent classes
    carrying their auxiliary's proposals."""
    from harmonizer.review import load_feedback, recompute_reviewed_table

    store = store or load_feedback(reference_id, compare_id)
    aff, rows = recompute_reviewed_table(reference_id, compare_id, store)
    rows = upgrade_reviewed_rows(reference_id, compare_id, rows, store)
    return aff, rows


def aux_affinity_for_class(
    reference_id: str, compare_id: str, reference_value: int
) -> AffinityResult | None:
    """The first active auxiliary's sub-affinity containing this reference
    class, or ``None``. Used when confirming edges for a class only an
    auxiliary evidences, so the frozen retained probability comes from the
    sub-matrix that actually proposed the edge instead of defaulting to 0."""
    rv = int(reference_value)
    for entry in active_auxiliaries():
        try:
            aaff = aux_affinity(reference_id, compare_id, entry["name"])
        except FileNotFoundError:
            continue
        if rv in aaff.reference_classes and aaff.compare_classes:
            return aaff
    return None
