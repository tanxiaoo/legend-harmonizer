"""GMM modeling (Stage 3).

Turns each class's embedding vectors into a probability distribution, so classes
can be compared as distributions rather than point clouds (docs/PIPELINE.md,
Stage 3).

This stage **consumes** the per-map point tables cached by Stage 2
(``cache/samples_<product>.npz`` plus the JSON sidecar) as-is and **does not
resample** -- no GEE access here. It reads the full 64-dim embedding vectors
already sampled and never drops dimensions; parameter-hunger of a full 64x64
covariance is held in check by the Stage 2 point floor, the K auto-cap, and the
simpler-covariance fallback for under-floor classes below.

Per class this module:
  1. skips classes flagged ``absent`` in Stage 2 -- they are carried through with
     no GMM and handled downstream, never force-fit;
  2. caps K so each component has at least ``min_points_per_component`` points,
     recording when a class was capped;
  3. forces ``covariance_type`` to ``diag`` for any class whose point count is
     **below the floor** (``CONFIG.sampling.points_floor`` -- the same threshold
     Stage 2 used), regardless of the configured default, to avoid overfitting a
     full covariance on few points; other classes use the configured default
     (``full``);
  4. fits one ``sklearn`` ``GaussianMixture`` with the configured ``reg_covar``
     and fixed seed for reproducibility;
  5. propagates the under-floor / ``relaxed`` low-confidence marker through to the
     output so it reaches the Stage 4 matching table.

The fitted parameters (weights, means, covariances) for every class of every map
are saved to ``cache/`` for Stage 4.

Constants come from ``CONFIG.gmm`` and ``CONFIG.sampling`` (section 2), never
hardcoded here.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
from sklearn.mixture import GaussianMixture

from harmonizer.config import CONFIG

# --------------------------------------------------------------------------- #
# Result types
# --------------------------------------------------------------------------- #


@dataclass
class ClassGMM:
    """A fitted GMM (or a skipped ``absent`` class) for one class of one map.

    The component parameters are stored as plain nested lists so the whole object
    is JSON-serialisable for the cache sidecar; Stage 4 rebuilds arrays from them.
    ``weights``/``means``/``covariances`` are ``None`` for an ``absent`` class,
    which carries through unmodelled.

    ``covariances`` shape depends on ``covariance_type_used``: ``(K, D, D)`` for
    ``full`` and ``(K, D)`` for ``diag`` -- exactly what sklearn stores, so Stage 4
    can reconstruct each component's full DxD covariance the same way sklearn does.
    """

    class_value: int
    n_points: int

    # Fit outcome.
    fitted: bool = False              # False only for absent (skipped) classes
    n_components_requested: int = 0
    n_components_used: int = 0
    k_capped: bool = False            # requested K reduced to respect the floor
    covariance_type_used: str = ""    # "full" | "diag" (diag forced under-floor)

    # Propagated Stage 2 markers.
    status: str = "ok"               # "ok" | "relaxed" | "absent" (from Stage 2)
    buffer_relaxed: bool = False
    absent: bool = False
    under_floor: bool = False         # point count < CONFIG.sampling.points_floor
    low_confidence: bool = False      # under-floor or relaxed -> lower-confidence match

    # Fitted parameters (None for absent classes).
    weights: list[float] | None = None
    means: list[list[float]] | None = None
    covariances: list | None = None


@dataclass
class MapGMM:
    """All fitted class GMMs for one map, with the config used to fit them."""

    product_id: str
    working_year: int
    floor: int
    n_components_requested: int
    covariance_type_default: str
    classes: dict[int, ClassGMM] = field(default_factory=dict)


# --------------------------------------------------------------------------- #
# Loading Stage 2's cached samples
# --------------------------------------------------------------------------- #


def samples_cache_path(product_id: str) -> Path:
    """Path to the Stage 2 ``.npz`` sample table for a product."""
    return CONFIG.cache_dir / f"samples_{product_id}.npz"


@dataclass
class _LoadedSamples:
    product_id: str
    working_year: int
    floor: int
    target: int
    # Per class value: stacked embeddings and the Stage 2 sidecar bookkeeping.
    embeddings: dict[int, np.ndarray]
    meta: dict[int, dict]


def load_samples(product_id: str) -> _LoadedSamples:
    """Load a Stage 2 sample table (``.npz`` + JSON sidecar) for one product.

    Returns the embedding matrix per class and that class's Stage 2 bookkeeping
    (status, flags, counts). No resampling and no GEE access -- Stage 3 reads what
    Stage 2 wrote (docs/PIPELINE.md, Stage 3 -> Inputs).
    """
    npz_path = samples_cache_path(product_id)
    if not npz_path.exists():
        raise FileNotFoundError(
            f"Stage 2 sample cache not found for {product_id!r}: {npz_path}. "
            "Run scripts/verify_stage2.py first."
        )
    sidecar = npz_path.with_suffix(".json")
    summary = json.loads(sidecar.read_text(encoding="utf-8"))

    with np.load(npz_path) as data:
        emb_all = np.asarray(data["embeddings"], dtype=float)
        class_values = np.asarray(data["class_values"], dtype=int)

    embeddings: dict[int, np.ndarray] = {}
    meta: dict[int, dict] = {}
    for cv_str, m in summary["classes"].items():
        cv = int(cv_str)
        meta[cv] = m
        embeddings[cv] = emb_all[class_values == cv]

    return _LoadedSamples(
        product_id=summary["product_id"],
        working_year=int(summary["working_year"]),
        floor=int(summary["floor"]),
        target=int(summary["target"]),
        embeddings=embeddings,
        meta=meta,
    )


# --------------------------------------------------------------------------- #
# Per-class fitting
# --------------------------------------------------------------------------- #


def _cap_k(requested_k: int, n_points: int) -> tuple[int, bool]:
    """Cap K so each component has >= min_points_per_component points.

    Returns the K actually usable and whether it was reduced. Never goes below 1
    (a class that reaches fitting has at least one point).
    """
    min_per = CONFIG.gmm.min_points_per_component
    max_k = max(1, n_points // min_per)
    used = min(requested_k, max_k)
    return used, used < requested_k


def fit_class(
    class_value: int,
    embeddings: np.ndarray,
    meta: dict,
) -> ClassGMM:
    """Fit one class's GMM, applying the Stage 3 rules.

    ``meta`` is that class's Stage 2 sidecar entry (status/flags/counts). Absent
    classes are skipped; under-floor classes are forced to diagonal covariance and
    marked low-confidence; K is auto-capped for every fitted class.
    """
    n_points = int(embeddings.shape[0])
    status = str(meta.get("status", "ok"))
    buffer_relaxed = bool(meta.get("buffer_relaxed", False))
    absent = bool(meta.get("absent", False))
    floor = CONFIG.sampling.points_floor
    under_floor = n_points < floor

    result = ClassGMM(
        class_value=int(class_value),
        n_points=n_points,
        status=status,
        buffer_relaxed=buffer_relaxed,
        absent=absent,
        under_floor=under_floor,
        # Low-confidence marker reaching Stage 4: under-floor (by count) OR the
        # relaxed 1-pixel-buffer flag from Stage 2.
        low_confidence=under_floor or buffer_relaxed,
        n_components_requested=CONFIG.gmm.n_components,
    )

    # Absent classes are not force-fit; they carry through with no GMM.
    if absent or n_points == 0:
        result.fitted = False
        result.absent = True
        result.status = "absent"
        return result

    used_k, capped = _cap_k(CONFIG.gmm.n_components, n_points)

    # Simpler covariance for under-floor classes, regardless of the configured
    # default, to avoid overfitting a full 64x64 covariance on few points.
    cov_type = "diag" if under_floor else CONFIG.gmm.covariance_type

    gmm = GaussianMixture(
        n_components=used_k,
        covariance_type=cov_type,
        reg_covar=CONFIG.gmm.reg_covar,
        random_state=CONFIG.gmm.random_seed,
    )
    gmm.fit(embeddings)

    result.fitted = True
    result.n_components_used = used_k
    result.k_capped = capped
    result.covariance_type_used = cov_type
    result.weights = gmm.weights_.tolist()
    result.means = gmm.means_.tolist()
    result.covariances = gmm.covariances_.tolist()
    return result


# --------------------------------------------------------------------------- #
# Whole-map fitting
# --------------------------------------------------------------------------- #


def fit_map(product_id: str) -> MapGMM:
    """Fit a GMM per class for one map, from its Stage 2 sample cache."""
    loaded = load_samples(product_id)

    mg = MapGMM(
        product_id=loaded.product_id,
        working_year=loaded.working_year,
        floor=loaded.floor,
        n_components_requested=CONFIG.gmm.n_components,
        covariance_type_default=CONFIG.gmm.covariance_type,
    )
    for cv in sorted(loaded.embeddings):
        mg.classes[cv] = fit_class(cv, loaded.embeddings[cv], loaded.meta[cv])
    return mg


# --------------------------------------------------------------------------- #
# Persistence
# --------------------------------------------------------------------------- #


def gmm_cache_path(product_id: str) -> Path:
    return CONFIG.cache_dir / f"gmm_{product_id}.json"


def save_map_gmm(mg: MapGMM) -> Path:
    """Persist a MapGMM to ``cache/`` as JSON.

    Component parameters are stored as nested lists (see ``ClassGMM``) so Stage 4
    can reload weights/means/covariances directly. The point count, capped-K flag,
    covariance type actually used, and the low-confidence / ``relaxed`` marker are
    all recorded per class (docs/PIPELINE.md, Stage 3 -> Output artifact).
    """
    CONFIG.cache_dir.mkdir(parents=True, exist_ok=True)
    path = gmm_cache_path(mg.product_id)

    payload = {
        "product_id": mg.product_id,
        "working_year": mg.working_year,
        "floor": mg.floor,
        "n_components_requested": mg.n_components_requested,
        "covariance_type_default": mg.covariance_type_default,
        "reg_covar": CONFIG.gmm.reg_covar,
        "min_points_per_component": CONFIG.gmm.min_points_per_component,
        "random_seed": CONFIG.gmm.random_seed,
        "classes": {str(cv): asdict(cs) for cv, cs in mg.classes.items()},
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path
