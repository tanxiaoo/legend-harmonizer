"""Central configuration for the legend-harmonizer pipeline.

This module is the single in-code home for the constants defined in
`docs/PIPELINE.md` section 2 ("Decisions and default constants"). Stages must
read values from here rather than hardcoding them. Values marked ``(tune)`` in
the pipeline document are starting points to calibrate against real data.

Nothing in this module does work; it only holds configuration.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"
CACHE_DIR = REPO_ROOT / "cache"


# --------------------------------------------------------------------------- #
# Maps and embedding
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class MapsConfig:
    """Cross-map, run-level map settings only.

    Per-map facts -- asset ids, bands, class values, footprints, resolutions, and
    legends -- have moved to the **product registry** (one YAML per map under
    ``harmonizer/registry/products/``, docs/PIPELINE.md section 2.5), the single
    source of truth. This class keeps only settings that are shared across maps or
    belong to the *run* rather than to any one map.
    """

    # Where a local reference raster (HRLC) is dropped. The specific tile/path is a
    # per-map fact in the registry; this is only the default search directory.
    hrlc_data_dir: Path = DATA_DIR

    # Embedding dimensionality (AlphaEarth), used to reshape sampled vectors. The
    # embedding source's asset id lives in the registry (alphaearth.yaml).
    embedding_dims: int = 64

    # Google Cloud project that Earth Engine runs under. Required by ee.Initialize
    # for current GEE accounts. Override via the EE_PROJECT environment variable
    # if your project id differs.
    gee_project: str = "legend-harmonization"

    # Working year 2021 for the test swap: WorldCover v200 is a 2021 product, so
    # the Dynamic World composite and AlphaEarth sampling match that year. This is
    # a run parameter (which year to sample), not a per-map fact.
    working_year: int = 2021
    target_crs: str = "EPSG:4326"


# --------------------------------------------------------------------------- #
# Sampling
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class SamplingConfig:
    points_floor: int = 300          # minimum points per class
    points_target: int = 1500        # target points per class (tune)
    # Spatial declustering (starting values, (tune)):
    #   min_spacing_m  - minimum spacing between points within a class, in metres.
    #   grid_cell_deg  - side of the grid cells the overlap is divided into; points
    #                    are drawn per cell so one large patch cannot dominate a
    #                    class's sample.
    min_spacing_m: float = 100.0     # (tune)
    grid_cell_deg: float = 0.25      # (tune)
    # Raster scale (metres) the masks, erosion, and stratified sampling run at.
    # The MVP label maps are 10 m native, so 10.0 is the faithful value. It is
    # exposed here (rather than hardcoded) so a test run can trade spatial
    # fidelity for speed by sampling/eroding at a coarser scale, e.g. 30-100 m,
    # without touching the point floor/target.
    sample_scale_m: float = 10.0


# --------------------------------------------------------------------------- #
# Buffering
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class BufferingConfig:
    erode_pixels: int = 2            # erode class mask inward before sampling
    relaxed_erode_pixels: int = 1    # relaxed buffer if a class is starved
    homogeneous_window: int = 3      # require a homogeneous NxN neighbourhood


# --------------------------------------------------------------------------- #
# GMM
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class GMMConfig:
    n_components: int = 4            # default K per class, user-settable (1-10)
    covariance_type: str = "full"    # under-floor classes fall back to "diag" (Stage 3)
    min_points_per_component: int = 50   # auto-cap K to respect this floor
    reg_covar: float = 1e-6          # added to covariance diagonals to avoid singularity (tune)
    random_seed: int = 42


# --------------------------------------------------------------------------- #
# Affinity and decision
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class AffinityConfig:
    # Distance: closed-form Bures-Wasserstein between Gaussians, combined across
    # GMM components. Raw similarity from distance: s = 1 / (1 + d) -- kept as the
    # absolute-match signal for the orphan floor.
    #
    # Row probabilities use a temperature-scaled softmax over negative distances,
    # P_ij = exp(-d_ij / T) / sum_k exp(-d_ik / T), NOT linear normalisation of the
    # raw similarities. The reason: s = 1/(1+d) compresses the distance range into
    # a narrow band (~0.47-0.91 on the first real matrix), so linear
    # row-normalisation flattens every row to near-uniform (entropy ~0.99) and
    # entropy can no longer separate `strong` from `mixed`. The softmax temperature
    # sharpens the winner; smaller T -> sharper (tune).
    softmax_temperature: float = 0.25       # T in the row-softmax (tune)
    # strong/mixed split is the top1-top2 margin of the normalised (softmax) row,
    # NOT entropy. Entropy penalises tail-spread and inverts on two-way ties: a
    # clean single winner with a long thin tail can score higher entropy than a
    # genuine two-way split, so no entropy threshold separates them. The margin
    # (gap between the best and second-best probability) reads a clean winner as
    # large and a two-way tie as small, which is exactly the strong/mixed line.
    # margin >= margin_threshold -> strong, else mixed (orphan still comes from the
    # absolute-affinity floor on the raw similarity).
    margin_threshold: float = 0.18          # top1-top2 margin (tune)
    # Entropy is still computed and reported per row, just not used to classify.
    high_entropy_threshold: float = 0.65     # reported only; no longer the classifier (tune)
    # Absolute-affinity floor on the row's best raw similarity; below it the
    # class is an orphan. Calibrated against the observed raw-similarity
    # distribution from the first Stage 4 run (bunched 0.47-0.91; 0.60 sits below
    # every genuine best match and above the weakest, non-matching rows).
    absolute_affinity_floor: float | None = 0.60  # (tune)


# --------------------------------------------------------------------------- #
# Review (Stage 6 -- human-in-the-loop evidence explorer)
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class ReviewConfig:
    """Constants for the Stage 6 evidence explorer (docs/PIPELINE.md section 6.3).

    The evidence explorer samples a few representative locations per class-pair and
    shows a fixed patch at each. These are the tunable defaults for that sampling;
    the spatial declustering itself reuses the Stage 2 ``min_spacing_m`` /
    ``grid_cell_deg`` so evidence points are spread the same way sample points are.
    """

    # Representative locations sampled per class-pair (docs section 6.3: "N = 9-12
    # -- few enough to review quickly, enough to be representative"). (tune)
    patches_per_pair: int = 10
    # Fixed patch window shown at each location, in pixels of the label map, so the
    # amount of ground context is consistent regardless of location (docs 6.3:
    # "256 x 256 px ~= 2.5 km at 10 m"). The metres-per-pixel is the label map's own
    # resolution; the window in metres is patch_window_px * that resolution. (tune)
    patch_window_px: int = 256
    # LIVE-path oversample factor: the server-side draw asks for
    # ``ceil(live_oversample * n)`` candidates so that, after declustering and the
    # native-resolution exactness filter drop boundary points, ~n locations still
    # survive. Raise it when scarce/fragmented class combinations come back short
    # of the requested n (costs a larger point table in the same round trip).
    # User-overridable per query from the Review UI. (tune)
    live_oversample: float = 2.0
    # Scale (metres) the candidate LOCATIONS are *found* at. The explorer only needs
    # to locate a handful of representative places, not sample precisely, so the
    # ``stratifiedSample`` draw runs at this coarse scale (fast) while each location's
    # reported label is still read back at the map's native resolution (exact). This
    # is the dominant cost lever for "Find evidence": at 10 m the draw scans every
    # pixel of a class across the AOI; 100 m is ~100x fewer pixels. (tune)
    draw_scale_m: float = 100.0


# --------------------------------------------------------------------------- #
# Absence (Stage 7 -- multi-AOI absence handling)
# --------------------------------------------------------------------------- #

# Reasons a class carries status "absent" (docs/PIPELINE.md, section 2 ->
# Multi-AOI absence handling). Distinct from the status itself: the status says
# the class could not be modelled, the reason says why, which decides whether an
# auxiliary AOI could help.
#   not_in_aoi - declared in the registry legend but no pixels observed in any of
#                the run's AOIs. An auxiliary AOI covering the class would help.
#   too_rare   - observed, but below the point floor even before eroding (the
#                Stage 2 absent-vs-buffered-away rule). A bigger AOI might help;
#                relaxing the buffer would not.
REASON_NOT_IN_AOI = "not_in_aoi"
REASON_TOO_RARE = "too_rare"


@dataclass(frozen=True)
class AbsenceConfig:
    """Constants for Stage 7 multi-AOI absence handling (docs/PIPELINE.md).

    There is deliberately **no absence-check scale**: absence is read from the
    Stage 2/3 caches, not from a separate Earth Engine query. Sampling is itself
    the observation (see ``harmonizer.absence``), so the check costs nothing and
    needs no scale of its own.
    """

    # Cap on auxiliary AOIs per run, so a run cannot fan out into an unbounded
    # number of GEE sampling passes. (tune)
    max_auxiliary_aois: int = 3


# --------------------------------------------------------------------------- #
# Aggregate configuration
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Config:
    maps: MapsConfig = field(default_factory=MapsConfig)
    sampling: SamplingConfig = field(default_factory=SamplingConfig)
    buffering: BufferingConfig = field(default_factory=BufferingConfig)
    gmm: GMMConfig = field(default_factory=GMMConfig)
    affinity: AffinityConfig = field(default_factory=AffinityConfig)
    review: ReviewConfig = field(default_factory=ReviewConfig)
    absence: AbsenceConfig = field(default_factory=AbsenceConfig)

    data_dir: Path = DATA_DIR
    cache_dir: Path = CACHE_DIR


# Default singleton other modules import.
CONFIG = Config()
