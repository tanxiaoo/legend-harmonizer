"""Affinity matrix via Mixture-Wasserstein distance (Stage 4, part 1).

Compares every reference class to every compare class as *distributions*, using
the closed-form 2-Wasserstein (Bures-Wasserstein) distance between Gaussians as
the ground cost and a Mixture-Wasserstein (MW2) optimal transport between the two
GMMs' component-weight vectors on top of it (docs/PIPELINE.md, Stage 4).

Pipeline of this module:
  1. Load the fitted GMMs both maps produced in Stage 3 (``cache/gmm_<id>.json``).
  2. Reconstruct every component's covariance to a full DxD matrix, branching on
     ``covariance_type_used`` -- a ``diag`` fit is expanded with ``np.diag`` and
     treated as full internally, so diagonal-vs-full and diagonal-vs-diagonal
     pairs compute correctly instead of raising on the shape mismatch.
  3. Per class pair, compute the component-wise squared Bures-Wasserstein cost
     matrix, then the MW2 distance as the optimal transport between the two weight
     vectors under that cost.
  4. Turn each distance into a similarity ``s = 1 / (1 + d)``, forming the M x N
     raw similarity matrix (reference rows by compare columns).
  5. Row-normalise so each reference row is a probability distribution, and
     compute each row's normalised mapping entropy.

The single combination method (MW2 optimal transport) is applied consistently
across every class pair; it reduces to the plain Bures-Wasserstein distance when
both GMMs have a single component, which is the sanity check. ``absent`` classes
carry no GMM and are excluded from the matrix.

``scipy.stats.wasserstein_distance`` is deliberately **not** used: it is
one-dimensional only and cannot compare 64-dim distributions.

The ``s = 1 / (1 + d)`` form follows section 2; no tunables are hardcoded here.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from harmonizer.config import CONFIG
from harmonizer.modeling import gmm_cache_path

# --------------------------------------------------------------------------- #
# Readable class names (for the matching table / verification only)
# --------------------------------------------------------------------------- #
# The distance maths never touches these; they exist so the emitted CSVs and the
# verification read as "water -> water" rather than "80 -> 0". Names come from the
# product registry (the per-map YAML legend, docs/PIPELINE.md section 2.5) -- the
# single source of truth -- not from a dict hardcoded here.

from harmonizer.registry.legends import class_name  # noqa: E402,F401  (re-export)


# --------------------------------------------------------------------------- #
# Loaded GMM component blocks
# --------------------------------------------------------------------------- #


@dataclass
class ClassComponents:
    """A fitted class's GMM as arrays, with every covariance expanded to full DxD.

    ``weights`` is ``(K,)``, ``means`` is ``(K, D)``, ``covariances`` is
    ``(K, D, D)`` -- diagonal fits are expanded here so downstream code never
    branches on covariance shape again.
    """

    product_id: str
    class_value: int
    weights: np.ndarray
    means: np.ndarray
    covariances: np.ndarray  # (K, D, D), full
    low_confidence: bool
    covariance_type_used: str
    n_points: int
    status: str


@dataclass
class LoadedMapGMM:
    product_id: str
    working_year: int
    # Only classes that were actually fitted (absent classes excluded).
    classes: dict[int, ClassComponents] = field(default_factory=dict)
    # Classes present in the map but carried through unmodelled (absent/skipped).
    absent_classes: dict[int, dict] = field(default_factory=dict)


def _expand_covariances(cov: list, covariance_type_used: str) -> np.ndarray:
    """Return covariances as ``(K, D, D)`` full matrices.

    ``full`` fits are already ``(K, D, D)``; ``diag`` fits are ``(K, D)`` and are
    expanded with ``np.diag`` so a diagonal covariance is treated as full
    internally (docs/PIPELINE.md, Stage 4 -> Mixed covariance types).
    """
    arr = np.asarray(cov, dtype=float)
    if covariance_type_used == "diag":
        return np.stack([np.diag(row) for row in arr], axis=0)  # (K, D) -> (K, D, D)
    if covariance_type_used == "full":
        return arr
    raise ValueError(
        f"Unknown covariance_type_used {covariance_type_used!r}; "
        "expected 'full' or 'diag'."
    )


def load_map_gmm(product_id: str) -> LoadedMapGMM:
    """Load a Stage 3 ``gmm_<product>.json`` cache, expanding covariances to full."""
    path = gmm_cache_path(product_id)
    if not path.exists():
        raise FileNotFoundError(
            f"Stage 3 GMM cache not found for {product_id!r}: {path}. "
            "Run scripts/verify_stage3.py first."
        )
    payload = json.loads(path.read_text(encoding="utf-8"))

    loaded = LoadedMapGMM(
        product_id=payload["product_id"],
        working_year=int(payload["working_year"]),
    )
    for cv_str, cs in payload["classes"].items():
        cv = int(cv_str)
        if not cs.get("fitted", False) or cs.get("absent", False):
            loaded.absent_classes[cv] = cs
            continue
        loaded.classes[cv] = ClassComponents(
            product_id=product_id,
            class_value=cv,
            weights=np.asarray(cs["weights"], dtype=float),
            means=np.asarray(cs["means"], dtype=float),
            covariances=_expand_covariances(
                cs["covariances"], cs["covariance_type_used"]
            ),
            low_confidence=bool(cs.get("low_confidence", False)),
            covariance_type_used=str(cs["covariance_type_used"]),
            n_points=int(cs["n_points"]),
            status=str(cs.get("status", "ok")),
        )
    return loaded


# --------------------------------------------------------------------------- #
# Bures-Wasserstein between single Gaussians
# --------------------------------------------------------------------------- #


def _sqrtm_psd(mat: np.ndarray) -> np.ndarray:
    """Symmetric PSD matrix square root via eigendecomposition.

    Robust and fast for the symmetric PSD covariances here, and it never leaves a
    complex part from round-off the way a general ``scipy.linalg.sqrtm`` can.
    Negative eigenvalues from numerical noise are clipped to 0.
    """
    mat = 0.5 * (mat + mat.T)
    vals, vecs = np.linalg.eigh(mat)
    vals = np.clip(vals, 0.0, None)
    return (vecs * np.sqrt(vals)) @ vecs.T


def bures_wasserstein_sq(
    m1: np.ndarray,
    cov1: np.ndarray,
    m2: np.ndarray,
    cov2: np.ndarray,
) -> float:
    """Squared 2-Wasserstein (Bures-Wasserstein) distance between two Gaussians.

    ``W2^2 = ||m1 - m2||^2 + Tr(cov1 + cov2 - 2 (cov1^1/2 cov2 cov1^1/2)^1/2)``
    (docs/PIPELINE.md, Stage 4). Both covariances are full DxD here (diagonal
    fits were expanded on load). The result is clipped at 0 so numerical
    round-off cannot produce a tiny negative squared distance.
    """
    mean_term = float(np.sum((m1 - m2) ** 2))

    s1 = _sqrtm_psd(cov1)
    inner = s1 @ cov2 @ s1
    inner_sqrt = _sqrtm_psd(inner)
    cov_term = float(np.trace(cov1) + np.trace(cov2) - 2.0 * np.trace(inner_sqrt))

    return max(mean_term + cov_term, 0.0)


# --------------------------------------------------------------------------- #
# Mixture-Wasserstein (MW2) between two GMMs
# --------------------------------------------------------------------------- #


def _component_cost_matrix(a: ClassComponents, b: ClassComponents) -> np.ndarray:
    """Squared Bures-Wasserstein cost between every component of ``a`` and ``b``.

    Returns a ``(Ka, Kb)`` matrix used as the OT ground cost.
    """
    ka = a.weights.shape[0]
    kb = b.weights.shape[0]
    cost = np.empty((ka, kb), dtype=float)
    for i in range(ka):
        for j in range(kb):
            cost[i, j] = bures_wasserstein_sq(
                a.means[i], a.covariances[i], b.means[j], b.covariances[j]
            )
    return cost


def _solve_transport(wa: np.ndarray, wb: np.ndarray, cost: np.ndarray) -> float:
    """Exact optimal transport cost ``min_T <T, cost>`` between two weight vectors.

    Balanced transportation problem: ``T >= 0`` with row marginals ``wa`` and
    column marginals ``wb`` (both sum to 1). Solved exactly as a min-cost max-flow
    on a bipartite network -- sources are ``a``'s components (supply ``wa``),
    sinks are ``b``'s components (demand ``wb``), the source->sink edges carry the
    Bures-Wasserstein ground cost -- with successive shortest augmenting paths
    (Bellman-Ford / SPFA to allow the zero-cost marginal edges). Total supply
    equals total demand, so the max flow saturates every marginal and its min cost
    is the exact OT value.

    For the modest component counts here (K up to ~10 per side) this is exact and
    fast. It is *the* single combination method used for every class pair.
    """
    ka = len(wa)
    kb = len(wb)
    n = ka + kb + 2
    src = ka + kb
    snk = ka + kb + 1

    # Each edge: [to, capacity, cost, index_of_reverse_edge_in_graph[to]].
    graph: list[list[list[float]]] = [[] for _ in range(n)]

    def add_edge(u: int, v: int, cap: float, cst: float) -> None:
        graph[u].append([v, cap, cst, len(graph[v])])
        graph[v].append([u, 0.0, -cst, len(graph[u]) - 1])

    for i in range(ka):
        add_edge(src, i, float(wa[i]), 0.0)
    for j in range(kb):
        add_edge(ka + j, snk, float(wb[j]), 0.0)
    for i in range(ka):
        for j in range(kb):
            add_edge(i, ka + j, float("inf"), float(cost[i, j]))

    INF = float("inf")
    total_cost = 0.0
    while True:
        # SPFA (queue-based Bellman-Ford) for a min-cost augmenting path src->snk.
        dist = [INF] * n
        in_queue = [False] * n
        prev_v = [-1] * n
        prev_e = [-1] * n
        dist[src] = 0.0
        queue = [src]
        in_queue[src] = True
        while queue:
            u = queue.pop(0)
            in_queue[u] = False
            du = dist[u]
            for ei, edge in enumerate(graph[u]):
                v, cap, cst, _ = edge
                if cap > 1e-15 and du + cst < dist[v] - 1e-15:
                    dist[v] = du + cst
                    prev_v[v] = u
                    prev_e[v] = ei
                    if not in_queue[v]:
                        queue.append(v)
                        in_queue[v] = True
        if dist[snk] == INF:
            break  # no more augmenting paths -> all marginals saturated

        # Bottleneck capacity along the path.
        f = INF
        v = snk
        while v != src:
            u = prev_v[v]
            f = min(f, graph[u][prev_e[v]][1])
            v = u
        # Push flow and update residuals.
        v = snk
        while v != src:
            u = prev_v[v]
            edge = graph[u][prev_e[v]]
            edge[1] -= f
            graph[v][edge[3]][1] += f
            v = u
        total_cost += f * dist[snk]

    return total_cost


def mixture_wasserstein(a: ClassComponents, b: ClassComponents) -> float:
    """MW2 distance between two GMMs.

    Ground cost between components is the squared Bures-Wasserstein distance; the
    mixture distance is the exact optimal transport between the two component
    weight vectors under that cost (docs/PIPELINE.md, Stage 4 -> GMM-to-GMM
    distance). Returned as a (non-squared) distance ``d = sqrt(MW2^2)`` so it is
    on a metric scale before ``s = 1 / (1 + d)``.

    Reduces to ``sqrt(bures_wasserstein_sq)`` when both GMMs have a single
    component: the weights ``[1.0]`` transport all mass through the one cost cell.
    """
    cost = _component_cost_matrix(a, b)
    wa = a.weights / a.weights.sum()  # defensive; sklearn weights already sum to 1
    wb = b.weights / b.weights.sum()
    mw2_sq = _solve_transport(wa, wb, cost)
    return float(np.sqrt(max(mw2_sq, 0.0)))


# --------------------------------------------------------------------------- #
# Full affinity matrix
# --------------------------------------------------------------------------- #


@dataclass
class AffinityResult:
    """The Stage 4 affinity computation, before decision/classification.

    ``raw_similarity`` is the M x N grid of ``s = 1 / (1 + d)`` (reference rows,
    compare columns) -- kept unchanged as the absolute-match signal for the orphan
    floor. ``normalized_affinity`` is the per-row **softmax over negative
    distances** ``P_ij = exp(-d_ij / T) / sum_k exp(-d_ik / T)`` (not a linear
    normalisation of the raw similarities), so each reference row is a probability
    distribution sharp enough for entropy to separate ``strong`` from ``mixed``.
    ``row_entropy`` is the per-row normalised mapping entropy of that softmax
    distribution in [0, 1]. ``distance`` keeps the underlying MW2 distances.

    **Semantic prior (Stage 8b).** ``semantic_prior`` is the M x N directed prior
    from ``harmonizer.semantics`` and ``alpha`` the exponent it enters the softmax
    with; ``normalized_affinity`` is the **fused** result. ``normalized_affinity_aef``
    keeps the unfused (alpha = 0) row-softmax so the Stage 8c disagreement report can
    put observational and fused tables side by side without a second run.
    ``semantic_orphan`` marks rows whose best prior falls below
    ``semantic_orphan_floor`` -- reported alongside, never replacing, the
    observational orphan status, which stays driven by the raw-similarity floor.
    """

    reference_id: str
    compare_id: str
    reference_classes: list[int]
    compare_classes: list[int]
    distance: np.ndarray             # (M, N) MW2 distances
    raw_similarity: np.ndarray       # (M, N) s = 1/(1+d)
    normalized_affinity: np.ndarray  # (M, N) row-normalised (fused when alpha > 0)
    row_entropy: np.ndarray          # (M,) normalised entropy in [0, 1]
    reference_low_confidence: dict[int, bool] = field(default_factory=dict)
    compare_low_confidence: dict[int, bool] = field(default_factory=dict)
    # -- Stage 8b: semantic prior ------------------------------------------- #
    semantic_prior: np.ndarray | None = None       # (M, N) directed prior pi
    alpha: float = 0.0                             # exponent on the prior
    semantic_orphan: np.ndarray | None = None      # (M,) bool
    normalized_affinity_aef: np.ndarray | None = None  # (M, N) the alpha = 0 rows


def _normalised_entropy(prob_row: np.ndarray) -> float:
    """Shannon entropy of a probability row, normalised by ``log(N)`` into [0, 1].

    With a single compare class (N == 1) entropy is undefined (``log 1 == 0``);
    return 0.0 -- a one-column row is maximally focused, not maximally spread.
    """
    n = prob_row.shape[0]
    if n <= 1:
        return 0.0
    p = prob_row[prob_row > 0.0]
    ent = -float(np.sum(p * np.log(p)))
    return ent / float(np.log(n))


def _softmax_rows(
    distance: np.ndarray,
    temperature: float,
    *,
    prior: np.ndarray | None = None,
    alpha: float = 0.0,
) -> np.ndarray:
    """Per-row softmax over negative distances: ``exp(-d/T) / sum_k exp(-d/T)``.

    Replaces linear normalisation of the raw similarities, which flattened every
    row to near-uniform because ``s = 1/(1+d)`` compresses the distance range.
    The max is subtracted per row for numerical stability (it cancels in the
    ratio). ``temperature`` must be > 0; smaller sharpens the winner.

    With a ``prior`` and ``alpha > 0`` the logits become the Stage 8 fused form
    ``-d_ij / T + alpha * log pi_ij``: the semantic prior enters as a *power*
    prior, so alpha scales how much a definition mismatch may move a row.
    ``alpha = 0`` drops the term entirely and reproduces the observational
    behaviour bit-for-bit, which is what makes the Stage 8b regression exact.
    """
    if temperature <= 0:
        raise ValueError(f"softmax temperature must be > 0, got {temperature!r}")
    logits = -distance / temperature
    if prior is not None and alpha:
        # Clip before the log: pi is in (0, 1] by construction, but a zero would
        # send the logit to -inf and silently annihilate the row's other terms.
        eps = CONFIG.affinity.semantic_prior_epsilon
        logits = logits + alpha * np.log(np.clip(prior, eps, None))
    logits -= logits.max(axis=1, keepdims=True)
    exp = np.exp(logits)
    return exp / exp.sum(axis=1, keepdims=True)


def compute_affinity(
    reference_id: str,
    compare_id: str,
    *,
    temperature: float | None = None,
    alpha: float | None = None,
) -> AffinityResult:
    """Compute the raw and normalised affinity matrices for a map pair.

    Reference classes index the rows, compare classes the columns. Only fitted
    (non-absent) classes on each side enter the matrix. ``raw_similarity`` stays
    ``s = 1/(1+d)`` (the orphan-floor signal); ``normalized_affinity`` is the
    per-row softmax over negative distances with ``temperature`` (default
    ``CONFIG.affinity.softmax_temperature``).

    ``alpha`` (default ``CONFIG.affinity.semantic_prior_alpha``) weights the
    Stage 8 semantic prior in the fused logits ``-d/T + alpha * log pi``. At
    ``alpha = 0`` the prior is computed and reported but does not affect
    ``normalized_affinity``. The unfused rows are always kept in
    ``normalized_affinity_aef`` for the disagreement report.
    """
    if temperature is None:
        temperature = CONFIG.affinity.softmax_temperature
    if alpha is None:
        alpha = CONFIG.affinity.semantic_prior_alpha

    ref = load_map_gmm(reference_id)
    cmp = load_map_gmm(compare_id)

    ref_classes = sorted(ref.classes)
    cmp_classes = sorted(cmp.classes)
    m, n = len(ref_classes), len(cmp_classes)

    distance = np.empty((m, n), dtype=float)
    for i, rc in enumerate(ref_classes):
        for j, cc in enumerate(cmp_classes):
            distance[i, j] = mixture_wasserstein(ref.classes[rc], cmp.classes[cc])

    raw_similarity = 1.0 / (1.0 + distance)

    # Stage 8: the directed semantic prior over the same class ordering, so it
    # aligns with `distance` cell for cell. An unencoded pair yields ones (and a
    # warning), which leaves the fused logits identical to the unfused ones.
    from harmonizer.semantics import semantic_prior as _semantic_prior
    from harmonizer.semantics import semantic_orphans as _semantic_orphans

    prior = _semantic_prior(reference_id, compare_id, ref_classes, cmp_classes)

    # Row probabilities: temperature-scaled softmax over negative distances (not a
    # linear normalisation of raw similarities, which flattens the rows), fused
    # with the semantic prior when alpha > 0.
    if n == 0:
        normalized = np.zeros((m, 0), dtype=float)
        normalized_aef = np.zeros((m, 0), dtype=float)
    else:
        normalized = _softmax_rows(distance, temperature, prior=prior, alpha=alpha)
        # The alpha = 0 rows, kept so the disagreement report needs no second run.
        normalized_aef = (
            normalized if not alpha else _softmax_rows(distance, temperature)
        )

    row_entropy = np.array(
        [_normalised_entropy(normalized[i]) for i in range(m)], dtype=float
    )

    return AffinityResult(
        reference_id=reference_id,
        compare_id=compare_id,
        reference_classes=ref_classes,
        compare_classes=cmp_classes,
        distance=distance,
        raw_similarity=raw_similarity,
        normalized_affinity=normalized,
        row_entropy=row_entropy,
        semantic_prior=prior,
        alpha=float(alpha),
        semantic_orphan=_semantic_orphans(prior),
        normalized_affinity_aef=normalized_aef,
        reference_low_confidence={
            cv: ref.classes[cv].low_confidence for cv in ref_classes
        },
        compare_low_confidence={
            cv: cmp.classes[cv].low_confidence for cv in cmp_classes
        },
    )
