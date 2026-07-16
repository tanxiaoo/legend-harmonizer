"""Stage 3 verification: GMM modeling.

Fits one Gaussian Mixture Model per class for both MVP label maps -- the
test-swap reference (ESA WorldCover) and the compare map (Dynamic World) -- from
the point tables Stage 2 cached (``cache/samples_<product>.npz`` + JSON sidecar).
No resampling and no GEE access here; this reads what Stage 2 wrote.

For each class it prints: the number of components actually used, the point
count, the covariance type used, whether K was capped, and the propagated
low-confidence marker. The check (docs/PIPELINE.md, Stage 3 -> Verification) is:

  * every under-floor class (point count < floor) was fit with the simpler
    (diagonal) covariance and carries the low-confidence / ``relaxed`` flag through
    to the output;
  * ``absent`` classes were skipped rather than force-fit (no GMM);
  * K-capping only happened where expected (points // min_per_component < K).

Fitted GMMs for both maps are written to ``cache/gmm_<product>.json`` for Stage 4.

Run:  python scripts/verify_stage3.py

Stage 2 must have run first (its sample caches must exist). This stage is pure
CPU -- no network.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from harmonizer.config import CONFIG
from harmonizer.modeling import MapGMM, fit_map, save_map_gmm

# Same two MVP label products Stage 2 sampled.
PRODUCTS = ["worldcover", "dynamicworld"]


def _print_map(mg: MapGMM) -> bool:
    """Print the per-class fit summary and return whether all checks passed."""
    floor = mg.floor
    min_per = CONFIG.gmm.min_points_per_component
    req_k = mg.n_components_requested

    print()
    print("=" * 92)
    print(
        f"Map: {mg.product_id}   working year {mg.working_year}   floor {floor}   "
        f"requested K {req_k}   default cov {mg.covariance_type_default}"
    )
    print("=" * 92)

    header = (
        f"{'class':>6}  {'points':>7}  {'K used':>6}  {'K capped':>8}  "
        f"{'cov':>5}  {'under-floor':>11}  {'low-conf':>8}  {'status':>8}  check"
    )
    print(header)
    print("-" * len(header))

    all_ok = True
    for cv in sorted(mg.classes):
        cs = mg.classes[cv]
        problems: list[str] = []

        if cs.absent or not cs.fitted:
            # Absent classes must be skipped, not force-fit.
            if cs.fitted:
                problems.append("absent-but-fitted")
            cov_shown = "-"
            k_shown = "-"
            capped_shown = "-"
        else:
            cov_shown = cs.covariance_type_used
            k_shown = str(cs.n_components_used)
            capped_shown = "yes" if cs.k_capped else "no"

            # Under-floor classes must use diagonal covariance and be low-conf.
            if cs.under_floor and cs.covariance_type_used != "diag":
                problems.append("under-floor-not-diag")
            if cs.under_floor and not cs.low_confidence:
                problems.append("under-floor-not-low-conf")
            # Relaxed classes must be marked low-confidence too.
            if cs.buffer_relaxed and not cs.low_confidence:
                problems.append("relaxed-not-low-conf")
            # At-or-above-floor classes should keep the configured default cov.
            if not cs.under_floor and cs.covariance_type_used != mg.covariance_type_default:
                problems.append("floor-ok-cov-changed")

            # K-capping should match the deterministic rule.
            expected_k = min(req_k, max(1, cs.n_points // min_per))
            if cs.n_components_used != expected_k:
                problems.append(
                    f"K={cs.n_components_used}!=expected {expected_k}"
                )
            if cs.k_capped != (expected_k < req_k):
                problems.append("k-capped-flag-mismatch")

        ok = not problems
        all_ok = all_ok and ok
        verdict = "OK" if ok else "!! " + ", ".join(problems)

        print(
            f"{cv:>6}  {cs.n_points:>7}  {k_shown:>6}  {capped_shown:>8}  "
            f"{cov_shown:>5}  {str(cs.under_floor):>11}  "
            f"{str(cs.low_confidence):>8}  {cs.status:>8}  {verdict}"
        )

    print("-" * len(header))
    print(f"All Stage 3 rules hold for this map: {'YES' if all_ok else 'NO'}")
    return all_ok


def main() -> None:
    print("=" * 92)
    print("Stage 3 verification - GMM modeling")
    print(
        f"requested K={CONFIG.gmm.n_components}  default cov={CONFIG.gmm.covariance_type}  "
        f"min points/component={CONFIG.gmm.min_points_per_component}  "
        f"reg_covar={CONFIG.gmm.reg_covar}  seed={CONFIG.gmm.random_seed}"
    )
    print(
        f"floor={CONFIG.sampling.points_floor} (same threshold Stage 2 used; "
        "under-floor -> diagonal covariance + low-confidence marker)"
    )
    print("=" * 92)

    overall_ok = True
    for product_id in PRODUCTS:
        try:
            mg = fit_map(product_id)
        except FileNotFoundError as exc:
            print()
            print(f"{product_id}: {exc}")
            overall_ok = False
            continue
        except Exception as exc:  # noqa: BLE001 - report and continue
            print()
            print(f"{product_id}: fit failed -> {type(exc).__name__}: {exc}")
            overall_ok = False
            continue

        overall_ok = _print_map(mg) and overall_ok
        path = save_map_gmm(mg)
        print(f"saved -> {path}")

    print()
    print(
        "Confirm: every under-floor class was fit with diagonal covariance and "
        "carries the low-confidence (relaxed / under-floor) flag; absent classes "
        "were skipped (no GMM); and K-capping only happened where the points // "
        "min-per-component rule required it."
    )
    print()
    print(f"ALL CHECKS PASSED: {'YES' if overall_ok else 'NO'}")


if __name__ == "__main__":
    main()
