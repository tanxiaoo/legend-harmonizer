# Working instructions for this project

This repository is built from a staged plan. Follow these rules every session.

## Before implementing
- Read `docs/PIPELINE.md` first. It is the source of truth for what to build.
- Pull every constant and default value from the "Decisions and default constants"
  section of that document (loaded via `harmonizer/config.py`). Never hardcode
  these values inside a stage, and never invent new ones — if a value is missing,
  ask rather than guessing.

## How to build
- Implement strictly **one stage at a time**, in order (Stage 1 through Stage 6).
  Do not start a later stage until the current one is done and verified.
- A later stage may use earlier stages' modules, but must not modify them without
  saying so.
- After finishing a stage, write the small **verification** described for that
  stage in `docs/PIPELINE.md` (a short script that prints or saves a checkable
  artifact), so the human can confirm it works before moving on.
- Keep changes scoped to the stage being built. Do not scaffold future stages
  ahead of time.

## Method reminders (do not drift from these)
- The pipeline is embedding + GMM throughout. Do not reintroduce a co-occurrence /
  pixel-overlay approach.
- Distribution distance is the closed-form Bures–Wasserstein between Gaussians.
  Do not use `scipy.stats.wasserstein_distance` (it is one-dimensional only).
- HRLC is read locally with rasterio and needs explicit CRS handling; Dynamic
  World and AlphaEarth come from GEE. Only coordinates and small point tables
  cross the network — never rasters.
- Keep the full 64 embedding dimensions; rely on the per-class point floor and the
  K auto-cap rather than dropping dimensions.
- The primary deliverable is the legend matching table (CSV) plus the affinity
  matrix (CSV). The harmonized-raster output is out of scope for the MVP.

## After each stage
- Stop and summarise what was built and how to run its verification.
- Let the human run the verification and commit before you continue.
