#!/bin/bash
# Batch conversion of every local-raster product to COGs (tools/to_cog.py).
#
# Run this instead of converting on a login node: the full set is ~600 files of
# a few minutes each, and the shared login nodes routinely sit at load ~100, so
# the work gets almost no CPU there (and long CPU-heavy jobs are not what those
# nodes are for). A dcgp node is CPU-only, which is all this needs -- there is
# no GPU work in a COG conversion.
#
#   sbatch tools/to_cog.slurm
#   squeue -u $USER
#   tail -f logs/to_cog-<jobid>.out
#
# Note the account: this project has two, one per partition family --
# ``iscrc_hsigfm`` for boost (GPU) and ``iscrc_hsigfm_0`` for dcgp (CPU). This
# job runs on dcgp, so it needs the ``_0`` one; submitting with the default
# account fails with a misleading "invalid account or expired budget" even
# though the budget is fine.
#
# The script is restartable: already-valid COGs are skipped, so if the job hits
# the wall clock, just submit it again and it resumes where it stopped.
#
#SBATCH --job-name=to_cog
#SBATCH --account=iscrc_hsigfm_0
#SBATCH --partition=dcgp_usr_prod
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=24:00:00
#SBATCH --output=logs/to_cog-%j.out
#SBATCH --error=logs/to_cog-%j.out

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
mkdir -p logs

# Where the converted tree is written. Must NOT be the home filesystem: that
# volume has a 50 GB quota and is already ~95% full, whereas the work area has
# terabytes free. Override by exporting HARMONIZER_COG_DIR before sbatch.
export HARMONIZER_COG_DIR="${HARMONIZER_COG_DIR:-/leonardo_work/IscrC_HSIGFM/$USER/legend-harmonizer-cog}"

# Keep GDAL's scratch off the small home volume too.
export TMPDIR="${TMPDIR:-/tmp}"

# One worker per pair of cores: each conversion is a mix of decompress (CPU) and
# Lustre reads (I/O), so oversubscribing cores slightly keeps both busy.
WORKERS="${WORKERS:-16}"

echo "host=$(hostname) workers=${WORKERS}"
echo "cog_dir=${HARMONIZER_COG_DIR}"

.venv/bin/python tools/to_cog.py --workers "${WORKERS}" "$@"

echo "done: $(find "${HARMONIZER_COG_DIR}" -name '*.tif' | wc -l) COG(s)"
